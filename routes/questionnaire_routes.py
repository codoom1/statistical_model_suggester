"""
Questionnaire Designer Service Routes
This module provides routes for the questionnaire design service,
allowing users to create, preview, and edit questionnaires.
"""
import hashlib
import logging
import secrets
from datetime import datetime, timedelta, timezone

from flask import (
    Blueprint,
    current_app,
    flash,
    redirect,
    render_template,
    request,
    send_file,
    session,
    url_for,
)
from flask_login import login_required, current_user
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm.attributes import flag_modified

from models import db, Questionnaire, QuestionnaireDraft
from utils.ai_service import is_ai_enabled
from utils.ai_usage import consume_user_ai_quota
from utils.export_utils import export_to_word
from utils.questionnaire_generator import generate_questionnaire

# Try to import PDF export functionality
try:
    from utils.export_utils import export_to_pdf
    PDF_EXPORT_AVAILABLE = True
except ImportError:
    PDF_EXPORT_AVAILABLE = False

logger = logging.getLogger(__name__)
questionnaire_bp = Blueprint('questionnaire', __name__, url_prefix='/questionnaire')

DRAFT_SESSION_KEY = 'questionnaire_draft_id'
DRAFT_MAX_AGE = timedelta(days=7)


def _load_draft():
    """Return the current server-side draft when it belongs to this session."""
    draft_id = session.get(DRAFT_SESSION_KEY)
    if not draft_id:
        return None
    draft = db.session.get(QuestionnaireDraft, draft_id)
    if draft is None:
        session.pop(DRAFT_SESSION_KEY, None)
        return None
    if draft.updated_at < datetime.utcnow() - DRAFT_MAX_AGE:
        db.session.delete(draft)
        db.session.commit()
        session.pop(DRAFT_SESSION_KEY, None)
        return None
    if (
        draft.user_id is not None
        and (
            not current_user.is_authenticated
            or draft.user_id != current_user.id
        )
    ):
        session.pop(DRAFT_SESSION_KEY, None)
        return None
    return draft


def _save_draft(content):
    """Persist a questionnaire working copy and keep only its ID in session."""
    draft = _load_draft()
    if draft is None:
        draft = QuestionnaireDraft(
            id=secrets.token_urlsafe(32),
            user_id=(
                current_user.id if current_user.is_authenticated else None
            ),
            content=content,
        )
        db.session.add(draft)
    else:
        draft.content = content
        flag_modified(draft, 'content')
        if draft.user_id is None and current_user.is_authenticated:
            draft.user_id = current_user.id
    db.session.commit()
    session[DRAFT_SESSION_KEY] = draft.id
    return draft


def _draft_content_or_redirect():
    draft = _load_draft()
    if draft is None:
        flash('Please design a questionnaire first.', 'warning')
        return None
    return draft.content


@questionnaire_bp.route('/')
def index():
    """Landing page for the questionnaire design service."""
    return render_template('questionnaire/index.html')
@questionnaire_bp.route('/design', methods=['GET', 'POST'])
def design():
    """
    GET: Show form to enter research description
    POST: Process form input and generate questionnaire
    """
    if request.method == 'POST':
        research_topic = request.form.get('research_topic', '')
        research_description = request.form.get('research_description', '')
        target_audience = request.form.get('target_audience', '')
        questionnaire_purpose = request.form.get('questionnaire_purpose', '')
        if not all(
            [
                research_topic.strip(),
                research_description.strip(),
                target_audience.strip(),
                questionnaire_purpose.strip(),
            ]
        ):
            flash('Please complete all required questionnaire fields.', 'warning')
            return redirect(url_for('questionnaire.design'))
        # Check if AI enhancement was requested
        use_ai = request.form.get('use_ai_enhancement', 'off') == 'on'
        # Get the total number of focused AI questions to add.
        num_ai_questions = 3 # Default value
        if use_ai:
            if not current_user.is_authenticated:
                flash('Please log in before using AI questionnaire enhancement.', 'info')
                return redirect(url_for('auth.login', next=url_for('questionnaire.design')))
            try:
                num_ai_questions = int(request.form.get('num_ai_questions', 3))
                # Clamp the value between 1 and 5
                num_ai_questions = max(1, min(num_ai_questions, 5))
            except ValueError:
                num_ai_questions = 3 # Fallback to default if conversion fails
            if is_ai_enabled():
                try:
                    allowed, _ = consume_user_ai_quota(current_user.id)
                except SQLAlchemyError:
                    db.session.rollback()
                    logger.exception(
                        "Could not record questionnaire AI usage for user %s.",
                        current_user.id,
                    )
                    flash(
                        'AI usage tracking is not initialized. Please contact the administrator.',
                        'danger',
                    )
                    return redirect(url_for('questionnaire.design'))
                if not allowed:
                    flash(
                        'You have reached the hourly AI usage limit. Please try again later.',
                        'warning',
                    )
                    return redirect(url_for('questionnaire.design'))
        safety_identifier = None
        if use_ai and current_user.is_authenticated:
            safety_identifier = hashlib.sha256(
                (
                    f"{current_app.config['SECRET_KEY']}:"
                    f"{current_user.id}"
                ).encode()
            ).hexdigest()
        # Generate questionnaire based on research description
        questionnaire = generate_questionnaire(
            research_description,
            research_topic,
            target_audience,
            questionnaire_purpose,
            use_ai_enhancement=use_ai,
            num_ai_questions=num_ai_questions,
            safety_identifier=safety_identifier,
        )
        ai_applied = any(
            question.get('ai_created') or question.get('ai_enhanced')
            for section in questionnaire
            for question in section.get('questions', [])
        )
        if use_ai and not ai_applied:
            flash(
                'AI enhancement was unavailable, so a complete rules-based '
                'questionnaire was generated instead.',
                'warning',
            )
        _save_draft({
            'questionnaire': questionnaire,
            'research_topic': research_topic,
            'research_description': research_description,
            'target_audience': target_audience,
            'questionnaire_purpose': questionnaire_purpose,
            'used_ai_enhancement': ai_applied,
        })
        return redirect(url_for('questionnaire.preview'))
    return render_template('questionnaire/design.html')
@questionnaire_bp.route('/preview')
def preview():
    """Preview the generated questionnaire."""
    draft = _draft_content_or_redirect()
    if draft is None:
        return redirect(url_for('questionnaire.design'))
    return render_template(
        'questionnaire/preview.html',
        questionnaire=draft['questionnaire'],
        research_topic=draft.get('research_topic', ''),
        research_description=draft.get('research_description', ''),
        target_audience=draft.get('target_audience', ''),
        questionnaire_purpose=draft.get('questionnaire_purpose', '')
    )
@questionnaire_bp.route('/edit', methods=['GET', 'POST'])
def edit():
    """
    GET: Show form to edit questionnaire
    POST: Process edits and update the questionnaire
    """
    draft = _draft_content_or_redirect()
    if draft is None:
        return redirect(url_for('questionnaire.design'))
    if request.method == 'POST':
        # Process the form data
        research_topic = request.form.get('research_topic', '')
        target_audience = request.form.get('target_audience', '')
        questionnaire_purpose = request.form.get('questionnaire_purpose', '')
        research_description = request.form.get('research_description', '')
        # Process sections data from the form
        sections_data = []
        form_data = request.form.to_dict(flat=False)
        # Get all unique section indices from form data
        section_indices = set()
        for key in form_data:
            if key.startswith('sections[') and '][title]' in key:
                section_index = key.split('[')[1].split(']')[0]
                section_indices.add(section_index)
        # Sort section indices to maintain order
        section_indices = sorted(section_indices, key=int)
        # Process each section
        for section_index in section_indices:
            section_title = request.form.get(f'sections[{section_index}][title]', '')
            section_description = request.form.get(f'sections[{section_index}][description]', '')
            # Get questions for this section
            questions = []
            question_indices = set()
            # Find all question indices for this section
            for key in form_data:
                if key.startswith(f'sections[{section_index}][questions][') and '][text]' in key:
                    question_index = key.split('[')[3].split(']')[0]
                    question_indices.add(question_index)
            # Sort question indices to maintain order
            question_indices = sorted(question_indices, key=int)
            # Process each question
            for question_index in question_indices:
                question_text = request.form.get(f'sections[{section_index}][questions][{question_index}][text]', '')
                question_type = request.form.get(f'sections[{section_index}][questions][{question_index}][type]', '')
                # Get options if applicable
                options = []
                if question_type in ['Multiple Choice', 'Checkbox']:
                    option_indices = set()
                    # Find all option indices for this question
                    for key in form_data:
                        if key.startswith(f'sections[{section_index}][questions][{question_index}][options]['):
                            option_index = key.split('[')[5].split(']')[0]
                            option_indices.add(option_index)
                    # Sort option indices to maintain order
                    option_indices = sorted(option_indices, key=int)
                    # Process each option
                    for option_index in option_indices:
                        option_text = request.form.get(f'sections[{section_index}][questions][{question_index}][options][{option_index}]', '')
                        if option_text:
                            options.append(option_text)
                # Preserve AI flags if they exist
                ai_enhanced = False
                ai_created = False
                # Check if this question was in the original questionnaire
                original_questionnaire = draft.get('questionnaire', [])
                if int(section_index) < len(original_questionnaire):
                    original_section = original_questionnaire[int(section_index)]
                    original_questions = original_section.get('questions', [])
                    if int(question_index) < len(original_questions):
                        original_question = original_questions[int(question_index)]
                        ai_enhanced = original_question.get('ai_enhanced', False)
                        ai_created = original_question.get('ai_created', False)
                question_data: dict = {
                    'text': question_text,
                    'type': question_type
                }
                if options:
                    question_data['options'] = options  # type: ignore
                if ai_enhanced:
                    question_data['ai_enhanced'] = True  # type: ignore
                if ai_created:
                    question_data['ai_created'] = True  # type: ignore
                if question_text:  # Only add non-empty questions
                    questions.append(question_data)
            if section_title:  # Only add sections with a title
                sections_data.append({
                    'title': section_title,
                    'description': section_description,
                    'questions': questions
                })
        draft.update({
            'questionnaire': sections_data,
            'research_topic': research_topic,
            'target_audience': target_audience,
            'questionnaire_purpose': questionnaire_purpose,
            'research_description': research_description,
        })
        _save_draft(draft)
        flash('Questionnaire updated successfully.', 'success')
        return redirect(url_for('questionnaire.preview'))
    return render_template(
        'questionnaire/edit.html',
        questionnaire=draft['questionnaire'],
        research_topic=draft.get('research_topic', ''),
        research_description=draft.get('research_description', ''),
        target_audience=draft.get('target_audience', ''),
        questionnaire_purpose=draft.get('questionnaire_purpose', '')
    )
@questionnaire_bp.route('/save', methods=['POST'])
@login_required
def save_questionnaire():
    """Save the current questionnaire to the database."""
    draft = _draft_content_or_redirect()
    if draft is None:
        return redirect(url_for('questionnaire.design'))
    questionnaire_data = draft['questionnaire']
    research_topic = draft.get('research_topic', 'Untitled Questionnaire')
    research_description = draft.get('research_description', '')
    target_audience = draft.get('target_audience', '')
    questionnaire_purpose = draft.get('questionnaire_purpose', '')
    is_ai_enhanced = draft.get('used_ai_enhancement', False)
    try:
        # Check if we're updating an existing questionnaire
        questionnaire_id = request.form.get('questionnaire_id')
        if questionnaire_id:
            # Find the existing questionnaire
            questionnaire = Questionnaire.query.filter_by(
                id=questionnaire_id,
                user_id=current_user.id
            ).first()
            if not questionnaire:
                flash('Questionnaire not found or you do not have permission to edit it.', 'error')
                return redirect(url_for('questionnaire.preview'))
            # Update the existing questionnaire
            questionnaire.title = research_topic
            questionnaire.topic = research_topic
            questionnaire.description = research_description
            questionnaire.target_audience = target_audience
            questionnaire.purpose = questionnaire_purpose
            questionnaire.content = questionnaire_data
            questionnaire.is_ai_enhanced = is_ai_enhanced
            questionnaire.updated_at = datetime.now(timezone.utc)
        else:
            # Create a new questionnaire
            questionnaire = Questionnaire(
                user_id=current_user.id,  # type: ignore
                title=research_topic,  # type: ignore
                topic=research_topic,  # type: ignore
                description=research_description,  # type: ignore
                target_audience=target_audience,  # type: ignore
                purpose=questionnaire_purpose,  # type: ignore
                content=questionnaire_data,  # type: ignore
                is_ai_enhanced=is_ai_enhanced  # type: ignore
            )
            db.session.add(questionnaire)
        db.session.commit()
        flash('Questionnaire saved successfully!', 'success')
        # Store the questionnaire ID in the session
        session['saved_questionnaire_id'] = questionnaire.id
        return redirect(url_for('questionnaire.preview'))
    except Exception as e:
        db.session.rollback()
        logger.error(f"Error saving questionnaire: {e}")
        flash(f'Error saving questionnaire: {str(e)}', 'error')
        return redirect(url_for('questionnaire.preview'))
@questionnaire_bp.route('/my-questionnaires')
@login_required
def my_questionnaires():
    """List all questionnaires created by the current user."""
    questionnaires = Questionnaire.query.filter_by(user_id=current_user.id).order_by(Questionnaire.created_at.desc()).all()
    return render_template('questionnaire/my_questionnaires.html', questionnaires=questionnaires)
@questionnaire_bp.route('/load/<int:questionnaire_id>')
@login_required
def load_questionnaire(questionnaire_id):
    """Load a saved questionnaire from the database."""
    questionnaire = Questionnaire.query.filter_by(id=questionnaire_id, user_id=current_user.id).first()
    if not questionnaire:
        flash('Questionnaire not found or you do not have permission to view it.', 'error')
        return redirect(url_for('questionnaire.my_questionnaires'))
    _save_draft({
        'questionnaire': questionnaire.content,
        'research_topic': questionnaire.title,
        'research_description': questionnaire.description,
        'target_audience': questionnaire.target_audience,
        'questionnaire_purpose': questionnaire.purpose,
        'used_ai_enhancement': questionnaire.is_ai_enhanced,
    })
    session['saved_questionnaire_id'] = questionnaire.id
    return redirect(url_for('questionnaire.preview'))
@questionnaire_bp.route('/delete/<int:questionnaire_id>', methods=['POST'])
@login_required
def delete_questionnaire(questionnaire_id):
    """Delete a saved questionnaire."""
    questionnaire = Questionnaire.query.filter_by(id=questionnaire_id, user_id=current_user.id).first()
    if not questionnaire:
        flash('Questionnaire not found or you do not have permission to delete it.', 'error')
        return redirect(url_for('questionnaire.my_questionnaires'))
    try:
        db.session.delete(questionnaire)
        db.session.commit()
        flash('Questionnaire deleted successfully.', 'success')
    except Exception as e:
        db.session.rollback()
        logger.error(f"Error deleting questionnaire: {e}")
        flash(f'Error deleting questionnaire: {str(e)}', 'error')
    return redirect(url_for('questionnaire.my_questionnaires'))
@questionnaire_bp.route('/export/word')
def export_word():
    """Export questionnaire to Word document."""
    draft = _draft_content_or_redirect()
    if draft is None:
        return redirect(url_for('questionnaire.design'))
    questionnaire = draft['questionnaire']
    research_topic = draft.get('research_topic', 'Questionnaire')
    research_description = draft.get('research_description', '')
    target_audience = draft.get('target_audience', '')
    questionnaire_purpose = draft.get('questionnaire_purpose', '')
    # Generate a filename
    filename = f"{research_topic.replace(' ', '_')}_questionnaire.docx"
    # Create the Word document
    file_path = export_to_word(
        questionnaire,
        research_topic,
        research_description,
        target_audience,
        questionnaire_purpose
    )
    # Send the file to the user
    return send_file(
        file_path,
        as_attachment=True,
        download_name=filename,
        mimetype='application/vnd.openxmlformats-officedocument.wordprocessingml.document'
    )
@questionnaire_bp.route('/export/pdf')
def export_pdf():
    """Export questionnaire to PDF document."""
    # Check if PDF export is available
    if not PDF_EXPORT_AVAILABLE:
        flash('PDF export is not available. Please install reportlab package.', 'error')
        return redirect(url_for('questionnaire.preview'))
    
    draft = _draft_content_or_redirect()
    if draft is None:
        return redirect(url_for('questionnaire.design'))
    questionnaire = draft['questionnaire']
    research_topic = draft.get('research_topic', 'Questionnaire')
    research_description = draft.get('research_description', '')
    target_audience = draft.get('target_audience', '')
    questionnaire_purpose = draft.get('questionnaire_purpose', '')
    # Generate a filename
    filename = f"{research_topic.replace(' ', '_')}_questionnaire.pdf"
    # Create the PDF document
    file_path = export_to_pdf(
        questionnaire,
        research_topic,
        research_description,
        target_audience,
        questionnaire_purpose
    )
    # Send the file to the user
    return send_file(
        file_path,
        as_attachment=True,
        download_name=filename,
        mimetype='application/pdf'
    )
