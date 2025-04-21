"""
Questionnaire Designer Service Routes

This module provides routes for the questionnaire design service,
allowing users to create, preview, and edit questionnaires.
"""

from flask import Blueprint, render_template, request, redirect, url_for, session, flash, Response, send_file, jsonify
from flask_login import login_required, current_user
from utils.questionnaire_generator import generate_questionnaire
from utils.export_utils import export_to_word, export_to_pdf
from models import db, Questionnaire
import json
import os
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

questionnaire_bp = Blueprint('questionnaire', __name__, url_prefix='/questionnaire')

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
        
        # Check if AI enhancement was requested
        use_ai = request.form.get('use_ai_enhancement', 'off') == 'on'
        
        # Get the number of AI questions per type (default to 3 if not provided or not using AI)
        num_ai_questions = 3 # Default value
        if use_ai:
            try:
                num_ai_questions = int(request.form.get('num_ai_questions', 3))
                # Clamp the value between 1 and 5
                num_ai_questions = max(1, min(num_ai_questions, 5))
            except ValueError:
                num_ai_questions = 3 # Fallback to default if conversion fails
        
        # Generate questionnaire based on research description
        questionnaire = generate_questionnaire(
            research_description,
            research_topic,
            target_audience,
            questionnaire_purpose,
            use_ai_enhancement=use_ai,
            num_ai_questions=num_ai_questions
        )
        
        # Store questionnaire data in session
        session['questionnaire'] = questionnaire
        session['research_topic'] = research_topic
        session['research_description'] = research_description
        session['target_audience'] = target_audience
        session['questionnaire_purpose'] = questionnaire_purpose
        session['used_ai_enhancement'] = use_ai
        
        return redirect(url_for('questionnaire.preview'))
    
    return render_template('questionnaire/design.html')

@questionnaire_bp.route('/preview')
def preview():
    """Preview the generated questionnaire."""
    # Check if questionnaire data exists in session
    if 'questionnaire' not in session:
        flash('Please design a questionnaire first.', 'error')
        return redirect(url_for('questionnaire.design'))
    
    return render_template(
        'questionnaire/preview.html',
        questionnaire=session['questionnaire'],
        research_topic=session.get('research_topic', ''),
        research_description=session.get('research_description', ''),
        target_audience=session.get('target_audience', ''),
        questionnaire_purpose=session.get('questionnaire_purpose', '')
    )

@questionnaire_bp.route('/edit', methods=['GET', 'POST'])
def edit():
    """
    GET: Show form to edit questionnaire
    POST: Process edits and update the questionnaire
    """
    # Check if questionnaire data exists in session
    if 'questionnaire' not in session:
        flash('Please design a questionnaire first.', 'error')
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
                original_questionnaire = session.get('questionnaire', [])
                if int(section_index) < len(original_questionnaire):
                    original_section = original_questionnaire[int(section_index)]
                    original_questions = original_section.get('questions', [])
                    if int(question_index) < len(original_questions):
                        original_question = original_questions[int(question_index)]
                        ai_enhanced = original_question.get('ai_enhanced', False)
                        ai_created = original_question.get('ai_created', False)
                
                question_data = {
                    'text': question_text,
                    'type': question_type
                }
                
                if options:
                    question_data['options'] = options
                
                if ai_enhanced:
                    question_data['ai_enhanced'] = True
                
                if ai_created:
                    question_data['ai_created'] = True
                
                if question_text:  # Only add non-empty questions
                    questions.append(question_data)
            
            if section_title:  # Only add sections with a title
                sections_data.append({
                    'title': section_title,
                    'description': section_description,
                    'questions': questions
                })
        
        # Update session with edited data
        session['questionnaire'] = sections_data
        session['research_topic'] = research_topic
        session['target_audience'] = target_audience
        session['questionnaire_purpose'] = questionnaire_purpose
        session['research_description'] = research_description
        
        flash('Questionnaire updated successfully.', 'success')
        return redirect(url_for('questionnaire.preview'))
    
    return render_template(
        'questionnaire/edit.html',
        questionnaire=session['questionnaire'],
        research_topic=session.get('research_topic', ''),
        research_description=session.get('research_description', ''),
        target_audience=session.get('target_audience', ''),
        questionnaire_purpose=session.get('questionnaire_purpose', '')
    )

@questionnaire_bp.route('/save', methods=['POST'])
@login_required
def save_questionnaire():
    """Save the current questionnaire to the database."""
    # Check if questionnaire data exists in session
    if 'questionnaire' not in session:
        flash('Please design a questionnaire first.', 'error')
        return redirect(url_for('questionnaire.design'))
    
    # Get questionnaire data from session
    questionnaire_data = session['questionnaire']
    research_topic = session.get('research_topic', 'Untitled Questionnaire')
    research_description = session.get('research_description', '')
    target_audience = session.get('target_audience', '')
    questionnaire_purpose = session.get('questionnaire_purpose', '')
    is_ai_enhanced = session.get('used_ai_enhancement', False)
    
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
            questionnaire.updated_at = datetime.utcnow()
            
        else:
            # Create a new questionnaire
            questionnaire = Questionnaire(
                user_id=current_user.id,
                title=research_topic,
                topic=research_topic,
                description=research_description,
                target_audience=target_audience,
                purpose=questionnaire_purpose,
                content=questionnaire_data,
                is_ai_enhanced=is_ai_enhanced
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
    
    # Store questionnaire data in session
    session['questionnaire'] = questionnaire.content
    session['research_topic'] = questionnaire.title
    session['research_description'] = questionnaire.description
    session['target_audience'] = questionnaire.target_audience
    session['questionnaire_purpose'] = questionnaire.purpose
    session['used_ai_enhancement'] = questionnaire.is_ai_enhanced
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
    # Check if questionnaire data exists in session
    if 'questionnaire' not in session:
        flash('Please design a questionnaire first.', 'error')
        return redirect(url_for('questionnaire.design'))
    
    # Get questionnaire data from session
    questionnaire = session['questionnaire']
    research_topic = session.get('research_topic', 'Questionnaire')
    research_description = session.get('research_description', '')
    target_audience = session.get('target_audience', '')
    questionnaire_purpose = session.get('questionnaire_purpose', '')
    
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
    # Check if questionnaire data exists in session
    if 'questionnaire' not in session:
        flash('Please design a questionnaire first.', 'error')
        return redirect(url_for('questionnaire.design'))
    
    # Get questionnaire data from session
    questionnaire = session['questionnaire']
    research_topic = session.get('research_topic', 'Questionnaire')
    research_description = session.get('research_description', '')
    target_audience = session.get('target_audience', '')
    questionnaire_purpose = session.get('questionnaire_purpose', '')
    
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