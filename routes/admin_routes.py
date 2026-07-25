from flask import Blueprint, render_template, request, redirect, url_for, flash, jsonify
from flask_login import login_required, current_user
from models import (
    db,
    User,
    ExpertApplication,
    Analysis,
    Consultation,
)
from functools import wraps
from datetime import datetime, timedelta, timezone
from utils.email_service import send_expert_approved_email, send_expert_rejected_email
from utils.ai_service import (
    call_openai_api, is_ai_enabled, OpenAIServiceError, get_openai_config
)
from utils.ai_usage import (
    ai_usage_storage_ready,
    initialize_ai_usage_storage,
)
from utils.validation import (
    is_valid_email,
    is_valid_username,
    normalize_email,
)
from sqlalchemy.exc import SQLAlchemyError
import logging
import re
admin = Blueprint('admin', __name__, url_prefix='/admin')
logger = logging.getLogger(__name__)
# Custom decorator for admin access
def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not current_user.is_authenticated or not current_user.is_admin:
            flash('You need to be an admin to access this page.', 'danger')
            return redirect(url_for('main.home'))
        return f(*args, **kwargs)
    return decorated_function
@admin.route('/')
@login_required
@admin_required
def index():
    """Admin dashboard index"""
    return redirect(url_for('admin.dashboard'))
@admin.route('/dashboard')
@login_required
@admin_required
def dashboard():
    """Admin dashboard with analytics"""
    # Count of all users by role
    admin_users_count = User.query.filter_by(_is_admin=True).count()
    experts_count = User.query.filter_by(_is_expert=True, is_approved_expert=True).count()
    regular_users_count = User.query.filter_by(_is_admin=False, _is_expert=False).count()
    active_consultations_count = Consultation.query.filter_by(status='in_progress').count()
    # Get recent users
    recent_users = User.query.order_by(User.created_at.desc()).limit(10).all()
    # Get pending expert applications (including those needing more info)
    pending_applications = ExpertApplication.query.filter(
        ExpertApplication.status.in_(['pending', 'needs_info'])
    ).all()
    # User growth over time (last 6 months)
    end_date = datetime.now(timezone.utc)
    start_date = end_date - timedelta(days=180)
    months = []
    regular_users_growth = []
    experts_growth = []
    current_date = start_date.replace(day=1)  # Start at beginning of month and keep timezone info
    while current_date <= end_date:
        month_end = datetime(current_date.year, current_date.month, 1, tzinfo=timezone.utc) + timedelta(days=32)
        month_end = datetime(month_end.year, month_end.month, 1, tzinfo=timezone.utc) - timedelta(days=1)
        # Format date string
        month_str = current_date.strftime('%b %Y')
        months.append(month_str)
        # Count users created before this month end
        regular_count = User.query.filter(
            User._is_admin.is_(False),
            User._is_expert.is_(False),
            User.created_at <= month_end
        ).count()
        expert_count = User.query.filter(
            User._is_expert.is_(True),
            User.is_approved_expert.is_(True),
            User.created_at <= month_end
        ).count()
        regular_users_growth.append(regular_count)
        experts_growth.append(expert_count)
        # Move to next month
        next_month = current_date.replace(day=1) + timedelta(days=32)
        current_date = next_month.replace(day=1)
    return render_template(
        'admin/dashboard.html',
        admin_users_count=admin_users_count,
        experts_count=experts_count,
        regular_users_count=regular_users_count,
        active_consultations_count=active_consultations_count,
        recent_users=recent_users,
        pending_applications=pending_applications,
        user_growth_dates=months,
        regular_users_growth=regular_users_growth,
        experts_growth=experts_growth
    )
@admin.route('/users')
@login_required
@admin_required
def users_list():
    """List all users"""
    users = User.query.order_by(User.created_at.desc()).all()
    return render_template('admin/users_list.html', users=users)
@admin.route('/edit-user/<int:user_id>', methods=['GET', 'POST'])
@login_required
@admin_required
def edit_user(user_id):
    """Edit user details"""
    user = User.query.get_or_404(user_id)
    if request.method == 'POST':
        # Update user information
        username = request.form.get('username', '').strip()
        email = normalize_email(request.form.get('email'))
        if not is_valid_username(username) or not is_valid_email(email):
            flash('Provide a valid username and email address.', 'danger')
            return render_template('admin/edit_user.html', user=user), 400
        user.username = username
        user.email = email
        role = request.form.get('role', 'user')
        user._is_admin = role == 'admin'
        if role == 'expert':
            user._is_expert = True
            user.is_approved_expert = request.form.get('is_approved_expert') == 'on'
            user.areas_of_expertise = request.form.get(
                'areas_of_expertise',
                request.form.get('expertise', ''),
            )
            user.institution = request.form.get('institution')
            user.bio = request.form.get('bio')
        else:
            user._is_expert = False
            user.is_approved_expert = False
        db.session.commit()
        flash(f'User {user.username} updated successfully.', 'success')
        return redirect(url_for('admin.users_list'))
    return render_template('admin/edit_user.html', user=user)
@admin.route('/delete-user/<int:user_id>', methods=['POST'])
@login_required
@admin_required
def delete_user(user_id):
    """Delete a user account"""
    if current_user.id == user_id:
        flash('You cannot delete your own admin account here.', 'danger')
        return redirect(url_for('admin.users_list'))
    user = User.query.get_or_404(user_id)
    # Delete user's analyses
    Analysis.query.filter_by(user_id=user.id).delete()
    # Delete user
    db.session.delete(user)
    db.session.commit()
    flash(f'User {user.username} has been deleted.', 'success')
    return redirect(url_for('admin.users_list'))
@admin.route('/expert-applications')
@login_required
@admin_required
def expert_applications():
    """List expert applications"""
    # Get all applications, not just pending ones
    applications = ExpertApplication.query.all()
    return render_template('admin/expert_applications.html', applications=applications)
@admin.route('/approve-expert/<int:user_id>', methods=['POST'])
@login_required
@admin_required
def approve_expert(user_id):
    """Approve an expert application"""
    # Find application regardless of status
    application = ExpertApplication.query.filter_by(user_id=user_id).first_or_404()
    user = application.user
    user._is_expert = True
    user.is_approved_expert = True
    user.areas_of_expertise = application.areas_of_expertise
    user.institution = application.institution
    user.bio = application.bio
    application.status = 'approved'
    db.session.commit()
    # Send approval email
    send_expert_approved_email(user, application.email)
    flash(f'Expert {user.username} has been approved.', 'success')
    return redirect(url_for('admin.expert_applications'))
@admin.route('/reject-expert/<int:user_id>', methods=['POST'])
@login_required
@admin_required
def reject_expert(user_id):
    """Reject an expert application"""
    # Find application regardless of status
    application = ExpertApplication.query.filter_by(user_id=user_id).first_or_404()
    user = application.user
    user._is_expert = False
    user.is_approved_expert = False
    application.status = 'rejected'
    db.session.commit()
    # Send rejection email
    send_expert_rejected_email(user, application.email)
    flash(f'Expert application from {user.username} has been rejected.', 'info')
    return redirect(url_for('admin.expert_applications'))
@admin.route('/manage-experts')
@login_required
@admin_required
def manage_experts():
    """Manage existing expert accounts"""
    experts = User.query.filter_by(_is_expert=True, is_approved_expert=True).all()
    return render_template('admin/manage_experts.html', experts=experts)
@admin.route('/revoke-expert/<int:user_id>', methods=['POST'])
@login_required
@admin_required
def revoke_expert(user_id):
    """Revoke expert status"""
    user = User.query.get_or_404(user_id)
    if not user._is_expert or not user.is_approved_expert:
        flash('This user is not an approved expert.', 'warning')
        return redirect(url_for('admin.manage_experts'))
    user._is_expert = False
    user.is_approved_expert = False
    db.session.commit()
    flash(f'Expert status for {user.username} has been revoked.', 'success')
    return redirect(url_for('admin.manage_experts'))
@admin.route('/email-templates')
@login_required
@admin_required
def email_templates():
    """View email templates used for expert applications"""
    return render_template('admin/email_templates.html')
@admin.route('/consultations')
@login_required
@admin_required
def consultations_list():
    """List all consultations"""
    consultations = Consultation.query.order_by(Consultation.created_at.desc()).all()
    experts = User.query.filter_by(_is_expert=True, is_approved_expert=True).all()
    return render_template(
        'admin/consultations_list.html',
        consultations=consultations,
        experts=experts
    )
@admin.route('/assign-consultation/<int:consultation_id>', methods=['POST'])
@login_required
@admin_required
def assign_consultation(consultation_id):
    """Assign a consultation to an expert"""
    consultation = Consultation.query.get_or_404(consultation_id)
    expert_id = request.form.get('expert_id', type=int)
    if not expert_id:
        flash('No expert selected.', 'danger')
        return redirect(url_for('admin.consultations_list'))
    expert = User.query.filter_by(id=expert_id, _is_expert=True, is_approved_expert=True).first()
    if not expert:
        flash('Selected expert not found or not approved.', 'danger')
        return redirect(url_for('admin.consultations_list'))
    consultation.expert_id = expert.id
    consultation.status = 'in_progress'
    db.session.commit()
    flash(f'Consultation assigned to {expert.username}.', 'success')
    return redirect(url_for('admin.consultations_list'))
@admin.route('/ai_settings', methods=['GET'])
@login_required
@admin_required
def ai_settings():
    """Display non-secret AI integration status."""
    current_api_key, current_model = get_openai_config()
    current_enabled = is_ai_enabled()
    storage_ready = ai_usage_storage_ready()
    current_settings = {
        'api_key_configured': bool(current_api_key),
        'model': current_model,
        'enabled': current_enabled,
        'storage_ready': storage_ready,
    }
    if not current_enabled:
        api_status = {
            'status': 'disabled',
            'label': 'Disabled',
            'last_error': 'AI enhancement is disabled.',
        }
    elif not current_api_key:
        api_status = {
            'status': 'no_key',
            'label': 'Missing API token',
            'last_error': 'OPENAI_API_KEY is missing.',
        }
    elif not storage_ready:
        api_status = {
            'status': 'needs_setup',
            'label': 'Needs initialization',
            'last_error': (
                'The AI usage database table has not been initialized.'
            ),
        }
    else:
        api_status = {
            'status': 'ready',
            'label': 'Ready',
            'last_error': None,
        }
    return render_template(
        'admin/ai_settings.html',
        current_settings=current_settings,
        api_status=api_status,
    )


@admin.route('/initialize-ai-storage', methods=['POST'])
@login_required
@admin_required
def initialize_ai_storage():
    """Initialize durable AI usage storage for serverless deployments."""
    data = request.get_json(silent=True)
    if not isinstance(data, dict) or data.get('confirm') is not True:
        return jsonify({
            'success': False,
            'error': 'Explicit initialization confirmation is required.',
        }), 400

    try:
        initialize_ai_usage_storage()
    except SQLAlchemyError:
        logger.exception(
            "AI storage initialization failed for admin user %s.",
            current_user.id,
        )
        return jsonify({
            'success': False,
            'error': (
                'The AI usage table could not be created. Verify that '
                'DATABASE_URL is correct and permits schema changes.'
            ),
        }), 503

    return jsonify({
        'success': True,
        'message': 'AI usage storage is initialized.',
        'storage_ready': ai_usage_storage_ready(),
    })


@admin.route('/test-ai-integration', methods=['POST'])
@login_required
@admin_required
def test_ai_integration():
    """Test the server-side AI credentials without exposing or replacing them."""
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({'success': False, 'error': 'No JSON data provided'}), 400
    test_prompt = str(data.get('prompt', '')).strip()
    if not test_prompt:
        return jsonify(
            {'success': False, 'error': 'Please provide a test prompt'}
        ), 400
    if len(test_prompt) > 1000:
        return jsonify(
            {'success': False, 'error': 'Test prompts must be 1000 characters or fewer.'}
        ), 400

    try:
        enhanced_text = call_openai_api(
            test_prompt,
            system_prompt=(
                'You are an expert questionnaire designer. Improve the user '
                'question so it is specific, neutral, and measurable. Return '
                'only the improved question.'
            ),
        )
        return jsonify({
            'success': True,
            'original': test_prompt,
            'enhanced': enhanced_text
        })
    except OpenAIServiceError as exc:
        status_code = exc.status_code if exc.status_code in {402, 429, 503, 504} else 502
        return jsonify({'success': False, 'error': str(exc)}), status_code


@admin.route('/update-application-notes/<int:application_id>', methods=['POST'])
@login_required
@admin_required
def update_application_notes(application_id):
    """Update admin notes for an expert application"""
    application = ExpertApplication.query.get_or_404(application_id)
    admin_notes = request.form.get('admin_notes', '')
    application.admin_notes = admin_notes
    db.session.commit()
    flash('Application notes updated successfully.', 'success')
    return redirect(url_for('admin.expert_applications'))
@admin.route('/request-resume/<int:application_id>', methods=['POST'])
@login_required
@admin_required
def request_resume(application_id):
    """Request resume from expert applicant"""
    application = ExpertApplication.query.get_or_404(application_id)
    application.status = 'needs_info'
    if not application.admin_notes:
        application.admin_notes = "Please upload your resume/CV to continue the evaluation process."
    else:
        application.admin_notes += "\n\nPlease upload your resume/CV to continue the evaluation process."
    db.session.commit()
    flash('Resume requested from applicant.', 'success')
    return redirect(url_for('admin.expert_applications'))
@admin.route('/request-additional-info/<int:application_id>', methods=['POST'])
@login_required
@admin_required
def request_additional_info(application_id):
    """Request additional information from expert applicant"""
    application = ExpertApplication.query.get_or_404(application_id)
    user = application.user
    # Set status to needs_info
    application.status = 'needs_info'
    # Ensure user's expert status is properly set
    if user.is_approved_expert:
        # If they were previously approved, revoke their expert status until info is provided
        user.is_approved_expert = False
    # If no admin notes are present, add a default message
    if not application.admin_notes:
        application.admin_notes = "The review committee needs additional information to process your application. Please provide the requested details."
    # Add timestamp to the notes
    current_time = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')
    application.admin_notes = f"{application.admin_notes}\n\n--- Admin Request ({current_time}) ---\nPlease provide more information about your expertise and experience."
    db.session.commit()
    # Optional: Send email notification to the applicant
    # send_application_update_email(application.user, "Additional Information Requested",
    #    "The review committee has requested additional information for your expert application. Please log in to view details.")
    flash('Additional information requested from applicant.', 'success')
    return redirect(url_for('admin.expert_applications'))
@admin.route('/process-additional-info/<int:application_id>', methods=['POST'])
@login_required
@admin_required
def process_additional_info(application_id):
    """Process additional information from an expert applicant"""
    application = ExpertApplication.query.get_or_404(application_id)
    action = request.form.get('action', '')
    if action == 'approve':
        # Approve the expert
        user = application.user
        user._is_expert = True
        user.is_approved_expert = True
        user.areas_of_expertise = application.areas_of_expertise
        user.institution = application.institution
        user.bio = application.bio
        application.status = 'approved'
        db.session.commit()
        # Send approval email
        send_expert_approved_email(user, application.email)
        flash(f'Expert {user.username} has been approved.', 'success')
    elif action == 'request_more_info':
        # Redirect to request more info page
        return redirect(url_for('admin.request_additional_info', application_id=application_id))
    elif action == 'reject':
        # Reject the expert
        user = application.user
        user._is_expert = False
        user.is_approved_expert = False
        application.status = 'rejected'
        db.session.commit()
        # Send rejection email
        send_expert_rejected_email(user, application.email)
        flash(f'Expert application for {user.username} has been rejected.', 'success')
    return redirect(url_for('admin.expert_applications'))
@admin.route('/application-details/<int:application_id>')
@login_required
@admin_required
def application_details(application_id):
    """View detailed expert application with communication history"""
    application = ExpertApplication.query.get_or_404(application_id)
    # Parse and format the communication history
    conversation_history = []
    if application.admin_notes:
        # Split the notes by the markers
        notes_content = application.admin_notes
        admin_pattern = r'---\s*Admin\s*Request\s*\(([^)]+)\)\s*---\s*([^-]*)'
        expert_pattern = r'---\s*Expert\s*Response\s*\(([^)]+)\)\s*---\s*([^-]*)'
        # Extract admin requests
        admin_requests = re.finditer(admin_pattern, notes_content, re.DOTALL)
        for match in admin_requests:
            timestamp = match.group(1)
            message = match.group(2).strip()
            if message:  # Only add if there's actual content
                conversation_history.append({
                    'role': 'admin',
                    'author': 'Admin',
                    'timestamp': timestamp,
                    'message': message
                })
        # Extract expert responses
        expert_responses = re.finditer(expert_pattern, notes_content, re.DOTALL)
        for match in expert_responses:
            timestamp = match.group(1)
            message = match.group(2).strip()
            if message:  # Only add if there's actual content
                conversation_history.append({
                    'role': 'expert',
                    'author': application.user.username,
                    'timestamp': timestamp,
                    'message': message
                })
        # Sort the conversation by timestamp
        conversation_history.sort(key=lambda x: datetime.strptime(x['timestamp'], '%Y-%m-%d %H:%M:%S'))
    return render_template('admin/application_details.html',
                          application=application,
                          conversation_history=conversation_history)
