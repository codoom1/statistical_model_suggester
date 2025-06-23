from flask import Blueprint, render_template, request, redirect, url_for, flash, current_app
from flask_login import login_required, current_user
from models import db, User, Analysis, Consultation, ExpertApplication
from datetime import datetime, timezone
from functools import wraps
from utils.email_service import send_expert_approved_email, send_expert_rejected_email
import re
expert = Blueprint('expert', __name__)
# Custom decorator for expert access
def expert_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not current_user.is_authenticated or not current_user.is_expert:
            flash('You need to be an approved expert to access this page.', 'danger')
            return redirect(url_for('main.index'))
        return f(*args, **kwargs)
    return decorated_function
# Custom decorator for admin access
def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not current_user.is_authenticated or not current_user.is_admin:
            flash('You need to be an admin to access this page.', 'danger')
            return redirect(url_for('main.index'))
        return f(*args, **kwargs)
    return decorated_function
@expert.route('/experts')
def experts_list():
    """Display list of all approved experts"""
    experts = User.query.filter_by(_is_expert=True, is_approved_expert=True).all()
    return render_template('experts_list.html', experts=experts)
@expert.route('/expert/<int:expert_id>')
def expert_profile(expert_id):
    """View an expert's profile"""
    expert = User.query.filter_by(id=expert_id, _is_expert=True, is_approved_expert=True).first_or_404()
    # Get public consultations by this expert
    consultations = Consultation.query.filter_by(
        expert_id=expert_id,
        is_public=True,
        status='completed'
    ).order_by(Consultation.updated_at.desc()).all()
    return render_template('expert_profile.html', expert=expert, consultations=consultations)
@expert.route('/consultations')
@login_required
@expert_required
def consultations():
    """View consultations for the current expert"""
    # Consultations the expert is providing
    expert_consultations = Consultation.query.filter_by(expert_id=current_user.id).all()
    # Consultations the user has requested
    requested_consultations = Consultation.query.filter_by(requester_id=current_user.id).all()
    return render_template('my_consultations.html', expert_consultations=expert_consultations, requested_consultations=requested_consultations)
@expert.route('/consultation/<int:consultation_id>')
@login_required
@expert_required
def view_consultation(consultation_id):
    """View a specific consultation"""
    consultation = Consultation.query.get_or_404(consultation_id)
    # Ensure the expert is authorized to view this consultation
    if consultation.expert_id != current_user.id:
        flash('You are not authorized to view this consultation.', 'danger')
        return redirect(url_for('expert.consultations'))
    # Use shared view_consultation template
    return render_template('view_consultation.html', consultation=consultation)
# Admin view for any consultation details
@expert.route('/admin/consultation/<int:consultation_id>')
@login_required
@admin_required
def admin_view_consultation(consultation_id):
    """Allow admin to view all consultation details"""
    consultation = Consultation.query.get_or_404(consultation_id)
    return render_template('view_consultation.html', consultation=consultation)
@expert.route('/apply-expert', methods=['GET', 'POST'])
@login_required
def apply_expert():
    """Allow users to apply to become an expert"""
    # Check if user is already an expert or has a pending application
    if current_user.is_expert:
        flash('You are already an approved expert.', 'info')
        return redirect(url_for('main.profile'))
    # Check for pending application
    pending_application = ExpertApplication.query.filter_by(
        user_id=current_user.id,
        status='pending'
    ).first()
    if pending_application:
        flash('Your expert application is pending approval.', 'info')
        return redirect(url_for('main.profile'))
    if request.method == 'POST':
        # Get form data
        email = request.form.get('email')
        expertise = request.form.get('expertise')
        institution = request.form.get('institution')
        bio = request.form.get('bio')
        # Create new expert application
        application = ExpertApplication(
            user_id=current_user.id,  # type: ignore
            email=email,  # type: ignore
            areas_of_expertise=expertise,  # type: ignore
            institution=institution,  # type: ignore
            bio=bio,  # type: ignore
            status='pending'  # type: ignore
        )
        db.session.add(application)
        db.session.commit()
        flash('Your expert application has been submitted for review.', 'success')
        return redirect(url_for('main.profile'))
    return render_template('apply_expert.html')
@expert.route('/admin/expert-applications')
@login_required
@admin_required
def admin_expert_applications():
    """Admin page to approve expert applications"""
    # Get all pending expert applications
    applications = ExpertApplication.query.filter_by(status='pending').all()
    return render_template('admin/expert_applications.html', applications=applications)
@expert.route('/admin/approve-expert/<int:user_id>', methods=['POST'])
@login_required
@admin_required
def approve_expert(user_id):
    """Approve an expert application"""
    application = ExpertApplication.query.filter_by(user_id=user_id, status='pending').first_or_404()
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
    flash(f'Expert status approved for {user.username}.', 'success')
    return redirect(url_for('expert.admin_expert_applications'))
@expert.route('/admin/reject-expert/<int:user_id>', methods=['POST'])
@login_required
@admin_required
def reject_expert(user_id):
    """Reject an expert application"""
    application = ExpertApplication.query.filter_by(user_id=user_id, status='pending').first_or_404()
    user = application.user
    user._is_expert = False
    user.is_approved_expert = False
    application.status = 'rejected'
    db.session.commit()
    # Send rejection email
    send_expert_rejected_email(user, application.email)
    flash(f'Expert application from {user.username} has been rejected.', 'info')
    return redirect(url_for('expert.admin_expert_applications'))
@expert.route('/request-consultation', methods=['GET', 'POST'])
@login_required
def request_consultation():
    """Request a consultation from an expert"""
    # Get the expert_id from query string if provided
    expert_id = request.args.get('expert_id', type=int)
    analysis_id = request.args.get('analysis_id', type=int)
    # Get the expert if specified
    selected_expert = None
    if expert_id:
        selected_expert = User.query.filter_by(
            id=expert_id,
            _is_expert=True,
            is_approved_expert=True
        ).first_or_404()
    # Get the analysis if specified
    selected_analysis = None
    if analysis_id:
        selected_analysis = Analysis.query.filter_by(
            id=analysis_id,
            user_id=current_user.id
        ).first_or_404()
    # Get all experts for dropdown
    experts = User.query.filter_by(_is_expert=True, is_approved_expert=True).all()
    # Get user's previous analyses for dropdown
    analyses = Analysis.query.filter_by(user_id=current_user.id).order_by(Analysis.created_at.desc()).all()
    if request.method == 'POST':
        # Get form data
        title = request.form.get('title')
        description = request.form.get('description')
        expert_id = request.form.get('expert_id', type=int)
        analysis_id = request.form.get('analysis_id', type=int)
        analysis_goal = request.form.get('analysis_goal')
        # Check if consultation should be public
        is_public = request.form.get('public') == '1'
        # Validate form data
        if not title or not description:
            flash('Please provide both a title and detailed description.', 'warning')
            return render_template('request_consultation.html',
                               experts=experts,
                               analyses=analyses,
                               selected_expert=selected_expert,
                               selected_analysis=selected_analysis)
        # Create new consultation
        consultation = Consultation(
            requester_id=current_user.id,  # type: ignore
            expert_id=expert_id if expert_id else None,  # type: ignore
            analysis_id=analysis_id if analysis_id else None,  # type: ignore
            title=title,  # type: ignore
            description=description,  # type: ignore
            status='in_progress' if expert_id else 'pending',  # type: ignore
            is_public=is_public,  # type: ignore
            analysis_goal=analysis_goal  # type: ignore
        )
        db.session.add(consultation)
        db.session.commit()
        flash('Your consultation request has been submitted.', 'success')
        return redirect(url_for('expert.my_consultations'))
    return render_template('request_consultation.html',
                           experts=experts,
                           analyses=analyses,
                           selected_expert=selected_expert,
                           selected_analysis=selected_analysis)
@expert.route('/my-consultations')
@login_required
def my_consultations():
    """View all consultations for the current user"""
    # Get consultations requested by this user
    user_consultations = Consultation.query.filter_by(requester_id=current_user.id).order_by(Consultation.created_at.desc()).all()
    # If user is an expert, also get consultations assigned to them
    expert_consultations = []
    if current_user.is_expert:
        expert_consultations = Consultation.query.filter_by(expert_id=current_user.id).order_by(Consultation.created_at.desc()).all()
    return render_template('my_consultations.html',
                           user_consultations=user_consultations,
                           expert_consultations=expert_consultations)
@expert.route('/consultation/<int:consultation_id>/respond', methods=['POST'])
@login_required
@expert_required
def respond_consultation(consultation_id):
    """Respond to a consultation as an expert"""
    # Get the consultation
    consultation = Consultation.query.get_or_404(consultation_id)
    # Security check - only the assigned expert can respond
    if consultation.expert_id != current_user.id and not current_user.is_admin:
        flash('You are not assigned to this consultation.', 'danger')
        return redirect(url_for('expert.my_consultations'))
    # Get form data
    response = request.form.get('response')
    make_public = request.form.get('public') == 'on'
    # Update consultation
    consultation.response = response
    consultation.status = 'completed'
    consultation.updated_at = datetime.now(timezone.utc)
    consultation.is_public = make_public
    db.session.commit()
    flash('Your response has been submitted.', 'success')
    return redirect(url_for('expert.view_consultation', consultation_id=consultation.id))
@expert.route('/consultation/<int:consultation_id>/assign', methods=['POST'])
@login_required
@expert_required
def assign_consultation(consultation_id):
    """Assign yourself to a pending consultation"""
    # Get the consultation
    consultation = Consultation.query.get_or_404(consultation_id)
    # Check if the consultation is pending
    if consultation.status != 'pending':
        flash('This consultation is already assigned.', 'warning')
        return redirect(url_for('expert.view_consultation', consultation_id=consultation.id))
    # Assign the expert
    consultation.expert_id = current_user.id
    consultation.status = 'in_progress'
    consultation.updated_at = datetime.now(timezone.utc)
    db.session.commit()
    flash('You have been assigned to this consultation.', 'success')
    return redirect(url_for('expert.view_consultation', consultation_id=consultation.id))
@expert.route('/application-status')
@login_required
def application_status():
    """View expert application status"""
    # If user is already an approved expert, redirect to their expert profile
    if current_user.is_expert:
        flash('You are already an approved expert.', 'info')
        return redirect(url_for('expert.my_profile'))
    # Find the most recent application for this user
    application = ExpertApplication.query.filter_by(user_id=current_user.id).order_by(
        ExpertApplication.created_at.desc()).first()
    if not application:
        flash('No expert applications found.', 'info')
        return redirect(url_for('main.profile'))
    # Redirect to the application details page
    return redirect(url_for('expert.application_details', application_id=application.id))
@expert.route('/application-details/<int:application_id>')
@login_required
def application_details(application_id):
    """View expert application details with communication history"""
    application = ExpertApplication.query.get_or_404(application_id)
    # Security check - ensure users can only view their own applications
    if application.user_id != current_user.id and not current_user.is_admin:
        flash('Unauthorized access.', 'danger')
        return redirect(url_for('main.profile'))
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
    return render_template('expert/application_details.html',
                          application=application,
                          conversation_history=conversation_history)
@expert.route('/upload-resume', methods=['POST'])
@login_required
def upload_resume():
    """Upload resume for expert application"""
    application_id = request.form.get('application_id')
    if not application_id:
        flash('Invalid application.', 'danger')
        return redirect(url_for('expert.application_status'))
    application = ExpertApplication.query.get_or_404(application_id)
    # Security check - ensure users can only upload to their own applications
    if application.user_id != current_user.id:
        flash('Unauthorized access.', 'danger')
        return redirect(url_for('main.profile'))
    # Check if a file was uploaded
    if 'resume' not in request.files:
        flash('No file selected.', 'danger')
        return redirect(url_for('expert.application_status'))
    resume_file = request.files['resume']
    # Check if the file has a name (browser might submit an empty file)
    if resume_file.filename == '':
        flash('No file selected.', 'danger')
        return redirect(url_for('expert.application_status'))
    # Check if the file has an allowed extension
    allowed_extensions = {'pdf', 'doc', 'docx'}
    if resume_file.filename and '.' in resume_file.filename:
        file_extension = resume_file.filename.rsplit('.', 1)[1].lower()
    else:
        file_extension = ''
    if file_extension not in allowed_extensions:
        flash('Invalid file type. Please upload a PDF, DOC, or DOCX file.', 'danger')
        return redirect(url_for('expert.application_status'))
    try:
        # Create a unique filename
        import os
        from datetime import datetime
        from werkzeug.utils import secure_filename
        if not resume_file.filename:
            flash('Invalid filename.', 'danger')
            return redirect(url_for('expert.application_status'))
        filename = secure_filename(resume_file.filename)
        timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
        unique_filename = f"{current_user.id}_{timestamp}_{filename}"
        # Ensure the upload directory exists
        static_folder = current_app.static_folder or 'static'
        upload_dir = os.path.join(static_folder, 'uploads', 'resumes')
        os.makedirs(upload_dir, exist_ok=True)
        # Save the file
        file_path = os.path.join(upload_dir, unique_filename)
        resume_file.save(file_path)
        # Update the application with the resume URL
        resume_url = url_for('static', filename=f'uploads/resumes/{unique_filename}')
        application.resume_url = resume_url
        # Update status from "needs_info" to "pending_review" if it was previously in "needs_info" state
        if application.status == 'needs_info':
            application.status = 'pending_review'
        db.session.commit()
        flash('Resume uploaded successfully. Your application is under review.', 'success')
    except Exception as e:
        current_app.logger.error(f"Resume upload error: {str(e)}")
        flash(f'Error uploading resume: {str(e)}', 'danger')
    return redirect(url_for('expert.application_status'))
@expert.route('/submit-additional-info', methods=['POST'])
@login_required
def submit_additional_info():
    """Submit additional information for expert application"""
    application_id = request.form.get('application_id')
    additional_info = request.form.get('additional_info')
    if not application_id or not additional_info:
        flash('Please provide all required information.', 'danger')
        return redirect(url_for('expert.application_status'))
    application = ExpertApplication.query.get_or_404(application_id)
    # Security check - ensure users can only submit info for their own applications
    if application.user_id != current_user.id:
        flash('Unauthorized access.', 'danger')
        return redirect(url_for('main.profile'))
    try:
        # Create a record of the additional information
        existing_notes = application.admin_notes or ""
        # Update the application with the additional info
        timestamp = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')
        application.admin_notes = f"{existing_notes}\n\n--- Expert Response ({timestamp}) ---\n{additional_info}"
        # Update application status to indicate it's ready for admin review
        if application.status == 'needs_info':
            application.status = 'pending_review'
        application.updated_at = datetime.now(timezone.utc)
        db.session.commit()
        # Send email notification to admin about the additional info (optional)
        # send_admin_notification_email(application)
        flash('Additional information submitted successfully. Your application is now under review.', 'success')
    except Exception as e:
        current_app.logger.error(f"Error submitting additional information: {str(e)}")
        flash(f'Error submitting information: {str(e)}', 'danger')
    return redirect(url_for('expert.application_status'))
@expert.route('/my-profile', methods=['GET', 'POST'])
@login_required
@expert_required
def my_profile():
    """View and edit expert profile"""
    # Verify user is an approved expert
    if not current_user.is_expert:
        flash('You need to be an approved expert to access this page.', 'danger')
        return redirect(url_for('main.profile'))
    if request.method == 'POST':
        # Update expert profile information
        institution = request.form.get('institution', '')
        areas_of_expertise = request.form.get('areas_of_expertise', '')
        bio = request.form.get('bio', '')
        # Update user with expert profile information
        current_user.institution = institution
        current_user.areas_of_expertise = areas_of_expertise
        current_user.bio = bio
        db.session.commit()
        flash('Expert profile updated successfully.', 'success')
        return redirect(url_for('expert.my_profile'))
    return render_template('expert/my_profile.html', expert=current_user)