from flask import Blueprint, render_template, request, redirect, url_for, flash, current_app
from flask_login import login_required, current_user
from models import db, User, Analysis, Consultation, ExpertApplication
from datetime import datetime
from functools import wraps
from utils.email import send_expert_approved_email, send_expert_rejected_email

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
    consultations = Consultation.query.filter_by(expert_id=current_user.id).all()
    return render_template('expert/consultations.html', consultations=consultations)

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
        
    return render_template('expert/view_consultation.html', consultation=consultation)

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
            user_id=current_user.id,
            email=email,
            areas_of_expertise=expertise,
            institution=institution,
            bio=bio,
            status='pending'
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
            requester_id=current_user.id,
            expert_id=expert_id if expert_id else None,
            analysis_id=analysis_id if analysis_id else None,
            title=title,
            description=description,
            status='in_progress' if expert_id else 'pending',
            is_public=is_public,
            analysis_goal=analysis_goal
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
    consultation.updated_at = datetime.utcnow()
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
    consultation.updated_at = datetime.utcnow()
    
    db.session.commit()
    
    flash('You have been assigned to this consultation.', 'success')
    return redirect(url_for('expert.view_consultation', consultation_id=consultation.id)) 