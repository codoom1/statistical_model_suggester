from flask import Blueprint, render_template, request, redirect, url_for, flash
from flask_login import login_required, current_user
from models import db, User, Analysis, Consultation, ExpertApplication
from functools import wraps
from datetime import datetime, timedelta
from sqlalchemy import func

admin = Blueprint('admin', __name__, url_prefix='/admin')

# Admin-only decorator
def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not current_user.is_authenticated or not current_user.is_admin:
            flash('You do not have permission to access this page.', 'danger')
            return redirect(url_for('main.index'))
        return f(*args, **kwargs)
    return decorated_function

@admin.route('/dashboard')
@login_required
@admin_required
def dashboard():
    # Get basic statistics
    admin_users_count = User.query.filter_by(_is_admin=True).count()
    regular_users_count = User.query.filter_by(_is_admin=False, _is_expert=False).count()
    experts_count = User.query.filter_by(_is_expert=True, is_approved_expert=True).count()
    active_consultations_count = Consultation.query.filter_by(status='in_progress').count()

    # Get recent users (last 10)
    recent_users = User.query.order_by(User.created_at.desc()).limit(10).all()

    # Get pending applications
    pending_applications = ExpertApplication.query.filter_by(status='pending').order_by(ExpertApplication.created_at.desc()).all()

    # Calculate user growth over time for different user types
    regular_users_growth = db.session.query(
        func.date(User.created_at).label('date'),
        func.count(User.id).label('count')
    ).filter_by(_is_admin=False, _is_expert=False).group_by('date').order_by('date').all()

    experts_growth = db.session.query(
        func.date(User.created_at).label('date'),
        func.count(User.id).label('count')
    ).filter_by(_is_expert=True, is_approved_expert=True).group_by('date').order_by('date').all()

    # Prepare data for the growth chart
    user_growth_dates = sorted(list(set([str(record.date) for record in regular_users_growth + experts_growth])))
    
    # Create growth data arrays with zeros for missing dates
    regular_users_growth_data = []
    experts_growth_data = []
    growth_dict = {str(record.date): record.count for record in regular_users_growth}
    for date in user_growth_dates:
        regular_users_growth_data.append(growth_dict.get(date, 0))
    
    growth_dict = {str(record.date): record.count for record in experts_growth}
    for date in user_growth_dates:
        experts_growth_data.append(growth_dict.get(date, 0))

    return render_template('admin/dashboard.html',
                         admin_users_count=admin_users_count,
                         regular_users_count=regular_users_count,
                         experts_count=experts_count,
                         active_consultations_count=active_consultations_count,
                         recent_users=recent_users,
                         pending_applications=pending_applications,
                         user_growth_dates=user_growth_dates,
                         regular_users_growth=regular_users_growth_data,
                         experts_growth=experts_growth_data)

@admin.route('/users')
@login_required
@admin_required
def users_list():
    """List all users"""
    users = User.query.order_by(User.username).all()
    return render_template('admin/users_list.html', users=users)

@admin.route('/edit-user/<int:user_id>', methods=['GET', 'POST'])
@login_required
@admin_required
def edit_user(user_id):
    """Edit user details"""
    user = User.query.get_or_404(user_id)
    
    if request.method == 'POST':
        # Update user information
        user.username = request.form.get('username')
        user.email = request.form.get('email')
        user.role = request.form.get('role')
        
        # Update expert fields if applicable
        if user.role == 'expert':
            user.is_approved_expert = request.form.get('is_approved_expert') == 'on'
            user.expertise = request.form.get('expertise')
            user.institution = request.form.get('institution')
            user.bio = request.form.get('bio')
        
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
    """List pending expert applications"""
    applications = ExpertApplication.query.filter_by(status='pending').all()
    return render_template('admin/expert_applications.html', applications=applications)

@admin.route('/approve-expert/<int:user_id>', methods=['POST'])
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
    
    flash(f'Expert {user.username} has been approved.', 'success')
    return redirect(url_for('admin.expert_applications'))

@admin.route('/reject-expert/<int:user_id>', methods=['POST'])
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
    
    flash(f'Expert application from {user.username} has been rejected.', 'info')
    return redirect(url_for('admin.expert_applications'))

@admin.route('/consultations')
@login_required
@admin_required
def consultations_list():
    """List all consultations"""
    consultations = Consultation.query.order_by(Consultation.created_at.desc()).all()
    experts = User.query.filter_by(role='expert', is_approved_expert=True).all()
    return render_template('admin/consultations_list.html', consultations=consultations, experts=experts)

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