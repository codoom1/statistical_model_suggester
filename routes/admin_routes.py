from flask import Blueprint, render_template, request, redirect, url_for, flash, current_app
from flask_login import login_required, current_user
from models import db, User, Analysis, Consultation, ExpertApplication
from functools import wraps
from datetime import datetime, timedelta
from sqlalchemy import func
from utils.email import send_expert_approved_email, send_expert_rejected_email

admin = Blueprint('admin', __name__, url_prefix='/admin')

# Custom decorator for admin access
def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not current_user.is_authenticated or not current_user.is_admin:
            flash('You need to be an admin to access this page.', 'danger')
            return redirect(url_for('main.index'))
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
    active_consultations_count = Consultation.query.filter_by(status='active').count()
    
    # Get recent users
    recent_users = User.query.order_by(User.created_at.desc()).limit(10).all()
    
    # Get pending expert applications
    pending_applications = ExpertApplication.query.filter_by(status='pending').all()
    
    # User growth over time (last 6 months)
    end_date = datetime.utcnow()
    start_date = end_date - timedelta(days=180)
    months = []
    regular_users_growth = []
    experts_growth = []
    
    current_date = start_date
    while current_date <= end_date:
        month_end = datetime(current_date.year, current_date.month, 1) + timedelta(days=32)
        month_end = datetime(month_end.year, month_end.month, 1) - timedelta(days=1)
        
        # Format date string
        month_str = current_date.strftime('%b %Y')
        months.append(month_str)
        
        # Count users created before this month end
        regular_count = User.query.filter(
            User._is_admin == False,
            User._is_expert == False,
            User.created_at <= month_end
        ).count()
        
        expert_count = User.query.filter(
            User._is_expert == True,
            User.is_approved_expert == True,
            User.created_at <= month_end
        ).count()
        
        regular_users_growth.append(regular_count)
        experts_growth.append(expert_count)
        
        # Move to next month
        current_date = datetime(current_date.year, current_date.month, 1) + timedelta(days=32)
        current_date = datetime(current_date.year, current_date.month, 1)
    
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
        user.username = request.form.get('username')
        user.email = request.form.get('email')
        
        # Update roles
        is_admin = request.form.get('is_admin') == 'on'
        is_expert = request.form.get('is_expert') == 'on'
        
        user._is_admin = is_admin
        
        if is_expert:
            user._is_expert = True
            user.is_approved_expert = request.form.get('is_approved_expert') == 'on'
            user.areas_of_expertise = request.form.get('areas_of_expertise')
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
    
    # Send approval email
    send_expert_approved_email(user, application.email)
    
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