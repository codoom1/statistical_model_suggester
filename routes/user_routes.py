from flask import Blueprint, render_template, request, redirect, url_for, flash
from flask_login import login_required, current_user
from models import db, User, Analysis
from werkzeug.security import generate_password_hash

user = Blueprint('user', __name__)

@user.route('/profile')
@login_required
def profile():
    """Display user profile with analyses history"""
    analyses = Analysis.query.filter_by(user_id=current_user.id).order_by(Analysis.timestamp.desc()).all()
    return render_template('profile.html', user=current_user, analyses=analyses)

@user.route('/edit-profile', methods=['GET', 'POST'])
@login_required
def edit_profile():
    """Edit user profile"""
    if request.method == 'POST':
        # Update basic profile information
        email = request.form.get('email')
        
        # Check if email already exists for another user
        if email != current_user.email:
            existing_user = User.query.filter_by(email=email).first()
            if existing_user:
                flash('Email already in use.', 'danger')
                return redirect(url_for('user.edit_profile'))
        
        # Update password if provided
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')
        
        if password:
            if password != confirm_password:
                flash('Passwords do not match.', 'danger')
                return redirect(url_for('main.profile'))
            
            current_user.password_hash = generate_password_hash(password, method='pbkdf2:sha256')
        
        # Update other fields
        current_user.email = email
        
        # If user is an expert, also update expertise fields
        if current_user.role == 'expert':
            current_user.institution = request.form.get('institution', '')
            current_user.expertise = request.form.get('expertise', '')
            current_user.bio = request.form.get('bio', '')
        
        db.session.commit()
        flash('Profile updated successfully.', 'success')
        return redirect(url_for('main.profile'))
    
    return render_template('edit_profile.html', user=current_user)

@user.route('/delete-account', methods=['POST'])
@login_required
def delete_account():
    """Delete user account"""
    # Confirm with password
    password = request.form.get('password')
    
    if not current_user.check_password(password):
        flash('Incorrect password. Account deletion cancelled.', 'danger')
        return redirect(url_for('main.profile'))
    
    # Delete user's analyses
    Analysis.query.filter_by(user_id=current_user.id).delete()
    
    # Delete user
    db.session.delete(current_user)
    db.session.commit()
    
    flash('Your account has been permanently deleted.', 'info')
    return redirect(url_for('main.index')) 