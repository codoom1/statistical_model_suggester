from flask import Blueprint, request, redirect, url_for, flash
from flask_login import login_required, current_user
from models import db

user = Blueprint('user', __name__)


@user.route('/profile')
@login_required
def profile():
    """Preserve the former URL while using the canonical profile route."""
    return redirect(url_for('main.profile'))


@user.route('/edit-profile', methods=['GET', 'POST'])
@login_required
def edit_profile():
    """Preserve the former URL without maintaining a second profile editor."""
    if request.method == 'POST':
        flash('Please update your details from the profile page.', 'info')
    return redirect(url_for('main.profile'))


@user.route('/delete-account', methods=['POST'])
@login_required
def delete_account():
    """Delete user account"""
    # Confirm with password
    password = request.form.get('password')
    if not password or not current_user.check_password(password):
        flash('Incorrect password. Account deletion cancelled.', 'danger')
        return redirect(url_for('main.profile'))
    db.session.delete(current_user)
    db.session.commit()
    flash('Your account has been permanently deleted.', 'info')
    return redirect(url_for('main.index'))
