from flask import Blueprint, render_template, request, redirect, url_for, flash
from flask_login import login_user, logout_user, login_required, current_user
from models import db, User
from werkzeug.security import generate_password_hash, check_password_hash
from utils.email_service import send_email
from itsdangerous import URLSafeTimedSerializer, SignatureExpired, BadSignature
import os
from flask import current_app
auth = Blueprint('auth', __name__)
@auth.route('/login', methods=['GET', 'POST'])
def login():
    """Handle user login"""
    if current_user.is_authenticated:
        return redirect(url_for('main.home'))
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        remember = request.form.get('remember', False) == 'on'
        if not username or not password:
            flash('Please provide both username and password.', 'danger')
            return render_template('login.html')
        user = User.query.filter_by(username=username).first()
        if user and check_password_hash(user.password_hash, password):
            login_user(user, remember=remember)
            next_page = request.args.get('next')
            flash('Login successful!', 'success')
            return redirect(next_page or url_for('main.home'))
        else:
            flash('Login failed. Please check your username and password.', 'danger')
    return render_template('login.html')
@auth.route('/register', methods=['GET', 'POST'])
def register():
    """Handle user registration"""
    if current_user.is_authenticated:
        return redirect(url_for('main.home'))
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')
        if not username or not email or not password or not confirm_password:
            flash('Please fill in all fields.', 'danger')
            return render_template('register.html')
        if password != confirm_password:
            flash('Passwords do not match!', 'danger')
            return render_template('register.html')
        user_exists = User.query.filter_by(username=username).first()
        email_exists = User.query.filter_by(email=email).first()
        if user_exists:
            flash('Username already exists!', 'danger')
        elif email_exists:
            flash('Email already registered!', 'danger')
        else:
            new_user = User(
                username=username,
                email=email,
                password_hash=generate_password_hash(password, method='pbkdf2:sha256')
            )
            db.session.add(new_user)
            db.session.commit()
            flash('Account created successfully! Please log in.', 'success')
            return redirect(url_for('auth.login'))
    return render_template('register.html')
@auth.route('/logout')
@login_required
def logout():
    """Handle user logout"""
    logout_user()
    flash('You have been logged out.', 'info')
    return redirect(url_for('main.home'))
@auth.route('/forgot-password', methods=['GET', 'POST'])
def forgot_password():
    """Handle forgot password requests"""
    if current_user.is_authenticated:
        return redirect(url_for('main.home'))
    if request.method == 'POST':
        email = request.form.get('email')
        if not email:
            flash('Please provide an email address.', 'danger')
            return render_template('forgot_password.html')
        user = User.query.filter_by(email=email).first()
        if user:
            # Generate a secure token
            serializer = URLSafeTimedSerializer(current_app.config['SECRET_KEY'])
            token = serializer.dumps(email, salt='password-reset-salt')
            # Build reset URL
            reset_url = url_for('auth.reset_password', token=token, _external=True)
            # Send reset email
            subject = "Password Reset Request"
            html_body = f"""
            <p>Hello {user.username},</p>
            <p>You recently requested to reset your password for your Statistical Model Suggester account.
            Click the link below to reset it:</p>
            <p><a href="{reset_url}">Reset Your Password</a></p>
            <p>If you did not request a password reset, please ignore this email or contact us if you have concerns.</p>
            <p>This password reset link is only valid for 1 hour.</p>
            <p>Best regards,<br>
            Statistical Model Suggester Team</p>
            """
            send_email(subject, email, html_body)
            flash('If your email exists in our system, you will receive a password reset link shortly.', 'info')
        else:
            # Don't reveal that the email doesn't exist for security reasons
            flash('If your email exists in our system, you will receive a password reset link shortly.', 'info')
        return redirect(url_for('auth.login'))
    return render_template('forgot_password.html')
@auth.route('/reset-password/<token>', methods=['GET', 'POST'])
def reset_password(token):
    """Handle password reset with token"""
    if current_user.is_authenticated:
        return redirect(url_for('main.home'))
    try:
        # Validate token (expires after 1 hour)
        serializer = URLSafeTimedSerializer(current_app.config['SECRET_KEY'])
        email = serializer.loads(token, salt='password-reset-salt', max_age=3600)
    except (SignatureExpired, BadSignature):
        flash('The password reset link is invalid or has expired.', 'danger')
        return redirect(url_for('auth.forgot_password'))
    user = User.query.filter_by(email=email).first()
    if not user:
        flash('User not found.', 'danger')
        return redirect(url_for('auth.login'))
    if request.method == 'POST':
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')
        if not password or not confirm_password:
            flash('Please fill in all fields.', 'danger')
            return render_template('reset_password.html', token=token)
        if password != confirm_password:
            flash('Passwords do not match!', 'danger')
            return render_template('reset_password.html', token=token)
        # Update password
        user.password_hash = generate_password_hash(password, method='pbkdf2:sha256')
        db.session.commit()
        flash('Your password has been updated! You can now log in with your new password.', 'success')
        return redirect(url_for('auth.login'))
    return render_template('reset_password.html', token=token)
@auth.route('/test-email-config')
def test_email_config():
    """Test route to check email configuration"""
    if not current_user.is_authenticated or not current_user.is_admin:
        flash('You must be an admin to access this page.', 'danger')
        return redirect(url_for('auth.login'))
    app = current_app._get_current_object()
    mail_config = {
        'MAIL_SERVER': app.config.get('MAIL_SERVER'),
        'MAIL_PORT': app.config.get('MAIL_PORT'),
        'MAIL_USE_TLS': app.config.get('MAIL_USE_TLS'),
        'MAIL_USERNAME': app.config.get('MAIL_USERNAME'),
        'MAIL_PASSWORD': '****' if app.config.get('MAIL_PASSWORD') else None,
        'MAIL_DEFAULT_SENDER': app.config.get('MAIL_DEFAULT_SENDER')
    }
    # Check if email env vars are available
    env_vars = {
        'MAIL_SERVER': os.environ.get('MAIL_SERVER'),
        'MAIL_PORT': os.environ.get('MAIL_PORT'),
        'MAIL_USE_TLS': os.environ.get('MAIL_USE_TLS'),
        'MAIL_USERNAME': os.environ.get('MAIL_USERNAME'),
        'MAIL_PASSWORD': '****' if os.environ.get('MAIL_PASSWORD') else None,
        'MAIL_DEFAULT_SENDER': os.environ.get('MAIL_DEFAULT_SENDER')
    }
    return render_template('admin/email_config.html',
                         app_config=mail_config,
                         env_vars=env_vars)
@auth.route('/send-test-email')
def send_test_email():
    """Send a test email to verify email configuration"""
    if not current_user.is_authenticated or not current_user.is_admin:
        flash('You must be an admin to access this page.', 'danger')
        return redirect(url_for('auth.login'))
    try:
        # Send test email to the current user
        subject = "Test Email from Statistical Model Suggester"
        html_body = f"""
        <p>Hello {current_user.username},</p>
        <p>This is a test email from the Statistical Model Suggester application.</p>
        <p>If you're receiving this email, your email configuration is working correctly!</p>
        <p>Email configuration:</p>
        <ul>
            <li>MAIL_SERVER: {current_app.config.get('MAIL_SERVER')}</li>
            <li>MAIL_PORT: {current_app.config.get('MAIL_PORT')}</li>
            <li>MAIL_USERNAME: {current_app.config.get('MAIL_USERNAME')}</li>
            <li>MAIL_DEFAULT_SENDER: {current_app.config.get('MAIL_DEFAULT_SENDER')}</li>
        </ul>
        <p>Best regards,<br>
        Statistical Model Suggester Team</p>
        """
        send_email(subject, current_user.email, html_body)
        flash(f'Test email sent to {current_user.email}. Please check your inbox.', 'success')
    except Exception as e:
        flash(f'Error sending test email: {str(e)}', 'danger')
    return redirect(url_for('auth.test_email_config'))