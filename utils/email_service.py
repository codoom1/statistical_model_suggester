from flask import current_app
from flask_mail import Message, Mail
from threading import Thread
import logging

# Initialize Mail instance at module level
mail = Mail()
logger = logging.getLogger(__name__)

def init_mail(app):
    """Initialize the mail extension with the Flask app"""
    mail.init_app(app)

def send_email_async(app, msg):
    """Send email asynchronously"""
    with app.app_context():
        try:
            mail.send(msg)
            logger.info(f"Email sent successfully to {msg.recipients}")
        except Exception as e:
            logger.error(f"Failed to send email: {e}")
            # Log the email content instead
            logger.info(f"Email would have been sent to: {msg.recipients}")
            logger.info(f"Subject: {msg.subject}")
            logger.info(f"Body: {msg.html or msg.body}")

def send_email(subject, recipient, html_body, text_body=None):
    """Send an email using the application mail server
    
    Args:
        subject: Email subject
        recipient: Email address of the recipient
        html_body: HTML content of the email
        text_body: Plain text content (optional)
    """
    app = current_app._get_current_object()
    
    # Check if we're in development mode or email credentials are missing
    mail_username = app.config.get('MAIL_USERNAME')
    mail_password = app.config.get('MAIL_PASSWORD')
    
    # If credentials are missing or set to defaults, just log the email
    if not mail_username or not mail_password or mail_username == 'your_gmail@gmail.com' or mail_password == 'your_app_password':
        logger.info(f"[DEV MODE] Email would be sent to: {recipient}")
        logger.info(f"[DEV MODE] Subject: {subject}")
        logger.info(f"[DEV MODE] Body: {html_body}")
        return
    
    # Otherwise, send the actual email
    msg = Message(
        subject=subject,
        recipients=[recipient],
        html=html_body,
        body=text_body or "Please view this email in a HTML-compatible email client."
    )
    
    # Send email in background thread to not block the request
    Thread(target=send_email_async, args=(app, msg)).start()

def send_expert_approved_email(user, email):
    """Send notification when expert application is approved
    
    Args:
        user: User object who applied to be an expert
        email: User's email address
    """
    subject = "Your Expert Application Has Been Approved!"
    html_body = f"""
    <p>Congratulations {user.username}!</p>
    
    <p>Your application to become an expert on the Statistical Model Suggester platform has been <strong>approved</strong>.</p>
    
    <p>You can now:</p>
    <ul>
        <li>Provide guidance to users about statistical models</li>
        <li>Offer consultations to users needing more in-depth assistance</li>
        <li>Share your expertise with the community</li>
    </ul>
    
    <p>Thank you for contributing your expertise to our platform!</p>
    
    <p>Best regards,<br>
    Statistical Model Suggester Team</p>
    """
    
    send_email(subject, email, html_body)

def send_expert_rejected_email(user, email):
    """Send notification when expert application is rejected
    
    Args:
        user: User object who applied to be an expert
        email: User's email address
    """
    subject = "Update on Your Expert Application"
    html_body = f"""
    <p>Dear {user.username},</p>
    
    <p>Thank you for your interest in becoming an expert on the Statistical Model Suggester platform.</p>
    
    <p>After reviewing your application, we regret to inform you that we are unable to approve your expert status at this time.</p>
    
    <p>This decision may be due to one of several factors:</p>
    <ul>
        <li>We're looking for specific expertise in certain statistical domains</li>
        <li>We require more detailed information about your background and qualifications</li>
        <li>We have reached capacity for experts in your particular field</li>
    </ul>
    
    <p>You are welcome to apply again in the future with additional information about your qualifications and experience.</p>
    
    <p>Thank you for your understanding.</p>
    
    <p>Best regards,<br>
    Statistical Model Suggester Team</p>
    """
    
    send_email(subject, email, html_body) 