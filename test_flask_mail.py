from flask import Flask
from flask_mail import Mail, Message
import os
from dotenv import load_dotenv
import logging
import sys

# Set up logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

def create_test_app():
    """Create a test Flask app with Flask-Mail configured"""
    app = Flask(__name__)
    
    # Configure Flask-Mail from environment variables
    app.config['MAIL_SERVER'] = os.environ.get('MAIL_SERVER', 'smtp.gmail.com')
    app.config['MAIL_PORT'] = int(os.environ.get('MAIL_PORT', 587))
    app.config['MAIL_USE_TLS'] = os.environ.get('MAIL_USE_TLS', 'True').lower() == 'true'
    app.config['MAIL_USE_SSL'] = os.environ.get('MAIL_USE_SSL', 'False').lower() == 'true'
    app.config['MAIL_USERNAME'] = os.environ.get('MAIL_USERNAME', '')
    app.config['MAIL_PASSWORD'] = os.environ.get('MAIL_PASSWORD', '')
    app.config['MAIL_DEFAULT_SENDER'] = os.environ.get('MAIL_DEFAULT_SENDER', 'noreply@example.com')
    
    # Print configuration for debugging
    print(f"Mail Server: {app.config['MAIL_SERVER']}")
    print(f"Mail Port: {app.config['MAIL_PORT']}")
    print(f"Use TLS: {app.config['MAIL_USE_TLS']}")
    print(f"Use SSL: {app.config['MAIL_USE_SSL']}")
    print(f"Username: {app.config['MAIL_USERNAME']}")
    print(f"Password: {'Set' if app.config['MAIL_PASSWORD'] else 'Not Set'}")
    print(f"Default Sender: {app.config['MAIL_DEFAULT_SENDER']}")
    
    return app

def test_send_email(app, recipient_email):
    """Test sending an email using Flask-Mail"""
    try:
        with app.app_context():
            # Initialize Mail
            mail = Mail(app)
            
            # Create message
            msg = Message(
                subject="Test Email from Flask-Mail in Statistical Model Suggester",
                recipients=[recipient_email],
                html="""
                <html>
                <body>
                    <h2>Flask-Mail Test</h2>
                    <p>This email was sent using Flask-Mail from the test_flask_mail.py script.</p>
                    <p>If you're seeing this, your email configuration is working correctly with Flask-Mail.</p>
                </body>
                </html>
                """,
                body="This email was sent using Flask-Mail from the test_flask_mail.py script."
            )
            
            print(f"Sending test email to {recipient_email}...")
            mail.send(msg)
            print("Email sent successfully!")
            return True
            
    except Exception as e:
        print(f"Error sending email: {str(e)}")
        
        # Provide helpful troubleshooting information
        if "Authentication" in str(e) or "Username and Password not accepted" in str(e):
            print("\nAuthentication error - check your MAIL_USERNAME and MAIL_PASSWORD")
            print("For Gmail, you need to use an App Password: https://myaccount.google.com/apppasswords")
        elif "Application-specific password required" in str(e):
            print("\nGmail requires an App Password. Generate one at: https://myaccount.google.com/apppasswords")
        elif "SMTP" in str(e):
            print(f"\nSMTP server error - check your MAIL_SERVER and MAIL_PORT settings")
        
        return False

if __name__ == "__main__":
    # Get recipient email from command line or prompt
    if len(sys.argv) > 1:
        recipient = sys.argv[1]
    else:
        recipient = input("Enter recipient email address: ")
    
    # Create and configure test app
    app = create_test_app()
    
    # Test sending email
    test_send_email(app, recipient) 