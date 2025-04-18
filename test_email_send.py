import os
import sys
from dotenv import load_dotenv
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# Load environment variables
load_dotenv()

# Get email configuration from environment variables
MAIL_SERVER = os.environ.get('MAIL_SERVER', 'smtp.gmail.com')
MAIL_PORT = int(os.environ.get('MAIL_PORT', 587))
MAIL_USE_TLS = os.environ.get('MAIL_USE_TLS', 'True').lower() == 'true'
MAIL_USERNAME = os.environ.get('MAIL_USERNAME', '')
MAIL_PASSWORD = os.environ.get('MAIL_PASSWORD', '')
MAIL_DEFAULT_SENDER = os.environ.get('MAIL_DEFAULT_SENDER', 'noreply@expertapplicationdecisionteam.com')

# Function to send a test email
def send_test_email(recipient_email):
    print(f"Attempting to send test email to {recipient_email}")
    print(f"Using MAIL_SERVER: {MAIL_SERVER}")
    print(f"Using MAIL_PORT: {MAIL_PORT}")
    print(f"Using MAIL_USERNAME: {MAIL_USERNAME}")
    print(f"Using MAIL_DEFAULT_SENDER: {MAIL_DEFAULT_SENDER}")
    print(f"TLS Enabled: {MAIL_USE_TLS}")
    
    try:
        # Create message
        msg = MIMEMultipart()
        msg['From'] = MAIL_DEFAULT_SENDER
        msg['To'] = recipient_email
        msg['Subject'] = "Test Email from Statistical Model Suggester"
        
        # Email body
        body = """
        <html>
          <body>
            <p>Hello,</p>
            <p>This is a test email from the Statistical Model Suggester application.</p>
            <p>If you're receiving this email, your email configuration is working correctly!</p>
            <p>Best regards,<br>
            Statistical Model Suggester Team</p>
          </body>
        </html>
        """
        
        msg.attach(MIMEText(body, 'html'))
        
        # Connect to the SMTP server
        print("Connecting to SMTP server...")
        if MAIL_USE_TLS:
            smtp = smtplib.SMTP(MAIL_SERVER, MAIL_PORT)
            smtp.ehlo()
            smtp.starttls()
            smtp.ehlo()
        else:
            smtp = smtplib.SMTP_SSL(MAIL_SERVER, MAIL_PORT)
        
        # Login
        print("Logging in...")
        smtp.login(MAIL_USERNAME, MAIL_PASSWORD)
        
        # Send the email
        print("Sending email...")
        smtp.sendmail(MAIL_DEFAULT_SENDER, recipient_email, msg.as_string())
        
        # Disconnect
        smtp.quit()
        
        print(f"Email sent successfully to {recipient_email}!")
        return True
        
    except Exception as e:
        print(f"Error sending email: {e}")
        if "Application-specific password required" in str(e):
            print("\nIMPORTANT: Gmail requires an App Password for this application.")
            print("Please follow these steps:")
            print("1. Go to https://myaccount.google.com/apppasswords")
            print("2. Sign in with your Google account")
            print("3. Select 'App' and choose 'Other (Custom name)'")
            print("4. Enter 'Statistical Model Suggester' and click 'Generate'")
            print("5. Copy the generated password")
            print("6. Update your .env file's MAIL_PASSWORD with this new App Password")
        elif "Username and Password not accepted" in str(e):
            print("\nInvalid credentials. Please check your username and password.")
        elif "SMTP server" in str(e) or "connection" in str(e).lower():
            print("\nUnable to connect to the mail server. Please check your network and mail server settings.")
        return False

if __name__ == "__main__":
    if len(sys.argv) > 1:
        recipient = sys.argv[1]
    else:
        recipient = input("Enter recipient email address: ")
    
    send_test_email(recipient) 