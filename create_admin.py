from app import create_app
from models import db, User
from werkzeug.security import generate_password_hash

def create_admin_user(username, email, password):
    """Create an admin user"""
    app = create_app()
    
    with app.app_context():
        # Check if user already exists
        existing_user = User.query.filter_by(username=username).first()
        if existing_user:
            print(f"User {username} already exists")
            return
        
        # Create new admin user
        admin = User(
            username=username,
            email=email,
            _is_admin=True,
            _is_expert=False,
            is_approved_expert=False,
            areas_of_expertise=None,
            institution=None,
            bio=None
        )
        admin.set_password(password)
        
        # Add to database
        db.session.add(admin)
        db.session.commit()
        
        print(f"Created admin user: {username}")

if __name__ == "__main__":
    username = "admin"
    email = "admin@example.com"
    password = "admin123"  # You should change this in production
    
    create_admin_user(username, email, password) 