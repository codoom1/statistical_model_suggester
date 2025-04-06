from app import create_app
from models import db, User
from werkzeug.security import generate_password_hash

def create_test_users():
    """Create test users including admin, regular user, and expert"""
    app = create_app()
    
    with app.app_context():
        # Create admin user
        admin = User(
            username='admin',
            email='admin@example.com',
            _is_admin=True,
            _is_expert=False
        )
        admin.set_password('admin123')
        
        # Create regular user
        regular_user = User(
            username='testuser',
            email='testuser@example.com',
            _is_admin=False,
            _is_expert=False
        )
        regular_user.set_password('testuser123')
        
        # Create expert user
        expert_user = User(
            username='testexpert',
            email='testexpert@example.com',
            _is_admin=False,
            _is_expert=True,
            is_approved_expert=True,
            areas_of_expertise='Statistical Analysis, Data Science',
            institution='Test University',
            bio='Experienced data scientist with expertise in statistical analysis.'
        )
        expert_user.set_password('testexpert123')
        
        # Add users to database
        db.session.add(admin)
        db.session.add(regular_user)
        db.session.add(expert_user)
        
        # Commit changes
        db.session.commit()
        
        print("Created admin user (username: admin, password: admin123)")
        print("Created regular user (username: testuser, password: testuser123)")
        print("Created expert user (username: testexpert, password: testexpert123)")

if __name__ == "__main__":
    create_test_users() 