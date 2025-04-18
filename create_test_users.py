from app import create_app
from models import db, User
from werkzeug.security import generate_password_hash

def create_test_users():
    """Create test users including admin, regular user, and expert"""
    app = create_app()
    
    with app.app_context():
        # Check if users already exist
        admin_exists = User.query.filter_by(username='admin').first() is not None
        regular_exists = User.query.filter_by(username='testuser').first() is not None
        expert_exists = User.query.filter_by(username='testexpert').first() is not None
        
        users_created = False
        
        # Create admin user if not exists
        if not admin_exists:
            admin = User(
                username='admin',
                email='admin@example.com',
                _is_admin=True,
                _is_expert=False
            )
            admin.set_password('admin123')
            db.session.add(admin)
            print("Created admin user (username: admin, password: admin123)")
            users_created = True
        else:
            print("Admin user already exists, skipping creation")
        
        # Create regular user if not exists
        if not regular_exists:
            regular_user = User(
                username='testuser',
                email='testuser@example.com',
                _is_admin=False,
                _is_expert=False
            )
            regular_user.set_password('testuser123')
            db.session.add(regular_user)
            print("Created regular user (username: testuser, password: testuser123)")
            users_created = True
        else:
            print("Regular user already exists, skipping creation")
        
        # Create expert user if not exists
        if not expert_exists:
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
            db.session.add(expert_user)
            print("Created expert user (username: testexpert, password: testexpert123)")
            users_created = True
        else:
            print("Expert user already exists, skipping creation")
        
        # Commit changes if any users were created
        if users_created:
            db.session.commit()
            print("All created users committed to the database")
        else:
            print("No new users created")

if __name__ == "__main__":
    create_test_users() 