"""
Test configuration and fixtures for the Statistical Model Suggester app.
"""
import pytest
import tempfile
import os
import sys
from pathlib import Path
# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
# Set testing environment variables before any imports
os.environ['TESTING'] = 'true'
os.environ['WTF_CSRF_ENABLED'] = 'false'
os.environ['SECRET_KEY'] = 'test-secret-key'
os.environ['MAIL_SUPPRESS_SEND'] = 'true'
from models import db, User
@pytest.fixture
def app():
    """Create and configure a new app instance for each test."""
    # Create a temporary file to use as the database
    db_fd, db_path = tempfile.mkstemp()
    
    # Set testing environment variables before app creation
    os.environ['TESTING'] = 'true'
    os.environ['WTF_CSRF_ENABLED'] = 'false'
    os.environ['SECRET_KEY'] = 'test-secret-key'
    os.environ['MAIL_SUPPRESS_SEND'] = 'true'
    # Force SQLite for testing to avoid PostgreSQL connection issues
    os.environ['DATABASE_URL'] = f'sqlite:///{db_path}'
    
    # Import app creation function after setting environment
    from app import create_app
    app = create_app()
    app.config.update({
        'TESTING': True,
        'SQLALCHEMY_DATABASE_URI': f'sqlite:///{db_path}',
        'WTF_CSRF_ENABLED': False,
        'SECRET_KEY': 'test-secret-key',
        'MAIL_SUPPRESS_SEND': True,
        'MAIL_BACKEND': 'locmem'
    })
    
    with app.app_context():
        db.create_all()
        yield app
        
    os.close(db_fd)
    os.unlink(db_path)
@pytest.fixture
def client(app):
    """A test client for the app."""
    return app.test_client()
@pytest.fixture
def runner(app):
    """A test runner for the app's Click commands."""
    return app.test_cli_runner()
@pytest.fixture
def test_user(app):
    """Create a test user."""
    with app.app_context():
        # Check if test user already exists
        existing_user = User.query.filter_by(email='test@example.com').first()
        if existing_user:
            # Return existing user data
            test_user_data = {
                'id': existing_user.id,
                'username': existing_user.username,
                'email': existing_user.email,
                'password': 'testpassword'
            }
            return test_user_data
        
        # Create new test user
        user = User()
        user.username = 'testuser'
        user.email = 'test@example.com'
        user.set_password('testpassword')
        db.session.add(user)
        db.session.commit()
        # Refresh the user to ensure it's attached to the session
        db.session.refresh(user)
        # Return the user ID instead of the user object to avoid session issues
        user_id = user.id
        db.session.expunge(user)  # Detach from session to avoid conflicts
        # Create a new user object that can be used in tests
        test_user_data = {
            'id': user_id,
            'username': 'testuser',
            'email': 'test@example.com',
            'password': 'testpassword'
        }
        return test_user_data
@pytest.fixture
def admin_user(app):
    """Create an admin user."""
    with app.app_context():
        # Check if admin user already exists
        existing_admin = User.query.filter_by(email='admin@example.com').first()
        if existing_admin:
            # Return existing admin data
            admin_user_data = {
                'id': existing_admin.id,
                'username': existing_admin.username,
                'email': existing_admin.email,
                'password': 'adminpassword',
                '_is_admin': True
            }
            return admin_user_data
        
        # Create new admin user
        admin = User()
        admin.username = 'admin'
        admin.email = 'admin@example.com'
        admin.set_password('adminpassword')
        admin._is_admin = True
        db.session.add(admin)
        db.session.commit()
        # Refresh the admin to ensure it's attached to the session
        db.session.refresh(admin)
        # Return admin ID instead of the admin object to avoid session issues
        admin_id = admin.id
        db.session.expunge(admin)  # Detach from session to avoid conflicts
        # Create a new admin object that can be used in tests
        admin_user_data = {
            'id': admin_id,
            'username': 'admin',
            'email': 'admin@example.com',
            'password': 'adminpassword',
            '_is_admin': True
        }
        return admin_user_data
@pytest.fixture
def expert_user(app):
    """Create an expert user."""
    with app.app_context():
        # Check if expert user already exists
        existing_expert = User.query.filter_by(email='expert@example.com').first()
        if existing_expert:
            # Return existing expert data
            expert_user_data = {
                'id': existing_expert.id,
                'username': existing_expert.username,
                'email': existing_expert.email,
                'password': 'expertpassword',
                '_is_expert': True,
                'is_approved_expert': True,
                'areas_of_expertise': 'Statistics, Machine Learning',
                'institution': 'Test University'
            }
            return expert_user_data
        
        # Create new expert user
        expert = User()
        expert.username = 'expert'
        expert.email = 'expert@example.com'
        expert.set_password('expertpassword')
        expert._is_expert = True
        expert.is_approved_expert = True
        expert.areas_of_expertise = 'Statistics, Machine Learning'
        expert.institution = 'Test University'
        db.session.add(expert)
        db.session.commit()
        # Refresh the expert to ensure it's attached to the session
        db.session.refresh(expert)
        # Return expert ID instead of the expert object to avoid session issues
        expert_id = expert.id
        db.session.expunge(expert)  # Detach from session to avoid conflicts
        # Create a new expert object that can be used in tests
        expert_user_data = {
            'id': expert_id,
            'username': 'expert',
            'email': 'expert@example.com',
            'password': 'expertpassword',
            '_is_expert': True,
            'is_approved_expert': True,
            'areas_of_expertise': 'Statistics, Machine Learning',
            'institution': 'Test University'
        }
        return expert_user_data
@pytest.fixture
def authenticated_client(client, test_user):
    """A client with an authenticated user."""
    response = client.post('/auth/login', data={
        'username': test_user['username'],
        'password': test_user['password']
    }, follow_redirects=True)
    # Verify login was successful
    assert response.status_code == 200
    return client
@pytest.fixture
def admin_client(client, admin_user):
    """A client with an authenticated admin user."""
    response = client.post('/auth/login', data={
        'username': admin_user['username'],
        'password': admin_user['password']
    }, follow_redirects=True)
    # Verify login was successful
    assert response.status_code == 200
    return client
@pytest.fixture
def sample_analysis_data():
    """Sample data for analysis form."""
    return {
        'research_question': 'What factors predict customer satisfaction?',
        'analysis_goal': 'predict',
        'dependent_variable_type': 'continuous',
        'independent_variables': 'mixed',
        'sample_size': '500',
        'missing_data': 'little',
        'data_distribution': 'normal',
        'relationship_type': 'linear',
        'variables_correlated': 'no'
    }
@pytest.fixture
def clustering_analysis_data():
    """Sample data for clustering analysis."""
    return {
        'research_question': 'How can I cluster my customer data?',
        'analysis_goal': 'cluster',
        'dependent_variable_type': '',
        'independent_variables': 'continuous',
        'sample_size': '150',
        'missing_data': 'none',
        'data_distribution': 'non_normal',
        'relationship_type': 'non_linear',
        'variables_correlated': 'unknown'
    }
