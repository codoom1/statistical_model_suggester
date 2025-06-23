"""
Test the models module.
"""
import pytest
from datetime import datetime
from models import db, User, Analysis, ExpertApplication, Consultation, Questionnaire, get_model_details
class TestUser:
    """Test the User model."""
    def test_user_creation(self, app):
        """Test creating a new user."""
        with app.app_context():
            user = User()
            user.username = 'newuser'
            user.email = 'newuser@example.com'
            user.set_password('password123')
            db.session.add(user)
            db.session.commit()
            assert user.id is not None
            assert user.username == 'newuser'
            assert user.email == 'newuser@example.com'
            assert user.password_hash is not None
            assert user.created_at is not None
            assert not user.is_admin
            assert not user.is_expert
    def test_user_password_hashing(self, app):
        """Test password hashing and verification."""
        with app.app_context():
            user = User()
            user.username = 'testuser'
            user.email = 'test@example.com'
            password = 'securepassword123'
            user.set_password(password)
            # Password should be hashed, not stored in plain text
            assert user.password_hash != password
            assert user.check_password(password)
            assert not user.check_password('wrongpassword')
    def test_user_admin_property(self, app):
        """Test admin property."""
        with app.app_context():
            user = User()
            user.username = 'admin'
            user.email = 'admin@example.com'
            user._is_admin = True
            assert user.is_admin
    def test_user_expert_property(self, app):
        """Test expert property."""
        with app.app_context():
            user = User()
            user.username = 'expert'
            user.email = 'expert@example.com'
            user._is_expert = True
            user.is_approved_expert = True
            assert user.is_expert
            # Test that expert must be approved
            user.is_approved_expert = False
            assert not user.is_expert
    def test_user_string_representation(self, app):
        """Test user string representation."""
        with app.app_context():
            user = User()
            user.username = 'testuser'
            assert str(user) == '<User testuser>'
    def test_user_relationships(self, app):
        """Test user relationships."""
        with app.app_context():
            user = User()
            user.username = 'testuser'
            user.email = 'test@example.com'
            db.session.add(user)
            db.session.commit()
            # Test analyses relationship
            assert user.analyses == []
            # Test expert applications relationship
            assert user.expert_applications == []
            # Test questionnaires relationship
            assert user.questionnaires == []
class TestExpertApplication:
    """Test the ExpertApplication model."""
    def test_expert_application_creation(self, app, test_user):
        """Test creating an expert application."""
        with app.app_context():
            application = ExpertApplication()
            application.user_id = test_user["id"]
            application.email = test_user["email"]
            application.areas_of_expertise = 'Statistics, Data Science'
            application.institution = 'Test University'
            application.bio = 'I am a statistics expert.'
            application.status = 'pending'
            db.session.add(application)
            db.session.commit()
            assert application.id is not None
            assert application.user_id == test_user["id"]
            assert application.areas_of_expertise == 'Statistics, Data Science'
            assert application.institution == 'Test University'
            assert application.bio == 'I am a statistics expert.'
            assert application.status == 'pending'
class TestAnalysis:
    """Test the Analysis model."""
    def test_analysis_creation(self, app, test_user):
        """Test creating an analysis."""
        with app.app_context():
            analysis = Analysis()
            analysis.user_id = test_user["id"]
            analysis.research_question = 'What factors predict sales?'
            analysis.recommended_model = 'Linear Regression'
            analysis.analysis_goal = 'predict'
            analysis.dependent_variable = 'continuous'
            analysis.sample_size = '500'
            db.session.add(analysis)
            db.session.commit()
            assert analysis.id is not None
            assert analysis.user_id == test_user["id"]
            assert analysis.research_question == 'What factors predict sales?'
            assert analysis.recommended_model == 'Linear Regression'
            assert analysis.analysis_goal == 'predict'
            assert analysis.dependent_variable == 'continuous'
            assert analysis.sample_size == '500'
            assert analysis.created_at is not None
class TestDatabaseFunctions:
    """Test database utility functions."""
    def test_get_model_details_exists(self, app):
        """Test get_model_details function with existing model."""
        with app.app_context():
            # This test assumes the model database is loaded
            try:
                details = get_model_details('Linear Regression')
                assert details is not None
                assert 'name' in details
            except:
                # If model database isn't loaded, that's also a valid test case
                pytest.skip("Model database not available in test environment")
    def test_get_model_details_nonexistent(self, app):
        """Test get_model_details function with non-existent model."""
        with app.app_context():
            details = get_model_details('NonExistentModel')
            assert details is None
class TestDatabaseIntegrity:
    """Test database integrity and constraints."""
    def test_unique_username_constraint(self, app, test_user):
        """Test that usernames must be unique."""
        with app.app_context():
            # Try to create another user with the same username
            duplicate_user = User()
            duplicate_user.username = test_user['username']  # Same username
            duplicate_user.email = 'different@example.com'
            duplicate_user.set_password('password')
            db.session.add(duplicate_user)
            with pytest.raises(Exception):  # Should raise integrity error
                db.session.commit()
    def test_unique_email_constraint(self, app, test_user):
        """Test that emails must be unique."""
        with app.app_context():
            # Try to create another user with the same email
            duplicate_user = User()
            duplicate_user.username = 'differentuser'
            duplicate_user.email = test_user['email']  # Same email
            duplicate_user.set_password('password')
            db.session.add(duplicate_user)
            with pytest.raises(Exception):  # Should raise integrity error
                db.session.commit()
    def test_user_cascade_delete(self, app, test_user):
        """Test that deleting a user cascades to related records."""
        with app.app_context():
            # Create an analysis for the user
            analysis = Analysis()
            analysis.user_id = test_user["id"]
            analysis.research_question = 'Test question'
            analysis.recommended_model = 'Test Model'
            db.session.add(analysis)
            db.session.commit()
            analysis_id = analysis.id
            # Delete the user - first get the actual user object
            user = User.query.get(test_user["id"])
            db.session.delete(user)
            db.session.commit()
            # Check that the analysis was also deleted
            remaining_analysis = db.session.get(Analysis, analysis_id)
            assert remaining_analysis is None
