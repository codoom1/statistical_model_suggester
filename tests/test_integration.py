"""
Integration tests for the Statistical Model Suggester application.
These tests verify that different components work together correctly.
"""
import pytest
from models import db, User, Analysis
class TestUserWorkflow:
    """Test complete user workflows."""
    def test_complete_user_registration_and_analysis(self, client, app):
        """Test complete workflow from registration to analysis."""
        # Step 1: Register a new user
        response = client.post('/auth/register', data={
            'username': 'newuser',
            'email': 'newuser@example.com',
            'password': 'password123',
            'confirm_password': 'password123'
        }, follow_redirects=True)
        assert response.status_code == 200
        # Step 2: Login
        response = client.post('/auth/login', data={
            'username': 'newuser',
            'password': 'password123'
        }, follow_redirects=True)
        assert response.status_code == 200
        # Step 3: Submit analysis
        analysis_data = {
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
        response = client.post('/results', data=analysis_data, follow_redirects=True)
        assert response.status_code == 200
        # Step 4: Verify analysis was saved
        with app.app_context():
            user = User.query.filter_by(username='newuser').first()
            assert user is not None
            analysis = Analysis.query.filter_by(user_id=user.id).first()
            assert analysis is not None
            assert analysis.research_question == analysis_data['research_question']
    def test_expert_application_workflow(self, client, app):
        """Test expert application workflow."""
        # Step 1: Register and login
        client.post('/auth/register', data={
            'username': 'expertuser',
            'email': 'expert@example.com',
            'password': 'password123',
            'confirm_password': 'password123'
        })
        client.post('/auth/login', data={
            'username': 'expertuser',
            'password': 'password123'
        })
        # Step 2: Apply to become expert
        response = client.post('/expert/apply-expert', data={
            'email': 'expert@example.com',
            'expertise': 'Statistics, Machine Learning',
            'institution': 'Test University',
            'bio': 'I am a statistics expert with 10 years of experience.'
        }, follow_redirects=True)
        assert response.status_code == 200
        # Step 3: Verify application was created
        with app.app_context():
            user = User.query.filter_by(username='expertuser').first()
            assert user is not None
            assert len(user.expert_applications) > 0
            application = user.expert_applications[0]
            assert application.areas_of_expertise == 'Statistics, Machine Learning'
            assert application.status == 'pending'
class TestAdminWorkflow:
    """Test admin workflows."""
    def test_admin_manages_expert_application(self, client, app):
        """Test admin managing expert applications."""
        # Step 1: Create expert application
        with app.app_context():
            user = User()
            user.username = 'expertcandidate'
            user.email = 'candidate@example.com'
            user.set_password('password')
            db.session.add(user)
            db.session.commit()
            from models import ExpertApplication
            application = ExpertApplication()
            application.user_id = user.id
            application.email = 'candidate@example.com'
            application.areas_of_expertise = 'Statistics'
            application.institution = 'Test University'
            application.bio = 'Test bio'
            application.status = 'pending'
            db.session.add(application)
            db.session.commit()
            user_id = user.id
            # Application ID stored for potential future use
            _ = application.id
        # Step 2: Admin login
        with app.app_context():
            admin = User()
            admin.username = 'admin'
            admin.email = 'admin@example.com'
            admin.set_password('adminpass')
            admin._is_admin = True
            db.session.add(admin)
            db.session.commit()
        client.post('/auth/login', data={
            'username': 'admin',
            'password': 'adminpass'
        })
        # Step 3: View applications
        response = client.get('/admin/expert-applications')
        assert response.status_code == 200
        # Step 4: Approve application
        response = client.post(f'/admin/approve-expert/{user_id}', follow_redirects=True)
        assert response.status_code == 200
        # Step 5: Verify approval
        with app.app_context():
            user = db.session.get(User, user_id)
            assert user.is_approved_expert is True
class TestAnalysisVariations:
    """Test various analysis scenarios."""
    def test_different_analysis_types(self, authenticated_client):
        """Test different types of analysis."""
        analysis_scenarios = [
            {
                'name': 'Linear Regression',
                'data': {
                    'research_question': 'What predicts sales?',
                    'analysis_goal': 'predict',
                    'dependent_variable_type': 'continuous',
                    'independent_variables': 'continuous',
                    'sample_size': '500',
                    'missing_data': 'little',
                    'data_distribution': 'normal',
                    'relationship_type': 'linear',
                    'variables_correlated': 'no'
                }
            },
            {
                'name': 'Logistic Regression',
                'data': {
                    'research_question': 'What predicts success?',
                    'analysis_goal': 'classify',
                    'dependent_variable_type': 'categorical',
                    'independent_variables': 'continuous',
                    'sample_size': '300',
                    'missing_data': 'little',
                    'data_distribution': 'normal',
                    'relationship_type': 'linear',
                    'variables_correlated': 'no'
                }
            },
            {
                'name': 'Clustering',
                'data': {
                    'research_question': 'How to segment customers?',
                    'analysis_goal': 'cluster',
                    'dependent_variable_type': '',
                    'independent_variables': 'continuous',
                    'sample_size': '200',
                    'missing_data': 'none',
                    'data_distribution': 'normal',
                    'relationship_type': 'non_linear',
                    'variables_correlated': 'unknown'
                }
            }
        ]
        for scenario in analysis_scenarios:
            response = authenticated_client.post('/results', data=scenario['data'])
            assert response.status_code == 200
            # Should contain some model recommendation
            assert b'recommend' in response.data.lower() or b'model' in response.data.lower()
    def test_edge_case_inputs(self, client):
        """Test edge case inputs."""
        edge_cases = [
            {
                'name': 'Very small sample',
                'data': {
                    'research_question': 'Small sample test',
                    'analysis_goal': 'predict',
                    'dependent_variable_type': 'continuous',
                    'independent_variables': 'continuous',
                    'sample_size': '10',  # Very small
                    'missing_data': 'substantial',
                    'data_distribution': 'non_normal',
                    'relationship_type': 'non_linear',
                    'variables_correlated': 'yes'
                }
            },
            {
                'name': 'Large sample',
                'data': {
                    'research_question': 'Large sample test',
                    'analysis_goal': 'predict',
                    'dependent_variable_type': 'continuous',
                    'independent_variables': 'continuous',
                    'sample_size': '10000',  # Very large
                    'missing_data': 'none',
                    'data_distribution': 'normal',
                    'relationship_type': 'linear',
                    'variables_correlated': 'no'
                }
            }
        ]
        for case in edge_cases:
            response = client.post('/results', data=case['data'])
            assert response.status_code == 200
            # Should handle gracefully
class TestSecurityIntegration:
    """Test security across the application."""
    def test_unauthorized_access_protection(self, client):
        """Test protection against unauthorized access."""
        protected_routes = [
            '/admin/dashboard',
            '/admin/users',
            '/profile',
            '/auth/logout'
        ]
        for route in protected_routes:
            response = client.get(route)
            assert response.status_code == 302  # Should redirect to login
    def test_role_based_access_control(self, client, app):
        """Test role-based access control."""
        # Create users with different roles
        with app.app_context():
            # Regular user
            regular_user = User()
            regular_user.username = 'regular'
            regular_user.email = 'regular@example.com'
            regular_user.set_password('password')
            db.session.add(regular_user)
            # Admin user
            admin_user = User()
            admin_user.username = 'admin'
            admin_user.email = 'admin@example.com'
            admin_user.set_password('password')
            admin_user._is_admin = True
            db.session.add(admin_user)
            # Expert user
            expert_user = User()
            expert_user.username = 'expert'
            expert_user.email = 'expert@example.com'
            expert_user.set_password('password')
            expert_user._is_expert = True
            expert_user.is_approved_expert = True
            db.session.add(expert_user)
            db.session.commit()
        # Test regular user access
        client.post('/auth/login', data={'username': 'regular', 'password': 'password'})
        response = client.get('/admin/dashboard')
        assert response.status_code in [302, 403]  # Should be denied
        client.get('/auth/logout')
        # Test admin access
        client.post('/auth/login', data={'username': 'admin', 'password': 'password'})
        response = client.get('/admin/dashboard')
        assert response.status_code == 200  # Should be allowed
        client.get('/auth/logout')
class TestDataIntegrity:
    """Test data integrity across operations."""
    def test_user_deletion_cascades(self, app):
        """Test that deleting a user cascades properly."""
        with app.app_context():
            # Create user with analysis
            user = User()
            user.username = 'testcascade'
            user.email = 'cascade@example.com'
            user.set_password('password')
            db.session.add(user)
            db.session.commit()
            analysis = Analysis()
            analysis.user_id = user.id
            analysis.research_question = 'Test cascade'
            analysis.recommended_model = 'Test Model'
            analysis.analysis_goal = 'predict'
            db.session.add(analysis)
            db.session.commit()
            # Store IDs for testing cascade deletion
            _ = user.id
            analysis_id = analysis.id
            # Delete user
            db.session.delete(user)
            db.session.commit()
            # Check that analysis was also deleted
            remaining_analysis = db.session.get(Analysis, analysis_id)
            assert remaining_analysis is None
    def test_database_constraints(self, app):
        """Test database constraints."""
        with app.app_context():
            # Test unique username constraint
            user1 = User()
            user1.username = 'uniquetest'
            user1.email = 'unique1@example.com'
            user1.set_password('password')
            db.session.add(user1)
            db.session.commit()
            # Try to create another user with same username
            user2 = User()
            user2.username = 'uniquetest'  # Same username
            user2.email = 'unique2@example.com'
            user2.set_password('password')
            db.session.add(user2)
            with pytest.raises(Exception):  # Should raise integrity error
                db.session.commit()
class TestPerformanceIntegration:
    """Test performance with realistic scenarios."""
    def test_multiple_concurrent_analyses(self, client):
        """Test handling multiple analyses."""
        analysis_data = {
            'research_question': 'Performance test question',
            'analysis_goal': 'predict',
            'dependent_variable_type': 'continuous',
            'independent_variables': 'continuous',
            'sample_size': '500',
            'missing_data': 'little',
            'data_distribution': 'normal',
            'relationship_type': 'linear',
            'variables_correlated': 'no'
        }
        # Submit multiple analyses
        for i in range(5):
            modified_data = analysis_data.copy()
            modified_data['research_question'] = f'Performance test question {i}'
            response = client.post('/results', data=modified_data)
            assert response.status_code == 200
    def test_large_form_data_handling(self, client):
        """Test handling of large form data."""
        large_question = 'What factors predict customer satisfaction? ' * 100  # Very long question
        analysis_data = {
            'research_question': large_question,
            'analysis_goal': 'predict',
            'dependent_variable_type': 'continuous',
            'independent_variables': 'continuous',
            'sample_size': '500',
            'missing_data': 'little',
            'data_distribution': 'normal',
            'relationship_type': 'linear',
            'variables_correlated': 'no'
        }
        response = client.post('/results', data=analysis_data)
        assert response.status_code == 200  # Should handle gracefully
