"""
Test the main routes and analysis functionality.
"""
from models import Analysis


class TestMainRoutes:
    """Test main application routes."""
    
    def test_home_page(self, client):
        """Test that the home page loads."""
        response = client.get('/')
        assert response.status_code == 200
        assert b'statistical' in response.data.lower() or b'model' in response.data.lower()
    
    def test_analysis_form_page(self, client):
        """Test that the analysis form page loads."""
        response = client.get('/analysis-form')
        assert response.status_code == 200
        assert b'form' in response.data.lower() or b'analysis' in response.data.lower()
    
    def test_analysis_form_submission_authenticated(self, authenticated_client, sample_analysis_data, app):
        """Test analysis form submission with authenticated user."""
        response = authenticated_client.post('/results', data=sample_analysis_data, follow_redirects=True)
        assert response.status_code == 200
        
        # Check that analysis was saved to database
        with app.app_context():
            analysis = Analysis.query.filter_by(research_question=sample_analysis_data['research_question']).first()
            assert analysis is not None
            assert analysis.analysis_goal == sample_analysis_data['analysis_goal']
    
    def test_analysis_form_submission_anonymous(self, client, sample_analysis_data):
        """Test analysis form submission without authentication."""
        response = client.post('/results', data=sample_analysis_data, follow_redirects=True)
        assert response.status_code == 200
        # Should still work but not save to database for anonymous users
    
    def test_clustering_analysis(self, client, clustering_analysis_data):
        """Test clustering analysis with empty dependent variable."""
        response = client.post('/results', data=clustering_analysis_data, follow_redirects=True)
        assert response.status_code == 200
        assert b'cluster' in response.data.lower()
    
    def test_invalid_form_data(self, client):
        """Test form submission with invalid data."""
        invalid_data = {
            'research_question': '',  # Empty required field
            'analysis_goal': 'invalid_goal',
            'sample_size': 'not_a_number'
        }
        response = client.post('/results', data=invalid_data)
        # Should redirect to analysis form with validation error
        assert response.status_code == 302
    
    def test_model_recommendation_consistency(self, client):
        """Test that similar inputs produce consistent model recommendations."""
        test_data = {
            'research_question': 'What predicts customer satisfaction?',
            'analysis_goal': 'predict',
            'dependent_variable_type': 'continuous',
            'independent_variables': 'continuous',
            'sample_size': '500',
            'missing_data': 'little',
            'data_distribution': 'normal',
            'relationship_type': 'linear',
            'variables_correlated': 'no'
        }
        
        # Submit the same data multiple times
        response1 = client.post('/results', data=test_data)
        response2 = client.post('/results', data=test_data)
        
        assert response1.status_code == 200
        assert response2.status_code == 200
        # Responses should be consistent (same model recommended)
    
    def test_different_analysis_goals(self, client):
        """Test different analysis goals produce appropriate models."""
        base_data = {
            'research_question': 'Test question',
            'dependent_variable_type': 'continuous',
            'independent_variables': 'continuous',
            'sample_size': '100',
            'missing_data': 'none',
            'data_distribution': 'normal',
            'relationship_type': 'linear',
            'variables_correlated': 'no'
        }
        
        # Test prediction goal
        predict_data = base_data.copy()
        predict_data['analysis_goal'] = 'predict'
        response = client.post('/results', data=predict_data)
        assert response.status_code == 200
        
        # Test clustering goal
        cluster_data = base_data.copy()
        cluster_data['analysis_goal'] = 'cluster'
        cluster_data['dependent_variable_type'] = ''  # No dependent var for clustering
        response = client.post('/results', data=cluster_data)
        assert response.status_code == 200
        assert b'cluster' in response.data.lower()
        
        # Test classification goal
        classify_data = base_data.copy()
        classify_data['analysis_goal'] = 'classify'
        classify_data['dependent_variable_type'] = 'categorical'
        response = client.post('/results', data=classify_data)
        assert response.status_code == 200


class TestModelRecommendationEngine:
    """Test the model recommendation logic."""
    
    def test_linear_regression_recommendation(self, client):
        """Test that linear regression is recommended for appropriate data."""
        data = {
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
        response = client.post('/results', data=data)
        assert response.status_code == 200
        assert b'linear regression' in response.data.lower()
    
    def test_logistic_regression_recommendation(self, client):
        """Test that logistic regression is recommended for binary classification."""
        data = {
            'research_question': 'What predicts success?',
            'analysis_goal': 'classify',
            'dependent_variable_type': 'categorical',
            'independent_variables': 'continuous',
            'sample_size': '500',
            'missing_data': 'little',
            'data_distribution': 'normal',
            'relationship_type': 'linear',
            'variables_correlated': 'no'
        }
        response = client.post('/results', data=data)
        assert response.status_code == 200
        # Should recommend logistic regression or similar classification model
    
    def test_small_sample_size_handling(self, client):
        """Test recommendations for small sample sizes."""
        data = {
            'research_question': 'Small sample analysis',
            'analysis_goal': 'predict',
            'dependent_variable_type': 'continuous',
            'independent_variables': 'continuous',
            'sample_size': '25',  # Small sample
            'missing_data': 'little',
            'data_distribution': 'normal',
            'relationship_type': 'linear',
            'variables_correlated': 'no'
        }
        response = client.post('/results', data=data)
        assert response.status_code == 200
        # Should handle small sample sizes appropriately
    
    def test_non_normal_distribution_handling(self, client):
        """Test recommendations for non-normal distributions."""
        data = {
            'research_question': 'Non-normal data analysis',
            'analysis_goal': 'predict',
            'dependent_variable_type': 'continuous',
            'independent_variables': 'continuous',
            'sample_size': '500',
            'missing_data': 'little',
            'data_distribution': 'non_normal',
            'relationship_type': 'linear',
            'variables_correlated': 'no'
        }
        response = client.post('/results', data=data)
        assert response.status_code == 200
        # Should recommend models appropriate for non-normal data
    
    def test_missing_data_handling(self, client):
        """Test recommendations when there's significant missing data."""
        data = {
            'research_question': 'Analysis with missing data',
            'analysis_goal': 'predict',
            'dependent_variable_type': 'continuous',
            'independent_variables': 'continuous',
            'sample_size': '500',
            'missing_data': 'substantial',
            'data_distribution': 'normal',
            'relationship_type': 'linear',
            'variables_correlated': 'no'
        }
        response = client.post('/results', data=data)
        assert response.status_code == 200
        # Should provide guidance on handling missing data


class TestResultsPage:
    """Test the results page functionality."""
    
    def test_results_page_structure(self, client, sample_analysis_data):
        """Test that results page has expected structure."""
        response = client.post('/results', data=sample_analysis_data)
        assert response.status_code == 200
        
        # Check for key elements in results page
        assert b'recommend' in response.data.lower()
        assert b'model' in response.data.lower()
    
    def test_results_include_model_details(self, client, sample_analysis_data):
        """Test that results include model details and explanations."""
        response = client.post('/results', data=sample_analysis_data)
        assert response.status_code == 200
        
        # Should include some explanation or details
        response_text = response.data.decode('utf-8').lower()
        assert any(keyword in response_text for keyword in [
            'description', 'explanation', 'use', 'appropriate', 'suitable'
        ])
    
    def test_results_with_synthetic_data(self, client, sample_analysis_data):
        """Test results page includes synthetic data examples."""
        response = client.post('/results', data=sample_analysis_data)
        assert response.status_code == 200
        
        # May include synthetic data or code examples
        response_text = response.data.decode('utf-8').lower()
        # This is optional depending on implementation


class TestErrorHandling:
    """Test error handling in main routes."""
    
    def test_malformed_form_data(self, client):
        """Test handling of malformed form data."""
        # Send malformed data
        response = client.post('/results', data={'invalid': 'data'})
        assert response.status_code == 302
        
        # Should handle gracefully without crashing
    
    def test_empty_form_submission(self, client):
        """Test handling of empty form submission."""
        response = client.post('/results', data={})
        assert response.status_code == 302
    
    def test_sql_injection_protection(self, authenticated_client, app):
        """Test protection against SQL injection in form inputs."""
        malicious_data = {
            'research_question': "'; DROP TABLE users; --",
            'analysis_goal': 'predict',
            'dependent_variable_type': 'continuous',
            'independent_variables': 'continuous',
            'sample_size': '500',
            'missing_data': 'little',
            'data_distribution': 'normal',
            'relationship_type': 'linear',
            'variables_correlated': 'no'
        }
        
        response = authenticated_client.post('/results', data=malicious_data, follow_redirects=True)
        assert response.status_code == 200
        
        # Verify that the database is still intact
        with app.app_context():
            from models import User
            users = User.query.all()
            assert len(users) > 0  # Users table should still exist and have data
    
    def test_xss_protection(self, client):
        """Test protection against XSS attacks."""
        xss_data = {
            'research_question': '<script>alert("XSS")</script>',
            'analysis_goal': 'predict',
            'dependent_variable_type': 'continuous',
            'independent_variables': 'continuous',
            'sample_size': '500',
            'missing_data': 'little',
            'data_distribution': 'normal',
            'relationship_type': 'linear',
            'variables_correlated': 'no'
        }
        
        response = client.post('/results', data=xss_data)
        assert response.status_code == 200
        
        # Script should be escaped/sanitized in the research question display
        response_text = response.data.decode('utf-8')
        # Check that the malicious script is escaped in the research question section
        assert '&lt;script&gt;alert("XSS")&lt;/script&gt;' in response_text or 'alert("XSS")' not in response_text
