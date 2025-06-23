#!/usr/bin/env python3
import os
import json
from flask import Flask
from app import create_app
import flask.cli

# Suppress Flask CLI warning messages during testing
flask.cli.show_server_banner = lambda *args: None

def test_clustering_form_submission():
    """Test submission of the analysis form with an empty dependent variable for clustering"""
    # Create the app
    app = create_app()
    
    # Configure app for testing
    app.config['TESTING'] = True
    app.config['WTF_CSRF_ENABLED'] = False  # Disable CSRF for testing
    
    # Create test client
    with app.test_client() as client:
        # Prepare form data with empty dependent_variable_type
        form_data = {
            'research_question': 'How can I cluster my customer data?',
            'analysis_goal': 'cluster',
            'dependent_variable_type': '',  # Empty dependent variable
            'independent_variables': 'continuous',  # Use string for form submission
            'sample_size': '150',
            'missing_data': 'none',
            'data_distribution': 'non_normal',
            'relationship_type': 'non_linear',
            'variables_correlated': 'unknown'
        }
        
        # Submit the form
        print("Submitting form with empty dependent variable for clustering...")
        
        try:
            response = client.post('/results', data=form_data, follow_redirects=True)
            
            # Check if the response contains the expected content
            if response.status_code == 200:
                content = response.data.decode('utf-8')
                
                # Look for evidence of successful handling
                if "Cluster Analysis" in content:
                    print("✅ SUCCESS: Form handled the empty dependent variable correctly")
                    print("Recommended model: Cluster Analysis")
                    return True
                else:
                    print("❌ ERROR: Form submission did not recommend a clustering model")
                    return False
            else:
                print(f"❌ ERROR: Form submission failed with status code {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ ERROR: Exception during form submission: {e}")
            return False

if __name__ == "__main__":
    print("Testing clustering form submission...")
    print("===================================")
    result = test_clustering_form_submission()
    print("===================================")
    print(f"Test {'passed' if result else 'failed'}") 