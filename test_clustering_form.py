#!/usr/bin/env python3
import sys
import json
import random
from flask import Flask, current_app
from routes.main_routes import get_model_recommendation

def test_clustering_with_no_dependent():
    """Test clustering model recommendation with no dependent variable"""
    app = Flask(__name__)
    
    # Load model database
    with open('model_database.json') as f:
        model_database = json.load(f)
    
    with app.app_context():
        app.config['MODEL_DATABASE'] = model_database
        
        # Test with empty dependent variable
        result = get_model_recommendation(
            analysis_goal='cluster',
            dependent_variable='',  # Empty dependent variable
            independent_variables=['continuous'],
            sample_size=100,
            missing_data='none',
            data_distribution='normal',
            relationship_type='linear',
            variables_correlated='no'
        )
        
        recommended_model = result[0]
        alternative_models = result[1]
        
        print(f"Test 1: Empty dependent variable")
        print(f"Recommended model: {recommended_model}")
        
        # Check if the recommended model is a clustering model
        clustering_models = ['Cluster Analysis', 'K-Means', 'Hierarchical Clustering', 
                           'DBSCAN', 'Gaussian Mixture Models']
        
        if recommended_model in clustering_models:
            print("✅ SUCCESS: Recommended a clustering model with empty dependent variable")
        else:
            print(f"❌ ERROR: Recommended {recommended_model}, which is not a clustering model")

def test_form_response():
    """Test the backend response for a form submission with no dependent variable"""
    app = Flask(__name__)
    app.config['TESTING'] = True
    
    # Load model database
    with open('model_database.json') as f:
        model_database = json.load(f)
    
    # Set up the app context and config
    with app.app_context():
        app.config['MODEL_DATABASE'] = model_database
        
        # Import routes only after app context is set up
        from routes.main_routes import results
        
        # Create a test client
        client = app.test_client()
        
        # Create a test request
        form_data = {
            'research_question': 'How can we segment customers?',
            'analysis_goal': 'cluster',
            'dependent_variable_type': '',  # Empty dependent variable
            'independent_variables': ['continuous'],
            'sample_size': '200',
            'missing_data': 'none',
            'data_distribution': 'normal',
            'relationship_type': 'non_linear',
            'variables_correlated': 'unknown'
        }
        
        # Mock the request handling
        from routes.main_routes import get_model_recommendation
        recommended_model, explanation, alternative_models = get_model_recommendation(
            form_data['analysis_goal'],
            form_data['dependent_variable_type'],
            form_data['independent_variables'],
            form_data['sample_size'],
            form_data['missing_data'],
            form_data['data_distribution'],
            form_data['relationship_type'],
            form_data['variables_correlated']
        )
        
        print(f"\nTest 2: Form submission test")
        print(f"Recommended model: {recommended_model}")
        
        # Check if the recommended model is a clustering model
        clustering_models = ['Cluster Analysis', 'K-Means', 'Hierarchical Clustering', 
                           'DBSCAN', 'Gaussian Mixture Models']
        
        if recommended_model in clustering_models:
            print("✅ SUCCESS: Form handling correctly recommended a clustering model with empty dependent variable")
        else:
            print(f"❌ ERROR: Form handling recommended {recommended_model}, which is not a clustering model")

if __name__ == "__main__":
    print("Testing clustering model changes...")
    print("===================================")
    test_clustering_with_no_dependent()
    test_form_response()
    print("===================================")
    print("Tests complete!") 