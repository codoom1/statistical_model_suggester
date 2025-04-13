#!/usr/bin/env python3
import sys
import json
import random
from flask import Flask, current_app
from routes.main_routes import get_model_recommendation

def main():
    """Test clustering model recommendation"""
    app = Flask(__name__)
    
    # Load model database
    with open('model_database.json') as f:
        model_database = json.load(f)
    
    with app.app_context():
        app.config['MODEL_DATABASE'] = model_database
        
        # Test with cluster analysis goal
        result = get_model_recommendation(
            analysis_goal='cluster',
            dependent_variable='continuous',
            independent_variables=['continuous'],
            sample_size=100,
            missing_data='none',
            data_distribution='normal',
            relationship_type='linear',
            variables_correlated='no'
        )
        
        recommended_model = result[0]
        alternative_models = result[1]
        
        print(f"Recommended model: {recommended_model}")
        print(f"Alternative models: {alternative_models}")
        
        # Check if the recommended model is a clustering model
        clustering_models = ['Cluster Analysis', 'K-Means', 'Hierarchical Clustering', 
                            'DBSCAN', 'Gaussian Mixture Models']
        
        if recommended_model in clustering_models:
            print("SUCCESS: Recommended a clustering model")
        else:
            print(f"ERROR: Recommended {recommended_model}, which is not a clustering model")

if __name__ == "__main__":
    main() 