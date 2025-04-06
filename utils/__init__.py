import json
import os
from flask import current_app
import logging

logger = logging.getLogger(__name__)

def get_statistics():
    """
    Calculate statistics for the application dashboard
    Returns a dictionary containing various statistics
    """
    try:
        # Get the path to model_database.json
        model_db_path = os.path.join(current_app.root_path, 'model_database.json')
        
        # Initialize statistics
        stats = {
            'models_count': 0,
            'access_hours': '24/7',  # Default value
            'verification_rate': '100%'  # Default value
        }
        
        # Read and count models from the database
        if os.path.exists(model_db_path):
            with open(model_db_path, 'r') as f:
                model_db = json.load(f)
                
                # Count all models recursively
                def count_models(data):
                    count = 0
                    if isinstance(data, dict):
                        # Count items that have 'implementation' or 'description' keys
                        if 'implementation' in data or 'description' in data:
                            count += 1
                        # Recursively count models in nested dictionaries
                        for value in data.values():
                            count += count_models(value)
                    return count
                
                stats['models_count'] = count_models(model_db)
                
        return stats
    
    except Exception as e:
        logger.error(f"Error calculating statistics: {e}")
        return {
            'models_count': 0,
            'access_hours': 'N/A',
            'verification_rate': 'N/A'
        }