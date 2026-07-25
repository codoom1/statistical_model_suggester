import logging
from pathlib import Path

from utils.model_catalog import load_model_catalog

logger = logging.getLogger(__name__)

def get_statistics():
    """
    Calculate statistics for the application dashboard
    Returns a dictionary containing various statistics
    """
    try:
        # Import Flask inside the function to avoid circular imports
        from flask import current_app        
        # Initialize statistics
        stats = {
            'models_count': 0,
            'access_hours': '24/7',  # Default value
            'verification_rate': '100%'  # Default value
        }
        
        model_db = load_model_catalog(Path(current_app.root_path))
        stats['models_count'] = len(model_db)
                
        return stats
    
    except Exception as e:
        logger.error(f"Error calculating statistics: {e}")
        return {
            'models_count': 0,
            'access_hours': 'N/A',
            'verification_rate': 'N/A'
        }
