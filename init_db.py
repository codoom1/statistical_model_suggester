#!/usr/bin/env python
"""
Database initialization and migration script for local development
"""
import logging
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def init_database():
    """Initialize the database and run migrations"""
    try:
        # Import migration function
        from migrations import run_migrations
        
        # Run migrations
        success = run_migrations()
        
        if success:
            logger.info("Database initialized successfully!")
            return 0
        else:
            logger.error("Database initialization failed")
            return 1
            
    except Exception as e:
        logger.error(f"Error initializing database: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(init_database()) 