#!/usr/bin/env python
"""
Render build script - executes necessary setup for the application
during deployment on Render.com
"""
import os
import sys
import logging
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def run_render_setup():
    logger.info("Starting Render build script")
    
    # Check environment
    is_production = os.environ.get('FLASK_ENV') == 'production'
    logger.info(f"Environment: {'Production' if is_production else 'Development'}")
    
    # Check database configuration
    database_url = os.environ.get('DATABASE_URL')
    if not database_url:
        logger.error("DATABASE_URL environment variable is not set")
        return 1
    
    # If PostgreSQL, make sure the URL is properly formatted
    if database_url.startswith('postgres://'):
        logger.info("Converting postgres:// URL to postgresql:// format for SQLAlchemy 1.4+")
        database_url = database_url.replace('postgres://', 'postgresql://', 1)
        os.environ['DATABASE_URL'] = database_url
    
    logger.info(f"Using database type: {'PostgreSQL' if 'postgresql://' in database_url else 'SQLite'}")
    
    # Wait for the database to be fully available (important for PostgreSQL on Render)
    if 'postgresql://' in database_url:
        logger.info("Waiting for PostgreSQL database to be ready...")
        retries = 5
        for i in range(retries):
            try:
                # Try importing psycopg2 for PostgreSQL connection
                try:
                    import psycopg2
                except ImportError:
                    logger.error("psycopg2 not installed. PostgreSQL connections will fail.")
                    if is_production:
                        return 1
                    else:
                        logger.warning("Skipping PostgreSQL connection test in development")
                        break
                
                conn_params = database_url.replace('postgresql://', '')
                user_pass, host_db = conn_params.split('@')
                if ':' in user_pass:
                    user, password = user_pass.split(':')
                else:
                    user, password = user_pass, ''
                
                if '/' in host_db:
                    host_port, db = host_db.split('/')
                    if ':' in host_port:
                        host, port = host_port.split(':')
                    else:
                        host, port = host_port, '5432'
                else:
                    host, port, db = host_db, '5432', ''
                
                # Connect to PostgreSQL to verify database is ready
                conn = psycopg2.connect(
                    host=host,
                    port=port,
                    user=user,
                    password=password,
                    dbname=db,
                    connect_timeout=5
                )
                conn.close()
                logger.info("Successfully connected to PostgreSQL database")
                break
            except Exception as e:
                logger.warning(f"Database connection attempt {i+1}/{retries} failed: {e}")
                if i < retries - 1:
                    wait_time = 5 * (i + 1)
                    logger.info(f"Waiting {wait_time} seconds before retrying...")
                    time.sleep(wait_time)
                else:
                    logger.error("Could not connect to database after multiple attempts")
                    if is_production:
                        return 1
      # Import the migration script
    try:
        from database_management.migrations import run_migrations
        
        # Run database migrations
        logger.info("Running database migrations...")
        success = run_migrations()
        
        if success:
            logger.info("Database migration completed successfully")
        else:
            logger.warning("Database migration had issues")
            # In development, don't fail the build for migration issues
            if not is_production:
                logger.info("Continuing with build despite migration issues (development mode)")
            else:
                logger.error("Database migration failed in production")
                return 1
    except Exception as e:
        logger.warning(f"Error running migrations: {e}")
        # In development, don't fail the build for migration issues
        if not is_production:
            logger.info("Continuing with build despite migration error (development mode)")
        else:
            logger.error("Migration error in production")
            return 1
    
    logger.info("Render build script completed successfully")
    return 0

if __name__ == "__main__":
    sys.exit(run_render_setup())