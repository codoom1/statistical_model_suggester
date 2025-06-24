#!/usr/bin/env python
"""
Render build script - executes necessary setup for the application
during deployment on Render.com
"""
import os
import sys
import logging
import time
import subprocess

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def install_ml_dependencies():
    """Install ML dependencies if enabled"""
    install_ml = os.environ.get('INSTALL_ML_DEPS', 'false').lower() == 'true'
    
    if install_ml:
        logger.info("Installing ML dependencies...")
        try:
            result = subprocess.run([
                sys.executable, '-m', 'pip', 'install', '-r', 'requirements-ml.txt'
            ], check=True, capture_output=True, text=True)
            logger.info("ML dependencies installed successfully")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to install ML dependencies: {e}")
            logger.error(f"STDOUT: {e.stdout}")
            logger.error(f"STDERR: {e.stderr}")
            return False
    else:
        logger.info("Skipping ML dependencies (INSTALL_ML_DEPS not set to 'true')")
        return True

def run_render_setup():
    logger.info("Starting Render build script")
    
    # Install ML dependencies first if requested
    if not install_ml_dependencies():
        logger.error("Failed to install ML dependencies")
        return 1
    
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
            logger.error("Database migration failed")
            if is_production:
                # In production, fail the build if migrations fail
                return 1
    except Exception as e:
        logger.error(f"Error running migrations: {e}")
        if is_production:
            # In production, fail the build if there's an error
            return 1
    
    logger.info("Render build script completed successfully")
    return 0

if __name__ == "__main__":
    sys.exit(run_render_setup())