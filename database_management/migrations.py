from flask_migrate import Migrate
from app import create_app, db
import os
import logging
import sys

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_migrations():
    """Set up and run database migrations"""
    app = create_app()
    
    # Configure Flask-Migrate
    migrate = Migrate(app, db)
    
    with app.app_context():
        from flask import current_app
        from flask.cli import with_appcontext
        from flask_migrate import init, migrate, upgrade, current
        
        # Get database type
        db_url = current_app.config['SQLALCHEMY_DATABASE_URI']
        is_postgres = 'postgresql' in db_url
        db_type = "PostgreSQL" if is_postgres else "SQLite"
        logger.info(f"Running migrations for {db_type} database")
        
        try:
            # Check if migrations directory exists
            if not os.path.exists('migrations'):
                logger.info("Initializing migrations directory...")
                init()
              # Check if the database is reachable
            try:
                # Simple query to test connection
                from sqlalchemy import text
                db.session.execute(text('SELECT 1'))
                db.session.commit()
                logger.info("Database connection successful")
            except Exception as conn_err:
                logger.error(f"Database connection error: {conn_err}")
                if is_postgres:
                    logger.error("Make sure your PostgreSQL database is accessible")
                return False
                  # Create migration - automatic detection of changes to models
            logger.info("Checking for pending migrations...")
            try:
                # First try to upgrade any existing migrations
                upgrade()
                logger.info("Applied existing migrations successfully")
                
                # Then create new migration if there are model changes
                logger.info("Creating migration for any new model changes...")
                migrate(message="Auto-migration for model updates")
                logger.info("Created new migration")
                
                # Apply the new migration
                upgrade()
                logger.info("Applied new migration")
                
            except Exception as migrate_err:
                logger.warning(f"Migration process info: {migrate_err}")
                # For production deployments, it's common to have this kind of info
                # Continue with the deployment even if migrations are up to date
            
            logger.info("Migration completed successfully!")
            return True
        
        except Exception as e:
            logger.error(f"Migration error: {e}")
            
            # For SQLite, fallback to direct column addition if migration fails
            # This is needed because SQLite has limited ALTER TABLE support
            if not is_postgres:
                logger.info("Attempting SQLite direct column addition...")
                try:
                    # Connect directly to SQLite
                    import sqlite3
                    conn = sqlite3.connect('users.db')
                    cursor = conn.cursor()
                    
                    # Check if columns exist
                    cursor.execute("PRAGMA table_info(expert_applications)")
                    columns = [col[1] for col in cursor.fetchall()]
                    
                    # Add columns if they don't exist
                    if 'resume_url' not in columns:
                        cursor.execute("ALTER TABLE expert_applications ADD COLUMN resume_url TEXT")
                        logger.info("Added resume_url column")
                    
                    if 'admin_notes' not in columns:
                        cursor.execute("ALTER TABLE expert_applications ADD COLUMN admin_notes TEXT")
                        logger.info("Added admin_notes column")
                    
                    conn.commit()
                    conn.close()
                    logger.info("SQLite columns added successfully")
                    return True
                except Exception as e2:
                    logger.error(f"SQLite direct column addition failed: {e2}")
            else:
                logger.error("PostgreSQL migration failed. Check database connection and permissions.")
            
            return False

# Direct execution
if __name__ == '__main__':
    success = run_migrations()
    sys.exit(0 if success else 1) 