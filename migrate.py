from flask_migrate import Migrate
from app import create_app, db
import os
import subprocess

def setup_migrations():
    """Set up and run database migrations"""
    app = create_app()
    
    # Configure Flask-Migrate
    migrate = Migrate(app, db)
    
    with app.app_context():
        # Check if migrations directory exists
        if not os.path.exists('migrations'):
            print("Initializing migrations directory...")
            result = subprocess.run(['flask', 'db', 'init'], check=True)
            if result.returncode != 0:
                print("Error initializing migrations")
                return
        
        # Create a migration script
        print("Creating migration script...")
        result = subprocess.run(['flask', 'db', 'migrate', '-m', "Add resume_url and admin_notes to expert applications"], check=True)
        if result.returncode != 0:
            print("Error creating migration script")
            return
        
        # Apply the migration
        print("Applying migration...")
        result = subprocess.run(['flask', 'db', 'upgrade'], check=True)
        if result.returncode != 0:
            print("Error applying migration")
            return
        
        print("Migration complete!")

if __name__ == '__main__':
    setup_migrations() 