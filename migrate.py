from app import create_app
from models import db
import os
from sqlalchemy import inspect

def migrate():
    """Recreate database tables"""
    app = create_app()
    
    # Get the database file path
    db_path = os.path.join(os.path.dirname(__file__), 'users.db')
    
    # Remove existing database if it exists
    if os.path.exists(db_path):
        os.remove(db_path)
        print(f"Removed existing database at {db_path}")
    
    # Create new database with updated schema
    with app.app_context():
        # Drop all tables first to ensure clean slate
        db.drop_all()
        
        # Create all tables with correct schema
        db.create_all()
        
        # Verify tables were created
        inspector = inspect(db.engine)
        tables = inspector.get_table_names()
        print(f"Created tables: {', '.join(tables)}")
        
        # Verify User table schema
        columns = [col['name'] for col in inspector.get_columns('user')]
        print(f"User table columns: {', '.join(columns)}")
        
        print("Created new database with updated schema")

if __name__ == "__main__":
    migrate() 