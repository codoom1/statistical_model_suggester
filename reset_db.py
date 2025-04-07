from app import create_app
from models import db

def reset_database():
    """Reset the database by dropping all tables and recreating them"""
    app = create_app()
    
    with app.app_context():
        # Drop all tables
        db.drop_all()
        print("Dropped all existing tables.")
        
        # Create all tables with new schema
        db.create_all()
        print("Created all tables with new schema.")
        
        print("Database has been reset successfully.")

if __name__ == "__main__":
    reset_database() 