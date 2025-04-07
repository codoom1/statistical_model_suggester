#!/usr/bin/env python3
from app import create_app
from models import db
import sqlite3

def migrate_database():
    """Add email column to expert_application table"""
    print("Starting database migration...")
    
    # Create app context
    app = create_app()
    
    with app.app_context():
        try:
            # Connect to SQLite database
            conn = sqlite3.connect('users.db')
            cursor = conn.cursor()
            
            # Check if email column exists
            cursor.execute("PRAGMA table_info(expert_application)")
            columns = [col[1] for col in cursor.fetchall()]
            
            if 'email' not in columns:
                print("Adding email column to expert_application table...")
                cursor.execute("ALTER TABLE expert_application ADD COLUMN email TEXT DEFAULT ''")
                conn.commit()
                print("Column added successfully.")
            else:
                print("Email column already exists in expert_application table.")
            
            conn.close()
            print("Migration completed successfully.")
        except Exception as e:
            print(f"Error during migration: {e}")
            raise

if __name__ == "__main__":
    migrate_database() 