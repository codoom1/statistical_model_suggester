from app import app, db
from models import Analysis
import sqlite3
import os
from flask import current_app
import glob

def list_sqlite_files():
    """Find all SQLite database files in the current directory and subdirectories"""
    sqlite_files = []
    
    # Look for .db files
    for file in glob.glob("**/*.db", recursive=True):
        sqlite_files.append(file)
    
    # Look for files without extension that might be SQLite
    for file in glob.glob("**/*", recursive=True):
        if os.path.isfile(file) and not os.path.splitext(file)[1]:
            try:
                conn = sqlite3.connect(file)
                cursor = conn.cursor()
                cursor.execute("PRAGMA database_list")
                conn.close()
                sqlite_files.append(file)
            except sqlite3.Error:
                pass
    
    return sqlite_files

def inspect_db_file(db_path):
    """Check tables in the given database file"""
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # List all tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        
        print(f"Tables in {db_path}:")
        for table in tables:
            print(f"  - {table[0]}")
            
            # List columns for analyses table if it exists
            if table[0] == 'analyses':
                cursor.execute(f"PRAGMA table_info({table[0]})")
                columns = cursor.fetchall()
                print(f"    Columns in 'analyses' table:")
                for column in columns:
                    print(f"      - {column[1]} ({column[2]})")
        
        conn.close()
        return [table[0] for table in tables]
    except sqlite3.Error as e:
        print(f"Error inspecting {db_path}: {e}")
        return []

def add_variables_correlated_column():
    """Add variables_correlated column to analyses table if it doesn't exist"""
    with app.app_context():
        # First try to get the database from config
        db_uri = current_app.config.get('SQLALCHEMY_DATABASE_URI', '')
        if db_uri.startswith('sqlite:///'):
            config_db_path = db_uri.replace('sqlite:///', '')
            if config_db_path:
                print(f"Database path from config: {config_db_path}")
                
                # Check if it's an absolute or relative path
                if not os.path.isabs(config_db_path):
                    # Try relative to the current directory
                    abs_path = os.path.join(os.getcwd(), config_db_path)
                    if os.path.exists(abs_path):
                        config_db_path = abs_path
                
                if os.path.exists(config_db_path):
                    tables = inspect_db_file(config_db_path)
                    if 'analyses' in tables:
                        update_db(config_db_path)
                        return
                else:
                    print(f"Database file from config not found: {config_db_path}")
        
        # If we get here, we need to search for database files
        print("Searching for SQLite database files...")
        db_files = list_sqlite_files()
        
        if not db_files:
            print("No SQLite database files found!")
            return
        
        print(f"Found {len(db_files)} potential database files:")
        for i, file in enumerate(db_files):
            print(f"{i+1}. {file}")
            tables = inspect_db_file(file)
            
            # If we find a database with the analyses table, use it
            if 'analyses' in tables:
                print(f"Found 'analyses' table in {file}")
                update_db(file)
                return
        
        print("No database with 'analyses' table found.")
        print("Please check if the application is correctly set up and the database has been initialized.")

def update_db(db_path):
    """Update the specific database file to add the variables_correlated column"""
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Check if column exists
        cursor.execute("PRAGMA table_info(analyses)")
        columns = [column[1] for column in cursor.fetchall()]
        
        if 'variables_correlated' not in columns:
            print(f"Adding 'variables_correlated' column to 'analyses' table in {db_path}...")
            cursor.execute("ALTER TABLE analyses ADD COLUMN variables_correlated VARCHAR(20) DEFAULT 'unknown'")
            conn.commit()
            print("Column added successfully!")
        else:
            print("Column 'variables_correlated' already exists.")
        
        conn.close()
        print("Database update completed.")
    except sqlite3.Error as e:
        print(f"Error updating database: {e}")

if __name__ == "__main__":
    add_variables_correlated_column() 