#!/usr/bin/env python3
import os
import sqlite3
from sqlite3 import Error
from dotenv import load_dotenv
import psycopg2

# Load environment variables
load_dotenv()

def create_connection():
    """Create a database connection based on DATABASE_URL."""
    database_url = os.environ.get('DATABASE_URL')
    
    if database_url and 'postgresql' in database_url:
        # For PostgreSQL
        if database_url.startswith('postgres://'):
            database_url = database_url.replace('postgres://', 'postgresql://', 1)
        try:
            conn = psycopg2.connect(database_url)
            return conn
        except Error as e:
            print(f"Error connecting to PostgreSQL: {e}")
            return None
    else:
        # For SQLite
        try:
            conn = sqlite3.connect('users.db')
            return conn
        except Error as e:
            print(f"Error connecting to SQLite database: {e}")
            return None

def add_columns(conn):
    """Add resume_url and admin_notes columns to expert_applications table."""
    try:
        cur = conn.cursor()
        
        # Check if columns already exist
        if isinstance(conn, sqlite3.Connection):
            # SQLite approach
            result = cur.execute("PRAGMA table_info(expert_applications)").fetchall()
            columns = [col[1] for col in result]
            
            if 'resume_url' not in columns:
                cur.execute("ALTER TABLE expert_applications ADD COLUMN resume_url TEXT")
                print("Added resume_url column to expert_applications table")
            
            if 'admin_notes' not in columns:
                cur.execute("ALTER TABLE expert_applications ADD COLUMN admin_notes TEXT")
                print("Added admin_notes column to expert_applications table")
        else:
            # PostgreSQL approach
            cur.execute("""
                DO $$
                BEGIN
                    IF NOT EXISTS (
                        SELECT 1 
                        FROM information_schema.columns 
                        WHERE table_name='expert_applications' AND column_name='resume_url'
                    ) THEN
                        ALTER TABLE expert_applications ADD COLUMN resume_url VARCHAR(500);
                        RAISE NOTICE 'Added resume_url column to expert_applications table';
                    END IF;
                    
                    IF NOT EXISTS (
                        SELECT 1 
                        FROM information_schema.columns 
                        WHERE table_name='expert_applications' AND column_name='admin_notes'
                    ) THEN
                        ALTER TABLE expert_applications ADD COLUMN admin_notes TEXT;
                        RAISE NOTICE 'Added admin_notes column to expert_applications table';
                    END IF;
                END $$;
            """)
            print("Checked and added columns to expert_applications table")
        
        conn.commit()
        return True
    except Error as e:
        print(f"Error adding columns: {e}")
        return False

def main():
    """Main function to run migrations."""
    conn = create_connection()
    if conn is not None:
        print("Connected to database successfully.")
        if add_columns(conn):
            print("Migration completed successfully.")
        else:
            print("Migration failed.")
        conn.close()
    else:
        print("Error connecting to database.")

if __name__ == '__main__':
    main() 