#!/usr/bin/env python
"""
Data migration script: SQLite to PostgreSQL
This script exports data from a local SQLite database and imports it to a PostgreSQL database.
Usage: python migrate_data_to_postgres.py
"""
import os
import json
import sys
import logging
from datetime import datetime
from dotenv import load_dotenv
import sqlalchemy as sa
from sqlalchemy import create_engine, MetaData, Table, select, insert
from sqlalchemy.orm import scoped_session, sessionmaker

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Constants
SQLITE_DB_PATH = "users.db"
EXPORT_PATH = "db_export.json"

def create_connection(url):
    """Create a database connection based on URL."""
    try:
        engine = create_engine(url)
        connection = engine.connect()
        return engine, connection
    except Exception as e:
        logger.error(f"Error connecting to database: {e}")
        return None, None

def get_all_tables(connection, metadata):
    """Get all tables from the database."""
    metadata.reflect(bind=connection)
    return metadata.tables

def export_data(sqlite_url):
    """Export all data from SQLite database to JSON file."""
    logger.info(f"Connecting to SQLite database: {sqlite_url}")
    engine, connection = create_connection(sqlite_url)
    
    if not connection:
        logger.error("Failed to connect to SQLite database")
        return False
    
    try:
        metadata = MetaData()
        tables = get_all_tables(connection, metadata)
        
        export_data = {}
        
        for table_name, table in tables.items():
            logger.info(f"Exporting table: {table_name}")
            
            # Get all rows
            query = select(table)
            result = connection.execute(query).fetchall()
            
            # Convert rows to dictionaries
            rows = []
            for row in result:
                row_dict = {}
                for column, value in row._mapping.items():
                    # Handle datetime objects
                    if isinstance(value, datetime):
                        row_dict[column] = value.isoformat()
                    else:
                        row_dict[column] = value
                rows.append(row_dict)
            
            export_data[table_name] = rows
            logger.info(f"Exported {len(rows)} rows from {table_name}")
        
        # Write to JSON file
        with open(EXPORT_PATH, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        logger.info(f"Data exported to {EXPORT_PATH}")
        return True
    
    except Exception as e:
        logger.error(f"Error exporting data: {e}")
        return False
    finally:
        connection.close()
        engine.dispose()

def import_data(postgres_url):
    """Import data from JSON file to PostgreSQL database."""
    logger.info(f"Connecting to PostgreSQL database")
    engine, connection = create_connection(postgres_url)
    
    if not connection:
        logger.error("Failed to connect to PostgreSQL database")
        return False
    
    try:
        # Read the exported data
        if not os.path.exists(EXPORT_PATH):
            logger.error(f"Export file {EXPORT_PATH} not found")
            return False
        
        with open(EXPORT_PATH, 'r') as f:
            export_data = json.load(f)
        
        metadata = MetaData()
        tables = get_all_tables(connection, metadata)
        
        # Start a transaction
        transaction = connection.begin()
        
        for table_name, rows in export_data.items():
            if table_name not in tables:
                logger.warning(f"Table {table_name} not found in PostgreSQL database. Skipping.")
                continue
            
            table = tables[table_name]
            logger.info(f"Importing {len(rows)} rows to table: {table_name}")
            
            # Insert each row
            for row in rows:
                # Handle SQLite rowid
                if 'rowid' in row and 'rowid' not in [c.name for c in table.columns]:
                    del row['rowid']
                
                # Convert ISO format strings back to datetime for datetime columns
                for column_name, value in list(row.items()):
                    column = table.columns.get(column_name)
                    if column and isinstance(column.type, sa.DateTime) and value:
                        try:
                            row[column_name] = datetime.fromisoformat(value)
                        except (ValueError, TypeError):
                            pass
                
                try:
                    stmt = insert(table).values(**row)
                    connection.execute(stmt)
                except Exception as e:
                    logger.error(f"Error inserting row into {table_name}: {e}")
                    logger.error(f"Row data: {row}")
                    # Continue with other rows
        
        # Commit the transaction
        transaction.commit()
        logger.info(f"Data imported successfully to PostgreSQL")
        return True
    
    except Exception as e:
        logger.error(f"Error importing data: {e}")
        if 'transaction' in locals():
            transaction.rollback()
        return False
    finally:
        connection.close()
        engine.dispose()

def main():
    """Main function."""
    # Load environment variables
    load_dotenv()
    
    # Get database URLs
    sqlite_url = f"sqlite:///{SQLITE_DB_PATH}"
    postgres_url = os.environ.get('DATABASE_URL')
    
    if not postgres_url:
        logger.error("DATABASE_URL environment variable not set")
        return 1
    
    # Make sure PostgreSQL URL is in the correct format
    if postgres_url.startswith('postgres://'):
        postgres_url = postgres_url.replace('postgres://', 'postgresql://', 1)
    
    # Export data from SQLite
    logger.info("Starting data export from SQLite...")
    export_success = export_data(sqlite_url)
    
    if not export_success:
        logger.error("Failed to export data from SQLite")
        return 1
    
    # Import data to PostgreSQL
    logger.info("Starting data import to PostgreSQL...")
    import_success = import_data(postgres_url)
    
    if not import_success:
        logger.error("Failed to import data to PostgreSQL")
        return 1
    
    logger.info("Data migration completed successfully")
    return 0

if __name__ == "__main__":
    sys.exit(main()) 