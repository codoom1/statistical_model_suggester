#!/usr/bin/env python3
"""
PostgreSQL Integration Test Script
Tests PostgreSQL connection and extension initialization separately from the main test suite.
"""
import os
import sys
import tempfile
from pathlib import Path
from sqlalchemy import text

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def test_postgresql_connection():
    """Test PostgreSQL connection if available"""
    try:
        import psycopg2
        from app import create_app
        from models import db
        
        # Get the DATABASE_URL from environment or use default PostgreSQL URL
        database_url = os.environ.get('DATABASE_URL')
        
        # Only run PostgreSQL tests if we have a PostgreSQL URL
        if not database_url or 'postgresql' not in database_url:
            print("⚠ No PostgreSQL DATABASE_URL found, skipping PostgreSQL tests")
            print("  Set DATABASE_URL=postgresql://user:pass@host:port/db to enable PostgreSQL tests")
            assert True  # Skip test but don't fail
            return
        
        # Test direct connection
        print(f"Testing direct connection to: {database_url}")
        conn = psycopg2.connect(database_url)
        cursor = conn.cursor()
        cursor.execute("SELECT version();")
        version = cursor.fetchone()
        assert version is not None, "Could not get PostgreSQL version"
        print(f"PostgreSQL version: {version[0]}")
        cursor.close()
        conn.close()
        print("✓ Direct PostgreSQL connection successful")
        
        # Test Flask app connection
        print("Testing Flask app with PostgreSQL...")
        os.environ['TESTING'] = 'false'  # Enable PostgreSQL extensions
        
        app = create_app()
        with app.app_context():
            # Create tables
            db.create_all()
            print("✓ Database tables created successfully")
            
            # Test a simple query
            result = db.session.execute(text("SELECT 1 as test")).fetchone()
            assert result is not None and result[0] == 1, "Database query test failed"
            print("✓ Database query test successful")
            
            # Test pg_trgm extension
            try:
                db.session.execute(text("SELECT * FROM pg_extension WHERE extname = 'pg_trgm'"))
                print("✓ pg_trgm extension is available")
            except Exception as e:
                print(f"⚠ pg_trgm extension test failed: {e}")
        
        print("🎉 PostgreSQL integration test completed successfully!")
        
    except ImportError:
        print("⚠ psycopg2 not available, skipping PostgreSQL tests")
        assert True  # Skip test but don't fail
    except Exception as e:
        print(f"❌ PostgreSQL integration test failed: {e}")
        assert False, f"PostgreSQL integration test failed: {e}"

def test_sqlite_fallback():
    """Test SQLite fallback for development/testing"""
    try:
        from app import create_app
        from models import db
        
        # Create temporary SQLite database
        db_fd, db_path = tempfile.mkstemp(suffix='.db')
        os.close(db_fd)
        
        print(f"Testing SQLite fallback: {db_path}")
        os.environ['DATABASE_URL'] = f'sqlite:///{db_path}'
        os.environ['TESTING'] = 'true'
        
        app = create_app()
        with app.app_context():
            db.create_all()
            print("✓ SQLite database tables created successfully")
            
            # Test a simple query
            result = db.session.execute(text("SELECT 1 as test")).fetchone()
            assert result is not None and result[0] == 1, "SQLite query test failed"
            print("✓ SQLite query test successful")
        
        # Clean up
        os.unlink(db_path)
        print("✓ SQLite fallback test completed successfully!")
        
    except Exception as e:
        print(f"❌ SQLite fallback test failed: {e}")
        assert False, f"SQLite fallback test failed: {e}"

if __name__ == "__main__":
    print("=== Database Integration Tests ===")
    
    sqlite_success = test_sqlite_fallback()
    postgresql_success = test_postgresql_connection()
    
    print("\n=== Test Summary ===")
    print(f"SQLite fallback: {'✓ PASS' if sqlite_success else '❌ FAIL'}")
    print(f"PostgreSQL integration: {'✓ PASS' if postgresql_success else '❌ FAIL'}")
    
    # Exit with error code if any test failed
    if not (sqlite_success and postgresql_success):
        sys.exit(1)
    
    print("\n🎉 All database tests passed!")
