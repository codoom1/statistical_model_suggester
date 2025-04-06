#!/usr/bin/env python3
import os
import sys

# Add parent directory to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from werkzeug.security import generate_password_hash
import datetime
from app import create_app
from models import db, User

def setup_test_data():
    """Set up test data for expert consultations feature"""
    
    app = create_app()
    
    with app.app_context():
        print("Setting up test data...")
        
        # Create admin user
        admin_exists = False
        admin_count = User.query.filter_by(role='admin').count()
        
        if admin_count == 0:
            print("Creating admin user...")
            admin = User(
                username="admin",
                email="admin@example.com",
                password_hash=generate_password_hash("admin123"),
                role="admin",
                created_at=datetime.datetime.utcnow()
            )
            db.session.add(admin)
            print("Admin user created with username 'admin' and password 'admin123'")
        else:
            print("Admin user already exists")
            admin_exists = True
        
        # Create sample expert user that's already approved
        expert_exists = User.query.filter_by(username='expert1').first() is not None
        
        if not expert_exists:
            print("Creating sample expert user...")
            expert = User(
                username="expert1",
                email="expert1@example.com",
                password_hash=generate_password_hash("expert123"),
                role="expert",
                is_approved_expert=True,
                expertise="Linear regression, Logistic regression, ANOVA",
                bio="PhD in Statistics with 10 years of experience in statistical consulting.",
                institution="University of Statistics",
                created_at=datetime.datetime.utcnow()
            )
            db.session.add(expert)
            print("Expert user created with username 'expert1' and password 'expert123'")
        else:
            print("Expert user already exists")
        
        # Create regular user
        user_exists = User.query.filter_by(username='user1').first() is not None
        
        if not user_exists:
            print("Creating sample regular user...")
            user = User(
                username="user1",
                email="user1@example.com",
                password_hash=generate_password_hash("user123"),
                role="user",
                created_at=datetime.datetime.utcnow()
            )
            db.session.add(user)
            print("Regular user created with username 'user1' and password 'user123'")
        else:
            print("Regular user already exists")
        
        # Create a pending expert application
        pending_expert_exists = User.query.filter_by(username='expert2').first() is not None
        
        if not pending_expert_exists:
            print("Creating sample pending expert user...")
            pending_expert = User(
                username="expert2",
                email="expert2@example.com",
                password_hash=generate_password_hash("expert456"),
                role="expert",
                is_approved_expert=False,
                expertise="Time series analysis, Survival analysis",
                bio="Master's degree in Biostatistics with 5 years of industry experience.",
                institution="Research Institute",
                created_at=datetime.datetime.utcnow()
            )
            db.session.add(pending_expert)
            print("Pending expert user created with username 'expert2' and password 'expert456'")
        else:
            print("Pending expert user already exists")
        
        # Commit changes
        db.session.commit()
        
        print("Test data setup completed successfully!")
        return True

if __name__ == "__main__":
    try:
        if setup_test_data():
            print("\nYou can now test the application with the following users:")
            print("Admin:   username='admin', password='admin123'")
            print("Expert:  username='expert1', password='expert123'")  
            print("Pending: username='expert2', password='expert456'")
            print("User:    username='user1', password='user123'")
        else:
            sys.exit(1)
    except Exception as e:
        print(f"Error during test data setup: {e}")
        sys.exit(1) 