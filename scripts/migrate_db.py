#!/usr/bin/env python3
import os
import sys
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime

# Get the absolute path to the database file
DB_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'users.db'))

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{DB_PATH}'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(120), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    role = db.Column(db.String(20), default='user')
    is_approved_expert = db.Column(db.Boolean, default=False)
    expertise = db.Column(db.String(500), nullable=True)
    bio = db.Column(db.Text, nullable=True)
    institution = db.Column(db.String(200), nullable=True)

class Analysis(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    analysis_type = db.Column(db.String(50), nullable=False)
    data_characteristics = db.Column(db.Text, nullable=False)
    recommended_model = db.Column(db.String(100), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class Consultation(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    requester_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    expert_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=True)
    analysis_id = db.Column(db.Integer, db.ForeignKey('analysis.id'), nullable=False)
    status = db.Column(db.String(20), default='pending')
    expert_notes = db.Column(db.Text, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

def init_db():
    """Initialize the database with all required tables"""
    print(f"Initializing database at {DB_PATH}")
    
    with app.app_context():
        # Create all tables
        db.create_all()
        print("Database tables created successfully")
    
    return True

if __name__ == "__main__":
    try:
        if init_db():
            print("Database migration completed successfully!")
            sys.exit(0)
        else:
            print("Database migration failed!")
            sys.exit(1)
    except Exception as e:
        print(f"Error during database migration: {e}")
        sys.exit(1) 