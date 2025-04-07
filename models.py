from flask_login import UserMixin
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
import json
from werkzeug.security import generate_password_hash, check_password_hash

db = SQLAlchemy()

class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(128))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    _is_admin = db.Column(db.Boolean, default=False)
    _is_expert = db.Column(db.Boolean, default=False)
    is_approved_expert = db.Column(db.Boolean, default=False)
    areas_of_expertise = db.Column(db.Text)
    institution = db.Column(db.String(200))
    bio = db.Column(db.Text)
    
    # Relationships
    analyses = db.relationship('Analysis', backref='user', lazy=True)
    requested_consultations = db.relationship('Consultation', backref='requester', foreign_keys='Consultation.requester_id')
    expert_consultations = db.relationship('Consultation', backref='expert', foreign_keys='Consultation.expert_id')
    expert_applications = db.relationship('ExpertApplication', backref='user', lazy=True)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

    @property
    def is_admin(self):
        return self._is_admin

    @property
    def is_expert(self):
        return self._is_expert and self.is_approved_expert

    def __repr__(self):
        return f'<User {self.username}>'

class ExpertApplication(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    email = db.Column(db.String(120), nullable=False)
    areas_of_expertise = db.Column(db.Text, nullable=False)
    institution = db.Column(db.String(200))
    bio = db.Column(db.Text)
    status = db.Column(db.String(20), default='pending')  # pending, approved, rejected
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class Analysis(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    research_question = db.Column(db.String(500), nullable=False)
    analysis_goal = db.Column(db.String(50))
    dependent_variable = db.Column(db.String(50))
    independent_variables = db.Column(db.Text)  # Store as JSON string
    sample_size = db.Column(db.String(20))
    missing_data = db.Column(db.String(50))
    data_distribution = db.Column(db.String(50))
    relationship_type = db.Column(db.String(50))
    recommended_model = db.Column(db.String(100))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    consultations = db.relationship('Consultation', backref='analysis', lazy=True)

    def __repr__(self):
        return f'<Analysis {self.id}: {self.research_question}>'

class Consultation(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    requester_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    expert_id = db.Column(db.Integer, db.ForeignKey('user.id'))
    analysis_id = db.Column(db.Integer, db.ForeignKey('analysis.id'), nullable=True)
    title = db.Column(db.String(200), nullable=False)
    description = db.Column(db.Text)
    status = db.Column(db.String(20), default='pending')  # pending, in_progress, completed, cancelled
    response = db.Column(db.Text)
    is_public = db.Column(db.Boolean, default=False)
    analysis_goal = db.Column(db.String(50))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def __repr__(self):
        return f'<Consultation {self.id}: {self.title}>'

def get_model_details(model_name):
    try:
        with open('model_database.json', 'r') as f:
            models = json.load(f)
        
        # Get the model directly from the dictionary
        if model_name in models:
            return models[model_name]
        return None
    except Exception as e:
        print(f"Error loading model details: {e}")
        return None 