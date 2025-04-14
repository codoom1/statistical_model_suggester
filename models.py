from flask_login import UserMixin
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
import json
from werkzeug.security import generate_password_hash, check_password_hash
from sqlalchemy import Index, TEXT, text

db = SQLAlchemy()

class User(db.Model, UserMixin):
    __tablename__ = 'users'
    
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False, index=True)
    email = db.Column(db.String(120), unique=True, nullable=False, index=True)
    password_hash = db.Column(db.String(256))  # Increased for future-proofing
    created_at = db.Column(db.DateTime, default=datetime.utcnow, index=True)
    _is_admin = db.Column(db.Boolean, default=False, index=True)
    _is_expert = db.Column(db.Boolean, default=False, index=True)
    is_approved_expert = db.Column(db.Boolean, default=False)
    areas_of_expertise = db.Column(db.Text)
    institution = db.Column(db.String(200))
    bio = db.Column(db.Text)
    last_login = db.Column(db.DateTime)
    
    # Relationships
    analyses = db.relationship('Analysis', backref='user', lazy=True, cascade='all, delete-orphan')
    requested_consultations = db.relationship('Consultation', backref='requester', foreign_keys='Consultation.requester_id', cascade='all, delete-orphan')
    expert_consultations = db.relationship('Consultation', backref='expert', foreign_keys='Consultation.expert_id')
    expert_applications = db.relationship('ExpertApplication', backref='user', lazy=True, cascade='all, delete-orphan')
    questionnaires = db.relationship('Questionnaire', backref='user', lazy=True, cascade='all, delete-orphan')

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
    __tablename__ = 'expert_applications'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id', ondelete='CASCADE'), nullable=False, index=True)
    email = db.Column(db.String(120), nullable=False)
    areas_of_expertise = db.Column(db.Text, nullable=False)
    institution = db.Column(db.String(200))
    bio = db.Column(db.Text)
    resume_url = db.Column(db.String(500))  # URL or path to uploaded resume file
    admin_notes = db.Column(db.Text)  # Notes from admin during review
    status = db.Column(db.String(20), default='pending', index=True)  # pending, needs_info, approved, rejected
    created_at = db.Column(db.DateTime, default=datetime.utcnow, index=True)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Add index for quick filtering of pending applications
    __table_args__ = (
        Index('idx_expert_applications_status_created', 'status', 'created_at'),
    )

class Analysis(db.Model):
    __tablename__ = 'analyses'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id', ondelete='CASCADE'), nullable=False, index=True)
    research_question = db.Column(db.String(500), nullable=False)
    analysis_goal = db.Column(db.String(50), index=True)
    dependent_variable = db.Column(db.String(50))
    independent_variables = db.Column(db.Text)  # Store as JSON string
    sample_size = db.Column(db.String(20))
    missing_data = db.Column(db.String(50))
    data_distribution = db.Column(db.String(50))
    relationship_type = db.Column(db.String(50))
    variables_correlated = db.Column(db.String(20), default='unknown')
    recommended_model = db.Column(db.String(100), index=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow, index=True)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    consultations = db.relationship('Consultation', backref='analysis', lazy=True, cascade='all, delete-orphan')

    # Add PostgreSQL-specific full-text search index for research questions
    __table_args__ = (
        Index('idx_analysis_research_question_gin', 'research_question', postgresql_using='gin', 
              postgresql_ops={'research_question': 'gin_trgm_ops'}),
    )

    def __repr__(self):
        return f'<Analysis {self.id}: {self.research_question}>'

class Consultation(db.Model):
    __tablename__ = 'consultations'
    
    id = db.Column(db.Integer, primary_key=True)
    requester_id = db.Column(db.Integer, db.ForeignKey('users.id', ondelete='CASCADE'), nullable=False, index=True)
    expert_id = db.Column(db.Integer, db.ForeignKey('users.id', ondelete=None), index=True)
    analysis_id = db.Column(db.Integer, db.ForeignKey('analyses.id', ondelete='SET NULL'), nullable=True, index=True)
    title = db.Column(db.String(200), nullable=False)
    description = db.Column(db.Text)
    status = db.Column(db.String(20), default='pending', index=True)  # pending, in_progress, completed, cancelled
    response = db.Column(db.Text)
    is_public = db.Column(db.Boolean, default=False, index=True)
    analysis_goal = db.Column(db.String(50))
    created_at = db.Column(db.DateTime, default=datetime.utcnow, index=True)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Composite index for finding consultations by expert and status
    __table_args__ = (
        Index('idx_consultations_expert_status', 'expert_id', 'status'),
    )
    
    def __repr__(self):
        return f'<Consultation {self.id}: {self.title}>'

class Questionnaire(db.Model):
    __tablename__ = 'questionnaires'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id', ondelete='CASCADE'), nullable=False, index=True)
    title = db.Column(db.String(200), nullable=False)
    topic = db.Column(db.String(200), nullable=False, index=True)
    description = db.Column(db.Text)
    target_audience = db.Column(db.String(200))
    purpose = db.Column(db.String(200))
    content = db.Column(db.JSON)  # Store the questionnaire structure as JSON
    is_ai_enhanced = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow, index=True)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Full-text search index for PostgreSQL
    __table_args__ = (
        Index('idx_questionnaire_title_description', 'title', 'topic', postgresql_using='gin',
              postgresql_ops={'title': 'gin_trgm_ops', 'topic': 'gin_trgm_ops'}),
    )
    
    def __repr__(self):
        return f'<Questionnaire {self.id}: {self.title}>'

# PostgreSQL-specific function to initialize extensions
def initialize_postgres_extensions(app):
    """Initialize PostgreSQL extensions for full-text search"""
    if 'postgresql' in app.config['SQLALCHEMY_DATABASE_URI']:
        try:
            with app.app_context():
                db.session.execute(text('CREATE EXTENSION IF NOT EXISTS pg_trgm'))
                db.session.commit()
                print("PostgreSQL extensions initialized successfully")
        except Exception as e:
            print(f"Error initializing PostgreSQL extensions: {e}")

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