import os
import sys
from flask import Flask, render_template, send_from_directory, request
from flask_login import LoginManager
from flask_mail import Mail
from flask_migrate import Migrate
from models import db, User, get_model_details, initialize_postgres_extensions
import json
import argparse
import logging
from utils.email_service import init_mail
from dotenv import load_dotenv
import datetime

# Load environment variables from .env file
load_dotenv()
print("Environment variables loaded:")
print(f"HUGGINGFACE_API_KEY: {'Set' if os.environ.get('HUGGINGFACE_API_KEY') else 'Not set'}")
print(f"HUGGINGFACE_MODEL: {'Set' if os.environ.get('HUGGINGFACE_MODEL') else 'Not set'}")
print(f"AI_ENHANCEMENT_ENABLED: {'Set' if os.environ.get('AI_ENHANCEMENT_ENABLED') else 'Not set'}")

def create_app():
    app = Flask(__name__)
    app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'default-dev-key-change-in-production')
    
    # Configure database - prefer PostgreSQL for production, fallback to SQLite for development
    database_url = os.environ.get('DATABASE_URL')
    if database_url and database_url.startswith('postgres://'):
        # Heroku-style PostgreSQL URL needs to be updated for SQLAlchemy 1.4+
        database_url = database_url.replace('postgres://', 'postgresql://', 1)
    
    app.config['SQLALCHEMY_DATABASE_URI'] = database_url or 'sqlite:///users.db'
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
    
    # Configure Flask-Mail
    app.config['MAIL_SERVER'] = os.environ.get('MAIL_SERVER', 'smtp.gmail.com')
    app.config['MAIL_PORT'] = int(os.environ.get('MAIL_PORT', 587))
    app.config['MAIL_USE_TLS'] = os.environ.get('MAIL_USE_TLS', 'True').lower() == 'true'
    app.config['MAIL_USERNAME'] = os.environ.get('MAIL_USERNAME', '')
    app.config['MAIL_PASSWORD'] = os.environ.get('MAIL_PASSWORD', '')
    app.config['MAIL_DEFAULT_SENDER'] = os.environ.get('MAIL_DEFAULT_SENDER', 'noreply@statisticalmodelsuggester.com')
    
    # Set up logging
    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger(__name__)
    
    # Initialize database
    db.init_app(app)
    
    # Initialize Flask-Migrate
    migrate = Migrate(app, db)
    
    # Initialize PostgreSQL extensions if using PostgreSQL
    if database_url and ('postgresql://' in database_url or 'postgres://' in database_url):
        logger.info("PostgreSQL database detected, initializing extensions...")
        initialize_postgres_extensions(app)
    
    # Initialize mail
    init_mail(app)
      # Initialize login manager
    login_manager = LoginManager()
    login_manager.login_view = 'auth.login'  # type: ignore
    login_manager.login_message_category = 'info'
    login_manager.init_app(app)

    @login_manager.user_loader
    def load_user(user_id):
        logger.debug(f"Loading user with ID: {user_id}")
        try:
            # Use Session.get() instead of Query.get() (SQLAlchemy 2.0 compatibility)
            return db.session.get(User, int(user_id))
        except Exception as e:
            logger.error(f"Error loading user: {e}")
            return None

    # Create database tables
    with app.app_context():
        logger.debug("Creating database tables...")
        try:
            db.create_all()
            logger.debug("Database tables created successfully")
            
            # Create admin user if it doesn't exist
            admin_username = os.environ.get('ADMIN_USERNAME', 'admin')
            admin_email = os.environ.get('ADMIN_EMAIL', 'admin@example.com')
            admin_password = os.environ.get('ADMIN_PASSWORD')
            
            # Check if admin credentials are properly configured
            if not admin_password:
                logger.warning("ADMIN_PASSWORD environment variable not set. Using default password for admin account.")
                logger.warning("This is insecure. Please set ADMIN_PASSWORD in your environment variables.")
                admin_password = 'admin123'  # Default password, should be changed
              # Check if admin user exists
            admin_user = User.query.filter_by(username=admin_username).first()
            if not admin_user:
                logger.info(f"Creating admin user '{admin_username}'")
                admin_user = User()
                admin_user.username = admin_username
                admin_user.email = admin_email
                admin_user._is_admin = True
                admin_user.set_password(admin_password)
                db.session.add(admin_user)
                db.session.commit()
                logger.info("Admin user created successfully")
            else:
                logger.info(f"Admin user '{admin_username}' already exists")
                
        except Exception as e:
            logger.error(f"Error creating database tables or admin user: {e}")
            raise

    # Register blueprints
    from routes.auth_routes import auth
    from routes.main_routes import main
    from routes.user_routes import user
    from routes.expert_routes import expert
    from routes.admin_routes import admin
    from routes.questionnaire_routes import questionnaire_bp
    from routes.chatbot_routes import chatbot_bp
    
    app.register_blueprint(auth, url_prefix='/auth')
    app.register_blueprint(main, url_prefix='/')
    app.register_blueprint(user, url_prefix='/user')
    app.register_blueprint(expert, url_prefix='/expert')
    app.register_blueprint(admin, url_prefix='/admin')
    app.register_blueprint(questionnaire_bp, url_prefix='/questionnaire')
    app.register_blueprint(chatbot_bp, url_prefix='/chatbot')

    # Make model_groups available globally to all templates
    @app.context_processor
    def inject_model_groups_global():
        from routes.main_routes import MODEL_GROUPS
        return dict(model_groups=MODEL_GROUPS)

    # Add Jinja2 custom filters
    @app.template_filter('nl2br')
    def nl2br_filter(s):
        if s is None:
            return ""
        return s.replace('\n', '<br>')

    # Define error handler
    @app.errorhandler(404)
    def page_not_found(e):
        logger.warning(f"404 error: {e}")
        return render_template('error.html', error="Page not found"), 404

    @app.errorhandler(500)
    def internal_server_error(e):
        logger.error(f"500 error: {e}")
        return render_template('error.html', error=str(e)), 500
        
    # Diagnostic route to check database connection
    @app.route('/system/check-db')
    def check_db_connection():
        try:
            # Try to query the database
            user_count = User.query.count()
            db_uri = app.config['SQLALCHEMY_DATABASE_URI']
            # Hide password in logs/output
            if 'postgresql://' in db_uri:
                # Mask the password in the connection string
                masked_uri = db_uri.replace('://', '://').split('@')
                credentials = masked_uri[0].split(':')
                if len(credentials) > 2:
                    masked_uri[0] = f"{credentials[0]}:{credentials[1]}:****"
                db_uri = '@'.join(masked_uri)
            else:
                db_uri = 'sqlite:///users.db' if 'sqlite:///users.db' in db_uri else 'custom-sqlite-path'
                
            return {
                'status': 'ok',
                'database_type': 'PostgreSQL' if 'postgresql://' in app.config['SQLALCHEMY_DATABASE_URI'] else 'SQLite', 
                'connection': db_uri,
                'user_count': user_count,
                'timestamp': datetime.datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Database connection error: {e}")
            return {
                'status': 'error',
                'error': str(e),
                'database_url': app.config['SQLALCHEMY_DATABASE_URI'].split('@')[0].split(':')[0] + '://****' 
            }, 500

    # Load model database
    try:
        model_db_path = os.path.join(os.path.dirname(__file__), 'data', 'model_database.json')
        logger.debug(f"Loading model database from {model_db_path}")
        
        if not os.path.exists(model_db_path):
            logger.warning(f"model_database.json file not found at {model_db_path}, creating a default empty database")
            # Create a default empty database file
            with open(model_db_path, 'w') as f:
                json.dump({}, f)
            app.config['MODEL_DATABASE'] = {}
        else:
            with open(model_db_path, 'r') as f:
                models_data = json.load(f)
                logger.debug(f"Loaded {len(models_data)} models from model_database.json")
                app.config['MODEL_DATABASE'] = models_data
    except Exception as e:
        logger.error(f"Error loading model database: {e}")
        app.config['MODEL_DATABASE'] = {}

    # Add static file serving verification
    @app.route('/test-image')
    def test_image():
        try:
            logger.debug("Testing image serving...")
            return send_from_directory('static/images', 'stats-background.png')
        except Exception as e:
            logger.error(f"Error serving image: {e}")
            return str(e), 500

    @app.before_request
    def log_static_requests():
        if request.path.startswith('/static/'):
            logger.debug(f"Serving static file: {request.path}")

    return app

# Create the app instance at the module level for Gunicorn
app = create_app()

if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run the Statistical Model Suggester application')
    parser.add_argument('--port', type=int, default=int(os.environ.get('PORT', 8084)), help='Port to run the application on')
    args = parser.parse_args()
    
    print(f"Starting application on port {args.port}")
    app.run(host='0.0.0.0', debug=os.environ.get('FLASK_DEBUG', 'True').lower() == 'true', port=args.port) 