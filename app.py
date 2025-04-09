import os
import sys
from flask import Flask, render_template, send_from_directory, request
from flask_login import LoginManager
from flask_mail import Mail
from models import db, User, get_model_details
import json
import argparse
import logging
from utils.email import init_mail

def create_app():
    app = Flask(__name__)
    app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'default-dev-key-change-in-production')
    app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
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
    
    # Initialize mail
    init_mail(app)
    
    # Initialize login manager
    login_manager = LoginManager()
    login_manager.login_view = 'auth.login'
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
        except Exception as e:
            logger.error(f"Error creating database tables: {e}")
            raise

    # Register blueprints
    from routes.auth_routes import auth
    from routes.main_routes import main
    from routes.user_routes import user
    from routes.expert_routes import expert
    from routes.admin_routes import admin
    from routes.questionnaire_routes import questionnaire_bp
    
    app.register_blueprint(auth, url_prefix='/auth')
    app.register_blueprint(main, url_prefix='/')
    app.register_blueprint(user, url_prefix='/user')
    app.register_blueprint(expert, url_prefix='/expert')
    app.register_blueprint(admin, url_prefix='/admin')
    app.register_blueprint(questionnaire_bp, url_prefix='/questionnaire')

    # Define error handler
    @app.errorhandler(404)
    def page_not_found(e):
        logger.warning(f"404 error: {e}")
        return render_template('error.html', error="Page not found"), 404

    @app.errorhandler(500)
    def internal_server_error(e):
        logger.error(f"500 error: {e}")
        return render_template('error.html', error=str(e)), 500

    # Load model database
    try:
        model_db_path = os.path.join(os.path.dirname(__file__), 'model_database.json')
        logger.debug(f"Loading model database from {model_db_path}")
        
        if not os.path.exists(model_db_path):
            logger.error(f"model_database.json file not found at {model_db_path}")
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
    parser.add_argument('--port', type=int, default=8084, help='Port to run the application on')
    args = parser.parse_args()
    
    print(f"Starting application on port {args.port}")
    app.run(debug=True, port=args.port) 