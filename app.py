import argparse
import datetime
import json
import logging
import os
from pathlib import Path

import click
from dotenv import load_dotenv
from flask import Flask, render_template
from flask_login import LoginManager
from flask_migrate import Migrate
from flask_wtf.csrf import CSRFError, CSRFProtect

from models import User, db, initialize_postgres_extensions
from utils.email_service import init_mail


BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")
csrf = CSRFProtect()


def _is_production() -> bool:
    return bool(os.environ.get("VERCEL")) or os.environ.get(
        "FLASK_ENV", ""
    ).lower() == "production"


def _database_url() -> str:
    database_url = os.environ.get("DATABASE_URL", "").strip()
    if database_url.startswith("postgres://"):
        database_url = database_url.replace("postgres://", "postgresql://", 1)

    if _is_production() and not database_url:
        raise RuntimeError(
            "DATABASE_URL is required in production. Configure a persistent "
            "PostgreSQL database before starting the application."
        )

    return database_url or "sqlite:///users.db"


def _secret_key() -> str:
    secret_key = os.environ.get("SECRET_KEY", "").strip()
    if _is_production() and not secret_key:
        raise RuntimeError("SECRET_KEY is required in production.")
    return secret_key or "local-development-only"


def _load_model_database() -> dict:
    model_db_path = BASE_DIR / "data" / "model_database.json"
    if not model_db_path.is_file():
        raise RuntimeError(f"Required model database is missing: {model_db_path}")

    with model_db_path.open(encoding="utf-8") as model_db_file:
        models_data = json.load(model_db_file)

    if not isinstance(models_data, dict) or not models_data:
        raise RuntimeError("The model database must be a non-empty JSON object.")
    return models_data


def _register_cli_commands(app: Flask) -> None:
    @app.cli.command("init-db")
    def init_db_command() -> None:
        """Create database tables and PostgreSQL extensions once."""
        initialize_postgres_extensions(app)
        db.create_all()
        click.echo("Database initialized.")

    @app.cli.command("create-admin")
    @click.option("--username", envvar="ADMIN_USERNAME", required=True)
    @click.option("--email", envvar="ADMIN_EMAIL", required=True)
    @click.password_option(envvar="ADMIN_PASSWORD", confirmation_prompt=False)
    def create_admin_command(username: str, email: str, password: str) -> None:
        """Create or promote an administrator explicitly."""
        admin_user = User.query.filter(
            (User.username == username) | (User.email == email)
        ).first()
        if admin_user is None:
            admin_user = User(username=username, email=email, _is_admin=True)
            db.session.add(admin_user)
        else:
            admin_user.username = username
            admin_user.email = email
            admin_user._is_admin = True

        admin_user.set_password(password)
        db.session.commit()
        click.echo(f"Administrator '{username}' is ready.")


def create_app() -> Flask:
    is_vercel = bool(os.environ.get("VERCEL"))
    app = Flask(
        __name__,
        static_folder=None if is_vercel else str(BASE_DIR / "public" / "static"),
        static_url_path="/static",
    )
    if is_vercel:
        # Vercel's CDN serves public/static; this rule only lets url_for build URLs.
        app.add_url_rule(
            "/static/<path:filename>", endpoint="static", build_only=True
        )

    app.config.update(
        SECRET_KEY=_secret_key(),
        SQLALCHEMY_DATABASE_URI=_database_url(),
        SQLALCHEMY_TRACK_MODIFICATIONS=False,
        SQLALCHEMY_ENGINE_OPTIONS={"pool_pre_ping": True},
        MAX_CONTENT_LENGTH=4 * 1024 * 1024,
        SESSION_COOKIE_HTTPONLY=True,
        SESSION_COOKIE_SAMESITE="Lax",
        SESSION_COOKIE_SECURE=_is_production(),
        REMEMBER_COOKIE_HTTPONLY=True,
        REMEMBER_COOKIE_SAMESITE="Lax",
        REMEMBER_COOKIE_SECURE=_is_production(),
        PERMANENT_SESSION_LIFETIME=datetime.timedelta(days=7),
        WTF_CSRF_ENABLED=os.environ.get(
            "WTF_CSRF_ENABLED", "true"
        ).lower() == "true",
        MAIL_SERVER=os.environ.get("MAIL_SERVER", "smtp.gmail.com"),
        MAIL_PORT=int(os.environ.get("MAIL_PORT", 587)),
        MAIL_USE_TLS=os.environ.get("MAIL_USE_TLS", "true").lower() == "true",
        MAIL_USERNAME=os.environ.get("MAIL_USERNAME", ""),
        MAIL_PASSWORD=os.environ.get("MAIL_PASSWORD", ""),
        MAIL_DEFAULT_SENDER=os.environ.get(
            "MAIL_DEFAULT_SENDER", ""
        ),
        EMAIL_PROVIDER=os.environ.get("EMAIL_PROVIDER", ""),
        RESEND_API_KEY=os.environ.get("RESEND_API_KEY", ""),
        MAIL_SUPPRESS_SEND=os.environ.get(
            "MAIL_SUPPRESS_SEND", "false"
        ).lower() == "true",
        MODEL_DATABASE=_load_model_database(),
    )

    logging.basicConfig(
        level=logging.INFO if _is_production() else logging.DEBUG
    )
    logger = app.logger

    db.init_app(app)
    Migrate(app, db)
    csrf.init_app(app)
    init_mail(app)

    login_manager = LoginManager()
    login_manager.login_view = "auth.login"  # type: ignore
    login_manager.login_message_category = "info"
    login_manager.init_app(app)

    @login_manager.user_loader
    def load_user(user_id):
        try:
            return db.session.get(User, int(user_id))
        except (TypeError, ValueError):
            return None

    from routes.admin_routes import admin
    from routes.auth_routes import auth
    from routes.chatbot_routes import chatbot_bp
    from routes.expert_routes import expert
    from routes.main_routes import main
    from routes.questionnaire_routes import questionnaire_bp
    from routes.user_routes import user

    app.register_blueprint(auth, url_prefix="/auth")
    app.register_blueprint(main, url_prefix="/")
    app.register_blueprint(user, url_prefix="/user")
    app.register_blueprint(expert, url_prefix="/expert")
    app.register_blueprint(admin, url_prefix="/admin")
    app.register_blueprint(questionnaire_bp, url_prefix="/questionnaire")
    app.register_blueprint(chatbot_bp, url_prefix="/chatbot")

    @app.context_processor
    def inject_model_groups_global():
        from routes.main_routes import MODEL_GROUPS

        return {"model_groups": MODEL_GROUPS}

    @app.template_filter("nl2br")
    def nl2br_filter(value):
        return "" if value is None else value.replace("\n", "<br>")

    @app.errorhandler(404)
    def page_not_found(error):
        logger.info("404: %s", error)
        return render_template("error.html", error="Page not found"), 404

    @app.errorhandler(413)
    def upload_too_large(_error):
        return render_template(
            "error.html", error="Uploads must be smaller than 4 MB."
        ), 413

    @app.errorhandler(CSRFError)
    def handle_csrf_error(error):
        logger.warning("CSRF validation failed: %s", error.description)
        return render_template(
            "error.html",
            error="This form expired or could not be verified. Please try again.",
        ), 400

    @app.errorhandler(500)
    def internal_server_error(error):
        logger.exception("Unhandled application error: %s", error)
        message = (
            "An unexpected error occurred. Please try again."
            if _is_production()
            else str(error)
        )
        return render_template("error.html", error=message), 500

    @app.route("/health")
    def health_check():
        try:
            db.session.execute(db.text("SELECT 1"))
            return {
                "status": "healthy",
                "timestamp": datetime.datetime.now(
                    datetime.timezone.utc
                ).isoformat(),
                "service": "statistical-model-suggester",
            }
        except Exception:
            logger.exception("Health check failed")
            return {
                "status": "unhealthy",
                "service": "statistical-model-suggester",
            }, 503

    _register_cli_commands(app)
    return app


# Recognized by Vercel and Gunicorn as the WSGI entry point.
app = create_app()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the Statistical Model Suggester application"
    )
    parser.add_argument(
        "--port", type=int, default=int(os.environ.get("PORT", 8084))
    )
    args = parser.parse_args()
    app.run(
        host="0.0.0.0",
        debug=os.environ.get("FLASK_DEBUG", "false").lower() == "true",
        port=args.port,
    )
