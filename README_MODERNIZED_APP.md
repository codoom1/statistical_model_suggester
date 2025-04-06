# Modernized Statistical Model Suggester

This document describes the modernized structure of the Statistical Model Suggester application.

## Application Structure

We've reorganized the application to use a more maintainable, modular structure with Flask blueprints:

```
statistical_model_suggester/
├── app_new.py               # New application entry point with blueprint registration
├── models.py                # SQLAlchemy models using Flask-SQLAlchemy
├── routes/                  # Blueprint route modules
│   ├── admin_routes.py      # Admin-specific routes
│   ├── auth_routes.py       # Authentication routes
│   ├── expert_routes.py     # Expert consultation routes
│   ├── main_routes.py       # Main application routes
│   └── user_routes.py       # User profile routes
├── scripts/                 # Utility scripts
│   ├── migrate_db.py        # Database migration script
│   └── test_setup.py        # Test data setup script
├── static/                  # Static assets
│   ├── css/
│   ├── js/
│   └── img/
├── templates/               # HTML templates
│   ├── admin/               # Admin templates
│   ├── base.html            # Base template with layout
│   ├── index.html           # Home page
│   └── [other templates]
├── model_database.json      # Statistical models database
└── run_app.sh               # Script to run the application
```

## Key Improvements

1. **Separation of Concerns**:
   - Routes organized into logical blueprint modules
   - Database models separated into dedicated file
   - Application configuration centralized in create_app function

2. **Maintainability**:
   - Smaller, more focused route modules
   - Improved organization makes it easier to extend functionality
   - Clear separation between different functional areas

3. **New Features**:
   - Expert consultation system
   - Admin management interface
   - User profile management
   - User role system (admin, expert, user)

## Routes Overview

- **auth_routes.py**: Login, registration, and logout functionality
- **main_routes.py**: Core model recommendation functionality
- **user_routes.py**: User profile management
- **expert_routes.py**: Expert consultation system
- **admin_routes.py**: Admin management features

## Database Models

- **User**: Extended with role, expert approval status, expertise fields
- **Analysis**: Statistical analysis records linked to users
- **Consultation**: User requests for expert consultations

## How to Use

1. Run the migration and setup script:
   ```
   ./run_app.sh
   ```

2. Access the application:
   ```
   The application will be available at http://localhost:8084
   ```

3. Login with test accounts:
   - Admin: username=`admin`, password=`admin123`
   - Expert: username=`expert1`, password=`expert123`
   - User: username=`user1`, password=`user123`

## Blueprint Registration

The blueprints are registered in app_new.py with appropriate URL prefixes:

```python
app.register_blueprint(auth)
app.register_blueprint(main)
app.register_blueprint(user, url_prefix='/user')
app.register_blueprint(expert)
app.register_blueprint(admin)
```

## Future Development

The modernized structure makes it easier to:

1. Add new features as separate blueprints
2. Implement unit and integration tests
3. Add API endpoints for mobile/client applications
4. Reorganize into a larger package structure if needed

These improvements enhance both the user experience and the developer experience for future maintenance and expansion of the application. 