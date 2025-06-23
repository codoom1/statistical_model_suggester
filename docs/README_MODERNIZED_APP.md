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

### Enhanced Database Models for PostgreSQL

The application's data models have been optimized for PostgreSQL to ensure optimal performance and data integrity:

1. **User Model**:
   - Indexing on frequently queried fields: username, email, created_at, role flags
   - Proper relationships with cascading deletes to maintain referential integrity
   - String length increased for password hash to accommodate future hashing algorithms
   - Tracking of last_login for user activity monitoring

2. **Analysis Model**:
   - Full-text search capabilities for research questions (using PostgreSQL GIN indexes)
   - Indexed fields for common filters (analysis_goal, recommended_model)
   - Proper cascading relationships to prevent orphaned records

3. **Consultation Model**:
   - Composite indexes for efficient filtering of expert consultations by status
   - Proper foreign key constraints with appropriate deletion behaviors
   - Indexed status and public flag fields for quick filtering

4. **ExpertApplication Model**:
   - Index on status for quick filtering of pending applications
   - Optimization for time-based queries with composite indexes

5. **Questionnaire Model (New)**:
   - Storage of questionnaire structure in native JSON format
   - Full-text search on titles and topics
   - Complete user attribution and metadata

### PostgreSQL-Specific Optimizations

The application includes PostgreSQL-specific enhancements:

1. **Full-Text Search**:
   - Utilizes PostgreSQL's `pg_trgm` extension for fuzzy text searching
   - GIN indexes on text fields for fast search operations
   - Optimized for searching research questions and questionnaire content

2. **JSON Data Type**:
   - Native JSON storage for complex questionnaire structures
   - Efficient retrieval and filtering of JSON data

3. **Intelligent Indexing**:
   - Strategic indexes on frequently queried fields
   - Composite indexes for common query patterns
   - Careful balance between query performance and write overhead

4. **Referential Integrity**:
   - Proper CASCADE and SET NULL behaviors on foreign keys
   - Automatic cleanup of related records when parent records are deleted
   - Prevention of orphaned records and data inconsistencies

5. **Extension Auto-Initialization**:
   - Automatic setup of required PostgreSQL extensions on application startup
   - Graceful fallback to basic functionality when extensions are unavailable

These optimizations ensure that the application runs efficiently on PostgreSQL while maintaining full compatibility with SQLite for development environments.

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

## Deployment to Render

### Setting Up PostgreSQL Database on Render

To ensure user data persists between service restarts, we'll use Render's free PostgreSQL database:

1. **Create a PostgreSQL Database**:
   - In your Render dashboard, click on "New" and select "PostgreSQL"
   - Set a name for your database (e.g., "stats-model-suggester-db")
   - Choose the free plan
   - Create the database

2. **Get the Connection Details**:
   - After creation, click on your database service
   - You'll see "Connection" details including:
     - Internal Database URL
     - External Database URL
     - PSQL Command
   - Copy the "External Database URL" - this is your `DATABASE_URL`

3. **Configure Your Web Service**:
   - Go to your web service in the Render dashboard
   - Click on "Environment"
   - Add the following environment variable:
     - Key: `DATABASE_URL`
     - Value: (paste the External Database URL you copied)

### Environment Variables for Render Deployment

Set the following environment variables in your Render dashboard:

| Key | Value | Description |
|-----|-------|-------------|
| `DATABASE_URL` | `postgres://...` | Your PostgreSQL connection URL from Render |
| `ADMIN_USERNAME` | `admin` (or your preferred username) | Username for the admin account |
| `ADMIN_EMAIL` | `your-email@example.com` | Email for the admin account |
| `ADMIN_PASSWORD` | `your-secure-password` | Password for the admin account (use a strong password!) |
| `SECRET_KEY` | `your-random-secret-key` | Secret key for Flask sessions |
| `HUGGINGFACE_API_KEY` | `your-api-key` | If you want to use AI-enhanced questionnaires |

### Important Notes on Render Deployment

1. **Database Persistence**: With PostgreSQL configured, all user data (accounts, analyses, questionnaires) will persist when the service restarts or goes into sleep mode.

2. **Initial Tables Creation**: The application will automatically create all necessary database tables on first startup.

3. **Admin Account**: An admin account will be created using the credentials in your environment variables if it doesn't already exist.

4. **Free Tier Limitations**:
   - The free PostgreSQL database on Render has a storage limit of 1GB
   - The database is automatically deleted after 90 days of inactivity
   - Consider backing up important data periodically 