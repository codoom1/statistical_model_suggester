# Statistical Model Suggester

A web application that helps users choose appropriate statistical models based on their data characteristics and analysis goals.

## Features

- **Model Recommendations**: Get suggestions for statistical models based on your data type and analysis goals
- **User Accounts**: Save analysis history and track previous recommendations
- **Expert Mode**: Advanced features for experienced statisticians
- **PDF/Word Export**: Export analysis results and recommendations
- **Admin Dashboard**: User management and system monitoring
- **Optional AI Chat**: Enhanced recommendations with AI assistance (requires heavy ML dependencies)

## Quick Start

1. **Clone and setup**:
```bash
git clone https://github.com/codoom1/statistical-model-suggester.git
cd statistical-model-suggester
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt        # Core app (~100MB)
pip install -r requirements-dev.txt    # + Development tools
```

3. **Run the app**:
```bash
export FLASK_SECRET_KEY='your-secret-key-here'
python app.py
# Visit: http://localhost:8084
```

## Dependencies

- **`requirements.txt`**: Core production dependencies (Flask, SQLAlchemy, scikit-learn, basic plotting)
The app gracefully handles missing optional dependencies (like PDF export libraries).

## Deployment

### Local Development
```bash
export FLASK_SECRET_KEY='your-secret-key-here'
python app.py
# Visit: http://localhost:8084
```

### Production (Render.com)

1. **Create a Web Service** on Render and connect your GitHub repository
2. **Build Command**: `pip install -r requirements.txt && python render_build.py`
3. **Start Command**: `gunicorn app:app`
4. **Add a PostgreSQL database** and Render will set `DATABASE_URL` automatically

**Required Environment Variables:**
```bash
FLASK_ENV=production
SECRET_KEY=<your-secure-random-key>
ADMIN_USERNAME=<your-admin-username>
ADMIN_EMAIL=<your-admin-email>
ADMIN_PASSWORD=<your-secure-admin-password>
```

**Optional (for email notifications):**
```bash
MAIL_SERVER=<smtp-server>
MAIL_USERNAME=<email>
MAIL_PASSWORD=<password>
```

**Optional (for AI features):**
```bash
AI_ENHANCEMENT_ENABLED=true
HUGGINGFACE_API_KEY=<your-key>
```

## Project Structure
```
├── app.py                 # Main Flask application
├── models.py             # Database models
├── requirements.txt      # Core dependencies
├── requirements-dev.txt  # Development tools
├── routes/              # Route handlers
├── templates/           # HTML templates
├── static/             # CSS, JS, images
├── utils/              # Helper functions
├── tests/              # Test suite
└── data/               # Model database
```

## Testing
```bash
pip install -r requirements-dev.txt
python -m pytest tests/ -v
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

MIT License - see LICENSE file for details.