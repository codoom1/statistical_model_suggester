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

3. **Initialize and run the app**:
```bash
cp .env.example .env
flask --app app init-db
flask --app app create-admin
python app.py
# Visit: http://localhost:8084
```

## Dependencies

- **`requirements.txt`**: Packages required by live web requests
- **`requirements-dev.txt`**: Tests plus offline diagnostic/plot-generation packages

## Deployment

### Local Development
```bash
export FLASK_SECRET_KEY='your-secret-key-here'
python app.py
# Visit: http://localhost:8084
```

### Production (Vercel)

1. Import this Git repository into Vercel with the Flask framework preset.
2. Add a pooled PostgreSQL connection string as `DATABASE_URL`.
3. Create a private Vercel Blob store. Vercel supplies
   `BLOB_READ_WRITE_TOKEN` to the selected environments.
4. Add the required environment variables below.
5. Initialize the database once from a trusted machine using production
   environment variables:

```bash
vercel env pull .env.production.local
set -a
source .env.production.local
set +a
flask --app app init-db
flask --app app create-admin
```

6. Deploy from the Vercel dashboard or run `vercel --prod`.

The application never creates tables or administrator accounts during a web
request or cold start. Uploaded résumés are private Blob objects and are
downloaded through an authenticated application route.

**Required Environment Variables:**
```bash
FLASK_ENV=production
SECRET_KEY=<your-secure-random-key>
DATABASE_URL=<pooled-postgresql-url>
BLOB_READ_WRITE_TOKEN=<created-by-vercel-blob>
```

Administrator credentials are only needed when running `flask create-admin`.
They do not need to remain in the deployed environment.

**Transactional email with Resend:**
```bash
EMAIL_PROVIDER=resend
RESEND_API_KEY=<your-resend-api-key>
MAIL_DEFAULT_SENDER="Statistical Model Suggester <noreply@your-domain.example>"
```

The sender domain must be verified in Resend. For local SMTP instead, set
`EMAIL_PROVIDER=smtp` and configure `MAIL_SERVER`, `MAIL_PORT`,
`MAIL_USE_TLS`, `MAIL_USERNAME`, and `MAIL_PASSWORD`.

**Optional OpenAI integration:**
```bash
AI_ENHANCEMENT_ENABLED=true
OPENAI_API_KEY=<your-openai-project-api-key>
OPENAI_MODEL=gpt-5.6-luna
OPENAI_REASONING_EFFORT=low
AI_REQUESTS_PER_USER_PER_HOUR=20
AI_REQUEST_TIMEOUT_SECONDS=45
AI_MAX_OUTPUT_TOKENS=400
```

The OpenAI key is used only by the server and must never be added to source
control or browser code. AI requests require an authenticated user and are
limited per user with durable database usage records. After deploying this change, run
`flask --app app init-db` once so the `ai_usage_events` table exists.

## Project Structure
```
├── app.py                 # Main Flask application
├── models.py             # Database models
├── requirements.txt      # Core dependencies
├── requirements-dev.txt  # Development tools
├── routes/              # Route handlers
├── templates/           # HTML templates
├── public/static/      # CDN-served CSS, JS, images
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
