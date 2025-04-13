# Statistical Model Suggester

A web application that helps users choose appropriate statistical models based on their data characteristics and analysis goals.

## Features

- Recommends appropriate statistical models based on:
  - Analysis goal (prediction, classification, exploration)
  - Data characteristics (variable types, sample size)
  - Research question
- Provides detailed information about recommended models
- Suggests alternative models
- Maintains history of previous recommendations
- Includes implementation details and documentation links

## Installation

1. Clone the repository:
```bash
git clone https://github.com/codoom1/statistical-model-suggester.git
cd statistical-model-suggester
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
```bash
export FLASK_SECRET_KEY='your-secret-key-here'  # On Windows: set FLASK_SECRET_KEY=your-secret-key-here
```

## Usage

1. Start the application:
```bash
python app.py
```

2. Open your web browser and navigate to:
```
http://localhost:8084
```

3. Fill in the form with your data characteristics and analysis goals.

4. View the recommended model and alternatives.

## Project Structure

```
statistical-model-suggester/
├── app.py                 # Main application file
├── requirements.txt       # Python dependencies
├── static/               # Static files (CSS, JS)
│   ├── styles.css
│   └── results.css
└── templates/            # HTML templates
    ├── index.html
    ├── results.html
    ├── history.html
    └── error.html
```

## Dependencies

- Flask
- scikit-learn
- statsmodels
- xgboost
- tensorflow
- lifelines
- python-dotenv
- gunicorn (for production deployment)
- psycopg2-binary (for PostgreSQL support)

## Deployment on Render

### Setting Up on Render

1. Create a new Web Service on Render
2. Connect your GitHub repository
3. Use the following settings:
   - **Name**: statistical-model-suggester (or your preferred name)
   - **Environment**: Python
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn app:app`

### Environment Variables

Set the following environment variables in Render:

```
FLASK_ENV=production
FLASK_DEBUG=0
SECRET_KEY=<your-secure-random-key>
ADMIN_USERNAME=<your-admin-username>
ADMIN_EMAIL=<your-admin-email>
ADMIN_PASSWORD=<your-secure-admin-password>
```

For email functionality:
```
MAIL_SERVER=<your-smtp-server>
MAIL_PORT=<your-smtp-port>
MAIL_USE_TLS=True
MAIL_USERNAME=<your-email-username>
MAIL_PASSWORD=<your-email-password>
MAIL_DEFAULT_SENDER=<your-default-sender-email>
```

For AI features (optional):
```
HUGGINGFACE_API_KEY=<your-huggingface-api-key>
HUGGINGFACE_MODEL=<your-preferred-model>
AI_ENHANCEMENT_ENABLED=true
```

### Database Setup

1. Create a PostgreSQL database on Render
2. Render will automatically add the `DATABASE_URL` environment variable to your web service

### Troubleshooting Deployment

- Check application logs in Render dashboard
- Verify all environment variables are set correctly
- Ensure model_database.json exists and is valid JSON
- Check for any Python dependencies that might be missing from requirements.txt

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details. 