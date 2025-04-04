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
git clone https://github.com/yourusername/statistical-model-suggester.git
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
http://localhost:8080
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

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details. 