#!/bin/bash

# Colors for terminal output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${YELLOW}Starting Statistical Model Suggester${NC}"
echo "-------------------------------------------------------------------"

# Activate virtual environment
source venv/bin/activate

# Install dependencies if needed
pip install -r requirements.txt

# Start the Flask application
export FLASK_APP=app.py
export FLASK_ENV=development
export FLASK_DEBUG=1
python -m flask run --port=8084 