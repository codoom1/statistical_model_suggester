#!/bin/bash

# Colors for terminal output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${YELLOW}Starting Statistical Model Suggester${NC}"
echo "-------------------------------------------------------------------"

# Check if we're running in a virtual environment
if [[ -z $VIRTUAL_ENV && -d "venv" ]]; then
    echo -e "${YELLOW}Activating virtual environment...${NC}"
    source venv/bin/activate
elif [[ -z $VIRTUAL_ENV ]]; then
    echo -e "${YELLOW}No virtual environment found, but proceeding anyway (might be in container environment)${NC}"
fi

# Install dependencies
echo -e "${YELLOW}Installing dependencies...${NC}"
pip install -r requirements.txt

# Check if gunicorn is installed (needed for Render deployment)
if ! pip show gunicorn > /dev/null; then
    echo -e "${YELLOW}Installing gunicorn for production deployment...${NC}"
    pip install gunicorn
fi

# Determine environment type
if [[ ! -z "$RENDER" ]]; then
    echo -e "${GREEN}Running in Render production environment${NC}"
    export FLASK_ENV=production
    export FLASK_DEBUG=0
    # Render will use the Procfile to start the app
    exit 0
fi

# Running locally - Get port from environment variable or .env file
if [[ -f .env ]]; then
    DEFAULT_PORT=$(grep APP_PORT .env | cut -d '=' -f2 || echo 8084)
else
    DEFAULT_PORT=8084
fi

PORT=${PORT:-$DEFAULT_PORT}

# Function to check if port is available
is_port_available() {
    nc -z localhost $1 >/dev/null 2>&1
    return $?
}

# Try to find an available port starting with the default
while ! is_port_available $PORT; do
    echo "Port $PORT is in use, trying alternative..."
    PORT=$((PORT + 1))
    # Avoid trying too many ports
    if [ $((PORT - DEFAULT_PORT)) -gt 10 ]; then
        echo "Could not find an available port after 10 attempts."
        echo "Please manually specify a port using the PORT environment variable."
        exit 1
    fi
done

echo "Starting app on port $PORT"
export FLASK_APP=app.py
export FLASK_ENV=development
export PORT=$PORT

# Local development uses Flask's built-in server
python app.py 