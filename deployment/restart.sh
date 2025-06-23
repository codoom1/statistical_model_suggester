#!/bin/bash

echo "Restarting the Statistical Model Suggester application..."

# Kill any running Flask process
echo "Stopping existing Flask processes..."
pkill -f "flask run" || echo "No Flask processes found."

# Wait a moment for the process to terminate
sleep 2

# Start the application again
echo "Starting Flask application..."
export FLASK_APP=app.py
export FLASK_ENV=development

# Run the Flask app in the background
flask run --host=127.0.0.1 --port=8084 &

echo "Application restart complete!"
echo "The application should now be available at http://127.0.0.1:8084" 