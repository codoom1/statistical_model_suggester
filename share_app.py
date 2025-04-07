import subprocess
import time
import sys
import os
from threading import Thread
from app import create_app

def run_flask_app():
    """Run the Flask application in a separate thread"""
    app = create_app()
    app.run(debug=False, host='0.0.0.0', port=8084)

def check_localtunnel():
    """Check if localtunnel is installed, install if not"""
    try:
        subprocess.run(['lt', '--version'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return True
    except FileNotFoundError:
        print("Localtunnel not found. Installing...")
        try:
            subprocess.run(['npm', 'install', '-g', 'localtunnel'], check=True)
            return True
        except subprocess.CalledProcessError:
            print("Error installing localtunnel. Make sure npm is installed.")
            return False
        except FileNotFoundError:
            print("npm not found. Please install Node.js from https://nodejs.org/")
            return False

def run_localtunnel():
    """Start localtunnel and return the public URL"""
    process = subprocess.Popen(
        ['lt', '--port', '8084'],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    # Read output to get the URL
    for line in process.stdout:
        if "your url is" in line.lower():
            url = line.strip().split("is: ")[1]
            return url, process
    
    return None, process

if __name__ == "__main__":
    # Check for localtunnel
    if not check_localtunnel():
        sys.exit(1)
    
    # Start Flask app in a separate thread
    print("Starting Flask application...")
    flask_thread = Thread(target=run_flask_app)
    flask_thread.daemon = True
    flask_thread.start()
    
    # Give Flask time to start
    time.sleep(2)
    
    # Start localtunnel
    print("Creating public URL...")
    url, lt_process = run_localtunnel()
    
    if url:
        print("\n" + "=" * 50)
        print("✅ Public URL for sharing the application:")
        print(f"🔗 {url}")
        print("=" * 50)
        print("\nShare this link with anyone who wants to test your application.")
        print("Press Ctrl+C to stop the server when done.\n")
        
        try:
            # Keep the script running
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nShutting down the server...")
            lt_process.terminate()
    else:
        print("Failed to create public URL. Please try again.") 