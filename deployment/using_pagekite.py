import os
import sys
import time
import threading
import subprocess
from app import create_app

def run_flask_app(port):
    """Run the Flask application"""
    app = create_app()
    app.run(host="127.0.0.1", port=port, debug=False)

def install_pagekite():
    """Install pagekite if not already installed"""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pagekite"])
        return True
    except subprocess.CalledProcessError:
        print("Failed to install pagekite. Please install it manually with: pip install pagekite")
        return False

if __name__ == "__main__":
    # Install pagekite
    print("Checking for pagekite...")
    if not install_pagekite():
        sys.exit(1)
    
    # Use a port that's unlikely to be in use
    port = 8088
    
    # Start Flask in a separate thread
    print("Starting Flask application...")
    flask_thread = threading.Thread(target=run_flask_app, args=(port,))
    flask_thread.daemon = True
    flask_thread.start()
    
    # Give Flask time to start
    time.sleep(2)
    
    print("Creating public URL using PageKite...")
    print("(This may take a few moments)")
    
    # Set up pagekite for free subdomain
    pagekite_cmd = f"python -m pagekite.pk --clean --defaults {port} yourname.pagekite.me"
    
    print("\n" + "=" * 50)
    print("When prompted:")
    print("1. Press Enter to use a random name")
    print("2. Accept the terms of service")
    print("Once connected, you'll see a URL like: http://something.pagekite.me")
    print("=" * 50 + "\n")
    
    # Run pagekite
    try:
        subprocess.call(pagekite_cmd, shell=True)
    except KeyboardInterrupt:
        print("\nShutting down the server...")
        sys.exit(0) 