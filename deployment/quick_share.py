import os
import sys
import time
import subprocess
from threading import Thread
from app import create_app

def install_pyngrok():
    """Install pyngrok if not already installed"""
    try:
        import pyngrok
        return True
    except ImportError:
        print("Installing pyngrok...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "pyngrok"])
            return True
        except subprocess.CalledProcessError:
            print("Failed to install pyngrok. Please install it manually with: pip install pyngrok")
            return False

def run_flask_app():
    """Run the Flask application"""
    app = create_app()
    app.run(port=8084)

if __name__ == "__main__":
    # Install pyngrok if needed
    if not install_pyngrok():
        sys.exit(1)
    
    # Import after installation
    from pyngrok import ngrok, conf
    
    # Start Flask in a separate thread
    print("Starting Flask application...")
    flask_thread = Thread(target=run_flask_app)
    flask_thread.daemon = True
    flask_thread.start()
    
    # Give Flask time to start
    time.sleep(2)
    
    # Configure ngrok
    auth_token = os.environ.get("NGROK_AUTH_TOKEN")
    if auth_token:
        conf.get_default().auth_token = auth_token
    
    # Open a ngrok tunnel to the Flask app
    print("Creating public URL...")
    public_url = ngrok.connect(8084).public_url
    
    print("\n" + "=" * 50)
    print("✅ Public URL for sharing the application:")
    print(f"🔗 {public_url}")
    print("=" * 50)
    print("\nShare this link with anyone who wants to test your application.")
    print("Note: This is a temporary URL that will expire after 2 hours if using the free plan.")
    print("Press Ctrl+C to stop the server when done.\n")
    
    try:
        # Keep the script running
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down the server...")
        ngrok.kill() 