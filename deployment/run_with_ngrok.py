import os
from app import create_app
import ngrok
from threading import Thread
import time

def start_flask_app():
    # Create and run the Flask app
    app = create_app()
    app.run(debug=False, port=8084)

if __name__ == "__main__":
    # Start Flask app in a separate thread
    flask_thread = Thread(target=start_flask_app)
    flask_thread.daemon = True
    flask_thread.start()
    
    # Give Flask time to start up
    time.sleep(2)
    
    # Start ngrok tunnel
    listener = ngrok.connect(8084, domain=os.getenv("NGROK_DOMAIN"))
    
    print("\n" + "=" * 50)
    print("✅ Public URL for sharing the application:")
    print(f"🔗 {listener.url()}")
    print("=" * 50)
    print("\nShare this link with anyone who wants to test your application.")
    print("Press Ctrl+C to stop the server when done.\n")
    
    try:
        # Keep the script running
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down the server...")
        ngrok.disconnect() 