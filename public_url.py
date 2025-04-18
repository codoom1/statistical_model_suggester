import os
import sys
import time
import threading
import subprocess
import re
from app import create_app

def run_flask_app(port):
    """Run the Flask application"""
    app = create_app()
    app.run(host="127.0.0.1", port=port, debug=False)

def run_cloudflared(port):
    """Run cloudflared for tunneling"""
    cmd = ["cloudflared", "tunnel", "--url", f"http://localhost:{port}"]
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True
    )
    
    # Find the URL in the output
    print("Starting cloudflared tunnel...")
    url = None
    for line in process.stdout:
        print(line.strip())
        if "trycloudflare.com" in line or "cloudflare.com" in line:
            # Extract URL
            parts = line.split()
            for part in parts:
                if "http" in part and ("trycloudflare.com" in part or "cloudflare.com" in part):
                    url = part.strip()
                    break
            if url:
                break
    
    return url, process

if __name__ == "__main__":
    # Use a port that's unlikely to be in use
    port = 8089
    
    # Start Flask in a separate thread
    print("Starting Flask application...")
    flask_thread = threading.Thread(target=run_flask_app, args=(port,))
    flask_thread.daemon = True
    flask_thread.start()
    
    # Give Flask time to start
    time.sleep(2)
    
    # Create tunnel
    url, cloudflared_process = run_cloudflared(port)
    
    if url:
        print("\n" + "=" * 50)
        print("✅ Public URL for sharing the application:")
        print(f"🔗 {url}")
        print("=" * 50)
        print("\nShare this link with anyone who wants to test your application.")
        print("This link should work in all browsers including Safari.")
        print("Press Ctrl+C to stop the server when done.\n")
        
        try:
            # Keep the script running
            cloudflared_process.wait()
        except KeyboardInterrupt:
            print("\nShutting down the server...")
            cloudflared_process.terminate()
    else:
        print("Failed to create tunnel. Please check the logs above.")
        sys.exit(1) 