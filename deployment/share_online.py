import os
import sys
import time
import subprocess
import threading
import socket
import re
from app import create_app

def run_flask_app(port):
    """Run the Flask application"""
    app = create_app()
    app.run(host="127.0.0.1", port=port, debug=False)

def setup_ssh_tunnel(port):
    """Set up SSH tunnel using localhost.run"""
    # Command to create SSH tunnel with auto-accepting SSH key, using HTTP instead of HTTPS
    cmd = f"ssh -o StrictHostKeyChecking=no -R 80:localhost:{port} nokey@localhost.run --http"
    
    # Run the command
    process = subprocess.Popen(
        cmd, 
        shell=True, 
        stdout=subprocess.PIPE, 
        stderr=subprocess.PIPE,
        universal_newlines=True
    )
    
    # Extract the public URL from the output
    public_url = None
    for line in process.stdout:
        print(line.strip())
        # Look for the URL in the output (now looking for HTTP URL)
        if "tunneled without" in line.lower():
            match = re.search(r'(http://[^\s]+)', line)
            if match:
                public_url = match.group(1)
                break
    
    return public_url, process

if __name__ == "__main__":
    # Use a port that's unlikely to be in use
    port = 8087
    
    # Start Flask in a separate thread
    print("Starting Flask application...")
    flask_thread = threading.Thread(target=run_flask_app, args=(port,))
    flask_thread.daemon = True
    flask_thread.start()
    
    # Give Flask time to start
    time.sleep(2)
    
    print("Creating public URL using localhost.run...")
    print("(This may take a few moments)")
    
    # Set up the SSH tunnel
    public_url, ssh_process = setup_ssh_tunnel(port)
    
    if public_url:
        print("\n" + "=" * 50)
        print("✅ Public URL for sharing the application:")
        print(f"🔗 {public_url}")
        print("=" * 50)
        print("\nShare this link with anyone who wants to test your application.")
        print("Press Ctrl+C to stop the server when done.\n")
    
    try:
        # Keep the script running
        ssh_process.wait()
    except KeyboardInterrupt:
        print("\nShutting down the server...")
        ssh_process.terminate()
        sys.exit(0) 