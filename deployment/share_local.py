import os
import socket
import subprocess
from app import create_app

def get_local_ip():
    """Get the local IP address of the machine"""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        # doesn't even have to be reachable
        s.connect(('10.255.255.255', 1))
        IP = s.getsockname()[0]
    except Exception:
        IP = '127.0.0.1'
    finally:
        s.close()
    return IP

if __name__ == "__main__":
    # Determine available port
    port = 8085  # Use a different port than the default
    
    # Get local IP
    local_ip = get_local_ip()
    
    print("\n" + "=" * 50)
    print("Starting Flask application...")
    print(f"Local URL: http://localhost:{port}")
    print(f"Network URL: http://{local_ip}:{port}")
    print("=" * 50)
    print("\nShare the Network URL with anyone on your local network.")
    print("For Internet access, consider using tailscale.com or ngrok.com")
    print("Press Ctrl+C to stop the server when done.\n")
    
    # Create and run app
    app = create_app()
    app.run(host='0.0.0.0', port=port, debug=False) 