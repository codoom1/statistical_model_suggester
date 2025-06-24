import subprocess
import re
import sys
import time

def create_tunnel(port):
    """Create a tunnel to localhost using cloudflared"""
    # Command to start cloudflared
    cmd = ["cloudflared", "tunnel", "--url", f"http://localhost:{port}"]
    
    # Run the command and capture both stdout and stderr
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
        bufsize=1
    )
    
    print(f"Creating tunnel to localhost:{port}...")
    print("Waiting for URL (this may take a few seconds)...")
    
    # Look for URL in both stdout and stderr
    url_regex = r'https?://[a-zA-Z0-9][-a-zA-Z0-9]*\.trycloudflare\.com'
    
    def check_output(stream):
        for line in stream:
            print(line.strip())
            match = re.search(url_regex, line)
            if match:
                return match.group(0)
        return None
    
    # Check stdout in a non-blocking way
    url = None
    start_time = time.time()
    timeout = 30  # seconds
    
    while time.time() - start_time < timeout:
        if process.stdout.readable():
            line = process.stdout.readline()
            if line:
                print(line.strip())
                match = re.search(url_regex, line)
                if match:
                    url = match.group(0)
                    break
        
        if process.stderr.readable():
            line = process.stderr.readline()
            if line:
                print(line.strip())
                match = re.search(url_regex, line)
                if match:
                    url = match.group(0)
                    break
        
        time.sleep(0.1)
    
    if url:
        return url, process
    else:
        process.terminate()
        return None, None

if __name__ == "__main__":
    # Port to tunnel to
    port = 8084  # Default Flask port
    
    if len(sys.argv) > 1:
        try:
            port = int(sys.argv[1])
        except ValueError:
            print(f"Invalid port: {sys.argv[1]}")
            sys.exit(1)
    
    # Create tunnel
    url, process = create_tunnel(port)
    
    if url:
        print("\n" + "=" * 50)
        print("✅ Public URL for sharing the application:")
        print(f"🔗 {url}")
        print("=" * 50)
        print("\nShare this link with anyone who wants to test your application.")
        print("This URL is compatible with all browsers including Safari.")
        print("Press Ctrl+C to stop the tunnel when done.\n")
        
        try:
            # Keep the script running until interrupted
            process.wait()
        except KeyboardInterrupt:
            print("\nShutting down tunnel...")
            process.terminate()
    else:
        print("Failed to get public URL. Please check if:")
        print("1. cloudflared is installed correctly (brew install cloudflare/cloudflare/cloudflared)")
        print("2. Your application is running on the specified port")
        print("3. You have an active internet connection") 