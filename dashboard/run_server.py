import os
import sys

# Add /app to sys.path so we can import 'dashboard' package
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from dashboard.api import create_app, socketio
except ImportError:
    # Fallback if running from inside the package
    from api import create_app, socketio

app = create_app()

if __name__ == "__main__":
    # Use socketio.run instead of app.run to enable WebSocket support
    socketio.run(app, host="0.0.0.0", port=8000, allow_unsafe_werkzeug=True)
