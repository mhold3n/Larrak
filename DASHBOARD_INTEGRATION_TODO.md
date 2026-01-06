"""Update the Larrak dashboard to integrate structured logging and session management.

Summary of changes needed in dashboard/api.py:

1. Update WebSocketLogHandler to write to session files
2. Add session log retrieval endpoints
3. Integrate session logging in start_sequence()
4. Configure root logging at app startup
"""

# This is a planning note for the dashboard integration

# The actual implementation will be done in dashboard/api.py directly

# Key changes

# 1. In create_app()

# - Add configure_root_logging() call at startup

# - Add SessionContextFilter to root logger

#

# 2. Update start_sequence()

# - Wrap orchestration in session_logging() context manager

# - Pass run_id to session

#

# 3. New endpoints

# - GET /api/logs/sessions - List all sessions

# - GET /api/logs/sessions/{run_id} - Get full log for session

# - GET /api/logs/sessions/{run_id}/metrics - Get metrics

# - GET /api/logs/errors?since=YYYY-MM-DD - Get errors since date
