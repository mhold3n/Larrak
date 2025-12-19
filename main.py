"""
Uvicorn entrypoint convenience shim.

Allows: `uvicorn main:app --reload`
"""

from dashboard_app.app import app  # noqa: F401



