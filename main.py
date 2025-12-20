"""
Uvicorn entrypoint convenience shim.

Allows: `uvicorn main:app --reload`
"""

from webapp.app import app  # noqa: F401



