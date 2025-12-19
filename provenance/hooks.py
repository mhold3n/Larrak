import os
import builtins
import hashlib
from typing import Optional, Any
import shutil
import functools
import pandas as pd
import datetime

from provenance.spec import FileEvent, FileRole, Artifact
from provenance.db import db
from provenance.context import get_current_run_id

# Store original functions to avoid recursion
_original_open = builtins.open
_original_read_csv = pd.read_csv
_original_to_csv = pd.Series.to_csv # DataFrame.to_csv calls this internally usually, or we patch DataFrame
_original_df_to_csv = pd.DataFrame.to_csv

def _calculate_hash(path: str) -> Optional[str]:
    """Calculate SHA256 of a file."""
    if not os.path.exists(path):
        return None
    sha256 = hashlib.sha256()
    try:
        with _original_open(path, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                sha256.update(chunk)
        return sha256.hexdigest()
    except Exception:
        return None

def _log_file_event(path: str, op: str):
    run_id = get_current_run_id()
    if not run_id:
        return

    # Calculate basic stats
    size = 0
    if os.path.exists(path):
        size = os.path.getsize(path)
    
    # We might want to hash inputs/outputs
    # For now, let's hash if it's small or if strictly required
    # Keeping it lightweight for this pass
    file_hash = None 

    event = FileEvent(
        run_id=run_id,
        path=os.path.abspath(path),
        op=op,
        size=size,
        file_hash=file_hash
    )
    db.log_event(event)

    # If it's a write, we might want to register it as an artifact
    if op in ("write", "create"):
        # We can do this async or post-hoc, but let's do simple registration here
        pass

# --- Wrappers ---

def patched_open(file, mode='r', buffering=-1, encoding=None, errors=None, newline=None, closefd=True, opener=None):
    # Perform the operation
    f = _original_open(file, mode, buffering, encoding, errors, newline, closefd, opener)
    
    # Log the event
    # Determine operation type
    op = "read"
    if 'w' in mode or 'a' in mode or 'x' in mode:
        op = "write"
    
    # We only log if it's a file path (not an int fd)
    if isinstance(file, (str, bytes, os.PathLike)):
        try:
            _log_file_event(str(file), op)
        except Exception as e:
            # Never break the app because of logging failure
            print(f"[Provenance] Error logging open({file}): {e}")
            
    return f

def patched_read_csv(*args, **kwargs):
    # Capture the path
    filepath = args[0] if len(args) > 0 else kwargs.get('filepath_or_buffer')
    
    # Run original
    df = _original_read_csv(*args, **kwargs)
    
    # Log event
    if isinstance(filepath, (str, os.PathLike)):
         try:
            _log_file_event(str(filepath), "read")
         except Exception as e:
            print(f"[Provenance] Error logging read_csv({filepath}): {e}")
            
    return df

def patched_to_csv(self, *args, **kwargs):
    # Capture path
    path = args[0] if len(args) > 0 else kwargs.get('path_or_buf')
    
    # Run original
    result = _original_df_to_csv(self, *args, **kwargs)
    
    # Log event
    if isinstance(path, (str, os.PathLike)):
         try:
            _log_file_event(str(path), "write")
            
            # Register Artifact
            run_id = get_current_run_id()
            if run_id:
                abs_path = os.path.abspath(path)
                file_hash = _calculate_hash(abs_path)
                artifact = Artifact(
                    artifact_id=f"{file_hash[:8]}_{run_id[:8]}", # Simple ID for now
                    path=abs_path,
                    content_hash=file_hash,
                    run_id=run_id,
                    producer_module_id="unknown", # We'd need to fetch this from context
                    role=FileRole.OUTPUT, # Assumption
                    size_bytes=os.path.getsize(abs_path),
                    creation_time=datetime.datetime.now()
                )
                db.register_artifact(artifact)

         except Exception as e:
            print(f"[Provenance] Error logging to_csv({path}): {e}")
            
    return result

# --- Patching Mechanism ---

def install():
    """Install monkey patches."""
    builtins.open = patched_open
    pd.read_csv = patched_read_csv
    pd.DataFrame.to_csv = patched_to_csv
    print("[Provenance] Hooks installed.")

def uninstall():
    """Remove monkey patches."""
    builtins.open = _original_open
    pd.read_csv = _original_read_csv
    pd.DataFrame.to_csv = _original_df_to_csv
    print("[Provenance] Hooks removed.")
