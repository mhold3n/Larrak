"""
Service for querying run history from provenance.db.
Provides data for the Run History dashboard view.
"""
import sqlite3
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

DB_PATH = Path("provenance.db")


def _format_duration(start: Optional[str], end: Optional[str]) -> str:
    """Calculate and format duration between two timestamps."""
    if not start:
        return "N/A"
    try:
        start_dt = datetime.fromisoformat(start)
        if not end:
            # Still running
            end_dt = datetime.now()
        else:
            end_dt = datetime.fromisoformat(end)
        
        delta = end_dt - start_dt
        total_seconds = int(delta.total_seconds())
        
        if total_seconds < 60:
            return f"{total_seconds}s"
        elif total_seconds < 3600:
            minutes = total_seconds // 60
            seconds = total_seconds % 60
            return f"{minutes}m {seconds}s"
        else:
            hours = total_seconds // 3600
            minutes = (total_seconds % 3600) // 60
            return f"{hours}h {minutes}m"
    except Exception:
        return "N/A"


def get_recent_runs(limit: int = 20) -> List[Dict[str, Any]]:
    """
    Get recent runs from the provenance database.
    
    Returns list of runs with:
    - run_id, module_id, status, start_time, end_time, duration
    """
    if not DB_PATH.exists():
        return []
    
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    
    try:
        c.execute('''
            SELECT run_id, module_id, status, start_time, end_time, args, env
            FROM runs
            ORDER BY start_time DESC
            LIMIT ?
        ''', (limit,))
        
        rows = c.fetchall()
        runs = []
        
        for row in rows:
            start_time = row['start_time']
            end_time = row['end_time']
            
            runs.append({
                'run_id': row['run_id'],
                'module_id': row['module_id'],
                'status': row['status'] or 'UNKNOWN',
                'start_time': start_time,
                'end_time': end_time,
                'duration': _format_duration(start_time, end_time),
            })
        
        return runs
    except sqlite3.OperationalError:
        # Table might not exist yet
        return []
    finally:
        conn.close()


def get_run_details(run_id: str) -> Optional[Dict[str, Any]]:
    """
    Get detailed information for a specific run, including linked artifacts.
    """
    if not DB_PATH.exists():
        return None
    
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    
    try:
        # Get run info
        c.execute('''
            SELECT run_id, module_id, status, start_time, end_time, args, env, tags
            FROM runs
            WHERE run_id = ?
        ''', (run_id,))
        
        row = c.fetchone()
        if not row:
            return None
        
        # Get linked artifacts
        c.execute('''
            SELECT artifact_id, path, role, content_hash, meta
            FROM artifacts
            WHERE run_id = ?
        ''', (run_id,))
        
        artifacts = []
        for art_row in c.fetchall():
            meta = {}
            if art_row['meta']:
                try:
                    meta = json.loads(art_row['meta'])
                except json.JSONDecodeError:
                    pass
            
            artifacts.append({
                'artifact_id': art_row['artifact_id'],
                'path': art_row['path'],
                'role': art_row['role'],
                'content_hash': art_row['content_hash'],
                'meta': meta
            })
        
        start_time = row['start_time']
        end_time = row['end_time']
        
        return {
            'run_id': row['run_id'],
            'module_id': row['module_id'],
            'status': row['status'] or 'UNKNOWN',
            'start_time': start_time,
            'end_time': end_time,
            'duration': _format_duration(start_time, end_time),
            'args': json.loads(row['args']) if row['args'] else [],
            'env': json.loads(row['env']) if row['env'] else {},
            'artifacts': artifacts
        }
    except sqlite3.OperationalError:
        return None
    finally:
        conn.close()


# For convenience, also expose summary stats
def get_run_stats() -> Dict[str, Any]:
    """Get aggregate statistics about runs."""
    if not DB_PATH.exists():
        return {'total': 0, 'success': 0, 'failure': 0, 'running': 0}
    
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    try:
        c.execute('SELECT status, COUNT(*) FROM runs GROUP BY status')
        counts = dict(c.fetchall())
        
        return {
            'total': sum(counts.values()),
            'success': counts.get('SUCCESS', 0),
            'failure': counts.get('FAILURE', 0),
            'running': counts.get('RUNNING', 0)
        }
    except sqlite3.OperationalError:
        return {'total': 0, 'success': 0, 'failure': 0, 'running': 0}
    finally:
        conn.close()
