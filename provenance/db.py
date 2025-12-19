import sqlite3
import json
import datetime
from pathlib import Path
from typing import List, Optional, Any, Dict
from dataclasses import asdict

from provenance.spec import (
    Event, RunStartEvent, RunEndEvent, FileEvent, CheckpointEvent, 
    EntityType, FileRole, Artifact
)

DB_PATH = Path("provenance.db")

class ProvenanceDB:
    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        """Initialize schema if not exists."""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        # Runs table
        c.execute('''
            CREATE TABLE IF NOT EXISTS runs (
                run_id TEXT PRIMARY KEY,
                module_id TEXT,
                start_time TIMESTAMP,
                end_time TIMESTAMP,
                status TEXT,
                args TEXT,
                env TEXT,
                tags TEXT
            )
        ''')

        # Events table (append-only log)
        c.execute('''
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT,
                timestamp TIMESTAMP,
                event_type TEXT,
                details JSON
            )
        ''')
        
        # Artifacts table (Generated files)
        c.execute('''
            CREATE TABLE IF NOT EXISTS artifacts (
                artifact_id TEXT PRIMARY KEY,
                run_id TEXT,
                path TEXT,
                role TEXT,
                content_hash TEXT,
                meta JSON,
                FOREIGN KEY(run_id) REFERENCES runs(run_id)
            )
        ''')

        conn.commit()
        conn.close()

    def start_run(self, run_id: str, module_id: str, args: List[str], env: Dict[str, str]):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute('''
            INSERT INTO runs (run_id, module_id, start_time, args, env, status)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            run_id, 
            module_id, 
            datetime.datetime.now(), 
            json.dumps(args), 
            json.dumps(env), 
            "RUNNING"
        ))
        conn.commit()
        conn.close()

    def end_run(self, run_id: str, status: str = "SUCCESS"):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute('''
            UPDATE runs 
            SET end_time = ?, status = ?
            WHERE run_id = ?
        ''', (datetime.datetime.now(), status, run_id))
        conn.commit()
        conn.close()

    def log_event(self, event: Event):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        # Serialize event details
        # Using a simple convention: verify type and store relevant fields
        details = asdict(event)
        # Remove common fields stored in columns
        details.pop('timestamp', None)
        details.pop('run_id', None)
        
        # Determine event type name
        event_type = event.__class__.__name__

        c.execute('''
            INSERT INTO events (run_id, timestamp, event_type, details)
            VALUES (?, ?, ?, ?)
        ''', (
            event.run_id,
            event.timestamp,
            event_type,
            json.dumps(details, default=str)
        ))
        conn.commit()
        conn.close()

    def register_artifact(self, artifact: Artifact):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute('''
            INSERT OR REPLACE INTO artifacts (artifact_id, run_id, path, role, content_hash, meta)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            artifact.artifact_id,
            artifact.run_id,
            artifact.path,
            artifact.role.value,
            artifact.content_hash,
            json.dumps(artifact.metadata, default=str)
        ))
        conn.commit()
        conn.close()

# Global instance for easy access
db = ProvenanceDB()
