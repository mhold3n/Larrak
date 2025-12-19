import os
import hashlib
import json
from typing import List, Dict
from dataclasses import asdict

from provenance.spec import OriginFile, FileRole, EntityType, OriginID

IGNORABLE_DIRS = {
    ".git", "__pycache__", ".ipynb_checkpoints", ".pytest_cache", 
    ".mypy_cache", "node_modules", ".venv", "venv", "env", ".idea", ".vscode"
}

IGNORABLE_EXTS = {".pyc", ".pyd", ".o", ".so", ".dll", ".class"}

DEFAULT_ROLES = {
    ".py": FileRole.SOURCE,
    ".js": FileRole.SOURCE,
    ".html": FileRole.SOURCE, # Or TEMPLATE
    ".css": FileRole.SOURCE,
    ".md": FileRole.CONFIG, # Or DOC
    ".yaml": FileRole.CONFIG,
    ".yml": FileRole.CONFIG,
    ".json": FileRole.CONFIG,
    ".csv": FileRole.INPUT, # Defaulting to input, could be dataset
    ".png": FileRole.BINARY,
    ".jpg": FileRole.BINARY,
}

def get_file_role(path: str) -> FileRole:
    """Heuristic to determine file role."""
    _, ext = os.path.splitext(path)
    # Check specific paths first
    if "dashboard" in path and ext == ".html":
        return FileRole.REPORT
    if "logs" in path:
        return FileRole.LOG
    
    return DEFAULT_ROLES.get(ext, FileRole.UNKNOWN)

def generate_origin_id(rel_path: str) -> OriginID:
    """Stable ID based on path."""
    return hashlib.sha256(rel_path.encode('utf-8')).hexdigest()[:16]

class Scanner:
    def __init__(self, root_dir: str):
        self.root_dir = os.path.abspath(root_dir)
        self.registry: List[OriginFile] = []

    def scan(self):
        print(f"Scanning {self.root_dir}...")
        for root, dirs, files in os.walk(self.root_dir):
            # Prune directories
            dirs[:] = [d for d in dirs if d not in IGNORABLE_DIRS]
            
            for file in files:
                _, ext = os.path.splitext(file)
                if ext in IGNORABLE_EXTS:
                    continue
                
                abs_path = os.path.join(root, file)
                rel_path = os.path.relpath(abs_path, self.root_dir)
                
                role = get_file_role(rel_path)
                origin_id = generate_origin_id(rel_path)
                
                origin_file = OriginFile(
                    path=rel_path,
                    origin_id=origin_id,
                    role=role,
                    description=None,
                    tags={"scanned_at": "now"} # Placehold
                )
                self.registry.append(origin_file)
        
        print(f"Scanned {len(self.registry)} files.")

    def export(self, path: str):
        """Export registry to JSON."""
        data = {
            "version": "1.0",
            "files": [
                {
                    "path": f.path, 
                    "origin_id": f.origin_id, 
                    "role": f.role.value,
                    "tags": f.tags
                } 
                for f in self.registry
            ]
        }
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"Registry exported to {path}")

if __name__ == "__main__":
    scanner = Scanner(os.getcwd())
    scanner.scan()
    scanner.export("provenance_registry.json")
