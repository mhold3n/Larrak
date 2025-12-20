import os
import json
from typing import List, Dict, Any
from pathlib import Path
from dataclasses import dataclass

@dataclass
class Page:
    page_id: str
    title: str
    path: str
    type: str # module_dashboard, report, graph
    timestamp: float
    tags: Dict[str, Any]

class Indexer:
    def __init__(self, root_dir: str):
        self.root_dir = Path(root_dir).resolve()
        
    def scan(self) -> List[Dict[str, Any]]:
        pages = []
        # Areas to scan
        scan_dirs = ["dashboard"] 
        
        print(f"Scanning directories: {scan_dirs} in {self.root_dir}")
        
        for d in scan_dirs:
            p = self.root_dir / d
            if not p.exists():
                continue
                
            for root, _, files in os.walk(p):
                for file in files:
                    if file.endswith(".html"):
                        abs_path = Path(root) / file
                        rel_path = abs_path.relative_to(self.root_dir)
                        
                        # Heuristic for type and title
                        name = file.replace(".html", "")
                        page_type = "report"
                        title = name.replace("_", " ").title()
                        
                        if "provenance_" in name:
                            page_type = "module_dashboard"
                            title = name.replace("provenance_", "").replace("_", " ").title() + " (Provenance)"
                        elif "graph" in name:
                            page_type = "graph"
                        elif "final_gear_set" in name:
                            page_type = "report" # Specialized report
                            title = "Final Gear Set"
                            
                        # Get modified time
                        mtime = abs_path.stat().st_mtime
                        
                        # Prepare web_path for consistent use
                        web_path = str(rel_path).replace('\\', '/')
                        
                        pages.append(asdict(Page(
                            page_id=web_path, # Use web_path here for consistency
                            title=title,
                            path=f"/artifacts/{web_path}",
                            type=page_type,
                            timestamp=mtime,
                            tags={}
                        )))
        
        # Sort by newest
        pages.sort(key=lambda x: x['timestamp'], reverse=True)
        return pages

def asdict(obj):
    return obj.__dict__
