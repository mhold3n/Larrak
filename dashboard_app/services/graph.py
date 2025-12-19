import sqlite3
import json
import yaml
import time
from pathlib import Path
from typing import Dict, Any, List, Set

# region agent log (graph)
_AGENT_DEBUG_LOG_PATH = r"c:\Users\maxed\OneDrive\Desktop\Github Projects\Larrak\.cursor\debug.log"
def _agent_log(hypothesisId: str, location: str, message: str, data: Dict[str, Any]):
    try:
        payload = {
            "sessionId": "debug-session",
            "runId": "graph",
            "hypothesisId": hypothesisId,
            "location": location,
            "message": message,
            "data": data,
            "timestamp": int(time.time() * 1000),
        }
        Path(_AGENT_DEBUG_LOG_PATH).parent.mkdir(parents=True, exist_ok=True)
        with open(_AGENT_DEBUG_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
    except Exception:
        pass
# endregion agent log

class GraphService:
    def __init__(self, root_dir: str):
        # DB is in the root directory
        self.root_dir = Path(root_dir)
        self.db_path = self.root_dir / "provenance.db"
        
    def build_graph(self) -> Dict[str, Any]:
        print(f"GraphService DB Path: {self.db_path} Exists: {self.db_path.exists()}")
        if not self.db_path.exists():
            return {"nodes": [], "links": []}
            
        nodes = {}
        links = []
        
        # Load registry for module ordering (sequence links), display_names, and to ensure all modules appear as nodes
        registry_modules: List[Dict[str, Any]] = []
        display_names: Dict[str, str] = {}
        registry_path = self.root_dir / "provenance" / "registry.yaml"
        if registry_path.exists():
            try:
                with open(registry_path, "r", encoding="utf-8") as f:
                    reg = yaml.safe_load(f) or {}
                    registry_modules = reg.get("modules", [])
                    # Build display_name map
                    for mod in registry_modules:
                        mod_id = mod.get("module_id")
                        if mod_id:
                            display_names[mod_id] = mod.get("display_name", mod_id)
            except Exception as e:
                print(f"Graph registry load error: {e}")

        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        c = conn.cursor()
        
        try:
            # 1. Get Latest Runs per Module
            # simple group by module_id
            c.execute('''
                SELECT run_id, module_id, status, start_time 
                FROM runs 
                ORDER BY start_time ASC
            ''')
            # Use dictionary to keep only latest per module
            latest_runs = {}
            for r in c.fetchall():
                latest_runs[r['module_id']] = dict(r)
                
            # Create Module Nodes (prefer registry order, fall back to runs-only)
            reg_ids = [m.get("module_id") for m in registry_modules if m.get("module_id")]
            all_mod_ids = reg_ids or list(latest_runs.keys())
            for mod_id in all_mod_ids:
                run = latest_runs.get(mod_id)
                # Use display_name from registry if available, otherwise use module_id
                label = display_names.get(mod_id, mod_id)
                nodes.setdefault(mod_id, {
                    "id": mod_id,
                    "type": "module",
                    "label": label,
                    "display_name": label,
                    "status": run['status'] if run else "unknown",
                    "run_id": run['run_id'] if run else None
                })
                
            # 2. Get Artifacts (Outputs) for these Runs
            # We only track artifacts from the *latest* runs to keep graph clean
            run_ids = [r['run_id'] for r in latest_runs.values()]
            if not run_ids:
                return {"nodes": [], "links": []}
                
            placeholders = ','.join('?' * len(run_ids))
            
            # Outputs (Artifacts)
            c.execute(f'''
                SELECT path, run_id, role 
                FROM artifacts 
                WHERE run_id IN ({placeholders})
            ''', tuple(run_ids))
            
            # Helper map: run_id -> module_id
            run_to_mod = {r['run_id']: r['module_id'] for r in latest_runs.values()}

            artifacts = c.fetchall()
            for art in artifacts:
                path = art['path']
                mod_id = run_to_mod.get(art['run_id'])
                if not mod_id:
                    continue
                    
                # Node for Artifact
                # Simplify path for display
                label = Path(path).name
                
                nodes[path] = {
                    "id": path,
                    "type": "artifact",
                    "label": label,
                    "role": art['role']
                }
                
                # Link Module -> Artifact
                links.append({
                    "source": mod_id,
                    "target": path,
                    "type": "produced"
                })
                
            # 3. Get Inputs (File Events) for these Runs
            # Helper to find which artifact (if any) corresponds to a Read event
            # For now, inputs are just paths. If that path exists as an artifact node, link it.
            # If not, create an "Origin" node.
            
            c.execute(f'''
                SELECT run_id, event_type, details, id 
                FROM events 
                WHERE run_id IN ({placeholders})
                ORDER BY id ASC
            ''', tuple(run_ids))
            
            events = c.fetchall()
            checkpoint_counter = {}
            checkpoints_by_run = {}
            loop_index = {}
            retry_count = {}
            for evt in events:
                evt_type = evt['event_type']
                details = json.loads(evt['details'])
                run = next((r for r in latest_runs.values() if r['run_id'] == evt['run_id']), None)

                if evt_type == 'FileEvent' and details.get('op') == 'read':
                    path = details['path']
                    
                    target_node_id = None
                    
                    # Try exact match
                    if path in nodes:
                        target_node_id = path
                    else:
                        # Try finding key in nodes that ends with this path or vice versa
                        for n_id in nodes:
                            if path.endswith(n_id) or n_id.endswith(path):
                                target_node_id = n_id
                                break
                    
                    # If still not found, it's an external/origin input
                    if not target_node_id:
                        # Create Origin Node
                        label = Path(path).name
                        if "Larrak" in path: 
                            target_node_id = path
                            nodes[target_node_id] = {
                                "id": target_node_id,
                                "type": "artifact",
                                "label": label,
                                "role": "input"
                            }
                    
                    if run and target_node_id:
                        links.append({
                            "source": target_node_id,
                            "target": run['module_id'],
                            "type": "consumed"
                        })

                elif evt_type == 'CheckpointEvent':
                    name = details.get('name', 'checkpoint')
                    # Track loop index but do not create nodes for loop_* starts
                    if name.startswith("loop_") and name.endswith("_start"):
                        try:
                            idx_num = int(name.split("_")[1])
                        except Exception:
                            idx_num = loop_index.get(evt['run_id'], 0) + 1
                        loop_index[evt['run_id']] = idx_num
                        continue

                    idx = checkpoint_counter.get(evt['run_id'], 0) + 1
                    checkpoint_counter[evt['run_id']] = idx
                    node_id = f"{evt['run_id']}::chk::{idx}"
                    label = f"{name}"
                    nodes[node_id] = {
                        "id": node_id,
                        "type": "checkpoint",
                        "label": label,
                        "run_id": evt['run_id'],
                        "passed": details.get('passed', True),
                        "expected": details.get('expected'),
                        "observed": details.get('observed')
                    }
                    if run:
                        links.append({
                            "source": run['module_id'],
                            "target": node_id,
                            "type": "checkpoint"
                        })
                    checkpoints_by_run.setdefault(evt['run_id'], []).append({
                        "node_id": node_id,
                        "name": name,
                        "passed": details.get('passed', True),
                        "module_id": run['module_id'] if run else None
                    })

                    # If convergence failed, emit a retry edge back to module
                    if name == "convergence_gate" and details.get('passed') is False and run:
                        retry_num = retry_count.get(evt['run_id'], 0) + 1
                        retry_count[evt['run_id']] = retry_num
                        links.append({
                            "source": node_id,
                            "target": run['module_id'],
                            "type": "loop_retry",
                            "iteration": loop_index.get(evt['run_id'], retry_num),
                            "label": f"retry {retry_num}"
                        })

            # Link checkpoints sequentially to visualize flow
            for run_id, cp_list in checkpoints_by_run.items():
                for i in range(len(cp_list) - 1):
                    links.append({
                        "source": cp_list[i]["node_id"],
                        "target": cp_list[i+1]["node_id"],
                        "type": "checkpoint_flow"
                    })

            # 4. Add sequence links from registry (to visualize pipeline even without artifacts)
            if reg_ids and len(reg_ids) > 1:
                for i in range(len(reg_ids) - 1):
                    src = reg_ids[i]
                    dst = reg_ids[i + 1]
                    # Avoid duplicates
                    if src in nodes and dst in nodes:
                        links.append({
                            "source": src,
                            "target": dst,
                            "type": "sequence"
                        })
                        
        except Exception as e:
            print(f"Graph build error: {e}")
        finally:
            conn.close()
            
        result = {
            "nodes": list(nodes.values()),
            "links": links
        }
        _agent_log(hypothesisId="G1", location="graph.build_graph", message="graph_built", data={"nodes": len(result["nodes"]), "links": len(result["links"]), "has_registry": bool(reg_ids)})
        return result

# Singleton
# Resolve root relative to this file: dashboard_app/services/graph.py -> root is 3 levels up
ROOT_DIR = Path(__file__).parent.parent.parent
graph_service = GraphService(str(ROOT_DIR))
