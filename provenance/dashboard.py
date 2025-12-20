import json
import os
import datetime
import yaml
import weaviate
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

from provenance.db import db # Use the global instance which has the client

@dataclass
class RunSummary:
    run_id: str
    module_id: str
    start_time: str
    end_time: str
    status: str
    inputs: List[Dict[str, Any]]
    outputs: List[Dict[str, Any]]
    events: List[Dict[str, Any]]

def get_latest_run(module_id: str) -> Optional[RunSummary]:
    if not db._client:
        return None
        
    runs = db._client.collections.get("Run")
    
    # Query for the specific module
    # We need to filter by the referenced module's module_id.
    # In v4, we can filter by nested properties.
    
    response = runs.query.fetch_objects(
        filters=weaviate.classes.query.Filter.by_ref("executed_module").by_property("module_id").equal(module_id),
        sort=weaviate.classes.query.Sort.by_property("start_time", ascending=False),
        limit=1,
        return_references=[
            weaviate.classes.query.QueryReference(
                link_on="generated_artifacts",
                return_properties=["path", "role", "content_hash", "artifact_id"]
            ),
             weaviate.classes.query.QueryReference(
                link_on="executed_module",
                return_properties=["module_id"]
            )
        ]
    )
    
    if not response.objects:
        return None
        
    run_obj = response.objects[0]
    props = run_obj.properties
    
    # Extract references
    artifacts = run_obj.references.get("generated_artifacts", [])
    module_ref = run_obj.references.get("executed_module", [])
    
    actual_module_id = module_ref[0].properties["module_id"] if module_ref else module_id
    
    # Map Artifacts (Outputs)
    output_list = []
    for art in artifacts:
        p = art.properties
        output_list.append({
            'path': p.get('path'),
            'role': p.get('role'),
            'hash': p.get('content_hash'),
            'id': p.get('artifact_id')
        })
        
    # Inputs - For now, we don't have a reliable way to get inputs unless they are explicitly linked
    # We will leave this empty for this iteration or implement 'used_input_artifacts' fetching if we had them.
    input_list = [] 
                
    # Events - Detailed events are not currently pushed to Weaviate as objects.
    # We might have stored them in a blob or just omit them for now.
    event_list = []
        
    return RunSummary(
        run_id=props.get('run_id'),
        module_id=actual_module_id,
        start_time=str(props.get('start_time')),
        end_time=str(props.get('end_time')),
        status=props.get('status'),
        inputs=input_list,
        outputs=output_list,
        events=event_list
    )

def load_expectations(module_id: str) -> Dict[str, Any]:
    # In a real app, use the Scanner/Registry class.
    # Here, we parse the yaml directly for simplicity.
    try:
        with open("provenance/registry.yaml", "r") as f:
            registry = yaml.safe_load(f)
            for mod in registry.get("modules", []):
                if mod["module_id"] == module_id:
                    return mod
    except Exception as e:
        print(f"Error loading registry: {e}")
    return {}

def get_display_name(module_id: str) -> str:
    """Get human-readable display name for a module from registry."""
    try:
        with open("provenance/registry.yaml", "r") as f:
            registry = yaml.safe_load(f)
            for mod in registry.get("modules", []):
                if mod["module_id"] == module_id:
                    return mod.get("display_name", module_id)
    except Exception:
        pass
    return module_id

def generate_html(summary: RunSummary, expected: Dict[str, Any]) -> str:
    # Basic CSS
    css = """
    body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; padding: 20px; background: #f4f6f8; color: #172b4d; }
    .container { max_width: 1000px; margin: 0 auto; background: white; padding: 30px; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.12); }
    h1 { border-bottom: 2px solid #ebecf0; padding-bottom: 10px; margin-top: 0; }
    h2 { color: #42526e; margin-top: 30px; font-size: 1.2em; text-transform: uppercase; letter-spacing: 0.05em; }
    .badge { padding: 4px 8px; border-radius: 4px; font-size: 0.85em; font-weight: bold; }
    .badge-success { background: #e3fcef; color: #006644; }
    .badge-running { background: #deebff; color: #0747a6; }
    .badge-failure { background: #ffebe6; color: #bf2600; }
    table { width: 100%; border-collapse: collapse; margin-top: 10px; }
    th, td { text-align: left; padding: 12px; border-bottom: 1px solid #ebecf0; }
    th { color: #5e6c84; font-size: 0.9em; }
    .diff-missing { color: #bf2600; background: #ffebe6; }
    .diff-extra { color: #006644; background: #e3fcef; }
    .section { margin-bottom: 30px; }
    .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
    .module-id { color: #5e6c84; font-size: 0.85em; font-weight: normal; }
    """
    
    # Status Badge
    status_class = "badge-success" if summary.status == "SUCCESS" else "badge-failure" if summary.status == "FAILURE" else "badge-running"
    
    # Get human-readable display name
    display_name = get_display_name(summary.module_id)
    
    # Expectation Matching (Naive)
    def normalize(p):
        try:
            return os.path.relpath(os.path.abspath(p), os.getcwd())
        except:
            return p

    expected_inputs = [normalize(e['path']) for e in expected.get('expected_inputs', [])]
    expected_outputs = [normalize(e['path']) for e in expected.get('expected_outputs', [])]
    
    actual_inputs = [normalize(i['path']) for i in summary.inputs]
    actual_outputs = [normalize(o['path']) for o in summary.outputs]
    
    # HTML Construction
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Provenance: {display_name}</title>
        <style>{css}</style>
    </head>
    <body>
        <div class="container">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <h1>{display_name} <span class="module-id">({summary.module_id})</span></h1>
                <span class="badge {status_class}">{summary.status}</span>
            </div>
            
            <p><strong>Run ID:</strong> {summary.run_id}</p>
            <p><strong>Time:</strong> {summary.start_time} &rarr; {summary.end_time}</p>
            
            <div class="grid">
                <div class="section">
                    <h2>Inputs</h2>
                    <p><em>(Input tracking via FileEvents is temporarily disabled in Weaviate migration)</em></p>
                    <table>
                        <thead><tr><th>Path</th><th>Status</th></tr></thead>
                        <tbody>
                    """
    
    # Expectation logic for inputs omitted for brevity as we don't have actual inputs
    html += """
                        </tbody>
                    </table>
                </div>
                
                <div class="section">
                    <h2>Outputs</h2>
                    <table>
                        <thead><tr><th>Path</th><th>Status</th></tr></thead>
                        <tbody>
    """
    
    # Render Outputs
    for path in expected_outputs:
        found = any(path == out for out in actual_outputs)
        cls = "" if found else "diff-missing"
        icon = "✓" if found else "MISSING"
        html += f"<tr class='{cls}'><td>{path}</td><td>{icon}</td></tr>"
        
    for out in actual_outputs:
        if out not in expected_outputs:
             html += f"<tr class='diff-extra'><td>{out}</td><td>EXTRA</td></tr>"

    html += """
                        </tbody>
                    </table>
                </div>
            </div>

            <div class="section">
                <h2>Event Log</h2>
                 <table>
                    <thead><tr><th>Time</th><th>Type</th><th>Details</th></tr></thead>
                    <tbody>
    """
    
    for ev in summary.events:
        html += f"<tr><td>{ev.get('timestamp')}</td><td>{ev.get('type')}</td><td><pre>{json.dumps(ev.get('details'), indent=2)}</pre></td></tr>"
        
    html += """
                    </tbody>
                </table>
            </div>
        </div>
    </body>
    </html>
    """
    
    return html

def main():
    module_id = "gear_profile_synthesis" 
    print(f"Generating dashboard for {module_id}...")
    
    try:
        summary = get_latest_run(module_id)
        if not summary:
            print("No run found.")
            return
            
        expected = load_expectations(module_id)
        
        html = generate_html(summary, expected)
        
        out_path = f"dashboard/provenance_{module_id}.html"
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(html)
            
        print(f"Report generated: {out_path}")
    except Exception as e:
        print(f"Error generating dashboard: {e}")

if __name__ == "__main__":
    main()
