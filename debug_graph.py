from dashboard_app.services.graph import graph_service
import json

print("Building graph...")
try:
    data = graph_service.build_graph()
    print("Graph Data Nodes:", len(data['nodes']))
    print("Graph Data Links:", len(data['links']))
    print(json.dumps(data, indent=2))
except Exception as e:
    print(f"Error: {e}")
