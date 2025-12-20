from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import os
from pathlib import Path

from webapp.services.indexer import Indexer

app = FastAPI(title="Larrak Dashboard")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Services
BASE_DIR = Path(os.getcwd())
indexer = Indexer(str(BASE_DIR))

# --- API Endpoints ---

@app.get("/api/pages")
async def get_pages():
    return indexer.scan()

# --- Run Endpoints ---

from webapp.services.runner import runner
from pydantic import BaseModel

class RunRequest(BaseModel):
    module_id: str
    params: dict

@app.get("/api/modules")
async def get_modules():
    return runner.get_modules()

@app.post("/api/runs")
async def start_run(req: RunRequest):
    return runner.run_module(req.module_id, req.params)

# --- WebSocket ---
from fastapi import WebSocket
from webapp.services.stream import stream_manager

@app.websocket("/ws/log")
async def websocket_endpoint(websocket: WebSocket):
    await stream_manager.connect(websocket)
    try:
        while True:
            # Keep connection alive, maybe receive commands later
            data = await websocket.receive_text()
    except:
        stream_manager.disconnect(websocket)

# --- Graph ---
from webapp.services.graph import graph_service

@app.get("/api/graph")
async def get_graph():
    return graph_service.build_graph()

# --- Documentation ---
from webapp.services.docs import docs_service

# --- Run History ---
from webapp.services import runs_service

@app.get("/api/docs/tree")
async def get_docs_tree():
    """Return the documentation file tree structure."""
    return docs_service.get_tree()

@app.get("/api/docs/render")
async def render_doc(path: str):
    """Render a markdown document to HTML."""
    result = docs_service.render_doc(path)
    if result is None:
        raise HTTPException(status_code=404, detail="Document not found")
    return result

# --- Run History Endpoints ---

@app.get("/api/runs/history")
async def get_runs_history(limit: int = 20):
    """Get recent run history with status and metrics."""
    return runs_service.get_recent_runs(limit)

@app.get("/api/runs/stats")
async def get_runs_stats():
    """Get aggregate run statistics."""
    return runs_service.get_run_stats()

@app.get("/api/runs/{run_id}")
async def get_run_detail(run_id: str):
    """Get details for a specific run including artifacts."""
    result = runs_service.get_run_details(run_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Run not found")
    return result

# --- Static File Serving ---

# 1. Serve the App Itself
app.mount("/static", StaticFiles(directory="webapp/static"), name="static")

# 2. Serve Artifacts (The reports we scan)
# We mount the root directory but only index specific folders. 
# PLEASE NOTE: In production, restricted mounting is safer.
app.mount("/artifacts", StaticFiles(directory="output"), name="artifacts")

@app.get("/")
async def read_index():
    return FileResponse("webapp/static/index.html")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
