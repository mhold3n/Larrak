from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import os
from pathlib import Path

from dashboard_app.services.indexer import Indexer

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

from dashboard_app.services.runner import runner
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
from dashboard_app.services.stream import stream_manager

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
from dashboard_app.services.graph import graph_service

@app.get("/api/graph")
async def get_graph():
    return graph_service.build_graph()

# --- Static File Serving ---

# 1. Serve the App Itself
app.mount("/static", StaticFiles(directory="dashboard_app/static"), name="static")

# 2. Serve Artifacts (The reports we scan)
# We mount the root directory but only index specific folders. 
# PLEASE NOTE: In production, restricted mounting is safer.
app.mount("/artifacts", StaticFiles(directory="."), name="artifacts")

@app.get("/")
async def read_index():
    return FileResponse("dashboard_app/static/index.html")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
