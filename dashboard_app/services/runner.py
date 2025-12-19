import subprocess
import os
import sys
import yaml
import asyncio
from typing import Dict, Any, List
from pathlib import Path
from dashboard_app.services.stream import stream_manager
import json
import time
import traceback
from os import makedirs
from os.path import dirname

# region agent log
_AGENT_DEBUG_LOG_PATH = r"c:\Users\maxed\OneDrive\Desktop\Github Projects\Larrak\.cursor\debug.log"
def _agent_log(hypothesisId: str, location: str, message: str, data: Dict[str, Any], runId: str = "pre-fix") -> None:
    try:
        makedirs(dirname(_AGENT_DEBUG_LOG_PATH), exist_ok=True)
        payload = {
            "sessionId": "debug-session",
            "runId": runId,
            "hypothesisId": hypothesisId,
            "location": location,
            "message": message,
            "data": data,
            "timestamp": int(time.time() * 1000),
        }
        with open(_AGENT_DEBUG_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
    except Exception as e:
        # If file-write logging is broken, we still need a visible signal in server stdout.
        try:
            print(f"[AgentLog] FAILED to write debug log: {type(e).__name__} {repr(e)}")
        except Exception:
            pass
# endregion agent log

# region agent log
_agent_log(
    hypothesisId="Z",
    location="dashboard_app/services/runner.py:module_import",
    message="Runner module imported",
    data={"pid": os.getpid(), "python": sys.executable},
    runId="pre-fix",
)
# endregion agent log

class Runner:
    def __init__(self, root_dir: str):
        self.root_dir = Path(root_dir).resolve()
        self.registry = self._load_registry()
        self.current_process = None
        
    def _load_registry(self):
        try:
            with open(self.root_dir / "provenance" / "registry.yaml", "r") as f:
                return yaml.safe_load(f)
        except:
            return {"modules": []}
            
    def get_modules(self):
        return self.registry.get("modules", [])
        
    async def run_module_async(self, module_id: str, params: Dict[str, Any]):
        # region agent log
        _agent_log(
            hypothesisId="F",
            location="dashboard_app/services/runner.py:run_module_async:entry",
            message="Entered run_module_async",
            data={
                "module_id": module_id,
                "pid": os.getpid(),
                "runner_file": __file__,
                "cwd": str(self.root_dir),
                "param_keys": list((params or {}).keys()),
            },
            runId="pre-fix",
        )
        # endregion agent log
        # Find module
        module = next((m for m in self.registry["modules"] if m["module_id"] == module_id), None)
        if not module:
            raise ValueError(f"Module {module_id} not found")
        
        # 1. Prepare Run Context
        import uuid
        import datetime
        from provenance.db import db
        
        run_id = str(uuid.uuid4())
        
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        # Windows consoles often default to cp1252; force UTF-8 so logs don't crash on symbols like '≈' or 'κ'.
        env.setdefault("PYTHONUTF8", "1")
        env.setdefault("PYTHONIOENCODING", "utf-8")
        env["LARRAK_RUN_ID"] = run_id
        
        for k, v in params.items():
            env[f"LARRAK_{k.upper()}"] = str(v)
            
        cmd = [sys.executable, module["entrypoint"]]

        # region agent log
        entrypoint = module.get("entrypoint")
        entrypoint_path = (self.root_dir / entrypoint).resolve() if isinstance(entrypoint, str) else None
        _agent_log(
            hypothesisId="B",
            location="dashboard_app/services/runner.py:run_module_async",
            message="Prepared run context",
            data={
                "module_id": module_id,
                "run_id": run_id,
                "cwd": str(self.root_dir),
                "entrypoint": entrypoint,
                "entrypoint_exists": bool(entrypoint_path.exists()) if entrypoint_path else None,
                "cmd": cmd,
                "param_keys": list(params.keys()),
            },
            runId="pre-fix",
        )
        # endregion agent log
        
        # 2. Log Start to DB
        try:
            db.start_run(run_id, module_id, cmd, {k:str(v) for k,v in env.items() if k.startswith("LARRAK_")})
        except Exception as e:
            print(f"[Runner] DB Start Log Failed: {e}")
        
        await stream_manager.broadcast({
            "type": "status", 
            "content": "running", 
            "module_id": module_id
        })
        
        try:
            # region agent log
            _agent_log(
                hypothesisId="C",
                location="dashboard_app/services/runner.py:run_module_async",
                message="Spawning subprocess (strategy selected)",
                data={
                    "module_id": module_id,
                    "run_id": run_id,
                    "cmd": cmd,
                    "cwd": str(self.root_dir),
                    "strategy": "popen_thread" if os.name == "nt" else "asyncio_subprocess",
                },
                runId="pre-fix",
            )
            # endregion agent log

            if os.name == "nt":
                # Windows: asyncio subprocess transport may be unavailable depending on loop policy.
                # Use Popen + thread offload for portable streaming.
                loop = asyncio.get_running_loop()
                process = subprocess.Popen(
                    cmd,
                    cwd=str(self.root_dir),
                    env=env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                )
                self.current_process = process

                def _run_and_stream() -> int:
                    try:
                        if process.stdout is not None:
                            for line in process.stdout:
                                decoded = (line or "").rstrip("\r\n")
                                if decoded:
                                    print(f"[Runner] {decoded}")  # Local echo
                                    asyncio.run_coroutine_threadsafe(
                                        stream_manager.broadcast({"type": "log", "content": decoded}),
                                        loop,
                                    )
                    finally:
                        try:
                            if process.stdout is not None:
                                process.stdout.close()
                        except Exception:
                            pass
                    return process.wait()

                returncode = await asyncio.to_thread(_run_and_stream)
                status = "SUCCESS" if returncode == 0 else "FAILURE"
                # region agent log
                _agent_log(
                    hypothesisId="G",
                    location="dashboard_app/services/runner.py:run_module_async",
                    message="Subprocess completed (windows popen_thread)",
                    data={"module_id": module_id, "run_id": run_id, "returncode": returncode, "status": status},
                    runId="pre-fix",
                )
                # endregion agent log
            else:
                # POSIX: asyncio subprocess works well.
                process = await asyncio.create_subprocess_exec(
                    *cmd,
                    cwd=self.root_dir,
                    env=env,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.STDOUT
                )
                self.current_process = process

                # Stream output
                while True:
                    line = await process.stdout.readline()
                    if not line:
                        break
                    decoded = line.decode().strip()
                    if decoded:
                        print(f"[Runner] {decoded}") # Local echo
                        await stream_manager.broadcast({
                            "type": "log", 
                            "content": decoded
                        })

                await process.wait()
                status = "SUCCESS" if process.returncode == 0 else "FAILURE"
                # region agent log
                _agent_log(
                    hypothesisId="G",
                    location="dashboard_app/services/runner.py:run_module_async",
                    message="Subprocess completed (asyncio)",
                    data={"module_id": module_id, "run_id": run_id, "returncode": process.returncode, "status": status},
                    runId="pre-fix",
                )
                # endregion agent log
            
            # 3. Log End to DB
            try:
                db.end_run(run_id, status)
            except Exception as e:
                print(f"[Runner] DB End Log Failed: {e}")
            
            await stream_manager.broadcast({
                "type": "status", 
                "content": status.lower(),
                "module_id": module_id
            })
            
        except Exception as e:
            print(f"[Runner] Failed: {e}")
            # region agent log
            _agent_log(
                hypothesisId="A",
                location="dashboard_app/services/runner.py:run_module_async",
                message="Runner exception",
                data={
                    "module_id": module_id,
                    "run_id": run_id,
                    "exc_type": type(e).__name__,
                    "exc_repr": repr(e),
                    "traceback": traceback.format_exc(),
                },
                runId="pre-fix",
            )
            # endregion agent log
            try:
                db.end_run(run_id, "FAILURE")
            except: pass
            
            await stream_manager.broadcast({
                "type": "status", 
                "content": "failure",
                "error": str(e),
                "error_type": type(e).__name__,
                "error_repr": repr(e),
                "module_id": module_id
            })

    def run_module(self, module_id: str, params: Dict[str, Any]):
        # region agent log
        try:
            loop_running = asyncio.get_running_loop().is_running()
        except Exception:
            loop_running = None
        _agent_log(
            hypothesisId="A",
            location="dashboard_app/services/runner.py:run_module",
            message="API requested run (creating task)",
            data={
                "module_id": module_id,
                "param_keys": list((params or {}).keys()),
                "loop_running": loop_running,
            },
            runId="pre-fix",
        )
        # endregion agent log
        # Fire and forget async task
        asyncio.create_task(self.run_module_async(module_id, params))
        return {"status": "started", "module_id": module_id}

# Singleton
runner = Runner(os.getcwd())
