"""
Calibration Database Logger.
Manages the storage of Simulation Runs in `Simulations/_runs/`.
Enforces the structure: _runs/<timestamp>_<hash>/[inputs.json, outputs.json, meta.json]
"""

import os
import json
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
from .io_schema import SimulationInput, SimulationOutput

RUNS_DIR = Path(__file__).parent.parent / "_runs"

class SimulationLogger:
    """Handles persistence of simulation artifacts."""
    
    def __init__(self, run_root: Path = RUNS_DIR):
        self.root = run_root
        self.run_dir: Optional[Path] = None
        self.input_model: Optional[SimulationInput] = None
        
    def start_run(self, input_model: SimulationInput) -> Path:
        """
        Initializes a new run directory.
        1. Computes Hash of input configuration.
        2. Creates directory `YYYYMMDD_HHMMSS_<short_hash>`.
        3. Writes `inputs.json`.
        """
        self.input_model = input_model
        
        # Create Hash of critical inputs (Geometry + Ops + BCs)
        # We dump to json with sort_keys to ensure deterministic hash
        inputs_json = input_model.model_dump_json()
        input_hash = hashlib.sha256(inputs_json.encode('utf-8')).hexdigest()[:8]
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"{timestamp}_{input_hash}"
        
        self.run_dir = self.root / run_name
        self.run_dir.mkdir(parents=True, exist_ok=True)
        
        # Save Inputs
        with open(self.run_dir / "inputs.json", "w") as f:
            f.write(inputs_json)
            
        print(f"[Logger] Started Run: {run_name}")
        return self.run_dir
    
    def log_output(self, output_model: SimulationOutput, meta: Dict[str, Any] = None):
        """
        Saves run results.
        1. Writes `outputs.json`.
        2. Writes `meta.json` (computational details, logs).
        """
        if not self.run_dir:
            raise RuntimeError("Run not started. Call start_run() first.")
            
        # Save Output
        with open(self.run_dir / "outputs.json", "w") as f:
            f.write(output_model.model_dump_json(indent=2))
            
        # Save Meta
        if meta is None:
            meta = {}
            
        # Enrich meta
        meta["timestamp_end"] = datetime.now().isoformat()
        meta["run_id"] = self.input_model.run_id
        
        with open(self.run_dir / "meta.json", "w") as f:
            json.dump(meta, f, indent=2)
            
        print(f"[Logger] Completed Run: {self.run_dir}")

    @staticmethod
    def load_run(run_path: str) -> Tuple[SimulationInput, SimulationOutput]:
        """Utility to load a past run."""
        p = Path(run_path)
        with open(p / "inputs.json", "r") as f:
            inp = SimulationInput.model_validate_json(f.read())
        with open(p / "outputs.json", "r") as f:
            out = SimulationOutput.model_validate_json(f.read())
        return inp, out
