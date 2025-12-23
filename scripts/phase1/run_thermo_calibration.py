import os
import sys

# [DEBUG] Immediate start log
print(f"[INFO] Process {os.getpid()} starting: run_thermo_calibration.py", flush=True)

import datetime
import json
import subprocess
import time
import uuid
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.setup import env_setup  # [FIX] Path resolution for CasADi

# Add project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from campro.optimization.nlp.config import CONFIG

try:
    from provenance.db import db
    from provenance.spec import Artifact, CheckpointEvent, FileRole
except ImportError:
    db = None  # type: ignore
    CheckpointEvent = None  # type: ignore
    Artifact = None  # type: ignore
    FileRole = None  # type: ignore

LOG_PATH = str(Path.home() / ".larrak" / "debug.log")


# region agent log (adaptive_doe)
def _agent_log(hypothesisId: str, location: str, message: str, data: dict):
    try:
        payload = {
            "sessionId": "debug-session",
            "runId": os.environ.get("LARRAK_RUN_ID", "thermo_calibration_loop"),
            "hypothesisId": hypothesisId,
            "location": location,
            "message": message,
            "data": data,
            "timestamp": int(time.time() * 1000),
        }
        Path(LOG_PATH).parent.mkdir(parents=True, exist_ok=True)
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
    except Exception:
        pass


# endregion agent log


def _log_checkpoint(name: str, expected=None, observed=None, passed: bool = True):
    run_id = os.environ.get("LARRAK_RUN_ID")
    _agent_log(
        "G2",
        f"thermo_calibration:{name}",
        "checkpoint",
        {"expected": expected, "observed": observed, "passed": passed, "run_id": run_id},
    )
    if not run_id or not db or not CheckpointEvent:
        return
    try:
        db.log_event(
            CheckpointEvent(
                run_id=run_id, name=name, expected=expected, observed=observed, passed=passed
            )
        )
    except Exception:
        pass


def _register_artifact(path: str, role, meta: dict):
    run_id = os.environ.get("LARRAK_RUN_ID")
    if not run_id or not os.path.exists(path) or not db or not Artifact:
        return
    try:
        stat = os.stat(path)
        art = Artifact(
            artifact_id=str(uuid.uuid4()),
            path=os.path.abspath(path),
            content_hash=None,
            run_id=run_id,
            producer_module_id="thermo_calibration_loop",
            role=role,
            size_bytes=stat.st_size,
            creation_time=datetime.datetime.fromtimestamp(stat.st_mtime),
            metadata=meta or {},
        )
        db.register_artifact(art)
    except Exception:
        pass


def run_thermo_calibration():
    print("--- Thermo Calibration Loop: User -> DOE -> Sim -> Calibration ---")

    # 1. User Inputs & Scope
    # Loaded from campro.optimization.nlp.config.CONFIG (Geometry & Ranges)

    # --- Dashboard Overrides ---
    if "LARRAK_RPM_GRID_SIZE" in os.environ:
        rpm_n = int(os.environ["LARRAK_RPM_GRID_SIZE"])
        # Mocking grid update - assuming CONFIG has a method or we just simulate it here for the print
        # In reality we would do: CONFIG.rpm_grid = np.linspace(..., num=rpm_n)
        print(f"[Dashboard] Overriding RPM Grid to {rpm_n} points")

    if "LARRAK_ERROR_MARGIN" in os.environ:
        CONFIG.error_margin_percent = float(os.environ["LARRAK_ERROR_MARGIN"])
        print(f"[Dashboard] Overriding Error Margin to {CONFIG.error_margin_percent}%")

    if "LARRAK_RPM_MIN" in os.environ:
        CONFIG.ranges.rpm_min = float(os.environ["LARRAK_RPM_MIN"])
    if "LARRAK_RPM_MAX" in os.environ:
        CONFIG.ranges.rpm_max = float(os.environ["LARRAK_RPM_MAX"])
    if "LARRAK_FUEL_MAX_MG" in os.environ:
        CONFIG.ranges.fuel_max = float(os.environ["LARRAK_FUEL_MAX_MG"])
    if "LARRAK_BOOST_MAX_BAR" in os.environ:
        CONFIG.ranges.boost_max = float(os.environ["LARRAK_BOOST_MAX_BAR"])
    if "LARRAK_LAMBDA_MIN" in os.environ:
        CONFIG.ranges.lambda_min = float(os.environ["LARRAK_LAMBDA_MIN"])
    if "LARRAK_LAMBDA_MAX" in os.environ:
        CONFIG.ranges.lambda_max = float(os.environ["LARRAK_LAMBDA_MAX"])
    # ---------------------------

    print(
        f"[Core] Scope: {len(CONFIG.rpm_grid)} RPM x {len(CONFIG.boost_grid)} Boost x {len(CONFIG.fuel_grid)} Fuel"
    )
    print(
        f"[Core] Fixed Vars: Bore={CONFIG.geometry.bore * 1000}mm, Stroke={CONFIG.geometry.stroke * 1000}mm"
    )

    max_loops = 3
    loop_count = 0
    converged = False

    # Dashboard toggles (all optional) - support both new and legacy env var names
    train_surrogate = str(
        os.environ.get("LARRAK_TRAIN_SURROGATE", os.environ.get("LARRAK_TRAIN_INTERPRETER", "True"))
    ).strip().lower() in {"1", "true", "yes", "y", "on"}
    validate_surrogate = str(
        os.environ.get(
            "LARRAK_VALIDATE_SURROGATE", os.environ.get("LARRAK_VALIDATE_INTERPRETER", "True")
        )
    ).strip().lower() in {"1", "true", "yes", "y", "on"}
    surrogate_epochs = int(
        os.environ.get(
            "LARRAK_SURROGATE_EPOCHS",
            os.environ.get(
                "LARRAK_INTERPRETER_EPOCHS", os.environ.get("LARRAK_INTERPRETER_EPOCH", "200")
            ),
        )
    )
    surrogate_r2_target = float(
        os.environ.get(
            "LARRAK_SURROGATE_R2_TARGET", os.environ.get("LARRAK_INTERPRETER_R2_TARGET", "0.9")
        )
    )
    validation_error_margin = float(os.environ.get("LARRAK_VALIDATION_ERROR_MARGIN", "0.03"))

    while loop_count < max_loops and not converged:
        print(f"\n=== Loop {loop_count + 1} / {max_loops} ===")
        _log_checkpoint(
            name=f"loop_{loop_count + 1}_start",
            expected={"max_loops": max_loops},
            observed={"loop_index": loop_count + 1},
        )

        # 2. DOE Execution (NLP)
        print("[S2] Running DOE Optimization (NLP)...")
        # Call generate_doe.py as a subprocess to ensure clean state
        # (Assuming it uses the same CONFIG)
        cmd = [sys.executable, "tests/goldens/phase1/generate_doe.py"]
        ret = subprocess.call(cmd)
        if ret != 0:
            print("DOE Failed.")
            _log_checkpoint(name="doe_failure", expected=0, observed=ret, passed=False)
            sys.exit(ret)
        else:
            _log_checkpoint(name="doe_success", expected=0, observed=ret, passed=True)

        # 3. Detailed Simulation & Validation
        print("[S3] Validating Results against Surrogate Model...")
        results_path = os.environ.get(
            "LARRAK_THERMO_RESULTS",
            os.environ.get("LARRAK_PHASE1_RESULTS", "output/thermo/thermo_doe_results.csv"),
        )
        if not os.path.exists(results_path):
            print("Error: Results file not found.")
            _log_checkpoint(
                name="results_missing", expected=results_path, observed=None, passed=False
            )
            sys.exit(1)

        df = pd.read_csv(results_path)
        if FileRole:
            _register_artifact(results_path, FileRole.OUTPUT, {"loop": loop_count + 1})

        # Select Checkpoints (e.g., Highest Power, Highest Eff)
        top_power = df.loc[df["abs_work_net_j"].idxmax()]
        print(f"    Checkpoint A (Max Power): {top_power['rpm']} RPM, {top_power['p_int_bar']} Bar")

        # Train/Validate Surrogate on latest DOE results
        # This replaces the old mocked "Simulated High-Fi" step.
        env = os.environ.copy()
        env["LARRAK_THERMO_RESULTS"] = results_path
        env["LARRAK_SURROGATE_EPOCHS"] = str(surrogate_epochs)
        env["LARRAK_SURROGATE_R2_TARGET"] = str(surrogate_r2_target)
        env["LARRAK_VALIDATION_ERROR_MARGIN"] = str(validation_error_margin)

        validation_passed = True

        if train_surrogate:
            print(f"[Surrogate] Training (epochs={surrogate_epochs})...")
            ret_t = subprocess.call([sys.executable, "scripts/phase5/train_surrogate.py"], env=env)
            if ret_t != 0:
                print("[Surrogate] Training failed.")
                _log_checkpoint(name="train_failed", expected=0, observed=ret_t, passed=False)
                sys.exit(ret_t if isinstance(ret_t, int) and ret_t != 0 else 1)
            else:
                _log_checkpoint(
                    name="train_complete", expected=surrogate_epochs, observed=ret_t, passed=True
                )

        if validate_surrogate:
            print(f"[Surrogate] Validating (R2 target >= {surrogate_r2_target:.2f})...")
            ret_v = subprocess.call(
                [sys.executable, "scripts/phase5/validate_surrogate.py"], env=env
            )
            if ret_v != 0:
                print("[Surrogate] Validation did not meet target; continuing loop.")
                _log_checkpoint(
                    name="validate_failed",
                    expected=surrogate_r2_target,
                    observed=ret_v,
                    passed=False,
                )
                validation_passed = False
            else:
                _log_checkpoint(
                    name="validate_passed",
                    expected=surrogate_r2_target,
                    observed=ret_v,
                    passed=True,
                )
                validation_passed = True
            if FileRole:
                _register_artifact(
                    "surrogate/model_artifacts/surrogate_model.pth",
                    FileRole.MODEL,
                    {"loop": loop_count + 1},
                )
                _register_artifact(
                    "surrogate/model_artifacts/scaler_params.json",
                    FileRole.MODEL,
                    {"loop": loop_count + 1},
                )

        # Use the trained surrogate to estimate work at the checkpoint
        try:
            import torch

            from truthmaker.surrogates.models.model import EngineSurrogateModel

            model_path = "surrogate/model_artifacts/surrogate_model.pth"
            scaler_path = "surrogate/model_artifacts/scaler_params.json"
            if not (os.path.exists(model_path) and os.path.exists(scaler_path)):
                print("[Surrogate] Missing model/scaler artifacts; cannot estimate checkpoint.")
                sys.exit(1)

            with open(scaler_path, "r") as f:
                s = json.load(f)
            min_in = np.array(s["min_in"], dtype=np.float32)
            max_in = np.array(s["max_in"], dtype=np.float32)
            min_out = np.array(s["min_out"], dtype=np.float32)
            max_out = np.array(s["max_out"], dtype=np.float32)

            x_in = np.array(
                [top_power["rpm"], top_power["p_int_bar"], top_power["fuel_mass_mg"]],
                dtype=np.float32,
            )
            x_norm = (x_in - min_in) / (max_in - min_in + 1e-6)

            model = EngineSurrogateModel()
            model.load_state_dict(torch.load(model_path, map_location="cpu"))
            model.eval()
            with torch.no_grad():
                y_norm = model(torch.tensor(x_norm).float().unsqueeze(0)).numpy().squeeze(0)

            y = y_norm * (max_out - min_out + 1e-6) + min_out
            interp_work = float(y[2])

        except Exception as e:
            print(f"[Surrogate] Failed to estimate checkpoint: {e}")
            sys.exit(1)

        nlp_val = float(top_power["abs_work_net_j"])
        error_pct = abs(interp_work - nlp_val) / max(abs(interp_work), 1e-9) * 100.0
        print(f"    NLP Work: {nlp_val:.2f} J")
        print(f"    Surrogate Work: {interp_work:.2f} J")
        print(f"    Error: {error_pct:.2f}% (Threshold: {CONFIG.error_margin_percent:.1f}%)")
        within_error = error_pct <= CONFIG.error_margin_percent
        _log_checkpoint(
            name="error_eval",
            expected=CONFIG.error_margin_percent,
            observed=error_pct,
            passed=within_error,
        )
        _log_checkpoint(
            name="convergence_gate",
            expected={"error_margin": CONFIG.error_margin_percent, "validation_required": True},
            observed={"error_pct": error_pct, "validation_passed": validation_passed},
            passed=within_error and validation_passed,
        )

        if within_error and validation_passed:
            print("    [✓] Error within margin and validation passed. Convergence Reached.")
            converged = True
        else:
            if not within_error:
                print("    [!] Error exceeds margin. Re-running DOE/Surrogate loop.")
            elif not validation_passed:
                print("    [!] Validation failed; re-running loop despite low error.")
            loop_count += 1

    if converged:
        print("\n--- Thermo Calibration Complete: Validated & Converged ---")
        print("Ready for Gear Kinematics input.")
        sys.exit(0)
    else:
        print("\n--- Thermo Calibration Warning: Loop Limit Reached without Convergence ---")
        sys.exit(2)


if __name__ == "__main__":
    run_thermo_calibration()
