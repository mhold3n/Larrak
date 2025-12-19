# Larrak: Breathing Gear Engine Design Framework
## User Manual & Workflow Guide

Larrak is a specialized engineering framework for designing "Breathing Gear" engines—internal combustion engines where the piston motion is governed by a non-circular ring/planet gear set with a variable center distance. This allows for completely custom, non-sinusoidal piston trajectories optimized for thermodynamic efficiency.

This manual outlines the 5-Phase Workflow to design, optimize, and validate these engines.

---

## 🚀 Quick Start

### 1. Installation
The project relies on a Conda environment with CasADi and IPOPT.

```bash
# 1. Create Environment
conda env create -f environment.yml

# 2. Activate
conda activate larrak

# 3. Verify Setup
python scripts/check_environment.py
```

### 2. The Dashboard
All key results are centralized in the `dashboard/` directory.
- **`dashboard/breathing_gear_sets.html`**: Interactive visualization of optimized gear profiles.
- **`dashboard/phase1/`**: Results from large-scale thermodynamic sweeps.

---

## 📚 The 5-Phase Workflow

The design process moves from abstract thermodynamics to concrete mechanical geometry.

### Phase 1: Physical DOE (Thermodynamic Scaling)
**Goal:** Find the optimal thermodynamic operating points (RPM, Boost, Fuel) assuming an "Ideal" custom piston motion.
**Script:** `tests/goldens/phase1/generate_doe.py`

*   **How it works:**  Runs a 0D Thermodynamic Optimization (NLP) to find the *best possible* piston trajectory for given inputs, ignoring mechanical constraints initially.
*   **How to Run:**
    ```bash
    python tests/goldens/phase1/generate_doe.py
    ```
*   **Output:** `dashboard/phase1/phase1_physical_results.csv` (13,000+ points).
*   **Tuning Inputs:**
    Open `tests/goldens/phase1/generate_doe.py` and modify `main()`:
    ```python
    # Example: Change RPM range
    rpm_levels = np.linspace(1000, 8000, 36) 
    # Example: Switch from Full Factorial to Smoke Test
    doe_list = doe_list[:10] 
    ```

### Phase 2: Kinematic Interpreter
**Goal:** Translate the abstract "Ideal Piston Motion" from Phase 1 into a specific "Gear Ratio Function".
**Key Concept:** The "Breathing Gear" mechanism modulates the gear ratio $i(\theta)$ and center distance $C(\theta)$ to achieve the target lift.
*   *Note: This phase is often integrated directly into Phase 3's optimization loop.*

### Phase 3: Conjugate Gear Design
**Goal:** Generate physical Ring and Planet gear profiles that produce the target motion *without* interference, slip, or gaps.
**Script:** `scripts/run_conjugate_optimization.py`

*   **How it works:** Uses a specialized NLP to optimize the pitch curves of the Ring and Planet simultaneously. Enforces:
    1.  **Conjugacy:** The gears mesh perfectly.
    2.  **2:1 Topology:** Standard engine cycle map.
    3.  **No Interference:** Profiles do not self-intersect.
    4.  **Symmetry:** Ensures balanced forces.
*   **How to Run:**
    ```bash
    python scripts/run_conjugate_optimization.py
    ```
*   **Output:** 
    - `dashboard/breathing_gear_sets.html`: The definitive visual check. Look for "Pitch Curves" that are smooth and closed.

### Phase 4: Valve Strategy
**Goal:** Optimize intake/exhaust valve timings for the new custom piston motion.
**Script:** `scripts/run_phase4_cycle.py`

*   **How it works:** Re-runs the thermodynamic simulation with the *actual* kinematic constraints derived from Phase 3, optimizing valve opening/closing angles.

### Phase 5: Surrogate Scaling (The "Mega-Run")
**Goal:** Run massive design sweeps (10k+ points) in minutes instead of days.
**Method:** "Calibrate-Then-Optimize".
1.  **Calibrate:** Run high-fidelity simulations on a small sample (e.g., 25 points).
    - Script: `scripts/run_phase5_campaign.py`
    - Output: `thermo/calibration/friction_map.v1.json`, `combustion_map.v1.json`.
2.  **Optimize:** Use these "Surrogate Maps" to instantly predict engine performance across the full design space.
    - Script: Uses `tests/goldens/phase1/generate_doe.py` (which automatically loads the maps).

---

## 🛠️ Tuning & Modification Guide

### "I want to change the engine geometry (Bore/Stroke)"
Edit: `Simulations/common/io_schema.py` or the specific config dictionary in the running script (e.g., inside `generate_doe.py` -> `phase1_test` -> `GeometryConfig`).

### "I want to change the optimization target (e.g., Maximize Power instead of Efficiency)"
Edit: `thermo/nlp.py`.
Likely in `_calculate_objectives`. You will see weights for `work`, `smoothness`, etc.
```python
# Example: Increase weight on Work
J = -10.0 * work_nondim + 1e-4 * jerk_penalty
```

### "I want to change the number of samples in the DOE"
Edit: `tests/goldens/phase1/generate_doe.py`.
Modify the `np.linspace` calls in `main()`:
```python
# Finer grid
rpm_levels = np.linspace(1000, 6000, 51) # 50 steps
```

### "The gears look weird / self-intersect"
This typically happens if the target motion is too aggressive (infinite jerk).
1.  Go to `thermo/nlp.py` and increase `jerk` penalty weight.
2.  Re-run Phase 1 to get a smoother target.
3.  Re-run Phase 3 (`run_conjugate_optimization.py`) to fit the new curve.

---

## 📂 File Stack Overview

```
Larrak/
├── dashboard/                  # <--- START HERE. All Visualizations.
│   ├── phase1/                 # Large scale plots
│   └── breathing_gear_sets.html
├── scripts/
│   ├── run_conjugate_optimization.py  # Phase 3 (Gear Gen)
│   ├── run_phase4_cycle.py            # Phase 4 (Valves)
│   └── run_phase5_campaign.py         # Phase 5 (Surrogate Training)
├── tests/goldens/phase1/
│   └── generate_doe.py         # The Main Execution Script for Scaling
├── thermo/
│   ├── nlp.py                  # The Core Physics Engine (CasADi)
│   └── calibration/            # Where Phase 5 saves its AI models
└── Simulations/
    └── common/                 # Shared schemas (Geometry, OperatingPoint)
```