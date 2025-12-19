# Phase 4: Simulation Framework Backlog

Derived from [phase4_strategy_UPDATED.md](./phase4_strategy_UPDATED.md).

## Sprint 1: Foundation & IO Contracts
- [x] **1.1 Define IO Schema**
    - [x] Create `Simulations/common/io_schema.py` using Pydantic models.
        - [x] Define `SimulationInput` (Geometry, BCs, Materials, Ops).
        - [x] Define `SimulationOutput` (Scalars, Signals, FitMetrics).
    - [x] Implement `Simulations/common/validate_io.py` script.
    - [x] Create unit tests for schema validation (`tests/simulations/test_io.py`).
    - [x] Define `Materials` schema for `Simulations/common/materials.yaml`.
- [x] **1.2 Exporter Implementation**
    - [x] Create `thermo/export_candidate.py`.
    - [x] Implement function to serializes a `thermo` candidate object to `SimulationInput` JSON.
- [x] **1.3 Calibration Database Setup**
    - [x] Create `Simulations/_runs/` directory logic (automatic timestamp/hashing).
    - [x] Implement `Simulations/common/logger.py` to write `inputs.json`, `outputs.json`, `meta.json` automatically on run completion.

## Sprint 2: Thermal Feasibility (The First Gate)
- [x] **2.1 Thermal Solver**
    - [x] Implement `Simulations/thermal/steady_transient.py`.
    - [x] Define simple 1D/2D heat resistor network or FEA stub.
    - [x] Output `T_crown_max`, `T_ring_max`.
- [x] **2.2 Optimization Integration**
    - [x] Implement `thermo/constraints.py` hook to load thermal map.
    - [x] Wire constraint into `thermo/nlp.py`.
    - [x] Expose `T_crown_max` in diagnostic outputs.
    - [ ] Add hard constraint `T_crown_max < 600K` to the NLP.

## Sprint 3: Structural & Friction
- [x] **3.1 Friction Solver**
    - [x] Implement `Simulations/structural/friction_contact.py`.
    - [x] Implement Reynolds equation solver (or approximate) for ring lubrication.
    - [x] Output `FMEP_cycle` and `friction_force_theta`.
- [x] **3.2 Calibration Map Fitting**
    - [x] Create `Simulations/common/surrogates.py` (Shared RBF/Polynomial fitting utilities).
    - [x] Create `Simulations/structural/fit_friction_map.py`.
    - [x] Fit `A`, `B` coefficients (Chen-Flynn) from simulation results.
    - [x] Save to `thermo/calibration/friction_map.v1.json`.

## Sprint 4: Combustion & CFD
- [x] **4.1 Reduced Prechamber Model**
    - [x] Implement `Simulations/combustion/prechamber_reduced.py`.
    - [x] Model mass exchange and jet turbulence intensity.
- [x] **4.2 Wiebe Parameter Fitting**
    - [x] Create `Simulations/combustion/fit_wiebe_map.py`.
    - [x] Map geometrical parameters (nozzle diameter, volume) to Wiebe `a` and `m`.

## Sprint 5: The Loop (SBO Logic)
- [x] **5.1 Trust Region Logic**
    - [x] Implement `thermo/calibration/registry.py` to manage active maps.
    - [x] Implement "Bounded Update" logic (max change per iteration).
- [x] **5.2 Automated Orchestrator**
    - [x] Create `scripts/run_phase4_cycle.py`.
    - [x] Implement "Diversity Sampling" (Select Top N + Distance-based exploration).
    - [x] Logic: Optimize -> Select Batch -> Simulate -> Fit -> Update.
