# Phase 4: High-Fidelity Simulation & Implementation Strategy (Updated)

## Overview
Phase 4 bridges **fast optimization (Phase 1–3)** and **truth-model physics** (2D/3D FEA/CFD). The optimizer (`thermo/`) remains the global search engine; Phase 4 provides **validated constraints** and **calibrated, geometry-aware correction models** that the optimizer can query cheaply.

**Core idea:** Phase 4 should not “overwrite one coefficient after one run.” It should produce **calibration maps / surrogates** with **bounded updates** and a **declared validity region**.

---

## Relationship with Optimization System

| Feature | **Optimization System** (`thermo/`) | **Simulation System** (`Simulations/`) |
| :--- | :--- | :--- |
| Physics | 0D / reduced-order (lumped mass, Wiebe combustion, reduced heat transfer, empirical friction) | 2D/3D (contact mechanics, FEA heat conduction, CFD / reacting flow as needed) |
| Speed | milliseconds–seconds per candidate (enables global search) | minutes–hours per candidate (enables validation + calibration) |
| Primary role | propose candidates, enforce *approximate* constraints, optimize objectives | establish feasibility limits, provide calibrated coefficients and constraint margins |
| Output to optimizer | objective + constraints evaluated cheaply | **surrogates / calibrated functions** + uncertainty + validity region |

---

## Phase 4 ↔ Optimizer Contract (Mandatory)

### Inputs exported from optimization → simulations
All Phase 4 runs must accept a single, versioned **IO bundle**:
- **Geometry**: key dimensions + derived shapes (ring/planet profiles if relevant)
- **Materials**: references to `Simulations/common/materials.yaml` (single source of truth)
- **Cycle-resolved boundary conditions** (as functions of crank/phase angle):
  - pressure `p(θ)`
  - temperature `T_gas(θ)` or heat-release `Q̇(θ)`
  - piston speed / sliding speed `U(θ)`
  - contact loading proxies `N(θ)` (or data needed to compute them)
- **Operating point**: RPM, AFR/λ, load, coolant/oil temps, etc.

### Outputs returned from simulations → optimization
Each run returns:
- **Scalars (gating constraints):**
  - `T_crown_max`, `T_ring_max`, etc.
  - peak von Mises / safety margin
- **Cycle-resolved signals (for fitting):**
  - friction force or FMEP contribution vs θ
  - heat flux / HTC equivalents vs θ (as applicable)
- **Fitted calibration products:**
  - friction correction model parameters (map coefficients)
  - heat-transfer correction parameters (map coefficients)
  - Wiebe `(a, m)` map or burn-rate surrogate outputs
- **Uncertainty + validity region:**
  - fit residuals / RMSE
  - bounds of input space where the calibration is trusted

**Implementation requirement:** define a schema (JSON/YAML) and validate it.
- `Simulations/common/io_schema.json`
- `Simulations/common/validate_io.py`

---

## Multi-Fidelity Coupling Strategy (How Phase 4 informs optimization)

### 1) Batch-based “calibrate → re-optimize”
1. Optimize with current reduced-order model set (Phase 1–3).
2. Select a small **batch** of candidates:
   - top N by objective
   - + “diversity” samples (spread across design space) to prevent overfit
3. Run Phase 4 simulations on the batch.
4. Fit/update **calibration maps** and **constraint margins**.
5. Re-run optimization using updated maps **within their validity region**.

### 2) Trust-region and bounded updates (prevents chasing noise)
Calibration updates must be **bounded**:
- update coefficients in log-space with a max step (e.g., ≤ 10–20% per iteration)
- only “promote” updates after repeated confirmation (e.g., 2–3 consistent runs)
- store every calibration as a versioned artifact; never hard-code into source

### 3) Constraints before objectives
Use Phase 4 primarily to establish **hard feasibility gates**:
- thermal limits (e.g., crown temperature)
- stress/contact limits
Only spend expensive combustion CFD on candidates that clear feasibility.

---

## Physics Modules (Simulation Deliverables)

### A) Structural / Friction (`Simulations/structural/`)
**Goal:** replace empirical friction with physics-backed correction models.

**Minimum outputs:**
- cycle-averaged friction contribution (e.g., FMEP term)
- friction vs θ (or vs sliding speed/contact pressure)
- optional sensitivities (finite-diff) for key knobs (ring tension, clearance, lubrication temperature)

**Deliverable to optimizer:** a **friction correction surrogate**
- Example: `FMEP_corr = f(T_oil, U(θ), p(θ), N(θ), roughness, ring_tension, …)`
- Store as: `thermo/calibration/friction_map.<version>.json`

### B) Thermal (`Simulations/thermal/`)
**Goal:** enforce thermal feasibility and calibrate heat-transfer closures.

**Inputs must be cycle-resolved** (not just total heat):
- `Q̇(θ)` or `q''(θ)` boundary condition
- coolant/oil boundary conditions

**Minimum outputs:**
- `T_crown_max`, `T_ring_max`, `T_liner_max`
- optional: effective HTC correction factors for reduced model (if using a reduced HTC law)

**Deliverable to optimizer:** constraint function and/or correction model
- `g_thermal(x) = T_crown_max(x) - T_limit ≤ 0`
- plus `HTC_corr = h(RPM, load, geometry, …)` if needed

### C) Combustion (`Simulations/combustion/`)
**Goal:** calibrate burn-rate parameters (Wiebe or equivalent), especially for prechamber ignition.

**Recommended escalation ladder:**
1. Reduced model (fast): 0D/1D prechamber + jet entrainment approximations → outputs `(a, m)` or burn-rate surrogate
2. Mid-fidelity: 2D axisymmetric where defensible
3. Full CFD only when reduced/mid fail validation

**Deliverable to optimizer:** `(a, m)` maps + uncertainty
- `wiebe_a = g1(λ, RPM, load, prechamber_geo, …)`
- `wiebe_m = g2(λ, RPM, load, prechamber_geo, …)`

---

## Calibration Database (Required for reproducibility)

Every simulation run writes a record:
- inputs hash (geometry + BCs + materials version + solver settings)
- raw outputs (signals + scalars)
- fitted surrogate parameters + fit metrics (RMSE, residuals)
- solver metadata (mesh size, timestep, convergence)

Suggested structure:
- `Simulations/_runs/<timestamp>_<hash>/inputs.json`
- `Simulations/_runs/<timestamp>_<hash>/outputs.json`
- `Simulations/_runs/<timestamp>_<hash>/fit.json`
- `Simulations/_runs/<timestamp>_<hash>/meta.json`

---

## Directory Structure Strategy
Progressive enhancement with strict shared truth:

- `Simulations/common/`
  - `materials.yaml` (single source of truth)
  - `io_schema.json` + `validate_io.py`
  - `mesh.py` (simple 2D mesh utilities)
  - `surrogates.py` (fit/interp utilities used by all domains)
- `Simulations/structural/`
  - `friction_contact.py` (core solver)
  - `fit_friction_map.py` (writes calibration artifact)
- `Simulations/thermal/`
  - `steady_transient.py`
  - `fit_thermal_map.py`
- `Simulations/combustion/`
  - `prechamber_reduced.py`
  - `fit_wiebe_map.py`
- `thermo/calibration/`
  - `friction_map.<version>.json`
  - `thermal_map.<version>.json`
  - `wiebe_map.<version>.json`
  - `calibration_registry.yaml` (active calibration + validity bounds)

---

## Next Steps (Implementation Order)

1. **Define the IO contract**
   - Implement `Simulations/common/io_schema.json`
   - Implement `Simulations/common/validate_io.py`
   - Add a minimal exporter in `thermo/` that writes `inputs.json` for a candidate.

2. **Stand up the calibration database**
   - Create `Simulations/_runs/` logging + hashing.
   - Ensure every solver writes `outputs.json` + `meta.json`.

3. **Thermal feasibility first**
   - Implement a minimal thermal solver and `T_crown_max` output.
   - Add a hard constraint hook in optimization reading `thermal_map`.

4. **Structural friction next**
   - Implement `Simulations/structural/friction_contact.py`.
   - Fit `friction_map` and integrate into `thermo/physics.py` via calibration override (do not hard-code constants).

5. **Combustion last (reduced model first)**
   - Implement `prechamber_reduced.py` producing `(a, m)` with fit metrics.
   - Only escalate to heavier CFD when reduced model mismatch is confirmed.

6. **Trust-region update logic**
   - Implement bounded updates + “promotion” rules in `thermo/calibration/registry.py` (or equivalent).
   - Require repeated confirmation runs for new calibrations.

---

## Definition of Done (Phase 4 ready-to-inform optimization)
- Optimizer can run with **calibration artifacts** (maps) without touching solver code.
- Every Phase 4 run is reproducible (inputs hash + materials version + solver settings).
- Calibration updates are bounded and validity-limited.
- Thermal constraints are enforced as feasibility gates before expensive CFD.
