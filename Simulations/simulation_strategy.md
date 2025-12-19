# Phase 4: High-Fidelity Simulation & Implementation Strategy

## Overview
Phase 4 represents the transition from **Lumped Parameter Optimization** (Phase 1-3) to **Spatially Resolved Physical Simulation**. 

While the Optimization engine (`thermo/nlp.py`) is designed to explore millions of design combinations using simplified physics (0D ODEs), the Simulation Framework (`Simulations/`) is designed to validate specific designs with high-fidelity physics (2D/3D FEA & CFD).

## Relationship with Optimization System
The two systems exist as separate but symbiotic entities:

| Feature | **Optimization System** (`thermo/`) | **Simulation System** (`Simulations/`) |
| :--- | :--- | :--- |
| **Physics** | 0D (Lumped Mass, Wiebe Combustion, Chen-Flynn Friction) | 2D/3D (Navier-Stokes, Contact Mechanics, Finite Element) |
| **Speed** | Milliseconds per cycle (enables global search) | Minutes/Hours per cycle (enables validation) |
| **Goal** | Find the *approximate* optimal geometry. | Determine *exact* performance limits and coefficients. |

### The "Calibration Loop"
The critical link is that **Simulations inform Optimization**.
1.  **Optimize**: `ma86` finds a candidate engine geometry (e.g., Bore=100mm, CR=15).
2.  **Simulate**: Use `Simulations/structural/friction.py` to run a 2D Contact FEA on this geometry.
3.  **Calibrate**: The FEA verifies that the friction is actually `1.8 bar` (vs the predicted `1.5 bar`).
4.  **Feedback**: Update the `A` coefficient in `thermo/physics.py` and re-run optimization.

## Implementation Plan: Using the Scaffolding

The scaffolding in `Simulations/` provides the standardized interfaces to build these detailed models without breaking the rest of the codebase.

### 1. Structural FEA (`Simulations/structural/`)
*   **Goal**: Replace empirical friction formulas with physics.
*   **Usage**: 
    *   Inherit from `BaseSimulation`.
    *   Use `MaterialLibrary` to get temperature-dependent Young's Modulus for the liner.
    *   **Implementation**: In `step()`, solve the system $K u = F$ to determine piston ring contact pressure and resulting shear stress.

### 2. Thermal FEA (`Simulations/thermal/`)
*   **Goal**: Determine material temperature limits (melting points).
*   **Usage**:
    *   Map the "Heat Release" ($Q$) from the Optimization results as a generic boundary condition.
    *   **Implementation**: Solve the Heat Diffusion equation to ensure the Piston Crown temperature $< 600K$.
    *   **Feedback**: If temp is too high, the Optimization must lower $Q_{total}$ or Compression Ratio.

### 3. Combustion CFD (`Simulations/combustion/`)
*   **Goal**: Realistic burn rates.
*   **Usage**:
    *   Simulate the "Prechamber Jet" using a 1D or 2D CFD model.
    *   **Implementation**: track the Flame Front surface area evolution.
    *   **Feedback**: Provide improved `a` and `m` coefficients for the Wiebe functions used in Optimization.

## Directory Structure Strategy
The folder structure is designed for "progressive enhancement":
*   **`common/`**: Holds the "Truth" (Materials, Mesh definitions). ensures different physics solvers agree on what "Steel" is.
*   **`domain/`**: Each domain (`fluid`, `structural`) can advance independently. We can have a complex CFD model but a simple Thermal model, or vice-versa.

## Next Steps
1.  **Meshing**: Implement `Simulations/common/mesh.py` to generate simple 2D grids.
2.  **Solver Integration**: Choose a "Engine" (e.g., `scipy.sparse` for simple FEA, or bindings to `FEniCS`).
3.  **Run**: Execute `Simulations/structural/friction.py` as a standalone script to generate the first "High-Fidelity" friction curve.
