# Optimization Pipeline Report

## Run Configuration
- **K**: 5
- **C**: 3
- **Scaling Method**: Strict Betts-style (clamped [1e-3, 1e3]) + Global Objective Scaling

## Key Metrics

### 1. Scaling Quality
- **Condition Number**: `2.206e+14`
- **Overscaled Entries**: `24` (Entries < 1e-10 in scaled Jacobian)
  > [!WARNING]
  > Presence of overscaled entries indicates some constraints or variables are scaled too aggressively small.

### 2. KKT Conditions & Feasibility
- **Primal Infeasibility (inf_pr)**: `1.872e-15`
- **Dual Infeasibility (inf_du)**: `inf`
- **Complementarity (mu)**: `nan`

### 3. Solver Status
- **Status**: `Solved_To_Acceptable_Level`

### 4. Diagnostics
- **NaN/Inf Checks**: `None`

## Raw Output Snippets
```text
2025-11-27 23:00:46,098 [INFO] [INFO][FREE-PISTON] Scaled Jacobian statistics: min=9.915e-15, p25=2.077e-02, p50=1.882e-01, p75=1.000e+00, p95=1.645e+00, p99=2.188e+00, max=2.188e+00, mean=8.836e-03, condition_number=2.206e+14, quality_score=0.588, overscaled_entries=24 (3.9%), underscaled_entries=64 (10.4%)
2025-11-27 23:00:46,099 [INFO] [INFO][FREE-PISTON] Scaled Jacobian statistics by constraint type
2025-11-27 23:00:50,433 [INFO] [INFO][FREE-PISTON] Final residuals: inf_pr=1.872e-15 inf_du=inf mu=nan
```
