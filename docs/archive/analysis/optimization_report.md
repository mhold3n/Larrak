# Optimization Pipeline Report

## Run Configuration
- **K**: 5
- **C**: 3
- **Scaling Method**: Strict Betts-style (clamped [1e-3, 1e3]) + Global Objective Scaling

## Key Metrics

### 1. Scaling Quality
- **Condition Number**: `3.025e+10`
- **Overscaled Entries**: `3` (Entries < 1e-10 in scaled Jacobian)
  > [!WARNING]
  > Presence of overscaled entries indicates some constraints or variables are scaled too aggressively small.

### 2. KKT Conditions & Feasibility
- **Primal Infeasibility (inf_pr)**: `2.285e-08`
- **Dual Infeasibility (inf_du)**: `inf`
- **Complementarity (mu)**: `nan`

### 3. Solver Status
- **Status**: `Maximum_Iterations_Exceeded`

### 4. Diagnostics
- **NaN/Inf Checks**: `None`

## Raw Output Snippets
```text
[INFO][FREE-PISTON] Scaled Jacobian statistics: min=3.306e-11, p25=2.542e-03, p50=6.250e-02, p75=7.010e-01, p95=9.988e-01, p99=1.000e+00, max=1.000e+00, mean=3.664e-03, condition_number=3.025e+10, quality_score=0.654, overscaled_entries=3 (0.3%), underscaled_entries=92 (9.3%)
[INFO][FREE-PISTON] Scaled Jacobian statistics by constraint type
[INFO][FREE-PISTON] Final residuals: inf_pr=2.285e-08 inf_du=inf mu=nan
```
