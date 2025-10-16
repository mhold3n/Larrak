# MA27 default and MA57-readiness analysis

- Default Ipopt linear solver is now `ma27` across the project. HSL path and `ipopt.opt` are centralized in `campro.constants`.
- An optional analysis can capture Ipopt logs and compute a heuristic grade indicating whether `ma57` may improve robustness/performance.

## Enable analysis (programmatic)
- Set `IPOPTOptions.enable_analysis = True` to route Ipopt logs to `logs/ipopt/<timestamp>.log` and enable timing statistics.
- After each solve, parse logs and stats using `campro.optimization.solver_analysis.analyze_ipopt_run(stats, output_file)`.

## CLI quick check
```bash
python scripts/analyze_ma27_vs_ma57.py
```
- Prints JSON with:
  - `grade`: low | medium | high
  - `reasons`: key indicators
  - `suggested_action`: next step
  - `stats`: selected Ipopt metrics and file path

## Heuristics (when MA57 might help)
- Linear solver dominates runtime (>50% of IPOPT time)
- Many refactorizations; inertia corrections; restoration phases
- Small pivot / near-singularity warnings
- High iteration counts or persistent infeasibility

## Paths
- `HSLLIB_PATH` and `IPOPT_OPT_PATH` in `campro.constants`.
- Update these per local environment.
