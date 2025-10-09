from __future__ import annotations

from pathlib import Path

from campro.freepiston.io.load import load_cfg
from campro.freepiston.opt.driver import solve_cycle
from campro.freepiston.opt.solution import Solution


def main() -> None:
    cfg = load_cfg(Path("cfg/defaults.yaml"))
    # Optional run directory for checkpoints and outputs
    run_dir = Path("runs/op_cycle_001")
    cfg["run_dir"] = str(run_dir)
    sol = solve_cycle(cfg)
    if isinstance(sol, Solution):
        sol.save(run_dir)


if __name__ == "__main__":
    main()



