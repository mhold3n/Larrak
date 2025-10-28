from __future__ import annotations

from time import perf_counter

from campro.freepiston.io.load import load_cfg
from campro.freepiston.opt.driver import solve_cycle


def main() -> None:
    P = load_cfg("cfg/defaults.yaml")
    t0 = perf_counter()
    _ = solve_cycle(P)
    _ = perf_counter() - t0


if __name__ == "__main__":
    main()
