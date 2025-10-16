from __future__ import annotations

import argparse
import json
from pathlib import Path

from campro.logging import get_logger
from CamPro_LitvinPlanetary import (
    GeometrySearchConfig,
    OptimizationOrder,
    PlanetSynthesisConfig,
    RadialSlotMotion,
    optimize_geometry,
    synthesize_planet_from_motion,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Synthesize Litvin planet from motion law")
    parser.add_argument("--ring-teeth", type=int, default=60)
    parser.add_argument("--planet-teeth", type=int, default=30)
    parser.add_argument("--pressure-angle", type=float, default=20.0)
    parser.add_argument("--addendum-factor", type=float, default=1.0)
    parser.add_argument("--R0", type=float, default=30.0)
    parser.add_argument("--samples", type=int, default=360)
    parser.add_argument("--order", type=int, default=0)
    parser.add_argument("--ring-candidates", type=str, default="60")
    parser.add_argument("--planet-candidates", type=str, default="30")
    parser.add_argument("--out", type=Path, default=Path("litvin_planet_profile.json"))
    args = parser.parse_args()

    motion = RadialSlotMotion(
        center_offset_fn=lambda th: 0.0,
        planet_angle_fn=lambda th: 2.0 * th,
    )

    cfg = PlanetSynthesisConfig(
        ring_teeth=args.ring_teeth,
        planet_teeth=args.planet_teeth,
        pressure_angle_deg=args.pressure_angle,
        addendum_factor=args.addendum_factor,
        base_center_radius=args.R0,
        samples_per_rev=args.samples,
        motion=motion,
    )

    if args.order == OptimizationOrder.ORDER0_EVALUATE:
        profile = synthesize_planet_from_motion(cfg)
        args.out.write_text(json.dumps({"points": profile.points}))
        log = get_logger(__name__)
        log.info("wrote %s", args.out)
        return

    ring_c = [int(s) for s in args.ring_candidates.split(",") if s]
    planet_c = [int(s) for s in args.planet_candidates.split(",") if s]
    gcfg = GeometrySearchConfig(
        ring_teeth_candidates=ring_c,
        planet_teeth_candidates=planet_c,
        pressure_angle_deg_bounds=(args.pressure_angle - 2.0, args.pressure_angle + 2.0),
        addendum_factor_bounds=(args.addendum_factor - 0.1, args.addendum_factor + 0.1),
        base_center_radius=args.R0,
        samples_per_rev=args.samples,
        motion=motion,
    )
    res = optimize_geometry(gcfg, order=args.order)
    out = {
        "feasible": res.feasible,
        "objective": res.objective_value,
        "best_config": None if res.best_config is None else {
            "ring_teeth": res.best_config.ring_teeth,
            "planet_teeth": res.best_config.planet_teeth,
            "pressure_angle_deg": res.best_config.pressure_angle_deg,
            "addendum_factor": res.best_config.addendum_factor,
        },
    }
    args.out.write_text(json.dumps(out))
    log = get_logger(__name__)
    log.info("optimization result written to %s", args.out)


if __name__ == "__main__":
    main()


