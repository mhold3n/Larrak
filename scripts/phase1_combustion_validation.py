#!/usr/bin/env python3
"""
Quick validation harness for Phase-1 combustion integration.

This script exercises the upgraded SimpleCycleAdapter with and without
combustion inputs, reports pressure-ratio statistics, and highlights
how a workload perturbation increases the guardrail loss. Run with:

    python scripts/phase1_combustion_validation.py

It prints summary diagnostics that should be mirrored inside the GUI
when the same inputs are entered, providing a lightweight regression
check for CA markers and pressure-ratio invariance.
"""

from __future__ import annotations

import math
import pathlib
import sys
from typing import Any

# Add project root to path for imports when run directly
_script_dir = pathlib.Path(__file__).resolve().parent
_project_root = _script_dir.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import numpy as np

from campro.physics.simple_cycle_adapter import (
    SimpleCycleAdapter,
    CycleGeometry,
    CycleThermo,
    WiebeParams,
)
from campro.physics.pr_template import compute_pr_template


def pressure_ratio_summary(
    p_trace: np.ndarray, 
    p_bounce: np.ndarray, 
    p_env_kpa: float,
    p_load_kpa: float = 0.0,
    p_cc_kpa: float = 0.0,
) -> tuple[dict[str, float], np.ndarray]:
    """Compute pressure-ratio trace Π(θ) with workload-aligned denominator and scalar statistics."""
    denom = p_load_kpa + p_cc_kpa + p_env_kpa + p_bounce
    pi_trace = p_trace / np.maximum(denom, 1e-6)
    stats = {
        "pi_mean": float(np.mean(pi_trace)),
        "pi_peak": float(np.max(pi_trace)),
        "pi_min": float(np.min(pi_trace)),
        "pi_std": float(np.std(pi_trace)),
    }
    return stats, pi_trace


def compute_workload_aligned_p_load(workload_j: float, area_mm2: float, stroke_mm: float) -> float:
    """Compute workload-aligned load pressure: p_load = w / (A·stroke)."""
    area_m2 = area_mm2 * 1e-6
    stroke_m = stroke_mm * 1e-3
    if workload_j > 0.0 and area_m2 > 0.0:
        p_load_pa = workload_j / max(area_m2 * stroke_m, 1e-12)
        return p_load_pa / 1000.0  # Convert to kPa
    return 0.0


def evaluate_cycle(
    adapter: SimpleCycleAdapter,
    theta: np.ndarray,
    position_mm: np.ndarray,
    fuel_multiplier: float,
    c_load: float,
    geom: CycleGeometry,
    thermo: CycleThermo,
    combustion: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Helper to evaluate the adapter with consistent gradient handling."""
    v_mm_per_theta = np.gradient(position_mm, theta)
    return adapter.evaluate(
        theta,
        position_mm,
        v_mm_per_theta,
        fuel_multiplier,
        c_load,
        geom,
        thermo,
        combustion=combustion,
        cycle_time_s=combustion.get("cycle_time_s") if combustion else None,
    )


def main() -> None:
    n_points = 360
    stroke_mm = 20.0
    cycle_time_s = 0.02

    theta = np.linspace(0.0, 2 * math.pi, n_points, endpoint=False)
    position = 0.5 * stroke_mm * (1.0 - np.cos(theta))  # Smooth single-stroke profile

    # Geometry and thermo representative of Phase 1 defaults
    bore_mm = 35.0
    area_mm2 = math.pi * (bore_mm / 2.0) ** 2
    clearance_volume_mm3 = 850.0
    geom = CycleGeometry(area_mm2=area_mm2, Vc_mm3=clearance_volume_mm3)
    thermo = CycleThermo(gamma_bounce=1.25, p_atm_kpa=101.325)

    adapter = SimpleCycleAdapter(
        wiebe=WiebeParams(a=5.0, m=2.0, start_deg=-5.0, duration_deg=25.0),
        alpha_fuel_to_base=1.0,
        beta_base=0.0,
    )

    # Baseline evaluation (legacy behaviour)
    legacy_out = evaluate_cycle(
        adapter,
        theta,
        position,
        fuel_multiplier=1.0,
        c_load=0.0,
        geom=geom,
        thermo=thermo,
        combustion=None,
    )
    legacy_stats, legacy_pi = pressure_ratio_summary(
        np.asarray(legacy_out.get("p_comb"), dtype=float),
        np.asarray(legacy_out.get("p_bounce"), dtype=float),
        thermo.p_atm_kpa,
    )

    # Combustion-enabled evaluation
    combustion_inputs = {
        "afr": 18.0,
        "fuel_mass": 5e-4,
        "cycle_time_s": cycle_time_s,
        "ignition_theta_deg": -5.0,
        "initial_temperature_K": 900.0,
        "initial_pressure_Pa": 101325.0,
    }
    combustion_out = evaluate_cycle(
        adapter,
        theta,
        position,
        fuel_multiplier=1.0,
        c_load=0.0,
        geom=geom,
        thermo=thermo,
        combustion=combustion_inputs,
    )
    # Get p_cyl with explicit None check (avoid ambiguous truth value with arrays)
    p_cyl = combustion_out.get("p_cyl")
    if p_cyl is None:
        p_cyl = combustion_out.get("p_comb")
    # Calculate p_load_kpa from actual cycle work (workload-aligned denominator)
    cycle_work_j = float(combustion_out.get("cycle_work_j", 0.0))
    p_load_kpa_baseline = (
        compute_workload_aligned_p_load(cycle_work_j, area_mm2, stroke_mm)
        if cycle_work_j > 0.0
        else 0.0
    )
    combustion_stats, combustion_pi = pressure_ratio_summary(
        np.asarray(p_cyl, dtype=float),
        np.asarray(combustion_out.get("p_bounce"), dtype=float),
        thermo.p_atm_kpa,
        p_load_kpa=p_load_kpa_baseline,
        p_cc_kpa=0.0,
    )

    ca_markers = combustion_out.get("ca_markers")
    if ca_markers is None:
        ca_markers = {}

    # Perturb workload by altering fuel scalar (Stage B guard proxy)
    perturbed_out = evaluate_cycle(
        adapter,
        theta,
        position,
        fuel_multiplier=0.9,
        c_load=0.0,
        geom=geom,
        thermo=thermo,
        combustion={**combustion_inputs, "fuel_mass": combustion_inputs["fuel_mass"] * 0.9},
    )
    # Get p_cyl with explicit None check
    p_cyl_pert = perturbed_out.get("p_cyl")
    if p_cyl_pert is None:
        p_cyl_pert = perturbed_out.get("p_comb")
    # Calculate p_load_kpa from perturbed cycle work
    cycle_work_pert_j = float(perturbed_out.get("cycle_work_j", 0.0))
    p_load_kpa_pert = (
        compute_workload_aligned_p_load(cycle_work_pert_j, area_mm2, stroke_mm)
        if cycle_work_pert_j > 0.0
        else 0.0
    )
    _, perturbed_pi = pressure_ratio_summary(
        np.asarray(p_cyl_pert, dtype=float),
        np.asarray(perturbed_out.get("p_bounce"), dtype=float),
        thermo.p_atm_kpa,
        p_load_kpa=p_load_kpa_pert,
        p_cc_kpa=0.0,
    )
    guard_loss = float(np.mean((perturbed_pi - combustion_pi) ** 2))

    print("=== Phase-1 Combustion Validation ===")
    print("Baseline (legacy) pressure ratio stats:")
    for key, value in legacy_stats.items():
        print(f"  {key:8s}: {value:.4f}")

    print("\nCombustion-aware pressure ratio stats:")
    for key, value in combustion_stats.items():
        print(f"  {key:8s}: {value:.4f}")

    if ca_markers:
        print("\nCA markers from combustion model:")
        for name in ("CA10", "CA50", "CA90", "CA100"):
            if name in ca_markers and ca_markers[name] is not None:
                print(f"  {name:5s}: {float(ca_markers[name]):6.2f} deg")
    else:
        print("\nCA markers unavailable (combustion inputs may be incomplete).")

    print(f"\nGuardrail loss for 10% fuel reduction: {guard_loss:.6f}")
    if guard_loss <= 0.0:
        raise SystemExit("Guardrail loss should be positive for workload perturbation.")

    # Test workload steps and Π invariance
    print("\n=== Workload Steps and Π Invariance Test ===")
    workload_steps = [50.0, 100.0, 150.0, 200.0]  # Joules
    pi_values = []
    ca50_values = []
    p_load_values = []
    
    for workload_j in workload_steps:
        p_load_kpa = compute_workload_aligned_p_load(workload_j, area_mm2, stroke_mm)
        p_load_values.append(p_load_kpa)
        
        # Scale fuel mass to approximate workload (rough proxy)
        fuel_scale = workload_j / 100.0  # Normalize to 100J baseline
        scaled_fuel = combustion_inputs["fuel_mass"] * fuel_scale
        
        workload_out = evaluate_cycle(
            adapter,
            theta,
            position,
            fuel_multiplier=fuel_scale,
            c_load=0.0,
            geom=geom,
            thermo=thermo,
            combustion={**combustion_inputs, "fuel_mass": scaled_fuel},
        )
        
        # Compute workload-aligned pressure ratio
        # Get p_cyl with explicit None check
        p_cyl_work = workload_out.get("p_cyl")
        if p_cyl_work is None:
            p_cyl_work = workload_out.get("p_comb")
        workload_pi_stats, workload_pi = pressure_ratio_summary(
            np.asarray(p_cyl_work, dtype=float),
            np.asarray(workload_out.get("p_bounce"), dtype=float),
            thermo.p_atm_kpa,
            p_load_kpa=p_load_kpa,
            p_cc_kpa=0.0,
        )
        pi_values.append(workload_pi_stats["pi_mean"])
        
        # Extract CA markers
        workload_ca = workload_out.get("ca_markers")
        if workload_ca is None:
            workload_ca = {}
        if "CA50" in workload_ca and workload_ca["CA50"] is not None:
            ca50_values.append(float(workload_ca["CA50"]))
        else:
            ca50_values.append(None)
    
    # Check Π invariance (should be relatively constant across workload steps)
    pi_mean_values = [p for p in pi_values]
    if len(pi_mean_values) > 1:
        pi_variation = max(pi_mean_values) - min(pi_mean_values)
        pi_tolerance = 0.1  # 10% variation tolerance
        pi_ref_mean = np.mean(pi_mean_values)
        pi_relative_variation = pi_variation / max(abs(pi_ref_mean), 1e-6)
        
        print(f"Pressure ratio across workload steps:")
        for i, (w, pi_mean, p_load) in enumerate(zip(workload_steps, pi_mean_values, p_load_values)):
            print(f"  Workload={w:6.1f} J: p_load={p_load:6.2f} kPa, Π_mean={pi_mean:.4f}")
        
        print(f"\nΠ invariance: variation={pi_relative_variation:.4f} (tolerance={pi_tolerance:.2f})")
        if pi_relative_variation > pi_tolerance:
            print(f"  WARNING: Pressure ratio variation exceeds tolerance!")
        else:
            print(f"  PASS: Pressure ratio stays within tolerance across workload steps.")
    
    # Check CA marker tolerance
    print(f"\nCA50 markers across workload steps:")
    ca50_valid = [c for c in ca50_values if c is not None]
    if len(ca50_valid) > 1:
        ca50_variation = max(ca50_valid) - min(ca50_valid)
        ca50_tolerance = 5.0  # degrees
        print(f"  CA50 values: {ca50_valid}")
        print(f"  CA50 variation: {ca50_variation:.2f} deg (tolerance={ca50_tolerance:.1f} deg)")
        if ca50_variation > ca50_tolerance:
            print(f"  WARNING: CA50 variation exceeds tolerance!")
        else:
            print(f"  PASS: CA50 markers stay within tolerance.")
    else:
        print(f"  WARNING: Insufficient CA50 data for validation.")
    
    # Verify workload-derived load pressure scaling
    print(f"\nWorkload-derived load pressure scaling:")
    for w, p_load in zip(workload_steps, p_load_values):
        expected_ratio = w / workload_steps[0] if workload_steps[0] > 0 else 1.0
        actual_ratio = p_load / p_load_values[0] if p_load_values[0] > 0 else 1.0
        print(f"  Workload={w:6.1f} J: p_load={p_load:6.2f} kPa, ratio={actual_ratio:.3f} (expected={expected_ratio:.3f})")
        if abs(actual_ratio - expected_ratio) > 0.01:
            print(f"    WARNING: Load pressure scaling mismatch!")
        else:
            print(f"    PASS: Load pressure scales correctly with workload.")

    # Test PR template generation
    print("\n=== PR Template Generation Test ===")
    theta_deg_test = np.degrees(theta)
    pr_template = compute_pr_template(
        theta=theta,
        stroke_mm=stroke_mm,
        bore_mm=bore_mm,
        clearance_volume_mm3=clearance_volume_mm3,
        compression_ratio=(clearance_volume_mm3 + area_mm2 * stroke_mm) / clearance_volume_mm3,
        p_load_kpa=0.0,
        p_cc_kpa=0.0,
        p_env_kpa=thermo.p_atm_kpa,
        expansion_efficiency_target=0.85,
        pr_peak_scale=1.5,
    )
    
    # Compare template with seed-derived PR (from combustion evaluation)
    seed_denom = thermo.p_atm_kpa  # Simplified denominator
    seed_pr = combustion_pi  # From earlier evaluation
    
    # Normalize both for comparison
    template_norm = pr_template / np.mean(pr_template) if np.mean(pr_template) > 0 else pr_template
    seed_pr_norm = seed_pr / np.mean(seed_pr) if np.mean(seed_pr) > 0 else seed_pr
    
    # Compute differences
    pr_diff = np.abs(template_norm - seed_pr_norm)
    max_diff = float(np.max(pr_diff))
    mean_diff = float(np.mean(pr_diff))
    
    print(f"PR Template vs Seed-derived comparison:")
    print(f"  Template mean: {np.mean(pr_template):.4f}, peak: {np.max(pr_template):.4f}")
    print(f"  Seed-derived mean: {np.mean(seed_pr):.4f}, peak: {np.max(seed_pr):.4f}")
    print(f"  Normalized difference - mean: {mean_diff:.4f}, max: {max_diff:.4f}")
    
    if max_diff > 0.1:
        print(f"  PASS: Template differs significantly from seed-derived (decoupled design)")
    else:
        print(f"  NOTE: Template and seed-derived are similar (may indicate good seed)")

    # Test injector-delay sweep
    print("\n=== Injector-Delay Sweep Test ===")
    injector_delays_deg = [-10.0, -5.0, 0.0, 5.0, 10.0]
    ca10_values_delay = []
    ca50_values_delay = []
    ca90_values_delay = []
    
    for delay_deg in injector_delays_deg:
        delay_inputs = {**combustion_inputs, "injector_delay_deg": delay_deg}
        delay_out = evaluate_cycle(
            adapter,
            theta,
            position,
            fuel_multiplier=1.0,
            c_load=0.0,
            geom=geom,
            thermo=thermo,
            combustion=delay_inputs,
        )
        delay_ca = delay_out.get("ca_markers") or {}
        if "CA10" in delay_ca and delay_ca["CA10"] is not None:
            ca10_values_delay.append(float(delay_ca["CA10"]))
        if "CA50" in delay_ca and delay_ca["CA50"] is not None:
            ca50_values_delay.append(float(delay_ca["CA50"]))
        if "CA90" in delay_ca and delay_ca["CA90"] is not None:
            ca90_values_delay.append(float(delay_ca["CA90"]))
    
    print(f"Injector delay sweep results:")
    print(f"  Delays tested: {injector_delays_deg}")
    if ca10_values_delay:
        print(f"  CA10 values: {ca10_values_delay}")
        print(f"  CA10 shift: {max(ca10_values_delay) - min(ca10_values_delay):.2f} deg")
    if ca50_values_delay:
        print(f"  CA50 values: {ca50_values_delay}")
        print(f"  CA50 shift: {max(ca50_values_delay) - min(ca50_values_delay):.2f} deg")
    if ca90_values_delay:
        print(f"  CA90 values: {ca90_values_delay}")
        print(f"  CA90 shift: {max(ca90_values_delay) - min(ca90_values_delay):.2f} deg")
    
    # Test gain-map spot checks
    print("\n=== Gain-Map Spot Check Test ===")
    if hasattr(adapter, '_update_gain_table'):
        # Create a small 2x2 grid
        test_phis = [0.8, 1.2]
        test_fuel_masses = [4e-4, 6e-4]
        for phi in test_phis:
            for fuel_m in test_fuel_masses:
                # Use fixed alpha/beta for test
                test_alpha = 30.0 + phi * 10.0
                test_beta = 100.0 + fuel_m * 100000.0
                adapter._update_gain_table(phi, fuel_m, test_alpha, test_beta)
        
        # Test interpolation at grid corners
        print("Grid corner tests:")
        for phi in test_phis:
            for fuel_m in test_fuel_masses:
                alpha, beta = adapter._get_scheduled_base_pressure(phi, fuel_m)
                print(f"  (phi={phi:.2f}, fuel={fuel_m:.4e}): alpha={alpha:.2f}, beta={beta:.2f}")
        
        # Test interpolation at interior point
        print("Interior point test:")
        alpha_interp, beta_interp = adapter._get_scheduled_base_pressure(1.0, 5e-4)
        print(f"  (phi=1.0, fuel=5e-4): alpha={alpha_interp:.2f}, beta={beta_interp:.2f}")
        
        # Test fallback when point outside grid
        print("Outside-grid test (should fallback to default):")
        alpha_fallback, beta_fallback = adapter._get_scheduled_base_pressure(0.5, 1e-4)
        print(f"  (phi=0.5, fuel=1e-4): alpha={alpha_fallback:.2f}, beta={beta_fallback:.2f}")
        if abs(alpha_fallback - adapter.alpha) < 0.01 and abs(beta_fallback - adapter.beta) < 0.01:
            print("  PASS: Fallback to default alpha/beta works")
        else:
            print("  WARNING: Fallback may not be working correctly")
        
        # Test reset
        adapter.reset_gain_table()
        alpha_empty, beta_empty = adapter._get_scheduled_base_pressure(1.0, 5e-4)
        if abs(alpha_empty - adapter.alpha) < 0.01 and abs(beta_empty - adapter.beta) < 0.01:
            print("  PASS: Reset clears table correctly")
        else:
            print("  WARNING: Reset may not be working correctly")
    else:
        print("  Gain scheduling not available on adapter")
    
    print(
        "\nNext steps:\n"
        "  1. Launch the GUI and enter the same AFR/fuel/ignition inputs.\n"
        "  2. Verify the diagnostics tab lists matching CA markers and pressure-ratio stats.\n"
        "  3. Adjust fuel mass in the GUI to confirm the guardrail violation indicator toggles.\n"
        "  4. Test workload sweeps in the GUI to verify Π invariance diagnostics.\n"
        "  5. Verify PR template is geometry-informed (check template shape matches engine geometry).\n"
    )


if __name__ == "__main__":
    main()
