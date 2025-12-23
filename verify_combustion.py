import traceback
import numpy as np
from campro.physics.simple_cycle_adapter import (
    CycleGeometry,
    CycleThermo,
    SimpleCycleAdapter,
    WiebeParams,
)


def test_combustion() -> None:
    # Setup
    theta = np.linspace(0, 2 * np.pi, 200)
    # Simple harmonic motion: Top Dead Center at theta=0?
    # Usually 0 is TDC.
    x_mm = 50.0 - 50.0 * np.cos(theta)  # 0 to 100 mm (BDC at pi)
    v_mm_per_theta = 50.0 * np.sin(theta)

    geom = CycleGeometry(area_mm2=5000.0, clearance_volume_mm3=50000.0)
    thermo = CycleThermo(gamma_bounce=1.4, p_atm_kpa=101.3)
    wiebe = WiebeParams(a=5.0, m=2.0, start_deg=10.0, duration_deg=60.0)

    adapter = SimpleCycleAdapter(wiebe=wiebe, alpha_fuel_to_base=1.0, beta_base=0.0)

    # Test with combustion
    combustion_inputs = {
        "afr": 14.5,
        "fuel_mass_kg": 5e-5,  # Small amount
        "fuel_type": "diesel",
        "ignition_theta_deg": 5.0,  # 5 deg ATDC
    }

    cycle_time_s = 0.04

    print("Running evaluate with combustion...")
    try:
        result = adapter.evaluate(
            theta=theta,
            x_mm=x_mm,
            v_mm_per_theta=v_mm_per_theta,
            fuel_multiplier=1.0,
            c_load=0.0,
            geom=geom,
            thermo=thermo,
            combustion=combustion_inputs,
            cycle_time_s=cycle_time_s,
        )
        print("Evaluate returned successfully.")
        print("Result keys:", list(result.keys()))
        if "p_cyl" in result:
            print(f"Max Cylinder Pressure: {np.max(result['p_cyl']):.2f} kPa")
        if "mass_fraction_burned" in result:
            print(f"Max MFB: {np.max(result['mass_fraction_burned']):.4f}")

    except Exception as e:
        print(f"Evaluate failed: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    test_combustion()
