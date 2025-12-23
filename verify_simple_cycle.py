import numpy as np

from campro.physics.simple_cycle_adapter import (
    CycleGeometry,
    CycleThermo,
    SimpleCycleAdapter,
    WiebeParams,
)


def test_adapter():
    print("Testing SimpleCycleAdapter...")
    wiebe = WiebeParams(a=5.0, m=2.0, start_deg=-10.0, duration_deg=40.0)
    adapter = SimpleCycleAdapter(wiebe=wiebe, alpha_fuel_to_base=1.0, beta_base=0.0)

    # Mock data
    theta = np.linspace(0, 4 * np.pi, 200)
    x = 100 * (1 - np.cos(theta)) / 2 + 50
    v = np.gradient(x, theta)

    geom = CycleGeometry(area_mm2=8000.0, clearance_volume_mm3=100000.0)
    thermo = CycleThermo(gamma_bounce=1.4, p_atm_kpa=100.0)

    # Run evaluate with combustion
    print("Evaluating with combustion...")
    try:
        res = adapter.evaluate(
            theta=theta,
            x_mm=x,
            v_mm_per_theta=v,
            fuel_multiplier=1.0,
            c_load=0.0,
            geom=geom,
            thermo=thermo,
            combustion={
                "fuel_type": "gasoline",
                "fuel_mass_kg": 0.001,
                "afr": 14.7,
                "T_init": 350.0,  # Creates initial_temperature_k
            },
            cycle_time_s=0.1,
        )
        print("Evaluation successful.")
        print("IMEP:", res.get("imep"))
        print("Peak Pressure:", np.max(res.get("p_cyl")))
    except Exception as e:
        print(f"FAILED: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    test_adapter()
