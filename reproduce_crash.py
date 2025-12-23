import casadi as ca
import numpy as np

from campro.physics.combustion_model import CombustionModel


def test_model():
    print("Instantiating model...")
    model = CombustionModel()

    print("Configuring model...")
    model.configure(
        fuel_type="gasoline",
        afr=14.7,
        bore_m=0.1,
        stroke_m=0.1,
        clearance_volume_m3=0.0001,
        fuel_mass_kg=0.001,
        cycle_time_s=0.1,
        initial_temperature_k=300.0,
        initial_pressure_pa=1e5,
    )

    print("Generating symbolic expressions...")
    # Mock symbolic variables
    time_s = ca.MX.sym("time_s")
    piston_speed = ca.MX.sym("u_piston")
    ignition_time = ca.MX.sym("t_ign")
    omega = ca.MX.sym("omega")

    res = model.symbolic_heat_release(
        ca=ca,
        time_s=time_s,
        piston_speed_m_per_s=piston_speed,
        ignition_time_s=ignition_time,
        omega_deg_per_s=omega,
    )

    print("Symbolic generation successful.")
    print("Keys:", res.keys())

    # Try evaluating
    f = ca.Function("f", [time_s, piston_speed, ignition_time, omega], [res["heat_release_rate"]])
    val = f(0.05, 10.0, 0.04, 1000.0)
    print("Evaluated:", val)


if __name__ == "__main__":
    test_model()
