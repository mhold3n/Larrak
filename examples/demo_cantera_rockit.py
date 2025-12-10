import numpy as np
from campro.optimization.freepiston.opt.thermo_casadi import IdealGasMixture, NASACoeffs
from rockit import MultipleShooting, Ocp


def demo_rockit_thermo():
    print("DEMO: Rockit + Cantera-CasADi Thermodynamics")
    print("============================================")

    # 1. Setup Thermodynamics (Nitrogen)
    # coeffs from standard GRI30 or similar
    # N2 Low T (200-1000K) coefficients roughly:
    n2_coeffs = NASACoeffs(
        a_low=np.array(
            [
                3.53100528e00,
                -1.23660988e-04,
                -5.02999433e-07,
                2.43530612e-09,
                -1.40881235e-12,
                -1.04697628e03,
                2.96747038e00,
            ]
        ),
        a_high=np.array(
            [
                2.95257637e00,
                1.39690040e-03,
                -4.92631603e-07,
                7.86010195e-11,
                -4.60755204e-15,
                -9.23948688e02,
                5.87188762e00,
            ]
        ),
        T_low=200.0,
        T_mid=1000.0,
        T_high=6000.0,
        W=0.028,  # kg/mol
    )
    gas = IdealGasMixture({"N2": n2_coeffs}, {"N2": 1.0})

    # 2. Setup Optimal Control Problem
    ocp = Ocp(t0=0, T=0.02)  # 20ms expansion

    # States: Scale them naturally! Rockit allows explicit scaling.
    # Volume V [m3] (approx 100cc to 500cc)
    V = ocp.state(scale=1e-4)
    ocp.set_initial(V, 1e-4)  # Start at 100cc

    # Temperature T [K]
    T = ocp.state(scale=1000.0)
    ocp.set_initial(T, 1000.0)  # Start hot

    # 3. Physics (Adiabatic Expansion)
    # dV/dt = A * v_piston (Assume constant piston velocity for demo)
    v_piston = 10.0  # m/s
    A_piston = 0.002  # m2
    dVdt = v_piston * A_piston

    ocp.set_der(V, dVdt)

    # Energy Equation: dU/dt = -P dV/dt (Adiabatic)
    # u(T) is internal energy per kg. U_total = m * u(T).
    # m is constant. m * du/dt = -P * dV/dt
    # du/dt = cv * dT/dt (if cv constant, but it's not!)
    # Better: Use Enthalpy h = u + Pv => u = h - RT

    # Equation: dT/dt = ( -P * dV/dt ) / (m * cv(T))
    # We need cv(T). cp(T) is available. cv = cp - R.

    mass = 1e-4  # kg (approx)

    # Calculation of P, cv, etc. from State T
    # Note: These are SYMBOLIC expressions using our thermo_casadi class!
    cp_val = gas.cp(T)
    cv_val = cp_val - gas.R_spec

    rho = mass / V
    P = rho * gas.R_spec * T

    # Dynamics for T
    dTdt = (-P * dVdt) / (mass * cv_val)
    ocp.set_der(T, dTdt)

    # 4. Constraints
    ocp.subject_to(T >= 200)  # Keep T reasonable
    ocp.subject_to(ocp.at_t0(V) == 1e-4)
    ocp.subject_to(ocp.at_t0(T) == 1000.0)

    # 5. Solver Setup
    # Rockit handles scaling automatically if we provide typical scales

    ocp.solver("ipopt", {"ipopt": {"linear_solver": "ma57", "print_level": 5}})
    ocp.method(MultipleShooting(N=20))

    # 6. Solve
    print("Solving OCP...")
    sol = ocp.solve()

    # 7. Post-Process
    ts, Ts = sol.sample(T, grid="control")
    ts, Vs = sol.sample(V, grid="control")
    ts, Ps = sol.sample(P, grid="control")

    print("\nResults:")
    print(f"  Final Temp: {Ts[-1]:.2f} K")
    print(f"  Final Vol:  {Vs[-1]:.2e} m3")
    print(f"  Final Pres: {Ps[-1]:.2e} Pa")

    # Verify conservation?
    # PV^gamma approx constant?
    idx_start = 0
    idx_end = -1
    P1, V1 = Ps[idx_start], Vs[idx_start]
    P2, V2 = Ps[idx_end], Vs[idx_end]
    gamma_approx = np.log(P1 / P2) / np.log(V2 / V1)
    print(
        f"  Effective Gamma: {gamma_approx:.4f} (Expected ~1.35-1.4 for N2 at high T)"
    )


if __name__ == "__main__":
    demo_rockit_thermo()
