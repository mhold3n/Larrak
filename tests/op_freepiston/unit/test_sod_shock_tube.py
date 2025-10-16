"""Sod shock tube test cases for validation of 1D gas dynamics solver."""

import math
from typing import List, Tuple

import pytest

from campro.freepiston.net1d.bc import non_reflecting_inlet_bc, non_reflecting_outlet_bc
from campro.freepiston.net1d.flux import hllc_flux, primitive_from_conservative


def sod_shock_tube_initial_conditions() -> Tuple[List[Tuple[float, float, float]], List[Tuple[float, float, float]]]:
    """Initial conditions for Sod shock tube problem.
    
    Returns
    -------
    U_left : List[Tuple[float, float, float]]
        Left state conservative variables [rho, rho*u, rho*E]
    U_right : List[Tuple[float, float, float]]
        Right state conservative variables [rho, rho*u, rho*E]
    """
    # Left state (high pressure, high density)
    rho_L = 1.0  # kg/m^3
    u_L = 0.0    # m/s
    p_L = 1.0    # Pa (normalized)

    # Right state (low pressure, low density)
    rho_R = 0.125  # kg/m^3
    u_R = 0.0      # m/s
    p_R = 0.1      # Pa (normalized)

    # Convert to conservative variables
    gamma = 1.4
    e_L = p_L / ((gamma - 1.0) * rho_L)  # Specific internal energy
    e_R = p_R / ((gamma - 1.0) * rho_R)

    E_L = e_L + 0.5 * u_L**2  # Specific total energy
    E_R = e_R + 0.5 * u_R**2

    U_L = (rho_L, rho_L * u_L, rho_L * E_L)
    U_R = (rho_R, rho_R * u_R, rho_R * E_R)

    return U_L, U_R


def sod_shock_tube_analytical_solution(x: float, t: float, x0: float = 0.5) -> Tuple[float, float, float]:
    """Analytical solution for Sod shock tube problem.
    
    Parameters
    ----------
    x : float
        Spatial coordinate [m]
    t : float
        Time [s]
    x0 : float
        Initial discontinuity position [m]
        
    Returns
    -------
    rho : float
        Density [kg/m^3]
    u : float
        Velocity [m/s]
    p : float
        Pressure [Pa]
    """
    # Initial conditions
    rho_L, u_L, p_L = 1.0, 0.0, 1.0
    rho_R, u_R, p_R = 0.125, 0.0, 0.1
    gamma = 1.4

    # Speed of sound
    c_L = math.sqrt(gamma * p_L / rho_L)
    c_R = math.sqrt(gamma * p_R / rho_R)

    # Riemann invariants
    A_L = 2.0 * c_L / (gamma - 1.0)
    A_R = 2.0 * c_R / (gamma - 1.0)

    # Pressure ratio
    p_ratio = p_L / p_R

    # Shock relations
    mu_sq = (gamma - 1.0) / (gamma + 1.0)
    p_star = p_R * ((2.0 * gamma / (gamma + 1.0)) * (p_ratio - 1.0) + 1.0)

    # Wave speeds
    c_star_L = c_L * (p_star / p_L) ** ((gamma - 1.0) / (2.0 * gamma))
    c_star_R = c_R * (p_star / p_R) ** ((gamma - 1.0) / (2.0 * gamma))

    # Contact wave speed
    u_star = u_L + A_L - c_star_L

    # Shock speed
    u_shock = u_R + c_R * math.sqrt((gamma + 1.0) / (2.0 * gamma) * (p_star / p_R - 1.0) + 1.0)

    # Wave positions
    x_contact = x0 + u_star * t
    x_shock = x0 + u_shock * t
    x_rarefaction_left = x0 + (u_L - c_L) * t
    x_rarefaction_right = x0 + (u_star - c_star_L) * t

    # Solution based on position
    if x < x_rarefaction_left:
        # Left state
        return rho_L, u_L, p_L
    if x < x_rarefaction_right:
        # Rarefaction wave
        u_rare = u_L + 2.0 * c_L / (gamma + 1.0) * (1.0 + (gamma - 1.0) / (2.0 * c_L) * (x - x0) / t)
        c_rare = c_L - (gamma - 1.0) / 2.0 * u_rare
        p_rare = p_L * (c_rare / c_L) ** (2.0 * gamma / (gamma - 1.0))
        rho_rare = rho_L * (p_rare / p_L) ** (1.0 / gamma)
        return rho_rare, u_rare, p_rare
    if x < x_contact:
        # Star region (left)
        rho_star_L = rho_L * (p_star / p_L) ** (1.0 / gamma)
        return rho_star_L, u_star, p_star
    if x < x_shock:
        # Star region (right)
        rho_star_R = rho_R * (p_star / p_R) ** (1.0 / gamma)
        return rho_star_R, u_star, p_star
    # Right state
    return rho_R, u_R, p_R


def test_sod_shock_tube_initial_conditions():
    """Test initial conditions for Sod shock tube."""
    U_L, U_R = sod_shock_tube_initial_conditions()

    # Check left state
    assert U_L[0] == 1.0  # rho_L
    assert U_L[1] == 0.0  # rho*u_L
    assert U_L[2] > 0.0   # rho*E_L

    # Check right state
    assert U_R[0] == 0.125  # rho_R
    assert U_R[1] == 0.0    # rho*u_R
    assert U_R[2] > 0.0     # rho*E_R

    # Check that left state has higher energy
    assert U_L[2] > U_R[2]


def test_sod_shock_tube_analytical_solution():
    """Test analytical solution for Sod shock tube."""
    # Test at different positions and times
    x_positions = [0.0, 0.25, 0.5, 0.75, 1.0]
    t = 0.1  # s

    for x in x_positions:
        rho, u, p = sod_shock_tube_analytical_solution(x, t)

        # Check physical constraints
        assert rho > 0.0
        assert p > 0.0
        assert not math.isnan(u)
        assert not math.isinf(rho)
        assert not math.isinf(p)
        assert not math.isinf(u)


def test_hllc_flux_sod_shock_tube():
    """Test HLLC flux with Sod shock tube initial conditions."""
    U_L, U_R = sod_shock_tube_initial_conditions()

    # Compute HLLC flux
    F_hat = hllc_flux(U_L, U_R)

    # Check that flux is finite and reasonable
    assert all(not math.isnan(f) and not math.isinf(f) for f in F_hat)
    assert all(abs(f) < 1e6 for f in F_hat)  # Reasonable magnitude

    # Check flux properties
    # Mass flux should be positive (flow from left to right)
    assert F_hat[0] > 0.0

    # Momentum flux should be positive (pressure difference drives flow)
    assert F_hat[1] > 0.0

    # Energy flux should be positive (energy flows from high to low)
    assert F_hat[2] > 0.0


def test_sod_shock_tube_1d_simulation():
    """Test 1D simulation of Sod shock tube problem."""
    # Grid parameters
    nx = 100
    x_min, x_max = 0.0, 1.0
    dx = (x_max - x_min) / nx
    x = [x_min + i * dx for i in range(nx)]

    # Time parameters
    dt = 1e-4  # s
    t_final = 0.1  # s
    n_steps = int(t_final / dt)

    # Initial conditions
    U_L, U_R = sod_shock_tube_initial_conditions()

    # Initialize solution
    U = []
    for i in range(nx):
        if x[i] < 0.5:
            U.append(U_L)
        else:
            U.append(U_R)

    # Time stepping
    for step in range(n_steps):
        U_new = []

        # Boundary conditions (non-reflecting)
        U_new.append(U[0])  # Left boundary
        U_new.append(U[-1])  # Right boundary

        # Interior points
        for i in range(1, nx - 1):
            # HLLC flux at left interface
            F_left = hllc_flux(U[i-1], U[i])

            # HLLC flux at right interface
            F_right = hllc_flux(U[i], U[i+1])

            # Conservative update
            U_new_i = []
            for j in range(3):
                dU_dt = -(F_right[j] - F_left[j]) / dx
                U_new_i.append(U[i][j] + dt * dU_dt)

            U_new.append(tuple(U_new_i))

        U = U_new

    # Check final solution
    for i in range(nx):
        rho, u, p = primitive_from_conservative(U[i])

        # Check physical constraints
        assert rho > 0.0
        assert p > 0.0
        assert not math.isnan(u)
        assert not math.isinf(rho)
        assert not math.isinf(p)
        assert not math.isinf(u)


def test_sod_shock_tube_convergence():
    """Test convergence of Sod shock tube solution."""
    # Different grid resolutions
    nx_list = [50, 100, 200]
    errors = []

    for nx in nx_list:
        # Grid parameters
        x_min, x_max = 0.0, 1.0
        dx = (x_max - x_min) / nx
        x = [x_min + i * dx for i in range(nx)]

        # Time parameters
        dt = dx * 0.1  # CFL condition
        t_final = 0.1  # s
        n_steps = int(t_final / dt)

        # Initial conditions
        U_L, U_R = sod_shock_tube_initial_conditions()

        # Initialize solution
        U = []
        for i in range(nx):
            if x[i] < 0.5:
                U.append(U_L)
            else:
                U.append(U_R)

        # Time stepping
        for step in range(n_steps):
            U_new = []

            # Boundary conditions
            U_new.append(U[0])
            U_new.append(U[-1])

            # Interior points
            for i in range(1, nx - 1):
                F_left = hllc_flux(U[i-1], U[i])
                F_right = hllc_flux(U[i], U[i+1])

                U_new_i = []
                for j in range(3):
                    dU_dt = -(F_right[j] - F_left[j]) / dx
                    U_new_i.append(U[i][j] + dt * dU_dt)

                U_new.append(tuple(U_new_i))

            U = U_new

        # Compute error against analytical solution
        error = 0.0
        for i in range(nx):
            rho_num, u_num, p_num = primitive_from_conservative(U[i])
            rho_ana, u_ana, p_ana = sod_shock_tube_analytical_solution(x[i], t_final)

            error += abs(rho_num - rho_ana) + abs(u_num - u_ana) + abs(p_num - p_ana)

        error /= nx
        errors.append(error)

    # Check convergence (error should decrease with grid refinement)
    assert errors[1] < errors[0]  # 100 points better than 50
    assert errors[2] < errors[1]  # 200 points better than 100


def test_sod_shock_tube_energy_conservation():
    """Test energy conservation in Sod shock tube simulation."""
    # Grid parameters
    nx = 100
    x_min, x_max = 0.0, 1.0
    dx = (x_max - x_min) / nx

    # Time parameters
    dt = 1e-4  # s
    t_final = 0.1  # s
    n_steps = int(t_final / dt)

    # Initial conditions
    U_L, U_R = sod_shock_tube_initial_conditions()

    # Initialize solution
    U = []
    for i in range(nx):
        if i < nx // 2:
            U.append(U_L)
        else:
            U.append(U_R)

    # Compute initial total energy
    E_initial = sum(U[i][2] for i in range(nx)) * dx

    # Time stepping with conservative scheme
    for step in range(n_steps):
        U_new = []

        # Interior points with conservative flux differencing
        for i in range(nx):
            if i == 0:
                # Left boundary: use ghost cell with same state
                F_left = hllc_flux(U[0], U[0])  # F(0,0) = 0
                F_right = hllc_flux(U[0], U[1])
            elif i == nx - 1:
                # Right boundary: use ghost cell with same state
                F_left = hllc_flux(U[nx-2], U[nx-1])
                F_right = hllc_flux(U[nx-1], U[nx-1])  # F(0,0) = 0
            else:
                # Interior points
                F_left = hllc_flux(U[i-1], U[i])
                F_right = hllc_flux(U[i], U[i+1])

            # Conservative update: dU/dt = -dF/dx
            U_new_i = []
            for j in range(3):
                # Use conservative flux differencing
                dF_dx = (F_right[j] - F_left[j]) / dx
                U_new_i.append(U[i][j] - dt * dF_dx)

            U_new.append(tuple(U_new_i))

        U = U_new

    # Compute final total energy
    E_final = sum(U[i][2] for i in range(nx)) * dx

    # Check energy conservation (should be conserved within numerical error)
    energy_error = abs(E_final - E_initial) / E_initial
    assert energy_error < 0.01  # 1% error tolerance


def test_sod_shock_tube_boundary_conditions():
    """Test boundary conditions for Sod shock tube."""
    # Test non-reflecting boundary conditions
    U_interior = (1.0, 0.0, 1000.0)  # High pressure state

    # Inlet boundary condition
    p_target, T_target = 1.0, 300.0
    U_inlet = non_reflecting_inlet_bc(U_interior, p_target, T_target)

    # Check that boundary state is physically reasonable
    rho_bc, u_bc, p_bc = primitive_from_conservative(U_inlet)
    assert rho_bc > 0.0
    assert p_bc > 0.0
    assert not math.isnan(u_bc)

    # Outlet boundary condition
    p_target_out = 0.1
    U_outlet = non_reflecting_outlet_bc(U_interior, p_target_out)

    # Check that boundary state is physically reasonable
    rho_bc_out, u_bc_out, p_bc_out = primitive_from_conservative(U_outlet)
    assert rho_bc_out > 0.0
    assert p_bc_out > 0.0
    assert not math.isnan(u_bc_out)


def test_sod_shock_tube_robustness():
    """Test robustness of Sod shock tube solver for extreme conditions."""
    # Test with very high pressure ratio
    U_L = (10.0, 0.0, 10000.0)  # Very high pressure
    U_R = (0.1, 0.0, 100.0)     # Very low pressure

    F_hat = hllc_flux(U_L, U_R)

    # Check that solver remains stable
    assert all(not math.isnan(f) and not math.isinf(f) for f in F_hat)
    assert all(abs(f) < 1e8 for f in F_hat)  # Reasonable magnitude

    # Test with very high velocity
    U_L = (1.0, 1000.0, 1000.0)  # Very high velocity
    U_R = (1.0, 0.0, 1000.0)     # Zero velocity

    F_hat = hllc_flux(U_L, U_R)

    # Check that solver remains stable
    assert all(not math.isnan(f) and not math.isinf(f) for f in F_hat)
    assert all(abs(f) < 1e8 for f in F_hat)


if __name__ == "__main__":
    pytest.main([__file__])
