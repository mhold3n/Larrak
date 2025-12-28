"""Thermal Finite Element Analysis Model.

Implements 2D axisymmetric heat conduction for piston/liner thermal analysis.
Uses scipy sparse solvers for efficient FEA computation.
"""

from typing import Any, Dict, Optional, Tuple

import numpy as np

try:
    from scipy.sparse import csr_matrix, lil_matrix
    from scipy.sparse.linalg import spsolve

    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

from Simulations.common.io_schema import SimulationInput, SimulationOutput
from Simulations.common.materials import MaterialLibrary
from Simulations.common.simulation import BaseSimulation, SimulationConfig


class ThermalFEAConfig(SimulationConfig):
    """Configuration for 2D Thermal FEA."""

    nr: int = 10  # Radial nodes
    nz: int = 15  # Axial nodes
    method: str = "steady"  # "steady" or "transient"
    dt: float = 0.001
    t_end: float = 0.1


class ThermalFEAModel(BaseSimulation):
    """
    2D Axisymmetric Finite Element Thermal Solver.

    Solves heat diffusion: rho*Cp*dT/dt = div(k*grad(T))

    Grid:
    - r: radial direction (0 = centerline, R = outer surface)
    - z: axial direction (0 = crown, L = skirt)

    Boundary Conditions:
    - Crown (z=0): convection from combustion gas (Woschni HTC)
    - Outer (r=R): conduction to liner + coolant convection
    - Bottom (z=L): oil cooling convection
    - Centerline (r=0): symmetric (dT/dr = 0)

    Source: Incropera & DeWitt, Heat and Mass Transfer.
    """

    def __init__(self, name: str, config: ThermalFEAConfig):
        super().__init__(name, config)
        self.config: ThermalFEAConfig = config
        self.input_data: Optional[SimulationInput] = None
        self.T: Optional[np.ndarray] = None
        self.mesh_r: Optional[np.ndarray] = None
        self.mesh_z: Optional[np.ndarray] = None

    def load_input(self, input_data: SimulationInput):
        """Load simulation input bundle."""
        self.input_data = input_data

    def setup(self):
        """
        Initialize mesh and material properties.

        1. Create axisymmetric mesh (r, z)
        2. Assign thermal properties from MaterialLibrary
        3. Initialize temperature field
        """
        if not self.input_data:
            raise ValueError("No input data loaded. Call load_input() first.")

        geo = self.input_data.geometry

        # Piston geometry approximation
        R = geo.bore / 2.0  # Piston radius [m]
        L = 0.15 * geo.bore  # Crown thickness estimate [m]

        nr, nz = self.config.nr, self.config.nz

        # Create mesh
        self.mesh_r = np.linspace(0, R, nr)
        self.mesh_z = np.linspace(0, L, nz)
        self.dr = R / (nr - 1) if nr > 1 else R
        self.dz = L / (nz - 1) if nz > 1 else L

        # Material properties
        self.mat = MaterialLibrary.get_aluminum_6061_t6()

        # Initialize temperature field (isothermal start)
        self.T = np.full((nr, nz), self.input_data.operating_point.T_coolant)

        self.results = {}

    def _build_stiffness_matrix(
        self,
        h_gas: float,
        T_gas: float,
        h_oil: float,
        T_oil: float,
        h_coolant: float,
        T_coolant: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build FEA stiffness matrix and RHS for steady-state heat equation.

        Uses finite difference discretization of axisymmetric heat equation:
        (1/r) * d/dr(r * k * dT/dr) + d/dz(k * dT/dz) = 0

        Returns:
            K: Stiffness matrix (n x n)
            F: Load vector (n,)
        """
        nr, nz = self.config.nr, self.config.nz
        n = nr * nz
        k = self.mat.thermal_conductivity

        # Use sparse matrix for efficiency
        if HAS_SCIPY:
            K = lil_matrix((n, n))
        else:
            K = np.zeros((n, n))
        F = np.zeros(n)

        def idx(i, j):
            """Convert 2D index to 1D."""
            return i * nz + j

        for i in range(nr):
            r = self.mesh_r[i]
            for j in range(nz):
                node = idx(i, j)

                # Interior nodes: 5-point stencil
                if 0 < i < nr - 1 and 0 < j < nz - 1:
                    # Radial terms with axisymmetric correction
                    r_plus = r + self.dr / 2
                    r_minus = (r - self.dr / 2) if r > self.dr / 2 else self.dr / 2

                    coef_r_plus = k * r_plus / (r * self.dr**2)
                    coef_r_minus = k * r_minus / (r * self.dr**2)
                    coef_z = k / self.dz**2

                    K[node, idx(i + 1, j)] = -coef_r_plus
                    K[node, idx(i - 1, j)] = -coef_r_minus
                    K[node, idx(i, j + 1)] = -coef_z
                    K[node, idx(i, j - 1)] = -coef_z
                    K[node, node] = coef_r_plus + coef_r_minus + 2 * coef_z

                # Crown boundary (j=0): gas convection
                elif j == 0:
                    h_eff = h_gas / self.dz
                    K[node, node] = k / self.dz + h_eff
                    if j + 1 < nz:
                        K[node, idx(i, j + 1)] = -k / self.dz
                    F[node] = h_eff * T_gas

                # Bottom boundary (j=nz-1): oil convection
                elif j == nz - 1:
                    h_eff = h_oil / self.dz
                    K[node, node] = k / self.dz + h_eff
                    if j - 1 >= 0:
                        K[node, idx(i, j - 1)] = -k / self.dz
                    F[node] = h_eff * T_oil

                # Centerline (i=0): symmetry dT/dr = 0
                elif i == 0:
                    K[node, node] = 1.0
                    K[node, idx(1, j)] = -1.0
                    F[node] = 0.0

                # Outer surface (i=nr-1): coolant convection
                elif i == nr - 1:
                    h_eff = h_coolant / self.dr
                    K[node, node] = k / self.dr + h_eff
                    if i - 1 >= 0:
                        K[node, idx(i - 1, j)] = -k / self.dr
                    F[node] = h_eff * T_coolant

        if HAS_SCIPY:
            K = csr_matrix(K)

        return K, F

    def _estimate_htc_woschni(self) -> Tuple[float, float]:
        """
        Estimate cycle-averaged HTC and effective gas temperature using Woschni correlation.

        Returns:
            h_gas_eff: Effective gas-side HTC [W/m²K]
            T_gas_eff: Effective gas temperature [K]
        """
        bcs = self.input_data.boundary_conditions

        p_arr = np.array(bcs.pressure_gas)
        T_arr = np.array(bcs.temperature_gas)

        # Woschni-like correlation: h ~ P^0.8 * T^-0.55
        h_rel = (p_arr**0.8) * (T_arr**-0.55)
        h_scale = 1000.0 / np.mean(h_rel)  # Normalize to ~1000 W/m²K
        h_trace = h_rel * h_scale

        # Weighted effective temperature
        T_gas_eff = np.sum(h_trace * T_arr) / np.sum(h_trace)
        h_gas_eff = np.mean(h_trace)

        return h_gas_eff, T_gas_eff

    def solve_steady_state(self) -> SimulationOutput:
        """
        Solve steady-state thermal field.

        Returns:
            SimulationOutput with T_crown_max, T_liner_max
        """
        self.setup()

        ops = self.input_data.operating_point

        # Estimate gas-side conditions
        h_gas, T_gas = self._estimate_htc_woschni()

        # Cooling conditions
        h_oil = 1500.0  # Oil squirter [W/m²K]
        h_coolant = 3000.0  # Coolant channel [W/m²K]
        T_oil = ops.T_oil
        T_coolant = ops.T_coolant

        # Build and solve system
        K, F = self._build_stiffness_matrix(h_gas, T_gas, h_oil, T_oil, h_coolant, T_coolant)

        if HAS_SCIPY:
            T_flat = spsolve(K, F)
        else:
            T_flat = np.linalg.solve(K, F)

        # Reshape to 2D
        self.T = T_flat.reshape((self.config.nr, self.config.nz))

        # Extract key results
        T_crown_max = float(np.max(self.T[:, 0]))  # Crown surface (z=0)
        T_liner_max = float(np.max(self.T[-1, :]))  # Outer surface (r=R)

        self.results = {
            "T_crown_max": T_crown_max,
            "T_liner_max": T_liner_max,
            "T_field": self.T.copy(),
            "h_gas_eff": h_gas,
            "T_gas_eff": T_gas,
        }

        return SimulationOutput(
            run_id=self.input_data.run_id,
            success=True,
            T_crown_max=T_crown_max,
            T_liner_max=T_liner_max,
        )

    def step(self, dt: float):
        """
        Advance transient thermal solution by one time step.

        Uses implicit Euler: (M/dt + K) * T_new = M/dt * T_old + F
        """
        if self.T is None:
            self.setup()

        # For now, just store steady-state result
        # Full transient would require mass matrix assembly
        self.results["T_max"] = float(np.max(self.T))

    def post_process(self) -> Dict[str, Any]:
        """Return thermal analysis results."""
        return self.results
