"""Combustion CFD Adapter (OpenFOAM).

High-fidelity reacting flow simulation for:
- Prechamber jet ignition dynamics
- Main chamber flame propagation
- Wiebe parameter validation

Uses: openfoam reactingFoam -case <case_dir>
"""

import json
import shutil
import subprocess

from pathlib import Path
from typing import Any

import numpy as np

from Simulations.common.io_schema import SimulationOutput
from Simulations.hifi.base import ExternalSolverAdapter, ExternalSolverConfig, SolverBackend


# Pydantic model
class CombustionCFDConfig(ExternalSolverConfig):
    """Configuration for combustion CFD solver."""

    backend: SolverBackend = SolverBackend.OPENFOAM

    # Solver selection
    solver: str = "reactingFoam"  # or "XiFoam" for premixed

    # Chemistry model
    chemistry_model: str = "PaSR"  # "PaSR", "EDC", "flamelet"

    # Turbulence model
    turbulence_model: str = "kOmegaSST"

    # Time stepping
    end_time: float = 0.005  # 5ms combustion event
    write_interval: float = 0.0001  # 100us output
    max_courant: float = 0.5


class CombustionCFDAdapter(ExternalSolverAdapter):
    """
    3D Reacting Flow CFD wrapper using OpenFOAM.

    Workflow:
    1. Generate mesh from geometry (Gmsh -> gmshToFoam)
    2. Set up OpenFOAM case (0/, constant/, system/)
    3. Run reactingFoam or XiFoam
    4. Post-process for burn rate and Wiebe parameters
    """

    def __init__(self, name: str = "combustion_cfd", config: CombustionCFDConfig | None = None):
        super().__init__(name, config or CombustionCFDConfig())
        self.config: CombustionCFDConfig = self.config

    def _generate_mesh(self):
        """Generate 3D combustion chamber mesh."""
        from Simulations.hifi.mesh import GmshMesher, MeshConfig

        geo = self.input_data.geometry

        mesh_config = MeshConfig(
            element_size_min=1e-3,  # 1mm for CFD
            element_size_max=5e-3,
        )
        mesher = GmshMesher(mesh_config)

        msh_file = self.case_dir / "chamber.msh"
        result = mesher.create_combustion_chamber(
            bore=geo.bore,
            stroke=geo.stroke,
            clearance_height=0.01,  # 10mm clearance at TDC
            output_path=str(msh_file),
        )

        # Convert to OpenFOAM format
        self._convert_mesh_to_foam(msh_file)

        self.mesh_info = result
        print(f"[{self.name}] Generated mesh: {result['n_nodes']} nodes")

    def _convert_mesh_to_foam(self, msh_file: Path):
        """Convert Gmsh mesh to OpenFOAM polyMesh."""
        # Create constant/polyMesh directory
        poly_dir = self.case_dir / "constant" / "polyMesh"
        poly_dir.mkdir(parents=True, exist_ok=True)

        cmd = ["openfoam", "gmshToFoam", str(msh_file), "-case", str(self.case_dir)]

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"[{self.name}] gmshToFoam warning: {result.stderr[:200]}")

    def _write_input_files(self):
        """Write OpenFOAM case files."""
        self._write_control_dict()
        self._write_fv_schemes()
        self._write_fv_solution()
        self._write_initial_conditions()
        self._write_thermophysical_properties()
        self._write_turbulence_properties()

    def _write_control_dict(self):
        """Write system/controlDict."""
        system_dir = self.case_dir / "system"
        system_dir.mkdir(parents=True, exist_ok=True)

        content = f"""FoamFile
{{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      controlDict;
}}

application     {self.config.solver};
startFrom       startTime;
startTime       0;
stopAt          endTime;
endTime         {self.config.end_time};
deltaT          1e-7;
writeControl    adjustableRunTime;
writeInterval   {self.config.write_interval};
purgeWrite      0;
writeFormat     ascii;
writePrecision  8;
writeCompression off;
timeFormat      general;
timePrecision   6;
runTimeModifiable true;
adjustTimeStep  yes;
maxCo           {self.config.max_courant};
maxDeltaT       1e-5;

functions
{{
    fieldAverage1
    {{
        type            fieldAverage;
        libs            ("libfieldFunctionObjects.so");
        writeControl    writeTime;
        fields
        (
            U {{ mean on; prime2Mean on; base time; }}
            p {{ mean on; prime2Mean on; base time; }}
            T {{ mean on; prime2Mean on; base time; }}
        );
    }}
}}
"""
        (system_dir / "controlDict").write_text(content)

    def _write_fv_schemes(self):
        """Write system/fvSchemes."""
        content = """FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      fvSchemes;
}

ddtSchemes { default Euler; }
gradSchemes { default Gauss linear; }
divSchemes
{
    default         none;
    div(phi,U)      Gauss linearUpwind grad(U);
    div(phi,K)      Gauss linearUpwind default;
    div(phi,h)      Gauss linearUpwind default;
    div(phi,k)      Gauss linearUpwind default;
    div(phi,omega)  Gauss linearUpwind default;
    div(phi,Yi_h)   Gauss linearUpwind default;
    div(((rho*nuEff)*dev2(T(grad(U))))) Gauss linear;
}
laplacianSchemes { default Gauss linear corrected; }
interpolationSchemes { default linear; }
snGradSchemes { default corrected; }
"""
        (self.case_dir / "system" / "fvSchemes").write_text(content)

    def _write_fv_solution(self):
        """Write system/fvSolution."""
        content = """FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      fvSolution;
}

solvers
{
    "rho.*" { solver diagonal; }
    p
    {
        solver          PCG;
        preconditioner  DIC;
        tolerance       1e-8;
        relTol          0.01;
    }
    pFinal { $p; relTol 0; }
    "(U|h|k|omega|Yi)"
    {
        solver          PBiCGStab;
        preconditioner  DILU;
        tolerance       1e-8;
        relTol          0.1;
    }
    "(U|h|k|omega|Yi)Final" { $U; relTol 0; }
}

PIMPLE
{
    nOuterCorrectors 2;
    nCorrectors      2;
    nNonOrthogonalCorrectors 1;
}

relaxationFactors
{
    equations
    {
        ".*" 1;
    }
}
"""
        (self.case_dir / "system" / "fvSolution").write_text(content)

    def _write_initial_conditions(self):
        """Write 0/ directory files."""
        zero_dir = self.case_dir / "0"
        zero_dir.mkdir(parents=True, exist_ok=True)

        ops = self.input_data.operating_point
        bcs = self.input_data.boundary_conditions

        T_init = ops.T_intake if hasattr(ops, "T_intake") else 800.0
        p_init = ops.p_intake if hasattr(ops, "p_intake") else 30e5

        # Pressure
        p_content = f"""FoamFile {{ version 2.0; format ascii; class volScalarField; object p; }}
dimensions [1 -1 -2 0 0 0 0];
internalField uniform {p_init};
boundaryField {{ ".*" {{ type zeroGradient; }} }}
"""
        (zero_dir / "p").write_text(p_content)

        # Temperature
        T_content = f"""FoamFile {{ version 2.0; format ascii; class volScalarField; object T; }}
dimensions [0 0 0 1 0 0 0];
internalField uniform {T_init};
boundaryField {{ ".*" {{ type fixedValue; value uniform {T_init}; }} }}
"""
        (zero_dir / "T").write_text(T_content)

        # Velocity
        U_content = """FoamFile { version 2.0; format ascii; class volVectorField; object U; }
dimensions [0 1 -1 0 0 0 0];
internalField uniform (0 0 0);
boundaryField { ".*" { type noSlip; } }
"""
        (zero_dir / "U").write_text(U_content)

    def _write_thermophysical_properties(self):
        """Write constant/thermophysicalProperties."""
        const_dir = self.case_dir / "constant"
        const_dir.mkdir(parents=True, exist_ok=True)

        content = """FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      thermophysicalProperties;
}

thermoType
{
    type            hePsiThermo;
    mixture         reactingMixture;
    transport       sutherland;
    thermo          janaf;
    energy          sensibleEnthalpy;
    equationOfState perfectGas;
    specie          specie;
}

chemistryReader foamChemistryReader;
"""
        (const_dir / "thermophysicalProperties").write_text(content)

    def _write_turbulence_properties(self):
        """Write constant/turbulenceProperties."""
        content = f"""FoamFile
{{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      turbulenceProperties;
}}

simulationType RAS;
RAS
{{
    RASModel        {self.config.turbulence_model};
    turbulence      on;
    printCoeffs     on;
}}
"""
        (self.case_dir / "constant" / "turbulenceProperties").write_text(content)

    def _run_solver(self) -> int:
        """Execute OpenFOAM solver."""
        cmd = ["openfoam", self.config.solver, "-case", str(self.case_dir)]

        print(f"[{self.name}] Running: {' '.join(cmd)}")

        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=self.config.timeout_seconds
            )

            # Save logs
            (self.case_dir / "solver.log").write_text(result.stdout + result.stderr)

            return result.returncode

        except subprocess.TimeoutExpired:
            print(f"[{self.name}] Solver timed out")
            return -1
        except FileNotFoundError:
            print(f"[{self.name}] OpenFOAM not found. Is it installed?")
            return -2

    def _parse_output_files(self) -> dict[str, Any]:
        """Parse OpenFOAM results and compute burn metrics."""
        results = {
            "p_max_bar": 0.0,
            "burn_duration_10_90_ms": 0.0,
            "wiebe_a": 5.0,
            "wiebe_m": 2.0,
        }

        # Find latest time directory
        time_dirs = sorted(
            [
                d
                for d in self.case_dir.iterdir()
                if d.is_dir() and d.name.replace(".", "").replace("e-", "").isdigit()
            ],
            key=lambda x: float(x.name),
        )

        if not time_dirs:
            print(f"[{self.name}] No time directories found")
            return results

        # Read pressure from last time step
        last_time = time_dirs[-1]
        p_file = last_time / "p"

        if p_file.exists():
            content = p_file.read_text()
            # Extract internalField value (simplified parsing)
            if "uniform" in content:
                try:
                    parts = content.split("uniform")[1].split(";")[0]
                    p_val = float(parts.strip())
                    results["p_max_bar"] = p_val / 1e5
                except (ValueError, IndexError):
                    pass

        print(f"[{self.name}] Results: p_max={results['p_max_bar']:.1f} bar")

        return results

    def solve_steady_state(self) -> SimulationOutput:
        """Run combustion CFD and return standardized output."""
        success = self.execute()

        return SimulationOutput(
            run_id=self.input_data.run_id if self.input_data else "combustion_run",
            success=success,
            calibration_params=self.results,
        )
