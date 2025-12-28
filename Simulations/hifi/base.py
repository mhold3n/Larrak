"""Base classes for external solver adapters.

Provides common interface for wrapping external FEA/CFD tools:
- CalculiX, Code_Aster, ElmerFEM (FEA)
- OpenFOAM, SU2 (CFD)
- chtMultiRegionFoam (CHT)

The adapter pattern ensures consistent input/output contracts
matching the optimization system's io_schema.
"""

from abc import ABC, abstractmethod
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from Simulations.common.io_schema import SimulationInput, SimulationOutput


class SolverBackend(Enum):
    """Supported external solver backends."""

    CALCULIX = "calculix"
    CODE_ASTER = "code_aster"
    ELMER = "elmer"
    OPENFOAM = "openfoam"
    SU2 = "su2"
    GMSH = "gmsh"  # Meshing only


class ExternalSolverConfig(BaseModel):
    """Configuration for external solver execution."""

    # Solver selection
    backend: SolverBackend = SolverBackend.OPENFOAM

    # Paths
    solver_executable: str = ""
    case_template_dir: str = ""
    working_dir: str = "/tmp/hifi_runs"

    # Execution
    n_procs: int = 4
    timeout_seconds: int = 3600
    cleanup_on_success: bool = True

    # Mesh settings
    mesh_refinement_level: int = 2
    use_cached_mesh: bool = True

    # Convergence
    max_iterations: int = 1000
    residual_tolerance: float = 1e-6

    class Config:
        use_enum_values = True


class ExternalSolverAdapter(ABC):
    """
    Abstract base class for external solver wrappers.

    Lifecycle:
    1. load_input() - Receive SimulationInput from optimization
    2. setup() - Generate mesh, write input files
    3. execute() - Run external solver (subprocess/MPI)
    4. parse_results() - Extract outputs from solver files
    5. post_process() - Return SimulationOutput

    Subclasses must implement:
    - _generate_mesh()
    - _write_input_files()
    - _run_solver()
    - _parse_output_files()
    """

    def __init__(self, name: str, config: ExternalSolverConfig):
        self.name = name
        self.config = config
        self.input_data: SimulationInput | None = None
        self.case_dir: Path | None = None
        self.results: dict[str, Any] = {}
        # Optional CEM client for adaptive learning
        self._cem_client: Any | None = None

    def set_cem_client(self, cem_client: Any) -> None:
        """
        Register CEM client for adaptive learning.

        When set, HiFi results will trigger CEM rule adaptation
        after each solve, enabling the CEM to learn from ground truth.
        """
        self._cem_client = cem_client

    def load_input(self, input_data: SimulationInput):
        """Load simulation input bundle."""
        self.input_data = input_data

    def setup(self):
        """
        Prepare solver case directory.

        1. Create working directory
        2. Generate or load mesh
        3. Write boundary conditions
        4. Write solver control files
        """
        if not self.input_data:
            raise ValueError("No input data loaded. Call load_input() first.")

        # Create case directory
        self.case_dir = Path(self.config.working_dir) / f"{self.name}_{self.input_data.run_id}"
        self.case_dir.mkdir(parents=True, exist_ok=True)

        # Template subclass methods
        self._generate_mesh()
        self._write_input_files()

    @abstractmethod
    def _generate_mesh(self):
        """Generate computational mesh using geometry from input_data."""
        raise NotImplementedError("Subclass must implement _generate_mesh()")

    @abstractmethod
    def _write_input_files(self):
        """Write solver-specific input files (BCs, material props, controls)."""
        raise NotImplementedError("Subclass must implement _write_input_files()")

    @abstractmethod
    def _run_solver(self) -> int:
        """
        Execute external solver via subprocess.

        Returns:
            Exit code from solver (0 = success)
        """
        raise NotImplementedError("Subclass must implement _run_solver()")

    @abstractmethod
    def _parse_output_files(self) -> dict[str, Any]:
        """
        Parse solver output files into structured results.

        Returns:
            Dictionary of scalar and array outputs
        """
        raise NotImplementedError("Subclass must implement _parse_output_files()")

    def execute(self) -> bool:
        """
        Run the external solver.

        Returns:
            True if solver completed successfully
        """
        self.setup()
        exit_code = self._run_solver()

        if exit_code != 0:
            print(f"[{self.name}] Solver failed with exit code {exit_code}")
            return False

        self.results = self._parse_output_files()
        return True

    def solve_steady_state(self) -> SimulationOutput:
        """
        Run solver and return standardized output.

        This is the main entry point matching the BaseSimulation interface.
        Triggers CEM adaptation if a CEM client is registered.
        """
        success = self.execute()

        run_id = self.input_data.run_id if self.input_data else "unknown"

        # Trigger CEM adaptation from HiFi results
        if success and self._cem_client is not None:
            self._trigger_cem_adaptation(run_id)

        return SimulationOutput(
            run_id=run_id,
            success=success,
            calibration_params=self.results,
        )

    def _trigger_cem_adaptation(self, run_id: str) -> None:
        """
        Feed HiFi results back to CEM for adaptive rule learning.

        This is the key integration point that closes the loop between
        ground truth simulations and CEM constraint limits.
        """
        if not hasattr(self._cem_client, "adapt_rules"):
            return

        # Build truth_data in the format expected by adapt_rules
        truth_data = [(self.results, 1.0)]  # (candidate_dict, objective)

        try:
            report = self._cem_client.adapt_rules(truth_data, run_id=run_id)
            if report and getattr(report, "any_adapted", False):
                print(
                    f"[{self.name}] CEM adapted {report.total_rules_adapted} rules from HiFi results"
                )
        except Exception as e:
            # Non-fatal - adaptation is optional
            print(f"[{self.name}] CEM adaptation skipped: {e}")

    def post_process(self) -> dict[str, Any]:
        """Return solver results."""
        return self.results
