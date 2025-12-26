"""Result parsers for external solver output files.

Provides utilities to extract structured data from:
- CalculiX .dat/.frd files
- OpenFOAM postProcessing directories
"""

from pathlib import Path
from typing import Any

import numpy as np

try:
    import meshio

    MESHIO_AVAILABLE = True
except ImportError:
    MESHIO_AVAILABLE = False
    meshio = None


class CalculiXResultParser:
    """Parse CalculiX output files (.dat, .frd)."""

    def __init__(self, case_dir: Path):
        self.case_dir = Path(case_dir)

    def parse_dat(self, filename: str = "model.dat") -> dict[str, Any]:
        """
        Parse CalculiX .dat text output file.

        Extracts:
        - Nodal displacements
        - Element stresses
        - Reaction forces
        """
        dat_file = self.case_dir / filename

        if not dat_file.exists():
            return {"error": f"File not found: {dat_file}"}

        results = {
            "displacements": [],
            "stresses": [],
            "max_displacement": 0.0,
            "max_von_mises": 0.0,
        }

        content = dat_file.read_text()
        lines = content.split("\n")

        current_section = None

        for line in lines:
            line = line.strip()

            # Detect section headers
            if "displacements" in line.lower():
                current_section = "displacements"
                continue
            elif "stresses" in line.lower():
                current_section = "stresses"
                continue
            elif line.startswith("*"):
                current_section = None
                continue

            # Parse data lines
            parts = line.split()
            if len(parts) < 3:
                continue

            try:
                if current_section == "displacements":
                    # Node, U1, U2, U3
                    node = int(parts[0])
                    u = [float(parts[i]) for i in range(1, min(4, len(parts)))]
                    results["displacements"].append({"node": node, "u": u})
                    mag = np.sqrt(sum(x**2 for x in u))
                    results["max_displacement"] = max(results["max_displacement"], mag)

                elif current_section == "stresses":
                    # Element, S11, S22, S33, S12, S13, S23
                    if len(parts) >= 7:
                        elem = int(parts[0])
                        s = [float(parts[i]) for i in range(1, 7)]
                        # von Mises from principal stresses
                        s11, s22, s33, s12, s13, s23 = s
                        vm = np.sqrt(
                            0.5
                            * (
                                (s11 - s22) ** 2
                                + (s22 - s33) ** 2
                                + (s33 - s11) ** 2
                                + 6 * (s12**2 + s13**2 + s23**2)
                            )
                        )
                        results["stresses"].append({"elem": elem, "von_mises": vm})
                        results["max_von_mises"] = max(results["max_von_mises"], vm)
            except (ValueError, IndexError):
                continue

        return results

    def parse_frd(self, filename: str = "model.frd") -> dict[str, Any]:
        """
        Parse CalculiX .frd binary/ASCII results file.

        Uses meshio for robust parsing of node/element data.
        """
        frd_file = self.case_dir / filename

        if not frd_file.exists():
            return {"error": f"File not found: {frd_file}"}

        if not MESHIO_AVAILABLE:
            return {"error": "meshio not installed. Run: pip install meshio"}

        try:
            mesh = meshio.read(str(frd_file))

            results = {
                "n_nodes": len(mesh.points),
                "n_cells": sum(len(c.data) for c in mesh.cells),
                "point_data": {},
                "cell_data": {},
            }

            # Extract point data (displacements, temperatures)
            for name, data in mesh.point_data.items():
                if data.ndim == 1:
                    results["point_data"][name] = {
                        "min": float(np.min(data)),
                        "max": float(np.max(data)),
                        "mean": float(np.mean(data)),
                    }
                elif data.ndim == 2:
                    # Vector field (e.g., U)
                    mag = np.linalg.norm(data, axis=1)
                    results["point_data"][name] = {
                        "max_magnitude": float(np.max(mag)),
                        "mean_magnitude": float(np.mean(mag)),
                    }

            # Extract cell data (stresses)
            for name, data_list in mesh.cell_data.items():
                all_data = np.concatenate(data_list) if data_list else np.array([])
                if len(all_data) > 0:
                    results["cell_data"][name] = {
                        "min": float(np.min(all_data)),
                        "max": float(np.max(all_data)),
                        "mean": float(np.mean(all_data)),
                    }

            return results

        except Exception as e:
            return {"error": f"Failed to parse {frd_file}: {e}"}


class OpenFOAMResultParser:
    """Parse OpenFOAM postProcessing and field data."""

    def __init__(self, case_dir: Path):
        self.case_dir = Path(case_dir)

    def get_time_directories(self) -> list[float]:
        """Get sorted list of time directories."""
        times = []
        for d in self.case_dir.iterdir():
            if d.is_dir():
                try:
                    t = float(d.name)
                    times.append(t)
                except ValueError:
                    continue
        return sorted(times)

    def parse_field_uniform(self, time: float | str, field: str) -> float | None:
        """
        Extract uniform field value from OpenFOAM field file.

        Args:
            time: Time directory name (float or string like "0")
            field: Field name (e.g., "p", "T", "U")
        """
        time_dir = self.case_dir / str(time)
        field_file = time_dir / field

        if not field_file.exists():
            return None

        content = field_file.read_text()

        # Look for "internalField uniform X" pattern
        if "uniform" in content:
            for line in content.split("\n"):
                if "internalField" in line and "uniform" in line:
                    # Extract value after "uniform"
                    parts = line.split("uniform")[-1].strip()
                    # Handle scalar
                    if parts.startswith("("):
                        # Vector - return magnitude
                        parts = parts.strip("();")
                        vals = [float(x) for x in parts.split()]
                        return np.sqrt(sum(v**2 for v in vals))
                    else:
                        # Scalar
                        val = parts.rstrip(";")
                        return float(val)
        return None

    def parse_probes(self, func_name: str = "probes") -> dict[str, np.ndarray]:
        """
        Parse postProcessing/probes data.

        Returns dict of {field_name: array of values per time step}.
        """
        probes_dir = self.case_dir / "postProcessing" / func_name

        if not probes_dir.exists():
            return {}

        results = {}

        # Find all time directories in postProcessing
        for time_dir in sorted(probes_dir.iterdir()):
            if not time_dir.is_dir():
                continue

            # Read each field file
            for field_file in time_dir.iterdir():
                if field_file.suffix or field_file.name.startswith("."):
                    continue

                field_name = field_file.name
                if field_name not in results:
                    results[field_name] = []

                # Parse probe data (time, values...)
                lines = field_file.read_text().strip().split("\n")
                for line in lines:
                    if line.startswith("#"):
                        continue
                    parts = line.split()
                    if len(parts) >= 2:
                        try:
                            values = [float(p) for p in parts]
                            results[field_name].append(values)
                        except ValueError:
                            continue

        # Convert to numpy arrays
        for k, v in results.items():
            results[k] = np.array(v) if v else np.array([])

        return results

    def parse_residuals(self, log_file: str = "solver.log") -> dict[str, list[float]]:
        """
        Extract residual history from solver log.
        """
        log_path = self.case_dir / log_file

        if not log_path.exists():
            return {}

        residuals = {
            "p": [],
            "U": [],
            "T": [],
            "k": [],
            "omega": [],
        }

        content = log_path.read_text()

        for line in content.split("\n"):
            # Look for "Solving for X, Initial residual = Y"
            if "Initial residual" in line:
                for field in residuals.keys():
                    if f"Solving for {field}" in line:
                        try:
                            res_str = line.split("Initial residual = ")[1].split(",")[0]
                            residuals[field].append(float(res_str))
                        except (IndexError, ValueError):
                            continue

        return {k: v for k, v in residuals.items() if v}
