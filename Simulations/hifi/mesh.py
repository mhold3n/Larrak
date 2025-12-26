"""Mesh Generation Module using Gmsh.

Provides mesh utilities for high-fidelity simulation adapters:
- 2D axisymmetric meshes for thermal/structural
- 3D tetrahedral meshes for CFD/FEA
- OpenFOAM mesh conversion
- CalculiX mesh export

Requires: gmsh (pip install gmsh, brew install gmsh)
"""

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import gmsh

    GMSH_AVAILABLE = True
except ImportError:
    GMSH_AVAILABLE = False
    gmsh = None

import numpy as np


@dataclass
class MeshConfig:
    """Configuration for mesh generation."""

    element_size_min: float = 0.5e-3  # 0.5mm min
    element_size_max: float = 5e-3  # 5mm max
    refinement_factor: float = 0.3  # For boundary layers
    mesh_algorithm: int = 6  # Frontal-Delaunay
    optimize_netgen: bool = True


class GmshMesher:
    """
    Gmsh-based mesh generator for FEA/CFD cases.

    Supports:
    - 2D quad/tri meshes
    - 3D tet/hex meshes
    - Export to OpenFOAM (foamMesh)
    - Export to CalculiX (.inp)
    """

    def __init__(self, config: MeshConfig | None = None):
        if not GMSH_AVAILABLE:
            raise ImportError("gmsh Python package not installed. Run: pip install gmsh")
        self.config = config or MeshConfig()
        self._initialized = False

    def initialize(self, model_name: str = "model"):
        """Initialize Gmsh session."""
        if self._initialized:
            gmsh.finalize()
        gmsh.initialize()
        gmsh.model.add(model_name)
        gmsh.option.setNumber("General.Terminal", 0)  # Suppress output
        self._initialized = True

    def finalize(self):
        """Clean up Gmsh session."""
        if self._initialized:
            gmsh.finalize()
            self._initialized = False

    def create_cylinder_2d(
        self,
        bore: float,
        height: float,
        wall_thickness: float = 0.01,
        output_path: str | None = None,
    ) -> dict[str, Any]:
        """
        Create 2D axisymmetric cylinder mesh (piston crown section).

        Args:
            bore: Cylinder bore diameter [m]
            height: Section height (piston crown thickness) [m]
            wall_thickness: Wall thickness for thermal analysis [m]
            output_path: Path to save mesh file

        Returns:
            Mesh metadata dict
        """
        self.initialize("cylinder_2d")

        r_inner = bore / 2
        r_outer = r_inner + wall_thickness

        # Create rectangular cross-section (axisymmetric around y-axis)
        # Points: inner-bottom, outer-bottom, outer-top, inner-top
        p1 = gmsh.model.geo.addPoint(0, 0, 0)  # Origin (axis)
        p2 = gmsh.model.geo.addPoint(r_inner, 0, 0)
        p3 = gmsh.model.geo.addPoint(r_outer, 0, 0)
        p4 = gmsh.model.geo.addPoint(r_outer, height, 0)
        p5 = gmsh.model.geo.addPoint(r_inner, height, 0)
        p6 = gmsh.model.geo.addPoint(0, height, 0)

        # Lines
        l1 = gmsh.model.geo.addLine(p1, p2)  # Bottom axis to inner
        l2 = gmsh.model.geo.addLine(p2, p3)  # Bottom inner to outer
        l3 = gmsh.model.geo.addLine(p3, p4)  # Outer wall
        l4 = gmsh.model.geo.addLine(p4, p5)  # Top outer to inner
        l5 = gmsh.model.geo.addLine(p5, p6)  # Top inner to axis
        l6 = gmsh.model.geo.addLine(p6, p1)  # Axis

        # Create surface
        loop = gmsh.model.geo.addCurveLoop([l1, l2, l3, l4, l5, l6])
        surf = gmsh.model.geo.addPlaneSurface([loop])

        gmsh.model.geo.synchronize()

        # Physical groups for boundary conditions
        gmsh.model.addPhysicalGroup(1, [l6], name="axis")
        gmsh.model.addPhysicalGroup(1, [l1, l2], name="bottom")
        gmsh.model.addPhysicalGroup(1, [l3], name="outer_wall")
        gmsh.model.addPhysicalGroup(1, [l4, l5], name="top")
        gmsh.model.addPhysicalGroup(2, [surf], name="domain")

        # Mesh settings
        gmsh.option.setNumber("Mesh.CharacteristicLengthMin", self.config.element_size_min)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", self.config.element_size_max)
        gmsh.option.setNumber("Mesh.Algorithm", self.config.mesh_algorithm)

        # Generate 2D mesh
        gmsh.model.mesh.generate(2)

        if self.config.optimize_netgen:
            gmsh.model.mesh.optimize("Netgen")

        # Get mesh statistics
        nodes = gmsh.model.mesh.getNodes()
        elements = gmsh.model.mesh.getElements()

        result = {
            "n_nodes": len(nodes[0]),
            "n_elements": sum(len(e) for e in elements[1]),
            "type": "2D_axisymmetric",
        }

        if output_path:
            # Use Gmsh v2 format for OpenFOAM compatibility (gmshToFoam)
            gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
            gmsh.write(output_path)
            result["file"] = output_path

        self.finalize()
        return result

    def create_piston_3d(
        self,
        bore: float,
        crown_thickness: float,
        skirt_length: float,
        pin_bore_diameter: float = 0.02,
        output_path: str | None = None,
    ) -> dict[str, Any]:
        """
        Create 3D piston mesh for FEA.

        Args:
            bore: Piston diameter [m]
            crown_thickness: Crown thickness [m]
            skirt_length: Skirt length below crown [m]
            pin_bore_diameter: Wrist pin bore diameter [m]
            output_path: Path to save mesh file

        Returns:
            Mesh metadata dict
        """
        self.initialize("piston_3d")

        r = bore / 2

        # Create cylinder (simplified piston)
        cyl = gmsh.model.occ.addCylinder(0, 0, 0, 0, 0, crown_thickness + skirt_length, r)

        # Pin bore holes (through both sides)
        pin_y = skirt_length / 2
        pin_r = pin_bore_diameter / 2
        pin1 = gmsh.model.occ.addCylinder(-r * 1.1, 0, pin_y, 2.2 * r, 0, 0, pin_r)

        # Boolean subtraction
        gmsh.model.occ.cut([(3, cyl)], [(3, pin1)])

        gmsh.model.occ.synchronize()

        # Mesh settings with finer resolution
        gmsh.option.setNumber("Mesh.CharacteristicLengthMin", self.config.element_size_min)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", self.config.element_size_max)
        gmsh.option.setNumber("Mesh.Algorithm3D", 1)  # Delaunay

        # Generate 3D mesh
        gmsh.model.mesh.generate(3)

        if self.config.optimize_netgen:
            gmsh.model.mesh.optimize("Netgen")

        nodes = gmsh.model.mesh.getNodes()
        elements = gmsh.model.mesh.getElements()

        result = {
            "n_nodes": len(nodes[0]),
            "n_elements": sum(len(e) for e in elements[1]),
            "type": "3D_tet",
        }

        if output_path:
            # Use Gmsh v2 format for OpenFOAM compatibility (gmshToFoam)
            gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
            gmsh.write(output_path)
            result["file"] = output_path

        self.finalize()
        return result

    def create_combustion_chamber(
        self,
        bore: float,
        stroke: float,
        clearance_height: float,
        piston_position: float = 0.0,
        output_path: str | None = None,
    ) -> dict[str, Any]:
        """
        Create 3D combustion chamber mesh for CFD.

        Args:
            bore: Cylinder bore [m]
            stroke: Piston stroke [m]
            clearance_height: TDC clearance height [m]
            piston_position: Piston position from TDC [m]
            output_path: Path to save mesh

        Returns:
            Mesh metadata dict
        """
        self.initialize("combustion_chamber")

        r = bore / 2
        height = clearance_height + piston_position

        # Simple cylindrical chamber
        chamber = gmsh.model.occ.addCylinder(0, 0, 0, 0, 0, height, r)

        gmsh.model.occ.synchronize()

        # Get boundary surfaces for BCs
        surfaces = gmsh.model.occ.getEntities(2)

        # Mesh settings
        gmsh.option.setNumber("Mesh.CharacteristicLengthMin", self.config.element_size_min)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", self.config.element_size_max * 2)

        gmsh.model.mesh.generate(3)

        nodes = gmsh.model.mesh.getNodes()

        result = {
            "n_nodes": len(nodes[0]),
            "type": "3D_tet",
            "volume_m3": np.pi * r**2 * height,
        }

        if output_path:
            gmsh.write(output_path)
            result["file"] = output_path

        self.finalize()
        return result

    def export_calculix(self, msh_file: str, inp_file: str):
        """
        Convert Gmsh .msh to CalculiX .inp format.

        Only exports 3D tetrahedral elements (C3D4). Shell/2D elements
        are skipped to avoid thickness specification issues.

        Args:
            msh_file: Input Gmsh mesh file
            inp_file: Output CalculiX input file
        """
        self.initialize("convert")
        gmsh.open(msh_file)

        # Get nodes
        node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
        node_coords = node_coords.reshape(-1, 3)

        # Get elements (tets = type 4)
        elem_types, elem_tags, elem_nodes = gmsh.model.mesh.getElements()

        # Track which nodes are used by 3D elements
        used_nodes = set()
        tet_elements = []

        for etype, etags, enodes in zip(elem_types, elem_tags, elem_nodes):
            if etype == 4:  # C3D4 tetrahedra only
                nodes_per_elem = 4
                enodes = enodes.reshape(-1, nodes_per_elem)
                for tag, nodes in zip(etags, enodes):
                    tet_elements.append((int(tag), [int(n) for n in nodes]))
                    used_nodes.update(int(n) for n in nodes)

        with open(inp_file, "w") as f:
            f.write("*HEADING\nGenerated by Gmsh\n")

            # Write nodes used by 3D elements
            f.write("*NODE, NSET=NALL\n")
            for i, (tag, coords) in enumerate(zip(node_tags, node_coords)):
                if int(tag) in used_nodes:
                    f.write(f"{int(tag)}, {coords[0]:.10e}, {coords[1]:.10e}, {coords[2]:.10e}\n")

            # Write only tetrahedral elements
            if tet_elements:
                f.write("*ELEMENT, TYPE=C3D4, ELSET=EALL\n")
                for tag, nodes in tet_elements:
                    f.write(f"{tag}, " + ", ".join(str(n) for n in nodes) + "\n")

        self.finalize()

    def convert_to_openfoam(self, msh_file: str, case_dir: str):
        """
        Convert Gmsh mesh to OpenFOAM format.

        Args:
            msh_file: Input Gmsh mesh file
            case_dir: OpenFOAM case directory
        """
        case_path = Path(case_dir)
        case_path.mkdir(parents=True, exist_ok=True)

        # Use gmshToFoam utility
        cmd = ["openfoam", "gmshToFoam", str(msh_file), "-case", str(case_dir)]
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            raise RuntimeError(f"gmshToFoam failed: {result.stderr}")


def create_mesh_for_adapter(
    adapter_type: str, geometry: dict[str, float], output_dir: str
) -> dict[str, Any]:
    """
    Factory function to create appropriate mesh for adapter type.

    Args:
        adapter_type: One of "structural", "combustion", "thermal", "port", "gear"
        geometry: Dict with bore, stroke, etc.
        output_dir: Directory for mesh files

    Returns:
        Mesh metadata including file paths
    """
    mesher = GmshMesher()
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    bore = geometry.get("bore", 0.085)
    stroke = geometry.get("stroke", 0.090)

    if adapter_type == "structural":
        msh_file = str(output_path / "piston.msh")
        result = mesher.create_piston_3d(
            bore=bore,
            crown_thickness=geometry.get("crown_thickness", 0.015),
            skirt_length=geometry.get("skirt_length", 0.05),
            output_path=msh_file,
        )
        # Also export CalculiX format
        inp_file = str(output_path / "piston.inp")
        mesher.export_calculix(msh_file, inp_file)
        result["calculix_file"] = inp_file

    elif adapter_type == "combustion":
        msh_file = str(output_path / "chamber.msh")
        result = mesher.create_combustion_chamber(
            bore=bore,
            stroke=stroke,
            clearance_height=geometry.get("clearance", 0.005),
            output_path=msh_file,
        )

    elif adapter_type == "thermal":
        msh_file = str(output_path / "thermal_2d.msh")
        result = mesher.create_cylinder_2d(
            bore=bore, height=geometry.get("crown_thickness", 0.015), output_path=msh_file
        )

    else:
        raise ValueError(f"Unknown adapter type: {adapter_type}")

    return result
