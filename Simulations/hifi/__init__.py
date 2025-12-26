"""High-Fidelity Simulation Package.

Contains adapters for external solver bindings:
- CalculiX (FEA structural/contact) via Docker
- OpenFOAM (CFD combustion, port flow, CHT) native

These modules generate training data for surrogates.
"""

from Simulations.hifi.base import ExternalSolverAdapter, ExternalSolverConfig, SolverBackend
from Simulations.hifi.combustion_cfd import CombustionCFDAdapter, CombustionCFDConfig
from Simulations.hifi.conjugate_ht import ConjugateHTAdapter, ConjugateHTConfig
from Simulations.hifi.example_inputs import create_simulation_input
from Simulations.hifi.gear_contact import GearContactConfig, GearContactFEAAdapter
from Simulations.hifi.mesh import GmshMesher, MeshConfig, create_mesh_for_adapter
from Simulations.hifi.port_flow_cfd import PortFlowCFDAdapter, PortFlowCFDConfig
from Simulations.hifi.result_parsers import CalculiXResultParser, OpenFOAMResultParser
from Simulations.hifi.structural_fea import StructuralFEAAdapter, StructuralFEAConfig

__all__ = [
    "ExternalSolverConfig",
    "ExternalSolverAdapter",
    "SolverBackend",
    "GmshMesher",
    "MeshConfig",
    "create_mesh_for_adapter",
    "CalculiXResultParser",
    "OpenFOAMResultParser",
    "create_simulation_input",
    "StructuralFEAAdapter",
    "StructuralFEAConfig",
    "CombustionCFDAdapter",
    "CombustionCFDConfig",
    "ConjugateHTAdapter",
    "ConjugateHTConfig",
    "PortFlowCFDAdapter",
    "PortFlowCFDConfig",
    "GearContactFEAAdapter",
    "GearContactConfig",
]
