"""Material Property Definitions for FEA/CFD."""

from dataclasses import dataclass

@dataclass
class MaterialProperties:
    name: str
    density: float  # kg/m^3
    youngs_modulus: float = 0.0  # Pa
    poisson_ratio: float = 0.0
    thermal_conductivity: float = 0.0  # W/(m*K)
    specific_heat: float = 0.0  # J/(kg*K)
    friction_coeff: float = 0.0

class MaterialLibrary:
    """Authoritative source for material properties (MMPDS/MatWeb)."""
    
    @staticmethod
    def get_aluminum_6061_t6() -> MaterialProperties:
        return MaterialProperties(
            name="Al6061-T6",
            density=2700.0,
            youngs_modulus=69e9,
            poisson_ratio=0.33,
            thermal_conductivity=167.0,
            specific_heat=896.0,
            friction_coeff=0.45 # Dry Al-Steel
        )
    
    @staticmethod
    def get_steel_4340() -> MaterialProperties:
        return MaterialProperties(
            name="Steel-4340",
            density=7850.0,
            youngs_modulus=205e9,
            poisson_ratio=0.29,
            thermal_conductivity=44.5,
            specific_heat=475.0,
            friction_coeff=0.45
        )
