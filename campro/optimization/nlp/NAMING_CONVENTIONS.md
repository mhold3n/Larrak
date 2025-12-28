# Geometry Module Naming Conventions

This file intentionally uses physics/engineering notation that deviates from Python naming conventions.

## Rationale

The following naming patterns are **intentional** and align with standard engineering notation:

### Method Names (PascalCase with underscores)

- `Volume(theta)` → Standard notation: V(θ)
- `dV_dtheta(theta)` → Standard notation: dV/dθ
- `Area_wall(theta)` → Standard notation: A_wall(θ)
- `Area_intake(theta)` → Standard notation: A_intake(θ)
- `Area_exhaust(theta)` → Standard notation: A_exhaust(θ)

### Variable Names (Single uppercase letters or short notation)

- `B` → Bore diameter [m]
- `S` → Stroke length [m]
- `L` → Connecting rod length [m]
- `CR` → Compression ratio
- `R` → Rod-to-crank ratio
- `V_disp` → Displaced volume [m³]
- `V_c` → Clearance volume [m³]
- `A_head` → Head area [m²]
- `A_piston` → Piston area [m²]
- `A_liner` → Liner area [m²]

## Why Not snake_case?

1. **Domain Clarity**: Engineers reading this code expect to see `V` for volume and `A` for area, not `volume` or `area`
2. **Equation Mapping**: Code directly maps to mathematical equations using standard symbols
3. **Industry Standard**: Matches notation in SAE papers, textbooks, and technical specifications
4. **Disambiguation**: `V` clearly means volume; `v` could be velocity

## Linter Configuration

These naming patterns generate linter warnings but should **NOT** be changed. The warnings can be safely ignored as they represent intentional domain-specific notation choices.

To suppress these warnings in your IDE, you can add a configuration to ignore naming convention warnings for this specific file or module.
