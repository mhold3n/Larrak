# Silent Fallback Audit Report

**Total Findings**: 183

**Files Affected**: 54

## Summary by Pattern Type

- comment_marker: 183

## Summary by Severity

- UNKNOWN: 183

## Detailed Findings


### /Users/maxholden/Documents/GitHub/Larrak/campro/config/system_builder.py

**Line 244** (comment_marker):
```python
# For now, return a placeholder
```
Context:
```python
 241:             component_instances[name] = component_class(parameters, name=name)
 242: 
 243:         # Create system (this would be implemented in a specific system class)
 244:         # For now, return a placeholder
 245:         log.info(
 246:             f"Created system {self.name} with {len(component_instances)} components",
 247:         )
```


### /Users/maxholden/Documents/GitHub/Larrak/campro/constants.py

**Line 73** (comment_marker):
```python
# If any error occurs, continue to fallback methods
```
Context:
```python
  70:             # If env_manager not available, skip this check
  71:             pass
  72:         except Exception:
  73:             # If any error occurs, continue to fallback methods
  74:             pass
  75: 
  76:         # Priority 2: Check environment variable
```


### /Users/maxholden/Documents/GitHub/Larrak/campro/diagnostics/feasibility.py

**Line 421** (comment_marker):
```python
# Fallback to heuristic in case of solver error
```
Context:
```python
 418: 
 419:         return report
 420:     except Exception:
 421:         # Fallback to heuristic in case of solver error
 422:         return check_feasibility(constraints, bounds)
```


### /Users/maxholden/Documents/GitHub/Larrak/campro/diagnostics/scaling.py

**Line 77** (comment_marker):
```python
# ---- CasADi helpers (stubs – safe to import lazily) -------------------------
```
Context:
```python
  74:     return out
  75: 
  76: 
  77: # ---- CasADi helpers (stubs – safe to import lazily) -------------------------
  78: 
  79: 
  80: def _import_casadi():  # pragma: no cover - lightweight lazy import
```

**Line 194** (comment_marker):
```python
# Fallback for older CasADi builds
```
Context:
```python
 191:         sym_kind = "SX" if isinstance(x, ca.SX) else "MX"
 192:     shape = getattr(x, "shape", None)
 193:     if not shape:
 194:         # Fallback for older CasADi builds
 195:         n = int(getattr(x, "size1", lambda: 0)())
 196:         m = int(getattr(x, "size2", lambda: 1)())
 197:         shape = (n, m)
```


### /Users/maxholden/Documents/GitHub/Larrak/campro/environment/env_manager.py

**Line 54** (comment_marker):
```python
# Fallback: assume current working directory
```
Context:
```python
  51:             import campro
  52:             project_root = Path(campro.__file__).parent.parent
  53:         except ImportError:
  54:             # Fallback: assume current working directory
  55:             project_root = Path.cwd()
  56: 
  57:     if is_local_conda_env_present(project_root):
```

**Line 224** (comment_marker):
```python
# Fallback: Check project CoinHSL directory using hsl_detector
```
Context:
```python
 221:             log.info(f"Found HSL library at: {search_path}")
 222:             return search_path
 223: 
 224:     # Fallback: Check project CoinHSL directory using hsl_detector
 225:     try:
 226:         from campro.environment.hsl_detector import get_hsl_library_path
 227: 
```

**Line 281** (comment_marker):
```python
# Fallback to global environment check
```
Context:
```python
 278:         )
 279:         return False
 280: 
 281:     # Fallback to global environment check
 282:     env_path = get_active_conda_env_path(project_root)
 283:     if env_path is not None:
 284:         log.info(f"Using global conda environment: {env_path}")
```


### /Users/maxholden/Documents/GitHub/Larrak/campro/environment/hsl_detector.py

**Line 30** (comment_marker):
```python
# Fallback: assume current working directory
```
Context:
```python
  27:         import campro
  28:         return Path(campro.__file__).parent.parent
  29:     except ImportError:
  30:         # Fallback: assume current working directory
  31:         return Path.cwd()
  32: 
  33: 
```

**Line 141** (comment_marker):
```python
# Fallback to bin directory
```
Context:
```python
 138:         # Try lib directory first (standard macOS location)
 139:         lib_path = coinhsl_dir / "lib" / "libcoinhsl.dylib"
 140:         if not lib_path.exists():
 141:             # Fallback to bin directory
 142:             lib_path = coinhsl_dir / "bin" / "libcoinhsl.dylib"
 143:     else:  # Linux
 144:         lib_path = coinhsl_dir / "lib" / "libcoinhsl.so"
```

**Line 268** (comment_marker):
```python
# Attempt to create solver with this linear solver
```
Context:
```python
 265: 
 266:         for solver_name in potentially_available:
 267:             try:
 268:                 # Attempt to create solver with this linear solver
 269:                 solver_opts["ipopt.linear_solver"] = solver_name
 270:                 solver_opts["ipopt.print_level"] = 0
 271:                 solver_opts["ipopt.sb"] = "yes"
```


### /Users/maxholden/Documents/GitHub/Larrak/campro/environment/platform_detector.py

**Line 94** (comment_marker):
```python
# Fallback for unknown systems
```
Context:
```python
  91:     elif system == "linux":
  92:         return "conda_env_linux"
  93:     else:
  94:         # Fallback for unknown systems
  95:         return f"conda_env_{system}"
  96: 
  97: 
```


### /Users/maxholden/Documents/GitHub/Larrak/campro/environment/validator.py

**Line 128** (comment_marker):
```python
# Fallback for CasADi builds without nlpsol_plugins attribute
```
Context:
```python
 125:                 ),
 126:             )
 127: 
 128:         # Fallback for CasADi builds without nlpsol_plugins attribute
 129:         try:
 130:             from campro.optimization.ipopt_factory import create_ipopt_solver
 131: 
```

**Line 381** (comment_marker):
```python
# Validate MA27 usage first - fail hard if a non-HSL fallback is detected
```
Context:
```python
 378: 
 379:     log.info("Starting environment validation")
 380: 
 381:     # Validate MA27 usage first - fail hard if a non-HSL fallback is detected
 382:     _validate_ma27_usage()
 383: 
 384:     results: dict[str, Any] = {
```


### /Users/maxholden/Documents/GitHub/Larrak/campro/freepiston/core/chem.py

**Line 185** (comment_marker):
```python
alpha = 2.0  # Temperature exponent
```
Context:
```python
 182: 
 183:     T_ref = 300.0  # K
 184:     p_ref = 1e5  # Pa
 185:     alpha = 2.0  # Temperature exponent
 186:     beta = -0.5  # Pressure exponent
 187: 
 188:     # Equivalence ratio dependence (simplified)
```


### /Users/maxholden/Documents/GitHub/Larrak/campro/freepiston/core/losses.py

**Line 261** (comment_marker):
```python
# Assume gas temperature is same as chamber temperature
```
Context:
```python
 258:     mdot_blowby = Cd * A_blowby * math.sqrt(2.0 * rho_gas * dp)
 259: 
 260:     # Energy loss (simplified)
 261:     # Assume gas temperature is same as chamber temperature
 262:     T_gas = 500.0  # K (simplified)
 263:     cp = 1005.0  # J/(kg K) (simplified)
 264:     E_loss_blowby = mdot_blowby * cp * T_gas
```


### /Users/maxholden/Documents/GitHub/Larrak/campro/freepiston/core/piston.py

**Line 53** (comment_marker):
```python
T_gas: float  # Gas temperature [K]
```
Context:
```python
  50: 
  51:     # Thermodynamic state
  52:     p_gas: float  # Gas pressure [Pa]
  53:     T_gas: float  # Gas temperature [K]
  54:     V_chamber: float  # Chamber volume [m^3]
  55: 
  56: 
```

**Line 497** (comment_marker):
```python
# Based on piston surface area and temperature difference
```
Context:
```python
 494:     mdot_piston = rho_gas * dV_dt
 495: 
 496:     # Heat transfer rate (simplified)
 497:     # Based on piston surface area and temperature difference
 498:     A_surface = math.pi * geometry.bore * geometry.stroke
 499:     h_conv = 100.0  # W/(m^2·K) - simplified
 500:     T_wall = 400.0  # K - simplified
```


### /Users/maxholden/Documents/GitHub/Larrak/campro/freepiston/core/thermo.py

**Line 93** (comment_marker):
```python
# Fallback to constant cp for species without JANAF data
```
Context:
```python
  90:                 )
  91:                 cp_total += frac * cp_species
  92:             else:
  93:                 # Fallback to constant cp for species without JANAF data
  94:                 gamma = 1.4  # Default
  95:                 R_species = R_UNIVERSAL / self.components[species]["W"]
  96:                 cp_species = gamma * R_species / (gamma - 1.0)
```

**Line 144** (comment_marker):
```python
# Entropy change due to temperature at constant pressure
```
Context:
```python
 141:         p_ref = 1e5  # 1 bar in Pa
 142:         R = self.gas_constant()
 143: 
 144:         # Entropy change due to temperature at constant pressure
 145:         s_T = 0.0
 146:         if abs(T - 298.15) > 1e-6:
 147:             # Integrate cp/T from 298.15 K to T
```

**Line 159** (comment_marker):
```python
# Entropy change due to pressure at constant temperature
```
Context:
```python
 156:                 dT = T_points[i + 1] - T_points[i]
 157:                 s_T += (cp_mid / T_mid) * dT
 158: 
 159:         # Entropy change due to pressure at constant temperature
 160:         s_p = -R * math.log(p / p_ref)
 161: 
 162:         return s_T + s_p
```


### /Users/maxholden/Documents/GitHub/Larrak/campro/freepiston/core/xfer.py

**Line 59** (comment_marker):
```python
swirl_factor = 1.0 + C2 * 1.0  # TODO: Add proper swirl ratio calculation
```
Context:
```python
  56:     h_base = C1 * (p**0.8) * (T**-0.53) * (w**0.8) * (B**-0.2)
  57: 
  58:     # Swirl enhancement (simplified - assumes w_m/sw = 1.0 for now)
  59:     swirl_factor = 1.0 + C2 * 1.0  # TODO: Add proper swirl ratio calculation
  60: 
  61:     return h_base * swirl_factor
  62: 
```


### /Users/maxholden/Documents/GitHub/Larrak/campro/freepiston/net1d/bc.py

**Line 269** (comment_marker):
```python
u_in = 0.0  # placeholder
```
Context:
```python
 266: def inlet_bc(p_in: float, T_in: float, rho_in: float) -> tuple[float, float, float]:
 267:     """Simple inlet boundary condition for 1D solver (legacy)."""
 268:     # Conservative variables at inlet
 269:     u_in = 0.0  # placeholder
 270:     rhoE_in = rho_in * (1.5 * 287.0 * T_in)  # simple ideal gas
 271:     return (rho_in, rho_in * u_in, rhoE_in)
 272: 
```

**Line 277** (comment_marker):
```python
return (0.0, 0.0, 0.0)  # placeholder
```
Context:
```python
 274: def outlet_bc(p_out: float) -> tuple[float, float, float]:
 275:     """Simple outlet boundary condition for 1D solver (legacy)."""
 276:     # Simple outlet: extrapolate from interior
 277:     return (0.0, 0.0, 0.0)  # placeholder
 278: 
 279: 
 280: def get_boundary_condition(method: str = "non_reflecting"):
```


### /Users/maxholden/Documents/GitHub/Larrak/campro/freepiston/net1d/stepper.py

**Line 983** (comment_marker):
```python
# Fallback: assume first and last cells are near walls
```
Context:
```python
 980:         )
 981:         return min_wall_distance < wall_distance_threshold
 982: 
 983:     # Fallback: assume first and last cells are near walls
 984:     return cell_index == 0 or cell_index == len(mesh.cell_centers) - 1
 985: 
 986: 
```

**Line 1006** (comment_marker):
```python
# Fallback: use cell size
```
Context:
```python
1003:         )
1004:         return min_wall_distance
1005: 
1006:     # Fallback: use cell size
1007:     if hasattr(mesh, "cell_size"):
1008:         return mesh.cell_size[cell_index]
1009: 
```

**Line 1010** (comment_marker):
```python
# Default fallback
```
Context:
```python
1007:     if hasattr(mesh, "cell_size"):
1008:         return mesh.cell_size[cell_index]
1009: 
1010:     # Default fallback
1011:     return 0.001  # 1 mm
1012: 
1013: 
```


### /Users/maxholden/Documents/GitHub/Larrak/campro/freepiston/net1d/wall.py

**Line 34** (comment_marker):
```python
T_wall: float = 300.0  # Wall temperature [K]
```
Context:
```python
  31: 
  32:     # Compressibility effects
  33:     M_wall: float = 0.0  # Wall Mach number
  34:     T_wall: float = 300.0  # Wall temperature [K]
  35:     T_ref: float = 300.0  # Reference temperature [K]
  36: 
  37:     # Heat transfer
```

**Line 35** (comment_marker):
```python
T_ref: float = 300.0  # Reference temperature [K]
```
Context:
```python
  32:     # Compressibility effects
  33:     M_wall: float = 0.0  # Wall Mach number
  34:     T_wall: float = 300.0  # Wall temperature [K]
  35:     T_ref: float = 300.0  # Reference temperature [K]
  36: 
  37:     # Heat transfer
  38:     h_conv: float = 100.0  # Convective heat transfer coefficient [W/(m^2·K)]
```

**Line 55** (comment_marker):
```python
return 0.0  # placeholder
```
Context:
```python
  52: def wall_shear_stress(rho: float, u: float, mu: float, y_plus: float) -> float:
  53:     """Wall shear stress tau_w (placeholder)."""
  54:     _ = (rho, u, mu, y_plus)
  55:     return 0.0  # placeholder
  56: 
  57: 
  58: def calculate_y_plus(
```

**Line 212** (comment_marker):
```python
# Temperature difference
```
Context:
```python
 209:     Returns:
 210:         Wall heat flux [W/m^2]
 211:     """
 212:     # Temperature difference
 213:     dT = T - T_wall
 214: 
 215:     if dT <= 0:
```

**Line 299** (comment_marker):
```python
# Temperature ratio
```
Context:
```python
 296:     Returns:
 297:         Dictionary with compressibility corrections
 298:     """
 299:     # Temperature ratio
 300:     T_ratio = T / T_wall
 301: 
 302:     # Compressibility factor
```

**Line 310** (comment_marker):
```python
# Temperature correction
```
Context:
```python
 307:         # Compressible
 308:         compressibility_factor = 1.0 + 0.2 * M**2
 309: 
 310:     # Temperature correction
 311:     temperature_factor = T_ratio**0.5
 312: 
 313:     # Combined correction
```

**Line 394** (comment_marker):
```python
# Temperature change due to heat transfer
```
Context:
```python
 391:     m_wall = rho_wall * thickness * area  # kg
 392:     C_wall = m_wall * cp_wall  # J/K
 393: 
 394:     # Temperature change due to heat transfer
 395:     dT_wall = q_wall * area * dt / C_wall
 396: 
 397:     # New wall temperature
```

**Line 397** (comment_marker):
```python
# New wall temperature
```
Context:
```python
 394:     # Temperature change due to heat transfer
 395:     dT_wall = q_wall * area * dt / C_wall
 396: 
 397:     # New wall temperature
 398:     T_wall_new = T_wall_old + dT_wall
 399: 
 400:     return T_wall_new
```

**Line 436** (comment_marker):
```python
# Temperature drop across layer (simplified)
```
Context:
```python
 433:         R_layer = thickness / (conductivity * area)  # K/W
 434:         total_resistance += R_layer
 435: 
 436:         # Temperature drop across layer (simplified)
 437:         dT_layer = 0.0  # Simplified for now
 438:         layer_temperatures.append(layer_temperatures[-1] - dT_layer)
 439: 
```

**Line 771** (comment_marker):
```python
# Fallback if mesh doesn't expose dx as expected
```
Context:
```python
 768:         # ALEMesh: dx is ndarray
 769:         y_dist = float(np.min(mesh.dx)) * 0.5
 770:     except Exception:
 771:         # Fallback if mesh doesn't expose dx as expected
 772:         y_dist = 1e-3
 773:     y_dist = max(y_dist, 1e-9)
 774: 
```


### /Users/maxholden/Documents/GitHub/Larrak/campro/freepiston/opt/__init__.py

**Line 58** (comment_marker):
```python
# Legacy imports for backward compatibility
```
Context:
```python
  55: )
  56: from .cons import comprehensive_path_constraints
  57: 
  58: # Legacy imports for backward compatibility
  59: from .driver import solve_cycle, solve_cycle_adaptive, solve_cycle_robust
  60: from .ipopt_solver import IPOPTOptions, IPOPTResult, IPOPTSolver
  61: from .nlp import build_collocation_nlp, build_collocation_nlp_with_1d_coupling
```

**Line 101** (comment_marker):
```python
# Legacy imports for backward compatibility
```
Context:
```python
  98:     "get_preset_config",
  99:     "create_engine_config",
 100:     "create_optimization_scenario",
 101:     # Legacy imports for backward compatibility
 102:     "solve_cycle",
 103:     "solve_cycle_robust",
 104:     "solve_cycle_adaptive",
```


### /Users/maxholden/Documents/GitHub/Larrak/campro/freepiston/opt/colloc.py

**Line 89** (comment_marker):
```python
# Fallback (should not reach)
```
Context:
```python
  86:         raise NotImplementedError(
  87:             "Gauss–Legendre implemented only for C=2 in this draft",
  88:         )
  89:     # Fallback (should not reach)
  90:     raise NotImplementedError(f"Unknown collocation kind: {kind}")
```


### /Users/maxholden/Documents/GitHub/Larrak/campro/freepiston/opt/cons.py

**Line 65** (comment_marker):
```python
# Temperature constraints
```
Context:
```python
  62:             lbg_path.append(p_min_pa)
  63:             ubg_path.append(p_max_pa)
  64: 
  65:     # Temperature constraints
  66:     if "temperature" in states:
  67:         for T in states["temperature"]:
  68:             g_path.append(T)
```

**Line 525** (comment_marker):
```python
# Check temperature bounds
```
Context:
```python
 522:             log.error("Invalid pressure bounds: p_min >= p_max")
 523:             return False
 524: 
 525:     # Check temperature bounds
 526:     if "T_min" in bounds and "T_max" in bounds:
 527:         if bounds["T_min"] >= bounds["T_max"]:
 528:             log.error("Invalid temperature bounds: T_min >= T_max")
```

**Line 563** (comment_marker):
```python
# Temperature bounds [K]
```
Context:
```python
 560:         # Pressure bounds [MPa] (normalized to match variable scaling reference)
 561:         "p_min": 0.01,  # 0.01 MPa = 1e4 Pa
 562:         "p_max": 10.0,  # 10.0 MPa = 1e7 Pa
 563:         # Temperature bounds [K]
 564:         "T_min": 200.0,
 565:         "T_max": 2000.0,
 566:         # Density bounds [kg/m^3]
```

**Line 594** (comment_marker):
```python
# Wall temperature bounds [K]
```
Context:
```python
 591:         "E_max": 100.0,
 592:         # Combustion bounds [kJ] (normalized from J to bring energy terms to O(1-10) range)
 593:         "Q_comb_max": 10.0,  # 10.0 kJ = 10000.0 J
 594:         # Wall temperature bounds [K]
 595:         "T_wall_min": 250.0,
 596:         "T_wall_max": 800.0,
 597:         # Scavenging bounds
```


### /Users/maxholden/Documents/GitHub/Larrak/campro/freepiston/opt/driver.py

**Line 178** (comment_marker):
```python
# Minimal residual evaluation at a nominal state (placeholder)
```
Context:
```python
 175: 
 176:     nlp_build_start = time.time()
 177: 
 178:     # Minimal residual evaluation at a nominal state (placeholder)
 179:     mech = MechState(x_L=0.05, v_L=0.0, x_R=0.15, v_R=0.0)
 180:     gas = {"rho": 1.2, "E": 2.5e5, "p": 1.0e5}
 181:     res = cv_residual(mech, gas, {"geom": P.get("geom", {}), "flows": {}})
```

**Line 306** (comment_marker):
```python
# This is a fallback - ideally nlp should be a dict
```
Context:
```python
 303:             )
 304:         else:
 305:             # If nlp is a CasADi Function, convert to dict format first
 306:             # This is a fallback - ideally nlp should be a dict
 307:             reporter.warning("NLP is not a dict, attempting to use unscaled NLP")
 308:             nlp_scaled = nlp
 309:             scale_f = 1.0  # No objective scaling if NLP not a dict
```

**Line 676** (comment_marker):
```python
# Temperatures
```
Context:
```python
 673:     # Densities
 674:     rho_min = bounds.get("rho_min", 0.1)
 675:     rho_max = bounds.get("rho_max", 10.0)
 676:     # Temperatures
 677:     T_min = bounds.get("T_min", 200.0)
 678:     T_max = bounds.get("T_max", 2000.0)
 679:     # Valve areas (lower bound is 0)
```

**Line 717** (comment_marker):
```python
T_initial = combustion_cfg.get("initial_temperature_K", 300.0)  # Initial temperature from user
```
Context:
```python
 714:     # Extract combustion model inputs
 715:     cycle_time = combustion_cfg.get("cycle_time_s", 1.0)  # Cycle time from user
 716:     p_initial = combustion_cfg.get("initial_pressure_Pa", 1e5)  # Initial pressure from user
 717:     T_initial = combustion_cfg.get("initial_temperature_K", 300.0)  # Initial temperature from user
 718:     ignition_timing = combustion_cfg.get("ignition_initial_s", None)  # Ignition timing from user
 719: 
 720:     # Calculate initial gas density from ideal gas law
```

**Line 738** (comment_marker):
```python
# At maximum expansion (back to V_max), pressure and temperature drop
```
Context:
```python
 735:     rho_compressed = p_compressed / (R * T_compressed)
 736: 
 737:     # Expansion phase: isentropic expansion (simplified - no combustion heat addition yet)
 738:     # At maximum expansion (back to V_max), pressure and temperature drop
 739:     p_expanded = p_compressed * ((V_compressed / V_max) ** gamma)
 740:     T_expanded = T_compressed * ((V_compressed / V_max) ** (gamma - 1))
 741:     rho_expanded = p_expanded / (R * T_expanded)
```

**Line 806** (comment_marker):
```python
# Early expansion: high pressure/temperature
```
Context:
```python
 803:             rho = rho_compressed * (1.0 - compression_progress * 0.1)  # Slight variation
 804:             T = T_compressed * (1.0 - compression_progress * 0.1)
 805:         elif phase < 0.75:
 806:             # Early expansion: high pressure/temperature
 807:             expansion_progress = (phase - 0.5) / 0.25
 808:             rho = rho_compressed * (1.0 - expansion_progress * 0.5)
 809:             T = T_compressed * (1.0 - expansion_progress * 0.3)
```

**Line 816** (comment_marker):
```python
# Apply to variables with NaN guards and interior point fallbacks
```
Context:
```python
 813:             rho = rho_expanded + (rho_initial - rho_expanded) * expansion_progress
 814:             T = T_expanded + (T_initial - T_expanded) * expansion_progress
 815: 
 816:         # Apply to variables with NaN guards and interior point fallbacks
 817:         if idx < n_vars:
 818:             x0[idx] = x_L if np.isfinite(x_L) else xL_interior
 819:         if idx + 1 < n_vars:
```

**Line 864** (comment_marker):
```python
x0[i] = 0.0  # Fallback
```
Context:
```python
 861:             elif var_idx_in_group == 5:  # T
 862:                 x0[i] = T_interior
 863:             else:
 864:                 x0[i] = 0.0  # Fallback
 865: 
 866:     log.info(
 867:         f"Generated physics-based initial guess for {n_vars} variables using "
```

**Line 1511** (comment_marker):
```python
# Gas temperature bounds
```
Context:
```python
1508:     p_min = constraints.get("p_min", 1e3)
1509:     p_max = constraints.get("p_max", 1e7)
1510: 
1511:     # Gas temperature bounds
1512:     T_min = constraints.get("T_min", 200.0)
1513:     T_max = constraints.get("T_max", 3000.0)
1514: 
```

**Line 1537** (comment_marker):
```python
lbx[i + 5] = T_min  # T (temperature)
```
Context:
```python
1534:             lbx[i + 4] = 0.1  # rho (density)
1535:             ubx[i + 4] = 100.0
1536:         if i + 5 < n_vars:
1537:             lbx[i + 5] = T_min  # T (temperature)
1538:             ubx[i + 5] = T_max
1539: 
1540: 
```


### /Users/maxholden/Documents/GitHub/Larrak/campro/freepiston/opt/ipopt_solver.py

**Line 291** (comment_marker):
```python
# Fallback to plugin list if available
```
Context:
```python
 288:                 self.ipopt_available = True
 289:                 return
 290:             except Exception:
 291:                 # Fallback to plugin list if available
 292:                 if hasattr(ca, "nlpsol_plugins"):
 293:                     try:
 294:                         plugins = ca.nlpsol_plugins()
```


### /Users/maxholden/Documents/GitHub/Larrak/campro/freepiston/opt/nlp.py

**Line 300** (comment_marker):
```python
T_out = T_c  # Assume outlet at chamber temperature
```
Context:
```python
 297: 
 298:     # Enthalpy of inlet/outlet streams
 299:     T_in = thermo.get("T_in", 300.0)
 300:     T_out = T_c  # Assume outlet at chamber temperature
 301:     h_in = cp * T_in
 302:     h_out = cp * T_out
 303: 
```

**Line 380** (comment_marker):
```python
# Gas properties (temperature-dependent)
```
Context:
```python
 377:     """
 378:     ca = _import_casadi()
 379: 
 380:     # Gas properties (temperature-dependent)
 381:     R = 287.0  # J/(kg K) - gas constant for air
 382:     cp_ref = 1005.0  # J/(kg K) - reference specific heat
 383:     cv_ref = cp_ref / ca.fmax(
```

**Line 387** (comment_marker):
```python
# Temperature-dependent specific heat (simplified linear model)
```
Context:
```python
 384:         gamma, CASADI_PHYSICS_EPSILON,
 385:     )  # J/(kg K) - reference specific heat at constant volume
 386: 
 387:     # Temperature-dependent specific heat (simplified linear model)
 388:     # In practice, this would use JANAF polynomial fits
 389:     cp = cp_ref * (1.0 + 0.0001 * (T - 300.0))  # Linear temperature dependence
 390:     cv = cp / ca.fmax(gamma, CASADI_PHYSICS_EPSILON)
```

**Line 389** (comment_marker):
```python
cp = cp_ref * (1.0 + 0.0001 * (T - 300.0))  # Linear temperature dependence
```
Context:
```python
 386: 
 387:     # Temperature-dependent specific heat (simplified linear model)
 388:     # In practice, this would use JANAF polynomial fits
 389:     cp = cp_ref * (1.0 + 0.0001 * (T - 300.0))  # Linear temperature dependence
 390:     cv = cp / ca.fmax(gamma, CASADI_PHYSICS_EPSILON)
 391: 
 392:     # Set default inlet/outlet temperatures
```

**Line 392** (comment_marker):
```python
# Set default inlet/outlet temperatures
```
Context:
```python
 389:     cp = cp_ref * (1.0 + 0.0001 * (T - 300.0))  # Linear temperature dependence
 390:     cv = cp / ca.fmax(gamma, CASADI_PHYSICS_EPSILON)
 391: 
 392:     # Set default inlet/outlet temperatures
 393:     if T_in is None:
 394:         T_in = 300.0  # K - ambient temperature
 395:     if T_out is None:
```

**Line 394** (comment_marker):
```python
T_in = 300.0  # K - ambient temperature
```
Context:
```python
 391: 
 392:     # Set default inlet/outlet temperatures
 393:     if T_in is None:
 394:         T_in = 300.0  # K - ambient temperature
 395:     if T_out is None:
 396:         T_out = T  # K - assume outlet at chamber temperature
 397: 
```

**Line 396** (comment_marker):
```python
T_out = T  # K - assume outlet at chamber temperature
```
Context:
```python
 393:     if T_in is None:
 394:         T_in = 300.0  # K - ambient temperature
 395:     if T_out is None:
 396:         T_out = T  # K - assume outlet at chamber temperature
 397: 
 398:     # Mass balance: d(rho*V)/dt = mdot_in - mdot_out
 399:     dm_dt = mdot_in - mdot_out
```

**Line 971** (comment_marker):
```python
rho_k1[i] += h * grid.weights[j] * 0.0  # Placeholder
```
Context:
```python
 968:         for j in range(C):
 969:             for i in range(n_cells):
 970:                 # Simplified gas state update
 971:                 rho_k1[i] += h * grid.weights[j] * 0.0  # Placeholder
 972:                 u_k1[i] += h * grid.weights[j] * 0.0  # Placeholder
 973:                 E_k1[i] += h * grid.weights[j] * 0.0  # Placeholder
 974: 
```

**Line 972** (comment_marker):
```python
u_k1[i] += h * grid.weights[j] * 0.0  # Placeholder
```
Context:
```python
 969:             for i in range(n_cells):
 970:                 # Simplified gas state update
 971:                 rho_k1[i] += h * grid.weights[j] * 0.0  # Placeholder
 972:                 u_k1[i] += h * grid.weights[j] * 0.0  # Placeholder
 973:                 E_k1[i] += h * grid.weights[j] * 0.0  # Placeholder
 974: 
 975:         # New state variables
```

**Line 973** (comment_marker):
```python
E_k1[i] += h * grid.weights[j] * 0.0  # Placeholder
```
Context:
```python
 970:                 # Simplified gas state update
 971:                 rho_k1[i] += h * grid.weights[j] * 0.0  # Placeholder
 972:                 u_k1[i] += h * grid.weights[j] * 0.0  # Placeholder
 973:                 E_k1[i] += h * grid.weights[j] * 0.0  # Placeholder
 974: 
 975:         # New state variables
 976:         xL_k = ca.SX.sym(f"xL_{k + 1}")
```

**Line 1359** (comment_marker):
```python
# Optional dynamic wall temperature state
```
Context:
```python
1356:     else:
1357:         dt_real = None
1358: 
1359:     # Optional dynamic wall temperature state
1360:     dynamic_wall = bool(walls_cfg.get("dynamic", False))
1361:     Cw = float(walls_cfg.get("capacitance", 0.0))  # [J/K]
1362:     T_wall_const = float(geometry.get("T_wall", 400.0))
```

**Line 1845** (comment_marker):
```python
# Temperature constraints
```
Context:
```python
1842:             lbg += [p_min_pa]
1843:             ubg += [p_max_pa]
1844: 
1845:             # Temperature constraints
1846:             T_kj = T_colloc[j]
1847:             g += [T_kj]  # Temperature constraint
1848:             lbg += [bounds.get("T_min", 200.0)]
```

**Line 1847** (comment_marker):
```python
g += [T_kj]  # Temperature constraint
```
Context:
```python
1844: 
1845:             # Temperature constraints
1846:             T_kj = T_colloc[j]
1847:             g += [T_kj]  # Temperature constraint
1848:             lbg += [bounds.get("T_min", 200.0)]
1849:             ubg += [bounds.get("T_max", 2000.0)]
1850: 
```

**Line 1883** (comment_marker):
```python
# Combustion timing constraints (optional placeholders)
```
Context:
```python
1880:                 lbg += [-bounds.get("a_max", 1000.0), -bounds.get("a_max", 1000.0)]
1881:                 ubg += [bounds.get("a_max", 1000.0), bounds.get("a_max", 1000.0)]
1882: 
1883:     # Combustion timing constraints (optional placeholders)
1884:     for k in range(K):
1885:         for j in range(C):
1886:             Q_comb_kj = Q_comb_stage[j]
```


### /Users/maxholden/Documents/GitHub/Larrak/campro/freepiston/opt/obj.py

**Line 158** (comment_marker):
```python
# Fallback to Simpson's rule for other point counts
```
Context:
```python
 155:         weights = [5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0]
 156:         points = [-math.sqrt(3.0 / 5.0), 0.0, math.sqrt(3.0 / 5.0)]
 157:     else:
 158:         # Fallback to Simpson's rule for other point counts
 159:         return indicated_work_simpson(p_series, V_series)
 160: 
 161:     W_ind = 0.0
```

**Line 490** (comment_marker):
```python
# This requires 1D model - placeholder for now
```
Context:
```python
 487:         )
 488: 
 489:     # 4. Scavenging Quality (uniformity of fresh charge distribution)
 490:     # This requires 1D model - placeholder for now
 491:     objectives["scavenging_uniformity"] = weights.get("uniformity", 0.5) * 0.0
 492: 
 493:     # 5. Blow-Down Efficiency (exhaust gas removal)
```


### /Users/maxholden/Documents/GitHub/Larrak/campro/freepiston/validation/config_validator.py

**Line 348** (comment_marker):
```python
# Check if temperature bounds are reasonable for the thermodynamics
```
Context:
```python
 345:                     "Maximum pressure may result in very high forces",
 346:                 )
 347: 
 348:         # Check if temperature bounds are reasonable for the thermodynamics
 349:         if "gamma" in thermo and "T_max" in bounds:
 350:             gamma = thermo["gamma"]
 351:             T_max = bounds["T_max"]
```

**Line 353** (comment_marker):
```python
# Check if temperature is reasonable for the gas
```
Context:
```python
 350:             gamma = thermo["gamma"]
 351:             T_max = bounds["T_max"]
 352: 
 353:             # Check if temperature is reasonable for the gas
 354:             if gamma < 1.3 and T_max > 1500:
 355:                 results["warnings"].append("High temperature for low gamma gas")
 356:             elif gamma > 1.4 and T_max < 500:
```


### /Users/maxholden/Documents/GitHub/Larrak/campro/freepiston/validation/cross_fidelity.py

**Line 223** (comment_marker):
```python
# This is a placeholder - in practice, you'd solve the full piston dynamics
```
Context:
```python
 220:             gas["T"] = gas["p"] / (gas["rho"] * 287.0)  # Ideal gas law
 221: 
 222:         # Update piston positions (simple dynamics)
 223:         # This is a placeholder - in practice, you'd solve the full piston dynamics
 224:         mech.x_L += dt * mech.v_L
 225:         mech.x_R += dt * mech.v_R
 226: 
```

**Line 303** (comment_marker):
```python
# Simple 1D time step (placeholder)
```
Context:
```python
 300: 
 301:     # Time integration
 302:     for i in range(1, n_steps + 1):
 303:         # Simple 1D time step (placeholder)
 304:         # In practice, this would use the full 1D gas dynamics solver
 305:         U = _simple_1d_time_step(U, dt, dx, problem_params)
 306: 
```

**Line 312** (comment_marker):
```python
# Compute pressure and temperature
```
Context:
```python
 309:         u_avg = np.mean(U[:, 1] / U[:, 0])
 310:         E_avg = np.mean(U[:, 2] / U[:, 0])
 311: 
 312:         # Compute pressure and temperature
 313:         p_avg = (1.4 - 1) * rho_avg * E_avg
 314:         T_avg = p_avg / (rho_avg * 287.0)
 315: 
```

**Line 344** (comment_marker):
```python
# This is a placeholder implementation
```
Context:
```python
 341:     problem_params: dict[str, Any],
 342: ) -> np.ndarray:
 343:     """Simple 1D time step (placeholder)."""
 344:     # This is a placeholder implementation
 345:     # In practice, this would use the full 1D gas dynamics solver with HLLC flux
 346: 
 347:     n_cells = U.shape[0]
```

**Line 416** (comment_marker):
```python
# Temperature error
```
Context:
```python
 413:         error_metrics["pressure_relative"] = np.max(pressure_relative_error)
 414:         error_metrics["pressure_absolute"] = np.max(pressure_error)
 415: 
 416:     # Temperature error
 417:     if validation_params.validate_temperature:
 418:         temperature_error = np.abs(
 419:             comparison.temperature_1d - comparison.temperature_0d,
```


### /Users/maxholden/Documents/GitHub/Larrak/campro/freepiston/validation/physics_validator.py

**Line 120** (comment_marker):
```python
# Use property with fallback to dict access
```
Context:
```python
 117:     ) -> None:
 118:         """Validate mass conservation."""
 119:         try:
 120:             # Use property with fallback to dict access
 121:             states = getattr(solution, "states", {})
 122:             if not states:
 123:                 # Fallback to direct dict access
```

**Line 123** (comment_marker):
```python
# Fallback to direct dict access
```
Context:
```python
 120:             # Use property with fallback to dict access
 121:             states = getattr(solution, "states", {})
 122:             if not states:
 123:                 # Fallback to direct dict access
 124:                 states = solution.data.get("states", {})
 125: 
 126:             if not states:
```

**Line 178** (comment_marker):
```python
# Use property with fallback to dict access
```
Context:
```python
 175:     ) -> None:
 176:         """Validate energy conservation."""
 177:         try:
 178:             # Use property with fallback to dict access
 179:             states = getattr(solution, "states", {})
 180:             if not states:
 181:                 # Fallback to direct dict access
```

**Line 181** (comment_marker):
```python
# Fallback to direct dict access
```
Context:
```python
 178:             # Use property with fallback to dict access
 179:             states = getattr(solution, "states", {})
 180:             if not states:
 181:                 # Fallback to direct dict access
 182:                 states = solution.data.get("states", {})
 183: 
 184:             if not states:
```

**Line 205** (comment_marker):
```python
# Calculate from temperature and mass
```
Context:
```python
 202:                 # Use total energy directly
 203:                 energies = states["E"]
 204:             else:
 205:                 # Calculate from temperature and mass
 206:                 T = states["T"]
 207:                 rho = states.get("rho", [1.0] * len(T))
 208:                 V = states.get("V", [1.0] * len(T))
```

**Line 243** (comment_marker):
```python
# Use property with fallback to dict access
```
Context:
```python
 240:     ) -> None:
 241:         """Validate momentum conservation."""
 242:         try:
 243:             # Use property with fallback to dict access
 244:             states = getattr(solution, "states", {})
 245:             if not states:
 246:                 # Fallback to direct dict access
```

**Line 246** (comment_marker):
```python
# Fallback to direct dict access
```
Context:
```python
 243:             # Use property with fallback to dict access
 244:             states = getattr(solution, "states", {})
 245:             if not states:
 246:                 # Fallback to direct dict access
 247:                 states = solution.data.get("states", {})
 248: 
 249:             if not states:
```

**Line 306** (comment_marker):
```python
# Use property with fallback to dict access
```
Context:
```python
 303:     ) -> None:
 304:         """Validate entropy increase (second law of thermodynamics)."""
 305:         try:
 306:             # Use property with fallback to dict access
 307:             states = getattr(solution, "states", {})
 308:             if not states:
 309:                 # Fallback to direct dict access
```

**Line 309** (comment_marker):
```python
# Fallback to direct dict access
```
Context:
```python
 306:             # Use property with fallback to dict access
 307:             states = getattr(solution, "states", {})
 308:             if not states:
 309:                 # Fallback to direct dict access
 310:                 states = solution.data.get("states", {})
 311: 
 312:             if not states:
```

**Line 321** (comment_marker):
```python
# Check if we have temperature and pressure data
```
Context:
```python
 318:                 )
 319:                 return
 320: 
 321:             # Check if we have temperature and pressure data
 322:             if "T" not in states or "p" not in states:
 323:                 results["warnings"].append(
 324:                     "Missing temperature or pressure data for entropy validation",
```

**Line 373** (comment_marker):
```python
# Use property with fallback to dict access
```
Context:
```python
 370:     ) -> None:
 371:         """Validate thermodynamic consistency."""
 372:         try:
 373:             # Use property with fallback to dict access
 374:             states = getattr(solution, "states", {})
 375:             if not states:
 376:                 # Fallback to direct dict access
```

**Line 376** (comment_marker):
```python
# Fallback to direct dict access
```
Context:
```python
 373:             # Use property with fallback to dict access
 374:             states = getattr(solution, "states", {})
 375:             if not states:
 376:                 # Fallback to direct dict access
 377:                 states = solution.data.get("states", {})
 378: 
 379:             if not states:
```

**Line 441** (comment_marker):
```python
# Use property with fallback to dict access
```
Context:
```python
 438:     ) -> None:
 439:         """Calculate additional physics metrics."""
 440:         try:
 441:             # Use property with fallback to dict access
 442:             states = getattr(solution, "states", {})
 443:             if not states:
 444:                 # Fallback to direct dict access
```

**Line 444** (comment_marker):
```python
# Fallback to direct dict access
```
Context:
```python
 441:             # Use property with fallback to dict access
 442:             states = getattr(solution, "states", {})
 443:             if not states:
 444:                 # Fallback to direct dict access
 445:                 states = solution.data.get("states", {})
 446: 
 447:             if not states:
```


### /Users/maxholden/Documents/GitHub/Larrak/campro/freepiston/validation/solution_validator.py

**Line 131** (comment_marker):
```python
# This is a placeholder for future implementation when the Solution structure
```
Context:
```python
 128: 
 129:         # For now, skip detailed state/control validation since the current Solution
 130:         # structure doesn't include states and controls in the expected format
 131:         # This is a placeholder for future implementation when the Solution structure
 132:         # is updated to include states and controls
 133:         results["warnings"].append(
 134:             "Detailed state/control validation not implemented for current Solution structure",
```

**Line 179** (comment_marker):
```python
# Check iteration count (use property with fallback)
```
Context:
```python
 176: 
 177:         results["convergence"] = success
 178: 
 179:         # Check iteration count (use property with fallback)
 180:         iterations = getattr(solution, "iterations", 0)
 181:         if iterations == 0:
 182:             # Fallback to metadata
```

**Line 182** (comment_marker):
```python
# Fallback to metadata
```
Context:
```python
 179:         # Check iteration count (use property with fallback)
 180:         iterations = getattr(solution, "iterations", 0)
 181:         if iterations == 0:
 182:             # Fallback to metadata
 183:             iterations = solution.meta.get("optimization", {}).get("iterations", 0)
 184: 
 185:         max_iter = self.config.get("optimization", {}).get("max_iterations", 1000)
```

**Line 189** (comment_marker):
```python
# Check objective function value (use property with fallback)
```
Context:
```python
 186:         if iterations >= max_iter:
 187:             results["warnings"].append(f"Maximum iterations reached: {iterations}")
 188: 
 189:         # Check objective function value (use property with fallback)
 190:         objective_value = getattr(solution, "objective_value", float("inf"))
 191:         if objective_value == float("inf"):
 192:             # Fallback to metadata
```

**Line 192** (comment_marker):
```python
# Fallback to metadata
```
Context:
```python
 189:         # Check objective function value (use property with fallback)
 190:         objective_value = getattr(solution, "objective_value", float("inf"))
 191:         if objective_value == float("inf"):
 192:             # Fallback to metadata
 193:             objective_value = solution.meta.get("optimization", {}).get(
 194:                 "f_opt", float("inf"),
 195:             )
```

**Line 224** (comment_marker):
```python
# Temperature constraints
```
Context:
```python
 221:                 elif p > self.default_limits["pressure_max"]:
 222:                     violations.append(f"High pressure at step {i}: {p:.0f} Pa")
 223: 
 224:         # Temperature constraints
 225:         if "temperature" in states:
 226:             temperatures = states["temperature"]
 227:             for i, T in enumerate(temperatures):
```

**Line 327** (comment_marker):
```python
# Check if performance metrics are available (use property with fallback)
```
Context:
```python
 324:         self, solution: Solution, results: dict[str, Any],
 325:     ) -> None:
 326:         """Validate performance constraints."""
 327:         # Check if performance metrics are available (use property with fallback)
 328:         perf_metrics = getattr(solution, "performance_metrics", {})
 329:         if not perf_metrics:
 330:             # Fallback to direct dict access
```

**Line 330** (comment_marker):
```python
# Fallback to direct dict access
```
Context:
```python
 327:         # Check if performance metrics are available (use property with fallback)
 328:         perf_metrics = getattr(solution, "performance_metrics", {})
 329:         if not perf_metrics:
 330:             # Fallback to direct dict access
 331:             perf_metrics = solution.meta.get("performance_metrics", {})
 332: 
 333:         if perf_metrics:
```

**Line 382** (comment_marker):
```python
# Temperature metrics
```
Context:
```python
 379:                 / len(pressures),
 380:             )
 381: 
 382:         # Temperature metrics
 383:         if "temperature" in states:
 384:             temperatures = states["temperature"]
 385:             metrics["temperature_max"] = max(temperatures)
```


### /Users/maxholden/Documents/GitHub/Larrak/campro/freepiston/zerod/cv.py

**Line 84** (comment_marker):
```python
T: float  # Temperature [K]
```
Context:
```python
  81: 
  82:     # Basic thermodynamic state
  83:     rho: float  # Density [kg/m^3]
  84:     T: float  # Temperature [K]
  85:     p: float  # Pressure [Pa]
  86:     u: float  # Internal energy [J/kg]
  87:     h: float  # Enthalpy [J/kg]
```

**Line 641** (comment_marker):
```python
# Calculate new temperature (assuming constant cp)
```
Context:
```python
 638:     cp = mix_props["cp"]
 639:     R = mix_props["R"]
 640: 
 641:     # Calculate new temperature (assuming constant cp)
 642:     T_new = (u_new + state.p / rho_new) / cp
 643: 
 644:     # Calculate new pressure
```


### /Users/maxholden/Documents/GitHub/Larrak/campro/litvin/optimization.py

**Line 584** (comment_marker):
```python
np.zeros(n_samples),  # Placeholder position - not used for indices
```
Context:
```python
 581:     # Get section indices for this theta array
 582:     section_indices = get_section_boundaries(
 583:         theta_full,
 584:         np.zeros(n_samples),  # Placeholder position - not used for indices
 585:         section_boundaries,
 586:     )
 587: 
```

**Line 785** (comment_marker):
```python
# Fallback: try to use motion object (will fail if multiprocessing is enabled)
```
Context:
```python
 782:             "theta_deg or position not available in config. "
 783:             "Falling back to using motion object directly (may fail with multiprocessing)."
 784:         )
 785:         # Fallback: try to use motion object (will fail if multiprocessing is enabled)
 786:         evaluate_batch = partial(
 787:             _evaluate_gear_combination_batch,
 788:             pa_lo=pa_lo,
```

**Line 832** (comment_marker):
```python
# CRITICAL: Check platform FIRST before attempting multiprocessing
```
Context:
```python
 829:         logger.info(f"Using {n_threads} threads for work-item level optimization")
 830: 
 831:     # Choose executor based on configuration
 832:     # CRITICAL: Check platform FIRST before attempting multiprocessing
 833:     # On Windows, skip multiprocessing entirely to avoid warnings
 834:     use_multiprocessing = getattr(config, "use_multiprocessing", False)
 835: 
```

**Line 843** (comment_marker):
```python
# Only attempt multiprocessing on Unix-like systems
```
Context:
```python
 840:     executor_class = ProcessPoolExecutor if use_multiprocessing else ThreadPoolExecutor
 841:     executor_kwargs: dict[str, Any] = {"max_workers": n_threads}
 842: 
 843:     # Only attempt multiprocessing on Unix-like systems
 844:     if use_multiprocessing:
 845:         # Unix-like systems: attempt fork context
 846:         global _fork_warning_logged, _fork_warning_lock
```

**Line 845** (comment_marker):
```python
# Unix-like systems: attempt fork context
```
Context:
```python
 842: 
 843:     # Only attempt multiprocessing on Unix-like systems
 844:     if use_multiprocessing:
 845:         # Unix-like systems: attempt fork context
 846:         global _fork_warning_logged, _fork_warning_lock
 847:         try:
 848:             mp_context = mp.get_context("fork")
```

**Line 933** (comment_marker):
```python
# No valid results for this section - use default fallback
```
Context:
```python
 930:     # For each section, find the combination with minimum objective
 931:     for section_name, results in section_results_dict.items():
 932:         if not results:
 933:             # No valid results for this section - use default fallback
 934:             section_results[section_name] = {
 935:                 "ratio": 2.0,  # Default 2:1 ratio
 936:                 "ring_teeth": 40,
```

**Line 1180** (comment_marker):
```python
# Fallback: use closest to weighted ratio
```
Context:
```python
1177:                 best_planet = zp
1178: 
1179:     if best_ring is None:
1180:         # Fallback: use closest to weighted ratio
1181:         integration_logger.warning("No integer pair found satisfying constraint, using closest to weighted ratio")
1182:         for zr in config.ring_teeth_candidates:
1183:             for zp in config.planet_teeth_candidates:
```

**Line 1292** (comment_marker):
```python
# Fallback: use previous point
```
Context:
```python
1289:             if len(phi_vals) > 0:
1290:                 seed = 0.5 * (phi_vals[-1] + phi_vals[0]) if len(phi_vals) > 0 else seed
1291:             else:
1292:                 # Fallback: use previous point
1293:                 seed = phi_vals[-1] if len(phi_vals) > 0 else seed
1294:         phi = _newton_solve_phi(flank, kin, theta, seed) or seed
1295:         phi_vals.append(phi)
```

**Line 1326** (comment_marker):
```python
# Create CasADi function for objective (placeholder - not used in hybrid approach)
```
Context:
```python
1323:     # Store reference for final validation
1324:     objective_with_physics = objective_function_with_physics
1325: 
1326:     # Create CasADi function for objective (placeholder - not used in hybrid approach)
1327:     # obj_func = ca.Function('obj', [phi], [ca.SX.sym('obj_val')])
1328: 
1329:     # For CasADi: use smoothness penalty as proxy, validate with physics
```


### /Users/maxholden/Documents/GitHub/Larrak/campro/litvin/section_analysis.py

**Line 135** (comment_marker):
```python
# Fallback to linear interpolation
```
Context:
```python
 132:             ca_value = brentq(residual, theta_min, theta_max, xtol=0.1)
 133:             return ca_value
 134:         except ValueError:
 135:             # Fallback to linear interpolation
 136:             idx_below = np.where(mass_fraction < target_fraction)[0]
 137:             idx_above = np.where(mass_fraction > target_fraction)[0]
 138:             if len(idx_below) > 0 and len(idx_above) > 0:
```


### /Users/maxholden/Documents/GitHub/Larrak/campro/optimization/cam_ring_optimizer.py

**Line 218** (comment_marker):
```python
# Initialize combustion parameters for legacy fallback only
```
Context:
```python
 215: 
 216:             ca_markers = primary_data.get("ca_markers")
 217: 
 218:             # Initialize combustion parameters for legacy fallback only
 219:             if ca_markers is None and self.constraints.combustion_params is None:
 220:                 from campro.utils.progress_logger import ProgressLogger
 221: 
```

**Line 218** (comment_marker):
```python
# Initialize combustion parameters for legacy fallback only
```
Context:
```python
 215: 
 216:             ca_markers = primary_data.get("ca_markers")
 217: 
 218:             # Initialize combustion parameters for legacy fallback only
 219:             if ca_markers is None and self.constraints.combustion_params is None:
 220:                 from campro.utils.progress_logger import ProgressLogger
 221: 
```


### /Users/maxholden/Documents/GitHub/Larrak/campro/optimization/cam_ring_processing.py

**Line 103** (comment_marker):
```python
# (e.g., legacy physics models). It is NOT part of the primary motion-law contract.
```
Context:
```python
 100:     # NOTE: Per-degree to per-second conversion for downstream modules only
 101:     # The primary optimization path uses per-degree units exclusively.
 102:     # This conversion is ONLY for downstream modules that explicitly require per-second units
 103:     # (e.g., legacy physics models). It is NOT part of the primary motion-law contract.
 104:     if len(velocity) > 0 and velocity_units in ("mm/deg", "m/deg"):
 105:         # Check if conversion is needed (if module requires mm/s)
 106:         needs_per_second = secondary_constraints.get("require_per_second_units", False)
```

**Line 212** (comment_marker):
```python
# This is a placeholder for more sophisticated optimization
```
Context:
```python
 209:     param_bounds = secondary_constraints.get("parameter_bounds", {})
 210: 
 211:     # For now, use the basic processing and add optimization logic
 212:     # This is a placeholder for more sophisticated optimization
 213:     results = process_linear_to_ring_follower(
 214:         primary_data,
 215:         secondary_constraints,
```


### /Users/maxholden/Documents/GitHub/Larrak/campro/optimization/casadi_motion_optimizer.py

**Line 950** (comment_marker):
```python
# Require duration_angle_deg - no fallback to default 360.0
```
Context:
```python
 947:         ValueError
 948:             If duration_angle_deg is missing from constraints.
 949:         """
 950:         # Require duration_angle_deg - no fallback to default 360.0
 951:         duration_angle_deg = constraints.get("duration_angle_deg")
 952:         if duration_angle_deg is None:
 953:             raise ValueError(
```


### /Users/maxholden/Documents/GitHub/Larrak/campro/optimization/casadi_problem_spec.py

**Line 205** (comment_marker):
```python
# Require duration_angle_deg - no fallback to default 360.0
```
Context:
```python
 202:             data.get("collocation_method", "legendre"),
 203:         )
 204: 
 205:         # Require duration_angle_deg - no fallback to default 360.0
 206:         duration_angle_deg = data.get("duration_angle_deg")
 207:         if duration_angle_deg is None:
 208:             raise ValueError(
```


### /Users/maxholden/Documents/GitHub/Larrak/campro/optimization/casadi_unified_flow.py

**Line 391** (comment_marker):
```python
# Fallback to default
```
Context:
```python
 388:             elif isinstance(manual_schedule, Iterable):
 389:                 schedule_source = tuple(max(self.settings.min_segments, int(seg)) for seg in manual_schedule)
 390:             else:
 391:                 # Fallback to default
 392:                 schedule_source = self.settings.coarse_resolution_segments
 393:         else:
 394:             schedule_source = self.settings.coarse_resolution_segments
```

**Line 566** (comment_marker):
```python
# Fallback
```
Context:
```python
 563:                 return max(self.settings.min_segments, int(first_item))
 564:         except (StopIteration, TypeError, ValueError):
 565:             pass
 566:         # Fallback
 567:         return self.settings.min_segments
 568: 
 569:     def _normalize_ladder(self, ladder: Iterable[int] | None) -> tuple[int, ...] | None:
```

**Line 596** (comment_marker):
```python
# Require duration_angle_deg - no fallback
```
Context:
```python
 593:         ValueError
 594:             If duration_angle_deg is missing or invalid.
 595:         """
 596:         # Require duration_angle_deg - no fallback
 597:         duration_angle_deg = constraints.get("duration_angle_deg")
 598:         if duration_angle_deg is None:
 599:             raise ValueError(
```

**Line 742** (comment_marker):
```python
# If failed, try fallback (when implemented)
```
Context:
```python
 739:         # Try direct collocation first
 740:         result = self.optimize_phase1(constraints, targets, **kwargs)
 741: 
 742:         # If failed, try fallback (when implemented)
 743:         if not result.successful:
 744:             message = (
 745:                 "Direct collocation failed and multiple shooting fallback "
```


### /Users/maxholden/Documents/GitHub/Larrak/campro/optimization/collocation.py

**Line 281** (comment_marker):
```python
# otherwise fall back to any legacy field on constraints.
```
Context:
```python
 278:             )
 279: 
 280:             # Get motion type preference: prefer explicit kwarg (from GUI/settings),
 281:             # otherwise fall back to any legacy field on constraints.
 282:             motion_type_str = (
 283:                 kwargs.get("motion_type")
 284:                 if isinstance(kwargs, dict) and "motion_type" in kwargs
```


### /Users/maxholden/Documents/GitHub/Larrak/campro/optimization/crank_center_optimizer.py

**Line 1110** (comment_marker):
```python
# Run Python-based optimization (safe fallback)
```
Context:
```python
1107:         """
1108:         log.info("Running optimization in CasADi physics validation mode")
1109: 
1110:         # Run Python-based optimization (safe fallback)
1111:         log.info("Running Python physics optimization...")
1112:         python_result = minimize(
1113:             objective,
```

**Line 1170** (comment_marker):
```python
# Defensive check for crank_angle key with fallback to theta
```
Context:
```python
1167:         unified_fn = create_unified_physics()
1168: 
1169:         # Prepare inputs for CasADi evaluation
1170:         # Defensive check for crank_angle key with fallback to theta
1171:         theta_vec = motion_law_data.get("crank_angle")
1172:         if theta_vec is None:
1173:             theta_vec = motion_law_data.get("theta")
```

**Line 1248** (comment_marker):
```python
# TODO: Add actual comparison with Python results and tolerance checking
```
Context:
```python
1245:         log.info(f"  Litvin objective: {casadi_validation['litvin_objective']:.6f}")
1246:         log.info(f"  Litvin closure: {casadi_validation['litvin_closure']:.6f}")
1247: 
1248:         # TODO: Add actual comparison with Python results and tolerance checking
1249:         log.info(
1250:             "Validation mode: Using Python results for optimization (CasADi for comparison only)",
1251:         )
```


### /Users/maxholden/Documents/GitHub/Larrak/campro/optimization/freepiston_phase1_adapter.py

**Line 589** (comment_marker):
```python
# Feature flag to allow fallback (default: use free-piston motion)
```
Context:
```python
 586:         K = int(meta_nlp.get("K", 20)) if isinstance(meta_nlp, dict) else 20
 587:         C = int(meta_nlp.get("C", 1)) if isinstance(meta_nlp, dict) else 1
 588: 
 589:         # Feature flag to allow fallback (default: use free-piston motion)
 590:         use_freepiston_motion = os.environ.get("PHASE1_USE_FREEPISTON_MOTION", "1").lower() not in _FALSEY
 591: 
 592:         try:
```

**Line 604** (comment_marker):
```python
# Fallback: uniform within element
```
Context:
```python
 601:                 try:
 602:                     nodes = getattr(grid_obj, "nodes", None)
 603:                     if nodes is None:
 604:                         # Fallback: uniform within element
 605:                         nodes = np.linspace(1.0 / (C + 1), 1.0, C, endpoint=True)
 606:                     else:
 607:                         nodes = np.asarray(nodes, dtype=float)
```

**Line 668** (comment_marker):
```python
# Cylinder pressure from densities and temperatures at collocation (ideal gas)
```
Context:
```python
 665:                 velocity_mm_per_s = v_follow_mps * 1000.0
 666:                 acceleration_mm_per_s2 = a_follow_mps2 * 1000.0
 667: 
 668:                 # Cylinder pressure from densities and temperatures at collocation (ideal gas)
 669:                 R_gas = 287.0
 670:                 # Initial + collocation for rho,T if available; else length-match to time with zeros
 671:                 rho_vals = x_opt_vec[den_idx] if den_idx else np.array([])
```


### /Users/maxholden/Documents/GitHub/Larrak/campro/optimization/grid.py

**Line 80** (comment_marker):
```python
# Fallback to finite differences
```
Context:
```python
  77:         # Check uniformity
  78:         d = np.diff(theta)
  79:         if not np.allclose(d, d.mean(), rtol=1e-3, atol=1e-6):
  80:             # Fallback to finite differences
  81:             return np.gradient(values, theta)
  82:         L = 2.0 * np.pi
  83:         k = np.fft.fftfreq(n, d=(L / n) / (2.0 * np.pi))  # cycles per 2π
```

**Line 162** (comment_marker):
```python
# Fallback: linear resample
```
Context:
```python
 159:         solves (Phi^T W Phi) c = Phi^T W f, returning c as values on to_theta.
 160:         """
 161:         if not _HAVE_SCIPY:
 162:             # Fallback: linear resample
 163:             return GridMapper.periodic_linear_resample(from_theta, from_values, to_theta)
 164:         # Build basis evaluation matrix Phi_{i,j} = L_j(from_theta_i)
 165:         n_to = len(to_theta)
```

**Line 236** (comment_marker):
```python
# Fallback coarse operators via linear interp matrices (dense sampling)
```
Context:
```python
 233:             P_u2g = np.linalg.pinv(Phi)
 234:             P_g2u = Phi
 235:             return P_u2g, P_g2u
 236:         # Fallback coarse operators via linear interp matrices (dense sampling)
 237:         # Construct P_u2g by solving n_to independent linear systems built by impulses at to nodes
 238:         P_g2u = np.zeros((n_from, n_to))
 239:         eye = np.eye(n_to)
```


### /Users/maxholden/Documents/GitHub/Larrak/campro/optimization/ipopt_factory.py

**Line 47** (comment_marker):
```python
_DEFAULT_LINEAR_SOLVER = "ma27"  # Fallback
```
Context:
```python
  44:             else:
  45:                 _DEFAULT_LINEAR_SOLVER = available[0]
  46:         else:
  47:             _DEFAULT_LINEAR_SOLVER = "ma27"  # Fallback
  48:     except Exception:
  49:         _DEFAULT_LINEAR_SOLVER = "ma27"  # Fallback
  50: 
```

**Line 49** (comment_marker):
```python
_DEFAULT_LINEAR_SOLVER = "ma27"  # Fallback
```
Context:
```python
  46:         else:
  47:             _DEFAULT_LINEAR_SOLVER = "ma27"  # Fallback
  48:     except Exception:
  49:         _DEFAULT_LINEAR_SOLVER = "ma27"  # Fallback
  50: 
  51:     return _DEFAULT_LINEAR_SOLVER
  52: 
```


### /Users/maxholden/Documents/GitHub/Larrak/campro/optimization/motion.py

**Line 103** (comment_marker):
```python
# Fallback simple resample via numpy interp for now; advanced mappers chosen from grid_spec later
```
Context:
```python
 100:                 "velocity": velocity,
 101:                 "acceleration": acceleration,
 102:             }
 103:         # Fallback simple resample via numpy interp for now; advanced mappers chosen from grid_spec later
 104:         import numpy as _np
 105: 
 106:         def _per_resample(th, vals, tgt):
```


### /Users/maxholden/Documents/GitHub/Larrak/campro/optimization/motion_law.py

**Line 140** (comment_marker):
```python
# Compatibility aliases for legacy/tests expecting different names
```
Context:
```python
 137:             raise ValueError("All arrays must have the same length")
 138:         # No additional validation here; compatibility helpers below
 139: 
 140:     # Compatibility aliases for legacy/tests expecting different names
 141:     @property
 142:     def theta(self) -> np.ndarray:
 143:         return self.cam_angle
```


### /Users/maxholden/Documents/GitHub/Larrak/campro/optimization/solver_selection.py

**Line 61** (comment_marker):
```python
# Fallback to MA27 if no solvers detected
```
Context:
```python
  58: 
  59:             self._available_solvers = detect_available_solvers(test_runtime=True)
  60:             if not self._available_solvers:
  61:                 # Fallback to MA27 if no solvers detected
  62:                 log.warning("No HSL solvers detected; defaulting to MA27")
  63:                 self._available_solvers = ["ma27"]
  64:         except ImportError:
```

**Line 100** (comment_marker):
```python
# Fallback if MA27 not available (shouldn't happen)
```
Context:
```python
  97:                     f"Solver selection for phase '{phase}': MA27 (small problem: {n_vars} vars)"
  98:                 )
  99:                 return SolverType.MA27
 100:             # Fallback if MA27 not available (shouldn't happen)
 101:             log.warning("MA27 not available; this should not happen")
 102:             return SolverType.MA27
 103: 
```

**Line 112** (comment_marker):
```python
# Fallback to MA27
```
Context:
```python
 109:                 )
 110:                 return SolverType.MA57
 111:             else:
 112:                 # Fallback to MA27
 113:                 log.debug(
 114:                     f"Solver selection for phase '{phase}': MA27 (MA57 not available, {n_vars} vars)"
 115:                 )
```

**Line 141** (comment_marker):
```python
# Fallback to MA27
```
Context:
```python
 138:                 )
 139:                 return SolverType.MA77
 140: 
 141:             # Fallback to MA27
 142:             log.debug(
 143:                 f"Solver selection for phase '{phase}': MA27 (no large-problem solvers available, {n_vars} vars)"
 144:             )
```


### /Users/maxholden/Documents/GitHub/Larrak/campro/optimization/thermal_efficiency_adapter.py

**Line 30** (comment_marker):
```python
# Expose patchable names for tests; real imports are attempted during setup
```
Context:
```python
  27: 
  28: log = get_logger(__name__)
  29: 
  30: # Expose patchable names for tests; real imports are attempted during setup
  31: ComplexMotionLawOptimizer = None  # type: ignore
  32: OptimizationConfig = None  # type: ignore
  33: 
```

**Line 450** (comment_marker):
```python
# Check temperature limits
```
Context:
```python
 447:                 )
 448:                 return False
 449: 
 450:             # Check temperature limits
 451:             max_temperature = complex_result.performance_metrics.get(
 452:                 "max_temperature", 0.0,
 453:             )
```


### /Users/maxholden/Documents/GitHub/Larrak/campro/optimization/unified_framework.py

**Line 81** (comment_marker):
```python
# Legacy/Universal-grid controls (predate CasADi ladder system):
```
Context:
```python
  78:     # Collocation settings (when using collocation methods)
  79:     collocation_degree: int = 3
  80:     collocation_tolerance: float = 1e-6
  81:     # Legacy/Universal-grid controls (predate CasADi ladder system):
  82:     # Universal grid size (GUI-controlled). Used for downstream comparisons and invariant checks.
  83:     # NOTE: This does NOT influence the CasADi ladder's segment counts. The ladder uses
  84:     # casadi_coarse_segments and casadi_resolution_ladder for its adaptive refinement.
```

**Line 98** (comment_marker):
```python
# Legacy/Universal-grid controls (predate CasADi ladder system):
```
Context:
```python
  95:     # Primary phase discretization parameters (for experimentation and debugging)
  96:     primary_min_segments: int = 10  # Minimum number of collocation segments K
  97:     primary_refinement_factor: int = 4  # K ≈ n_points / refinement_factor
  98:     # Legacy/Universal-grid controls (predate CasADi ladder system):
  99:     # Mapper method controls how solutions are resampled onto the universal grid for
 100:     # invariant checks. This mapping existed long before the CasADi flow.
 101:     # Options: linear, pchip, barycentric, projection
```

**Line 111** (comment_marker):
```python
# Note: This is a temporary global toggle controlled by the GUI. In a future iteration we will
```
Context:
```python
 108:     grid_family: str = "uniform"  # e.g., uniform, LGL, Radau, Chebyshev
 109:     grid_segments: int = 1
 110:     # Shared collocation method selection for all modules (GUI dropdown: 'legendre', 'radau', 'lobatto').
 111:     # Note: This is a temporary global toggle controlled by the GUI. In a future iteration we will
 112:     # allow granular per-stage method selection (primary/secondary/tertiary) and per-stage degrees.
 113:     collocation_method: str = "legendre"
 114: 
```

**Line 111** (comment_marker):
```python
# Note: This is a temporary global toggle controlled by the GUI. In a future iteration we will
```
Context:
```python
 108:     grid_family: str = "uniform"  # e.g., uniform, LGL, Radau, Chebyshev
 109:     grid_segments: int = 1
 110:     # Shared collocation method selection for all modules (GUI dropdown: 'legendre', 'radau', 'lobatto').
 111:     # Note: This is a temporary global toggle controlled by the GUI. In a future iteration we will
 112:     # allow granular per-stage method selection (primary/secondary/tertiary) and per-stage degrees.
 113:     collocation_method: str = "legendre"
 114: 
```

**Line 134** (comment_marker):
```python
# Phase-1: constant operating temperature (free-piston idealization)
```
Context:
```python
 131:     require_thermal_efficiency: bool = False
 132:     # Phase-1: constant load model (free piston against generator)
 133:     constant_load_value: float = 1.0
 134:     # Phase-1: constant operating temperature (free-piston idealization)
 135:     constant_temperature_K: float = 900.0
 136: 
 137:     # CasADi physics validation mode settings
```

**Line 159** (comment_marker):
```python
# PR template parameters (geometry-informed pressure ratio template)
```
Context:
```python
 156:     pressure_guard_epsilon: float = 1e-3
 157:     pressure_guard_lambda: float = 1e4
 158: 
 159:     # PR template parameters (geometry-informed pressure ratio template)
 160:     pr_template_expansion_efficiency: float = 0.85  # Target expansion efficiency (0-1)
 161:     pr_template_peak_scale: float = 1.5  # Peak PR scaling factor relative to baseline
 162:     pr_template_use_template: bool = True  # Use explicit template instead of seed-derived PR
```

**Line 162** (comment_marker):
```python
pr_template_use_template: bool = True  # Use explicit template instead of seed-derived PR
```
Context:
```python
 159:     # PR template parameters (geometry-informed pressure ratio template)
 160:     pr_template_expansion_efficiency: float = 0.85  # Target expansion efficiency (0-1)
 161:     pr_template_peak_scale: float = 1.5  # Peak PR scaling factor relative to baseline
 162:     pr_template_use_template: bool = True  # Use explicit template instead of seed-derived PR
 163: 
 164:     # Secondary collocation tracking weight (golden profile influence). The GUI can expose this as a
 165:     # user-tunable numeric field to adjust how strongly secondary tracks the golden motion profile.
```

**Line 894** (comment_marker):
```python
# Optional overrides for constant load and temperature from UI/input
```
Context:
```python
 891:             # Also convert to seconds for consistency
 892:             omega_avg = 360.0 / max(self.data.cycle_time, 1e-9)
 893:             self.data.injector_delay_s = float(injector_delay_deg) / omega_avg
 894:         # Optional overrides for constant load and temperature from UI/input
 895:         if "constant_load_value" in input_data:
 896:             try:
 897:                 self.settings.constant_load_value = float(
```

**Line 1414** (comment_marker):
```python
# Fallback to workload target if cycle work not available
```
Context:
```python
1411:                 p_load_pa_ref = cycle_work_seed__ / max(area_m2__ * stroke_m__, 1e-12)
1412:                 p_load_kpa_ref__ = p_load_pa_ref / 1000.0
1413:             elif workload_target_j__ and area_m2__ > 0.0:
1414:                 # Fallback to workload target if cycle work not available
1415:                 p_load_pa_ref = workload_target_j__ / max(area_m2__ * stroke_m__, 1e-12)
1416:                 p_load_kpa_ref__ = p_load_pa_ref / 1000.0
1417: 
```

**Line 1418** (comment_marker):
```python
# Compute PR reference: use geometry-informed template or seed-derived
```
Context:
```python
1415:                 p_load_pa_ref = workload_target_j__ / max(area_m2__ * stroke_m__, 1e-12)
1416:                 p_load_kpa_ref__ = p_load_pa_ref / 1000.0
1417: 
1418:             # Compute PR reference: use geometry-informed template or seed-derived
1419:             use_template = getattr(self.settings, "pr_template_use_template", True)
1420: 
1421:             if use_template:
```

**Line 1422** (comment_marker):
```python
# Compute geometry-informed PR template
```
Context:
```python
1419:             use_template = getattr(self.settings, "pr_template_use_template", True)
1420: 
1421:             if use_template:
1422:                 # Compute geometry-informed PR template
1423:                 from campro.physics.pr_template import compute_pr_template
1424: 
1425:                 # Calculate compression ratio from geometry
```

**Line 1435** (comment_marker):
```python
# Get template parameters from settings
```
Context:
```python
1432:                 # Compute bore from area
1433:                 bore_mm = np.sqrt(float(geom.area_mm2) * 4.0 / np.pi)
1434: 
1435:                 # Get template parameters from settings
1436:                 expansion_efficiency_target = float(
1437:                     getattr(self.settings, "pr_template_expansion_efficiency", 0.85)
1438:                 )
```

**Line 1441** (comment_marker):
```python
# Generate PR template
```
Context:
```python
1438:                 )
1439:                 pr_peak_scale = float(getattr(self.settings, "pr_template_peak_scale", 1.5))
1440: 
1441:                 # Generate PR template
1442:                 pi_ref__ = compute_pr_template(
1443:                     theta=theta_seed,
1444:                     stroke_mm=float(self.data.stroke),
```

**Line 1462** (comment_marker):
```python
# Fallback to seed-derived PR (legacy behavior)
```
Context:
```python
1459:                 p_cyl_seed__ = np.asarray(p_cyl_seed_raw__, dtype=float)
1460:                 p_bounce_seed__ = np.asarray(out0__.get("p_bounce"), dtype=float)
1461:             else:
1462:                 # Fallback to seed-derived PR (legacy behavior)
1463:                 p_cyl_seed_raw__ = out0__.get("p_cyl")
1464:                 if p_cyl_seed_raw__ is None:
1465:                     p_cyl_seed_raw__ = out0__.get("p_comb")
```

**Line 1462** (comment_marker):
```python
# Fallback to seed-derived PR (legacy behavior)
```
Context:
```python
1459:                 p_cyl_seed__ = np.asarray(p_cyl_seed_raw__, dtype=float)
1460:                 p_bounce_seed__ = np.asarray(out0__.get("p_bounce"), dtype=float)
1461:             else:
1462:                 # Fallback to seed-derived PR (legacy behavior)
1463:                 p_cyl_seed_raw__ = out0__.get("p_cyl")
1464:                 if p_cyl_seed_raw__ is None:
1465:                     p_cyl_seed_raw__ = out0__.get("p_comb")
```

**Line 1504** (comment_marker):
```python
# Fallback to scaling workload_target by fuel multiplier if available
```
Context:
```python
1501:                         p_load_pa_case = cycle_work_case / max(area_m2__ * stroke_m__, 1e-12)
1502:                         p_load_kpa_case = p_load_pa_case / 1000.0
1503:                     elif workload_target_j__ and area_m2__ > 0.0:
1504:                         # Fallback to scaling workload_target by fuel multiplier if available
1505:                         fuel_mult = float(out.get("fuel_multiplier", 1.0))
1506:                         if fuel_mult > 0.0:
1507:                             effective_work = workload_target_j__ * fuel_mult
```

**Line 3031** (comment_marker):
```python
# Require duration_angle_deg - no fallback
```
Context:
```python
3028:         cycle_time = max(1e-6, float(self.data.cycle_time))
3029:         stroke_m = max(1e-6, float(cam_constraints.stroke) * mm_to_m)
3030: 
3031:         # Require duration_angle_deg - no fallback
3032:         duration_angle_deg = getattr(self.data, "duration_angle_deg", None)
3033:         if duration_angle_deg is None:
3034:                 raise ValueError(
```

**Line 3256** (comment_marker):
```python
# Grids don't match - this indicates legacy grid logic is being used
```
Context:
```python
3253:                             vel_u = vel
3254:                             acc_u = acc
3255:                         else:
3256:                             # Grids don't match - this indicates legacy grid logic is being used
3257:                             raise ValueError(
3258:                                 f"Grid mismatch detected: solution grid has {len(cam_angle_rad_arr)} points, "
3259:                                 f"universal grid has {len(ug_th_arr)} points. "
```

**Line 3290** (comment_marker):
```python
# Fallback derivation from mapped acceleration on universal grid
```
Context:
```python
3287:                                 jrk_u = np.gradient(acc_u, dtheta_deg, edge_order=2)
3288:                             solution["jerk"] = jrk_u
3289:                         except Exception:
3290:                             # Fallback derivation from mapped acceleration on universal grid
3291:                             dtheta_deg = np.gradient(np.degrees(ug_th))
3292:                             solution["jerk"] = np.gradient(acc_u, dtheta_deg, edge_order=2)
3293:                         # Sensitivities: if gradients/Jacobians exist on internal grid, pull/push to U
```

**Line 3389** (comment_marker):
```python
# Fallback to constant_load_value from settings if no workload translation
```
Context:
```python
3386:                             load_value = p_load_kpa
3387:                         workload_target_j = pressure_meta.get("work_target_j")
3388: 
3389:                 # Fallback to constant_load_value from settings if no workload translation
3390:                 if load_value is None:
3391:                     try:
3392:                         load_value = float(
```

**Line 3409** (comment_marker):
```python
# Store constant operating temperature
```
Context:
```python
3406:                         self.data.primary_workload_metadata["workload_target_j"] = float(workload_target_j)
3407:                     if p_load_kpa is not None:
3408:                         self.data.primary_workload_metadata["p_load_kpa"] = float(p_load_kpa)
3409:             # Store constant operating temperature
3410:             try:
3411:                 self.data.primary_constant_temperature_K = float(
3412:                     getattr(self.settings, "constant_temperature_K", 900.0),
```

**Line 3456** (comment_marker):
```python
# Method 2: From optimized_parameters (legacy)
```
Context:
```python
3453:                 base_radius = solution.get("base_radius")
3454:                 log.debug(f"Extracted base_radius from solution: {base_radius}")
3455: 
3456:             # Method 2: From optimized_parameters (legacy)
3457:             if base_radius is None:
3458:                 optimized_params = solution.get("optimized_parameters", {})
3459:                 base_radius = optimized_params.get("base_radius")
```

**Line 3465** (comment_marker):
```python
# Method 3: From metadata (fallback)
```
Context:
```python
3462:                         f"Extracted base_radius from optimized_parameters: {base_radius}",
3463:                     )
3464: 
3465:             # Method 3: From metadata (fallback)
3466:             if base_radius is None and hasattr(result, "metadata") and result.metadata:
3467:                 gear_config = result.metadata.get("optimized_gear_config", {})
3468:                 base_radius = gear_config.get("base_center_radius")
```

**Line 3584** (comment_marker):
```python
# secondary_R_psi when available, since legacy consumers expect that signal.
```
Context:
```python
3581: 
3582:                 # Maintain backward compatibility: keep the Litvin planet trajectory in
3583: 
3584:                 # secondary_R_psi when available, since legacy consumers expect that signal.
3585: 
3586:                 chosen_R_psi = None
3587: 
```

**Line 3614** (comment_marker):
```python
# profile resampled onto the I^ grid so legacy code still receives data.
```
Context:
```python
3611: 
3612:                     # If R_psi is unavailable (unexpected), fall back to the synchronized ring
3613: 
3614:                     # profile resampled onto the I^ grid so legacy code still receives data.
3615: 
3616:                     from campro.optimization.grid import GridMapper
3617: 
```


### /Users/maxholden/Documents/GitHub/Larrak/campro/optimization/warmstart_manager.py

**Line 228** (comment_marker):
```python
# Strategy 3: Fallback - generate from simple motion profiles
```
Context:
```python
 225:                 bracketing_solutions, problem_params,
 226:             )
 227: 
 228:         # Strategy 3: Fallback - generate from simple motion profiles
 229:         log.debug("Using fallback initial guess generation")
 230:         return self._generate_fallback_guess(problem_params)
 231: 
```


### /Users/maxholden/Documents/GitHub/Larrak/campro/physics/casadi/litvin.py

**Line 435** (comment_marker):
```python
# Attempt Newton solve
```
Context:
```python
 432:             {'abstol': 1e-6, 'max_iter': 10}
 433:         )
 434: 
 435:         # Attempt Newton solve
 436:         phi_newton = newton_solver(phi_seed)
 437: 
 438:         # Check if Newton converged (simplified check)
```

**Line 446** (comment_marker):
```python
phi_seed  # Fallback to seed (would use interpolation in full implementation)
```
Context:
```python
 443:         phi_contact = ca.if_else(
 444:             newton_converged,
 445:             phi_newton,
 446:             phi_seed  # Fallback to seed (would use interpolation in full implementation)
 447:         )
 448: 
 449:     except Exception:
```

**Line 450** (comment_marker):
```python
# If Newton setup fails, use interpolation fallback
```
Context:
```python
 447:         )
 448: 
 449:     except Exception:
 450:         # If Newton setup fails, use interpolation fallback
 451:         phi_contact = phi_seed
 452: 
 453:     # Create function
```


### /Users/maxholden/Documents/GitHub/Larrak/campro/physics/combustion/model.py

**Line 531** (comment_marker):
```python
except Exception:  # pragma: no cover - fallback guard
```
Context:
```python
 528:                 ca_values[name] = float(
 529:                     np.interp(frac, mfb, theta_deg),
 530:                 )
 531:             except Exception:  # pragma: no cover - fallback guard
 532:                 ca_values[name] = None
 533: 
 534:         return ca_values
```


### /Users/maxholden/Documents/GitHub/Larrak/campro/physics/geometry/litvin.py

**Line 308** (comment_marker):
```python
# Fallback: approximate from target_average_radius vs cam average
```
Context:
```python
 305:         except Exception:
 306:             ratio_hint = None
 307:         if ratio_hint is None:
 308:             # Fallback: approximate from target_average_radius vs cam average
 309:             ratio_hint = float(target_average_radius) / max(r_cam_avg, 1e-9)
 310:         z_ring_ratio = int(np.round(z_cam * ratio_hint))
 311:         z_ring = max(min_teeth, max(z_ring_pitch, z_ring_ratio))
```


### /Users/maxholden/Documents/GitHub/Larrak/campro/physics/kinematics/constraints.py

**Line 43** (comment_marker):
```python
# Placeholder implementation
```
Context:
```python
  40:             Result containing constraint analysis
  41:         """
  42:         try:
  43:             # Placeholder implementation
  44:             log.info("Kinematic constraints component - placeholder implementation")
  45: 
  46:             return ComponentResult(
```


### /Users/maxholden/Documents/GitHub/Larrak/campro/physics/kinematics/time_kinematics.py

**Line 43** (comment_marker):
```python
# Placeholder implementation
```
Context:
```python
  40:             Result containing time kinematics
  41:         """
  42:         try:
  43:             # Placeholder implementation
  44:             log.info("Time kinematics component - placeholder implementation")
  45: 
  46:             return ComponentResult(
```


### /Users/maxholden/Documents/GitHub/Larrak/campro/physics/pr_template.py

**Line 114** (comment_marker):
```python
p_bounce_base = 0.0  # Will be computed per theta if needed, but for template we use base
```
Context:
```python
 111:     heat_loss_factor = 1.0 - 0.1 * sv_ratio  # Rough heat loss penalty (0-10% reduction)
 112: 
 113:     # Denominator for PR (constant across cycle)
 114:     p_bounce_base = 0.0  # Will be computed per theta if needed, but for template we use base
 115:     denom_base_kpa = p_load_kpa + p_cc_kpa + p_env_kpa + p_bounce_base
 116: 
 117:     # Initialize PR template
```

**Line 117** (comment_marker):
```python
# Initialize PR template
```
Context:
```python
 114:     p_bounce_base = 0.0  # Will be computed per theta if needed, but for template we use base
 115:     denom_base_kpa = p_load_kpa + p_cc_kpa + p_env_kpa + p_bounce_base
 116: 
 117:     # Initialize PR template
 118:     pi_template = np.ones_like(theta_deg, dtype=float)
 119: 
 120:     # Phase identification
```

**Line 141** (comment_marker):
```python
# For template: p_cyl ∝ (V_max/V)^γ
```
Context:
```python
 138:         v_comp = np.maximum(v_comp, v_min_mm3 * 1.001)  # Avoid division by zero
 139:         # Isentropic compression: P ∝ (V_max/V)^(γ)
 140:         # PR = p_cyl / p_denom, so we need to model p_cyl
 141:         # For template: p_cyl ∝ (V_max/V)^γ
 142:         pr_compression = (v_max_mm3 / v_comp) ** (gamma - 1.0)
 143:         # Normalize to baseline (start of compression) - ensure we have at least one point
 144:         if len(pr_compression) > 0 and pr_compression[0] > 0.0:
```


### /Users/maxholden/Documents/GitHub/Larrak/campro/physics/simple_cycle_adapter.py

**Line 184** (comment_marker):
```python
"fuel_multiplier": float(fuel_multiplier),  # Preserve for fallback path in _pressure_ratio
```
Context:
```python
 181:             "p_comb": p_cyl_abs_kpa,
 182:             "p_bounce": p_bounce,
 183:             "cycle_work_j": cycle_work_j,
 184:             "fuel_multiplier": float(fuel_multiplier),  # Preserve for fallback path in _pressure_ratio
 185:         }
 186:         if combustion_data is not None:
 187:             result.update(
```


### /Users/maxholden/Documents/GitHub/Larrak/campro/physics/thermal_efficiency_simple.py

**Line 282** (comment_marker):
```python
# Simplified temperature constraint
```
Context:
```python
 279:         max_temp_rise : float
 280:             Maximum temperature rise (K)
 281:         """
 282:         # Simplified temperature constraint
 283:         # Limit velocity to control temperature rise
 284:         # Note: velocity is in per-degree units, so constraint is scaled accordingly
 285:         # For typical engine speeds (~360 deg/s), 1 m/deg ≈ 360 m/s
```

**Line 283** (comment_marker):
```python
# Limit velocity to control temperature rise
```
Context:
```python
 280:             Maximum temperature rise (K)
 281:         """
 282:         # Simplified temperature constraint
 283:         # Limit velocity to control temperature rise
 284:         # Note: velocity is in per-degree units, so constraint is scaled accordingly
 285:         # For typical engine speeds (~360 deg/s), 1 m/deg ≈ 360 m/s
 286:         # Constraint scaled to work with per-degree units
```

**Line 342** (comment_marker):
```python
# Temperature constraints
```
Context:
```python
 339:         # Pressure rate constraints (using angular step)
 340:         self.add_pressure_rate_constraints(opti, acceleration, dtheta)
 341: 
 342:         # Temperature constraints
 343:         self.add_temperature_constraints(opti, velocity)
 344: 
 345:     def evaluate_efficiency(
```

