# NLP Structure Analysis: Constraint-Variable Relationship

## Executive Summary

The NLP formulation has **1906 variables** and **3151 constraints**, of which **2344 are equality constraints**. This results in **-438 degrees of freedom**, making the problem overconstrained and causing IPOPT to fail with "TOO_FEW_DEGREES_OF_FREEDOM".

**Key Finding**: The **continuity constraints (1170) are redundant** with the **collocation residuals (1170)**. Both enforce state continuity between stages, but collocation residuals already enforce this through the integration scheme.

## Problem Dimensions

- **K (stages)**: 90
- **C (collocation points per stage)**: 1
- **Total variables**: 1906
- **Total constraints**: 3151
- **Equality constraints**: 2344
- **Inequality constraints**: 807
- **Degrees of freedom**: -438 (overconstrained)

## Variable Inventory

### Variables by Category

Based on code analysis in `build_collocation_nlp()`:

#### 1. Initial State Variables (6 variables)
- `xL0, xR0, vL0, vR0`: Initial piston positions and velocities (4 vars)
- `rho0` (or `rho0_log`): Initial density (1 var, log-space if configured)
- `T0`: Initial temperature (1 var)

**Location**: Lines 1393-1418, tracked in `var_groups` at `var_idx = 0-5`

#### 2. Ignition Timing Variable (0 or 1 variable)
- `t_ign`: Ignition timing (1 var, only if `use_combustion_model=True`)

**Location**: Lines 1466-1485, tracked in `var_groups["ignition"]` at `var_idx = 6` (if present)

#### 3. Initial Valve Areas (2 variables)
- `Ain0` (or `Ain0_log`): Initial intake valve area (1 var, log-space if configured)
- `Aex0` (or `Aex0_log`): Initial exhaust valve area (1 var, log-space if configured)

**Location**: Lines 1487-1522, tracked in `var_groups["valve_areas"]` at `var_idx = 6-7` (or 7-8 if ignition present)

#### 4. Dynamic Wall Temperature (0 or 1 variable)
- `Tw0`: Initial wall temperature (1 var, only if `dynamic_wall=True`)

**Location**: Lines 1554-1568, tracked in `var_groups["temperatures"]` at `var_idx = 8` (if present)

#### 5. Scavenging/Timing Initial States (7 variables)
- `yF0`: Initial fresh fraction (1 var)
- `Mdel0`: Initial delivered mass (1 var)
- `Mlost0`: Initial lost mass (1 var)
- `AinInt0`: Initial intake area integral (1 var)
- `AinTmom0`: Initial intake area time moment (1 var)
- `AexInt0`: Initial exhaust area integral (1 var)
- `AexTmom0`: Initial exhaust area time moment (1 var)

**Location**: Lines 1570-1592, tracked in `var_groups["penalties"]` at `var_idx = 8-14` (or adjusted based on previous vars)

#### 6. Collocation Point Variables (K × C × 6 = 90 × 1 × 6 = 540 variables)

For each stage `k` (0 to K-1) and collocation point `j` (0 to C-1):
- `xL_{k}_{j}`, `xR_{k}_{j}`: Piston positions (2 vars)
- `vL_{k}_{j}`, `vR_{k}_{j}`: Piston velocities (2 vars)
- `rho_{k}_{j}` (or `rho_{k}_{j}_log`): Density (1 var, log-space if configured)
- `T_{k}_{j}`: Temperature (1 var)

**Location**: Lines 1647-1730, tracked in `var_groups` at `var_idx` updated per collocation point

**Count**: 90 stages × 1 collocation point × 6 states = **540 variables**

#### 7. Valve Area Control Variables (K × C × 2 = 90 × 1 × 2 = 180 variables)

For each stage `k` and collocation point `j`:
- `Ain_{k}_{j}` (or `Ain_{k}_{j}_log`): Intake valve area (1 var, log-space if configured)
- `Aex_{k}_{j}` (or `Aex_{k}_{j}_log`): Exhaust valve area (1 var, log-space if configured)

**Location**: Lines 1600-1631, tracked in `var_groups["valve_areas"]`

**Count**: 90 stages × 1 collocation point × 2 valves = **180 variables**

#### 8. Combustion Heat Release Variables (K × C = 90 × 1 = 90 variables, only if NOT using integrated model)

For each stage `k` and collocation point `j`:
- `Q_comb_{k}_{j}`: Combustion heat release (1 var, only if `use_combustion_model=False`)

**Location**: Lines 1633-1645

**Count**: 90 stages × 1 collocation point = **90 variables** (if not using integrated model)

**Note**: In the test case, `use_combustion_model=True`, so these are **0 variables**.

#### 9. Time-Step State Variables (K × 13 = 90 × 13 = 1170 variables)

For each stage `k` (1 to K, i.e., K variables):
- `xL_{k+1}`, `xR_{k+1}`: Piston positions (2 vars)
- `vL_{k+1}`, `vR_{k+1}`: Piston velocities (2 vars)
- `rho_{k+1}` (or `rho_{k+1}_log`): Density (1 var, log-space if configured)
- `T_{k+1}`: Temperature (1 var)
- `yF_{k+1}`: Fresh fraction (1 var)
- `Mdel_{k+1}`: Delivered mass (1 var)
- `Mlost_{k+1}`: Lost mass (1 var)
- `AinInt_{k+1}`: Intake area integral (1 var)
- `AinTmom_{k+1}`: Intake area time moment (1 var)
- `AexInt_{k+1}`: Exhaust area integral (1 var)
- `AexTmom_{k+1}`: Exhaust area time moment (1 var)

**Location**: Lines 2151-2268, **NOT tracked in `var_groups`** (this is the source of unaccounted variables!)

**Count**: 90 stages × 13 states = **1170 variables**

### Variable Count Summary

| Category | Count | Tracked in var_groups |
|----------|-------|----------------------|
| Initial states | 6 | Yes |
| Ignition timing | 1 | Yes (if present) |
| Initial valve areas | 2 | Yes |
| Dynamic wall temp | 1 | Yes (if present) |
| Scavenging initial | 7 | Yes |
| Collocation points | 540 | Yes |
| Valve controls | 180 | Yes |
| Combustion Q (if not integrated) | 90 | No (not tracked) |
| **Time-step states** | **1170** | **NO** |
| **Total** | **1906** | **736 tracked, 1170 untracked** |

**Note**: The 1170 unaccounted variables are the time-step state variables that are added in the loop but not tracked in `var_groups`.

## Constraint Inventory

### Constraints by Category

#### 1. Collocation Residuals (K × C × 13 = 90 × 1 × 13 = 1170 equality constraints)

For each stage `k` and collocation point `c`:
- `xL_c - rhs_xL = 0`
- `xR_c - rhs_xR = 0`
- `vL_c - rhs_vL = 0`
- `vR_c - rhs_vR = 0`
- `rho_c - rhs_rho = 0` (or `rho_colloc_log[c] - rhs_rho_log = 0` if log-space)
- `T_c - rhs_T = 0`
- `yF_k - rhs_yF = 0`
- `Mdel_k - rhs_Mdel = 0`
- `Mlost_k - rhs_Mlost = 0`
- `AinInt_k - rhs_AinInt = 0`
- `AinTmom_k - rhs_AinTmom = 0`
- `AexInt_k - rhs_AexInt = 0`
- `AexTmom_k - rhs_AexTmom = 0`

**Location**: Lines 2048-2084

**Count**: 90 stages × 1 collocation point × 13 residuals = **1170 equality constraints**

**Purpose**: Enforce that collocation point states satisfy the DAE system (integrated from previous time step).

#### 2. Continuity Constraints (K × 13 = 90 × 13 = 1170 equality constraints)

For each stage `k`:
- `xL_k - xL_k1 = 0`
- `xR_k - xR_k1 = 0`
- `vL_k - vL_k1 = 0`
- `vR_k - vR_k1 = 0`
- `rho_k - rho_k1 = 0` (or `rho_k_log - rho_k1 = 0` if log-space)
- `T_k - T_k1 = 0`
- `yF_k - yF_k1 = 0`
- `Mdel_k - Mdel_k1 = 0`
- `Mlost_k - Mlost_k1 = 0`
- `AinInt_k - AinInt_k1 = 0`
- `AinTmom_k - AinTmom_k1 = 0`
- `AexInt_k - AexInt_k1 = 0`
- `AexTmom_k - AexTmom_k1 = 0`

**Location**: Lines 2270-2307

**Count**: 90 stages × 13 states = **1170 equality constraints**

**Purpose**: Enforce continuity between time-step variables and next-stage states.

**⚠️ REDUNDANCY**: These constraints are **redundant** with collocation residuals. The collocation residuals already enforce that `state[k+1] = state[k] + integral(derivative)` through the integration scheme. The continuity constraints enforce `state[k+1] = state[k]` at boundaries, which is already satisfied by collocation.

#### 3. Periodicity Constraints (4 equality constraints)

- `xL_K - xL0 = 0`
- `xR_K - xR0 = 0`
- `vL_K - vL0 = 0`
- `vR_K - vR0 = 0`

**Location**: Lines 2401-2407

**Count**: **4 equality constraints**

**Purpose**: Enforce that the cycle is periodic (final state = initial state for positions and velocities).

#### 4. Path Constraints (807 inequality constraints)

##### 4a. Pressure Path Constraints (K = 90 inequality constraints)
- `p_min <= p_gas_k <= p_max` for each stage `k`

**Location**: Lines 2310-2320

**Count**: 90 stages = **90 inequality constraints**

##### 4b. Temperature Path Constraints (K = 90 inequality constraints)
- `T_min <= T_k <= T_max` for each stage `k`

**Location**: Lines 2322-2332

**Count**: 90 stages = **90 inequality constraints**

##### 4c. Velocity Path Constraints (K × 2 = 180 inequality constraints)
- `-v_max <= vL_{k}_{j} <= v_max` for each stage `k` and collocation point `j`
- `-v_max <= vR_{k}_{j} <= v_max` for each stage `k` and collocation point `j`

**Location**: Lines 2365-2373

**Count**: 90 stages × 1 collocation point × 2 pistons = **180 inequality constraints**

##### 4d. Acceleration Path Constraints ((K-1) × 2 = 178 inequality constraints)
- `-a_max <= aL_k <= a_max` for stages `k > 0`
- `-a_max <= aR_k <= a_max` for stages `k > 0`

**Location**: Lines 2375-2384

**Count**: (90-1) stages × 2 pistons = **178 inequality constraints**

##### 4e. Valve Rate Constraints ((K-1) × 2 = 178 inequality constraints)
- `-dA_dt_max <= dAin_dt <= dA_dt_max` for stages `k > 0`
- `-dA_dt_max <= dAex_dt <= dA_dt_max` for stages `k > 0`

**Location**: Lines 2349-2363

**Count**: (90-1) stages × 2 valves = **178 inequality constraints**

##### 4f. Clearance Path Constraint (1 inequality constraint)
- `xR_k - xL_k >= gap_min` (at final stage)

**Location**: Lines 2334-2347

**Count**: **1 inequality constraint**

##### 4g. Combustion Constraints (K = 90 inequality constraints)
- `0 <= Q_comb_{k}_{j} <= Q_comb_max` for each stage `k` and collocation point `j`

**Location**: Lines 2386-2399

**Count**: 90 stages × 1 collocation point = **90 inequality constraints**

**Total Path Constraints**: 90 + 90 + 180 + 178 + 178 + 1 + 90 = **807 inequality constraints**

### Constraint Count Summary

| Category | Count | Type | Redundant? |
|----------|-------|------|-----------|
| Collocation residuals | 1170 | Equality | No |
| **Continuity** | **1170** | **Equality** | **YES (with collocation)** |
| Periodicity | 4 | Equality | No |
| Path constraints | 807 | Inequality | No |
| **Total** | **3151** | **2344 equality, 807 inequality** | |

## Redundancy Analysis

### Hypothesis: Continuity Constraints are Redundant

**Evidence**:
1. **Collocation residuals** (1170) enforce: `state[k+1] = state[k] + integral(derivative)` through the integration scheme
2. **Continuity constraints** (1170) enforce: `state[k+1] = state[k]` at boundaries
3. Both constraint sets involve the same 13 state variables per stage
4. The collocation method already ensures continuity through the integration

**Mathematical Relationship**:
- Collocation residuals: `state_colloc - state_prev - h * f(state_colloc, controls) = 0`
- Continuity: `state_k - state_k1 = 0`
- If collocation is satisfied, then `state_k ≈ state_colloc[C-1]` (last collocation point), which should equal `state_k1` from the integration.

**Conclusion**: The continuity constraints are **redundant** with collocation residuals. Removing them would:
- Reduce equality constraints from 2344 to 1174
- Increase degrees of freedom from -438 to 732
- Make the problem well-posed

### Other Potential Redundancies

1. **Periodicity vs Continuity**: Periodicity constraints enforce `state[K] = state[0]` for positions/velocities. This is independent of continuity constraints, which enforce `state[k] = state[k-1]` between stages. Not redundant.

2. **Path Constraints**: These are inequality constraints enforcing bounds, not redundant with equality constraints.

## Degrees of Freedom Analysis

### Current State
- Variables: 1906
- Equality constraints: 2344
- **DOF = 1906 - 2344 = -438** (overconstrained)

### After Removing Redundant Continuity Constraints
- Variables: 1906
- Equality constraints: 2344 - 1170 = 1174
- **DOF = 1906 - 1174 = 732** (well-posed)

### Minimum Constraints Needed

For a well-posed collocation problem:
1. **Collocation residuals** (1170): Required to enforce DAE system
2. **Periodicity** (4): Required for cyclic boundary conditions
3. **Path constraints** (807): Required for physical bounds (inequality, don't reduce DOF)

**Minimum equality constraints**: 1170 + 4 = **1174**

**Expected DOF**: 1906 - 1174 = **732**

## Recommendations

### 1. Remove Redundant Continuity Constraints

**Action**: Remove or comment out the continuity constraint block (lines 2270-2307 in `nlp.py`).

**Impact**:
- Reduces equality constraints by 1170
- Increases DOF from -438 to 732
- Makes problem well-posed
- Should allow IPOPT to solve

**Risk**: Low - collocation residuals already enforce continuity

### 2. Track Time-Step Variables in var_groups

**Action**: Add tracking for time-step variables in `var_groups` when they are added to `w[]` (around line 2177-2213).

**Impact**:
- Improves diagnostic capabilities
- Enables proper validation of log-space variables
- Better understanding of variable structure

**Risk**: None - only affects metadata

### 3. Verify Collocation Method Correctness

**Action**: Verify that collocation residuals correctly enforce state continuity. If they don't, continuity constraints may be necessary but should be reformulated.

**Impact**: Ensures problem formulation is correct

**Risk**: Medium - requires understanding of collocation method

## Next Steps

1. **Implement recommendation 1**: Remove continuity constraints and test
2. **Implement recommendation 2**: Track time-step variables in var_groups
3. **Run diagnostics**: Verify DOF calculation and constraint counts
4. **Test optimization**: Run phase1 test to verify IPOPT can solve

## Constraint-Variable Dependency Mapping

### Collocation Residuals (1170 constraints)

Each collocation residual depends on:
- **Collocation point variables** (at stage k, collocation point c):
  - `xL_{k}_{c}`, `xR_{k}_{c}`, `vL_{k}_{c}`, `vR_{k}_{c}`: Piston states
  - `rho_{k}_{c}` (or `rho_{k}_{c}_log`): Density
  - `T_{k}_{c}`: Temperature
- **Previous time-step variables** (state at stage k):
  - `xL_k`, `xR_k`, `vL_k`, `vR_k`: Piston states
  - `rho_k` (or `rho_k_log`): Density
  - `T_k`: Temperature
  - `yF_k`, `Mdel_k`, `Mlost_k`: Scavenging states
  - `AinInt_k`, `AinTmom_k`, `AexInt_k`, `AexTmom_k`: Timing states
- **Control variables** (at stage k, collocation point c):
  - `Ain_{k}_{c}` (or `Ain_{k}_{c}_log`): Intake valve area
  - `Aex_{k}_{c}` (or `Aex_{k}_{c}_log`): Exhaust valve area
  - `Q_comb_{k}_{c}`: Combustion heat release (if not using integrated model)
- **Other variables**:
  - `t_ign`: Ignition timing (if using integrated combustion model)
  - `Tw_k`: Wall temperature (if dynamic_wall=True)

**Mathematical form**: `state_colloc - (state_prev + h * integral(derivative)) = 0`

### Continuity Constraints (1170 constraints) - REDUNDANT

Each continuity constraint depends on:
- **Time-step variables** (at stage k+1):
  - `xL_{k+1}`, `xR_{k+1}`, `vL_{k+1}`, `vR_{k+1}`: Piston states
  - `rho_{k+1}` (or `rho_{k+1}_log`): Density
  - `T_{k+1}`: Temperature
  - `yF_{k+1}`, `Mdel_{k+1}`, `Mlost_{k+1}`: Scavenging states
  - `AinInt_{k+1}`, `AinTmom_{k+1}`, `AexInt_{k+1}`, `AexTmom_{k+1}`: Timing states
- **Computed next states** (from integration at stage k):
  - `xL_k1`, `xR_k1`, `vL_k1`, `vR_k1`: Integrated piston states
  - `rho_k1` (or `rho_k1_log`): Integrated density
  - `T_k1`: Integrated temperature
  - `yF_k1`, `Mdel_k1`, `Mlost_k1`: Integrated scavenging states
  - `AinInt_k1`, `AinTmom_k1`, `AexInt_k1`, `AexTmom_k1`: Integrated timing states

**Mathematical form**: `state_{k+1} - state_k1 = 0` where `state_k1` is computed from integration

**Redundancy**: The collocation residuals already enforce that `state_{k+1} ≈ state_colloc[C-1]`, and `state_colloc[C-1]` is computed from `state_k1` through the integration. Therefore, continuity constraints are redundant.

### Periodicity Constraints (4 constraints)

Each periodicity constraint depends on:
- **Final time-step variables** (at stage K):
  - `xL_K`, `xR_K`, `vL_K`, `vR_K`: Final piston states
- **Initial state variables**:
  - `xL0`, `xR0`, `vL0`, `vR0`: Initial piston states

**Mathematical form**: `state_K - state_0 = 0`

### Path Constraints (807 inequality constraints)

#### Pressure Path Constraints (90 constraints)
- **Depend on**: `rho_k`, `T_k` (computed from time-step variables at stage k)
- **Form**: `p_min <= p(rho_k, T_k) <= p_max`

#### Temperature Path Constraints (90 constraints)
- **Depend on**: `T_k` (time-step variable at stage k)
- **Form**: `T_min <= T_k <= T_max`

#### Velocity Path Constraints (180 constraints)
- **Depend on**: `vL_{k}_{j}`, `vR_{k}_{j}` (collocation point variables)
- **Form**: `-v_max <= vL_{k}_{j} <= v_max`, `-v_max <= vR_{k}_{j} <= v_max`

#### Acceleration Path Constraints (178 constraints)
- **Depend on**: `vL_{k}_{j}`, `vR_{k}_{j}`, `vL_k`, `vR_k` (collocation and time-step variables)
- **Form**: `-a_max <= (vL_{k}_{j} - vL_k)/h <= a_max`

#### Valve Rate Constraints (178 constraints)
- **Depend on**: `Ain_{k}_{j}`, `Aex_{k}_{j}`, `Ain_{k-1}_{j}`, `Aex_{k-1}_{j}` (valve control variables)
- **Form**: `-dA_dt_max <= (Ain_{k}_{j} - Ain_{k-1}_{j})/h <= dA_dt_max`

#### Clearance Path Constraint (1 constraint)
- **Depend on**: `xL_K`, `xR_K` (final time-step variables)
- **Form**: `xR_K - xL_K >= gap_min`

#### Combustion Constraints (90 constraints)
- **Depend on**: `Q_comb_{k}_{j}` (combustion control variable, if not using integrated model)
- **Form**: `0 <= Q_comb_{k}_{j} <= Q_comb_max`

## Dependency Matrix Summary

| Constraint Type | Variables Depend On | Count |
|----------------|---------------------|-------|
| Collocation residuals | Collocation states, time-step states, controls | 1170 |
| Continuity (redundant) | Time-step states, integrated states | 1170 |
| Periodicity | Initial states, final time-step states | 4 |
| Path constraints | Time-step states, collocation states, controls | 807 |

## Appendix: Code Locations

### Variable Construction
- Initial states: Lines 1392-1425
- Ignition timing: Lines 1466-1485
- Valve areas: Lines 1487-1522
- Dynamic wall: Lines 1554-1568
- Scavenging initial: Lines 1570-1592
- Collocation points: Lines 1647-1730
- Valve controls: Lines 1600-1631
- Time-step states: Lines 2151-2268

### Constraint Construction
- Collocation residuals: Lines 2048-2084
- Continuity: Lines 2270-2307
- Periodicity: Lines 2401-2407
- Path constraints: Lines 2310-2399

