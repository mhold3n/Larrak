# Implementation Review

## Per-Degree Units Contract

**All motion-law inputs must be in per-degree units exclusively.**

### Core Contract

- **stroke**: in mm (or m)
- **duration_angle_deg**: **required**, in degrees (no fallback, no default)
- **upstroke_duration_percent**: percentage of cycle
- **Optional actuator limits**: in mm/degⁿ (velocity, acceleration, jerk)
  - `max_velocity`: mm/deg (or m/deg in SI)
  - `max_acceleration`: mm/deg² (or m/deg² in SI)
  - `max_jerk`: mm/deg³ (or m/deg³ in SI)
- **engine_speed_rpm**: optional, used to derive cycle_time
- **cycle_time**: derived from engine_speed_rpm and duration_angle_deg, not a primary input

### No Compatibility Mode

- No per-second units are accepted
- No auto-conversion from per-second to per-degree
- No fallback to cycle_time for duration_angle_deg
- No fallback to `casadi_duration_angle_deg` setting
- No heuristic validation that attempts to detect per-second units
- No time-based fallbacks in result handling

### Implementation Details

1. **UnifiedOptimizationData**: All inputs are per-degree or unitless
2. **CasADi Phase 1**: Requires `duration_angle_deg` explicitly; raises error if missing (no fallback to settings)
3. **Compression ratio limits**: Defaults based on clearance geometry, computed consistently in both `_build_casadi_primary_constraints` and `_create_problem_from_constraints`
4. **Result handling**: No time-based fallbacks; requires `cam_angle` or `theta_deg` in results
5. **Collocation wrapper**: Emits `cam_angle` and `theta_deg`, not `time` array
6. **Downstream conversions**: Per-degree to per-second conversion exists ONLY for legacy downstream modules that explicitly require it; not part of primary contract

### Error Handling

If `duration_angle_deg` is missing, the system raises a clear error:
```
duration_angle_deg is required for Phase 1 per-degree optimization.
It must be set in data.duration_angle_deg.
No fallback is allowed to prevent unit mixing.
```

If primary result lacks cam angle data:
```
Primary optimization result must include cam_angle or theta_deg.
Time-based fallback is not supported in per-degree-only contract.
```

### Removed Settings

- `casadi_duration_angle_deg`: Removed from `UnifiedOptimizationSettings` (no default fallback)
- `casadi_compatibility_mode`: Removed (no per-second auto-conversion)

