## Phase 2 Motion Mapping: Radial Center Stepping and No-Slip

### Summary
- Center radius stepping: C(θ) = C0 + x(θ), where C0 is the user-entered base center radius and x(θ) is the primary motion law (mm, ≥ 0).
- Angular synchronization: orbit angle equals ψ from Litvin synthesis; spin angle obeys base-circle no-slip dφ/dψ = rb_ring / rb_cam.
- Constant ring speed: the outer ring rotates at constant ω; no-slip implies rb_cam dφ/dt = rb_ring ω.

### Dataflow
1. Phase 1 produces θ (deg) and x(θ) (mm).
2. Phase 2 constructs cam profile r_c(θ) = r_b + x(θ); Litvin synthesis yields ψ(θ), R(ψ).
3. Assembly resamples x(θ) → offset(ψ), then sets planet center radius C(ψ) = C0 + offset(ψ) and center angle = ψ.
4. Spin angle φ(ψ) = sign · (rb_ring/rb_cam) · (ψ − ψ₀).

### Invariants
- No-slip at base circles: rb_cam dφ/dt = rb_ring ω (with constant ω).
- x(θ) is treated as non-negative displacement in addition to C0.
- Backwards compatibility: `center_distance` in state is kept as mean(C(ψ)).

### References
- `campro/physics/kinematics/litvin_assembly.py` for center stepping and kinematics.
- `campro/optimization/cam_ring_optimizer.py` for wiring assembly state after Litvin synthesis.

### Runtime animation path (no solving)
- Use optimized Phase‑2 outputs and primary motion to build relationships only:
  - Inputs: `θ[deg]`, `x(θ)[mm]`, `r_b`, `ψ[rad]`, `R(ψ)[mm]`, gear bases
  - Build: `r_c(θ)=r_b+x(θ)`, `C(ψ)=r_b+resample(x(θ)→ψ)`, orbit angle `= ψ`,
    spin `φ(ψ)=sign·(rb_ring/rb_cam)·(ψ−ψ₀)`
- Implemented by `campro/physics/kinematics/phase2_relationships.py::build_phase2_relationships`
- Consumed by GUI animation via `UnifiedOptimizationFramework.get_phase2_animation_inputs()`



