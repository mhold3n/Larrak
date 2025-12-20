# Optimization System Architecture

This document describes the multi-layer optimization system architecture for the Larrak/CamPro project.

## Overview

The optimization system uses a cascaded, three-layer architecture where secondary and tertiary optimizers are generic "shells" that receive their constraints, relationships, and targets from external sources, enabling maximum extensibility.

---

## Architecture Layers

### Layer 1: Primary Optimizer (Concrete Implementation)

**Purpose**: Core motion law optimization with full implementation.

**Component**: `MotionOptimizer` in `campro.optimization.motion`

**Features**:
- Direct collocation with CasADi integration
- Multiple motion law types (minimum time, energy, jerk)
- Comprehensive constraint handling
- Performance tracking

**Retains implementation** as it provides the foundation for the optimization chain.

### Layer 2: Secondary Optimizer Shell

**Purpose**: Generic shell for cascaded optimization using primary results.

**Component**: `SecondaryOptimizer` in `campro.optimization.secondary`

**Key Features**:
- Access to primary optimization results
- Receives external specifications for constraints, relationships, targets
- Receives external processing and objective functions
- Supports multiple refinement strategies

**Generic Shell Design**:
```python
class SecondaryOptimizer(BaseOptimizer):
    """
    Secondary collocation optimizer shell for cascaded optimization.
    Generic shell that receives external specifications.
    """
    
    def process_primary_result(
        self,
        primary_result_id: str,
        secondary_constraints: Dict[str, Any],
        secondary_relationships: Dict[str, Any],
        optimization_targets: Dict[str, Any],
        processing_function: Callable,
        objective_function: Callable
    ) -> OptimizationResult:
        ...
```

### Layer 3: Tertiary Optimizer Shell

**Purpose**: Advanced optimization with complete context visibility.

**Component**: `TertiaryOptimizer` in `campro.optimization.tertiary`

**Key Features**:
- Full visibility into optimization chain
- Access to results, constraints, rules, and settings from all layers
- Motion law tuning with complete history
- Linkage placement optimization

**Generic Shell Design**:
```python
class TertiaryOptimizer(BaseOptimizer):
    """
    Tertiary collocation optimizer shell for advanced optimization.
    Has full visibility into the optimization chain.
    """
    
    def process_optimization_context(
        self,
        tertiary_constraints: Dict[str, Any],
        tertiary_relationships: Dict[str, Any],
        optimization_targets: Dict[str, Any],
        processing_function: Callable,
        objective_function: Callable
    ) -> OptimizationResult:
        ...
```

---

## Result Storage System

### Component: `OptimizationRegistry` in `campro.storage`

**Purpose**: Centralized storage and sharing of optimization results.

**Features**:
- Result persistence with metadata
- Access control between layers
- Expiration management and cleanup
- Chain management for optimization workflows
- Performance tracking

**Storage with Complete Context**:
```python
storage_result = registry.store_result(
    optimizer_id="motion_optimizer",
    result_data=solution,
    metadata=metadata,
    constraints=constraints,           # Original constraints
    optimization_rules=optimization_rules,  # Optimization rules
    solver_settings=solver_settings,   # Solver configuration
    expires_in=3600
)
```

---

## Cascaded Optimization Workflow

```
┌─────────────────────────────────────────────────────────────┐
│                    Cascaded Optimization Flow               │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  [Primary Optimizer]                                        │
│         │                                                   │
│         ├──> Solve initial problem                         │
│         └──> Store result + context in registry            │
│                                                             │
│                     ↓                                       │
│                                                             │
│  [Secondary Optimizer Shell]                                │
│         │                                                   │
│         ├──> Retrieve primary result                       │
│         ├──> Apply external constraints/relationships      │
│         ├──> Execute external processing function          │
│         └──> Store refined result                          │
│                                                             │
│                     ↓                                       │
│                                                             │
│  [Tertiary Optimizer Shell]                                 │
│         │                                                   │
│         ├──> Access complete optimization context          │
│         ├──> Apply external constraints/targets            │
│         ├──> Execute external processing function          │
│         └──> Store final result                            │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## Optimization Strategies

### Secondary Layer Strategies

| Strategy | Purpose |
|----------|---------|
| Motion Law Refinement | Improve smoothness, efficiency, or accuracy |
| Multi-Objective | Balance multiple optimization goals |
| Constraint Tightening | Re-optimize with tighter constraints |

### Tertiary Layer Strategies

| Strategy | Purpose |
|----------|---------|
| Motion Law Tuning | Blend primary/secondary results with context-aware adjustments |
| Linkage Placement | Optimize follower linkage geometry |
| Combined Optimization | Simultaneous motion law and linkage optimization |

---

## Linkage Parameters System

**Purpose**: Manage follower linkage geometry and placement.

**Components**:
- **Cam Center Position**: (x, y) coordinates
- **Follower Center Position**: (x, y) coordinates
- **Linkage Geometry**: Length, angle, radius, offset
- **Optimization Bounds**: Constraints for linkage parameter optimization

---

## Benefits of Shell Architecture

### Modular Design
- Generic shells without hardcoded implementations
- External specifications for all constraints and targets
- Flexible processing functions define optimization logic
- Easy to add new strategies without modifying core code

### Complete Context Visibility
- Full access to optimization context at all layers
- Constraint propagation across layers
- Relationship understanding in the complete chain
- Robust decision making based on complete context

### Future-Ready Architecture
- **Combustion Physics Ready**: Shells can receive combustion-specific constraints
- **Valve Timing Ready**: Shells can receive valve timing relationships
- **Mechanical Analysis Ready**: Shells can receive mechanical processing functions
- **Multi-Objective Ready**: Shells can receive multi-objective specifications

---

## Performance Results

### Typical Cascaded Optimization Performance

| Layer | Status | Solve Time | Improvement |
|-------|--------|------------|-------------|
| Primary | Converged | 0.000s | Baseline |
| Secondary (Refinement) | Converged | 0.500s | -70.6% jerk |
| Secondary (Multi-Objective) | Converged | 0.000s | Balanced |
| Tertiary (Combined) | Converged | 0.000s | Further refined |

---

## Related Documentation

- **System Overview**: See `architecture/overview.md`
- **CasADi API**: See `architecture/casadi-api.md`
- **Development Roadmap**: See `development/roadmap.md`






