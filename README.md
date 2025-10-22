# Larrak: Optimal Motion Law Problems with CasADi and Ipopt

A comprehensive Python framework for solving optimal motion law problems using direct collocation methods with CasADi and Ipopt.

[![CI](https://github.com/yourusername/larrak/workflows/CI/badge.svg)](https://github.com/yourusername/larrak/actions)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**CI/CD**: This repository includes GitHub Actions workflows that automatically validate the environment setup, run tests, and perform code quality checks on every push and pull request.

## Prerequisites

- **Python 3.9+**: Required for all installations
- **Conda or Miniconda**: Recommended for best compatibility with ipopt solver
- **Git**: For cloning the repository

## Quick Installation

The fastest way to get started:

```bash
# Clone and setup
git clone <repository-url>
cd Larrak
python scripts/setup_environment.py
conda activate larrak

# Verify installation
python scripts/check_environment.py
```

**Note**: Large dependency folders (`casadi/`, `coinhsl-archive-2023.11.17/`, `ThirdParty-HSL/`) are excluded from GitHub clones. The setup script will install CasADi and IPOPT via conda. HSL solvers (MA27/MA57) are supported; non-HSL fallbacks are not permitted in this project.

## Modern Usage

### CLI

```bash
# Solve a motion law from a YAML/JSON spec
larrak solve --spec specs/example_min_jerk.yml --diagnose
```

This writes `runs/{RUN_ID}-report.json` and an Ipopt log to `runs/{RUN_ID}-ipopt.log`.

### Python API

```python
from campro.api import ProblemSpec, solve_motion

spec = ProblemSpec(
    stroke=20.0,
    cycle_time=1.0,
    phases={"upstroke_percent": 60.0, "zero_accel_percent": 0.0},
    bounds={"max_velocity": 100.0, "max_acceleration": 1000.0, "max_jerk": 10000.0},
    objective="minimum_jerk",
)
report = solve_motion(spec)
print(report.status, report.kkt)
```

> Legacy examples below (CamPro_OptimalMotion, older solver APIs) are kept for reference and will be updated as the façade API expands.

## Features

- **Multiple Motion Law Types**: Minimum time, energy, jerk, and custom objectives
- **True Collocation Methods**: Legendre, Radau, and Lobatto collocation
- **Flexible Constraints**: Position, velocity, acceleration, and jerk bounds
- **Boundary Conditions**: Full control over initial and final states
- **High Performance**: Leverages CasADi's symbolic computation and Ipopt's optimization
- **Comprehensive Testing**: Unit tests, integration tests, and property-based testing
- **Visualization**: Built-in plotting capabilities for solution analysis
- **Automated Setup**: One-command environment setup with dependency validation

### Basic Usage - Cam Motion Laws

```python
from CamPro_OptimalMotion import solve_cam_motion_law

# Solve a cam follower motion law problem
solution = solve_cam_motion_law(
    stroke=20.0,                    # 20mm follower stroke
    upstroke_duration_percent=60.0, # 60% of cycle for upstroke
    motion_type="minimum_jerk",     # Smooth motion
    cycle_time=1.0                  # 1 second cycle (360°)
)

# Access the solution
cam_angle = solution['cam_angle']    # 0 to 360 degrees
position = solution['position']      # Follower position
velocity = solution['velocity']      # Follower velocity
acceleration = solution['acceleration']  # Follower acceleration
jerk = solution['control']           # Follower jerk
```

### Advanced Cam Usage

```python
from CamPro_OptimalMotion import solve_cam_motion_law

# Cam with constraints and zero acceleration phase
solution = solve_cam_motion_law(
    stroke=25.0,                    # 25mm stroke
    upstroke_duration_percent=50.0, # 50% of cycle for upstroke
    motion_type="minimum_jerk",     # Smooth motion
    cycle_time=0.5,                 # 0.5 second cycle
    max_velocity=100.0,             # 100 mm/s max velocity
    max_acceleration=500.0,         # 500 mm/s² max acceleration
    zero_accel_duration_percent=10.0, # 10% of cycle with zero acceleration
    dwell_at_tdc=True,              # Dwell at TDC (0°)
    dwell_at_bdc=False              # No dwell at BDC (180°)
)
```

### Advanced Usage

```python
from CamPro_OptimalMotion import OptimalMotionSolver, MotionConstraints, CollocationSettings

# Configure solver with high accuracy
settings = CollocationSettings(
    degree=5,           # Higher collocation degree
    method="radau",     # Radau collocation
    max_iterations=1000,
    tolerance=1e-8
)

solver = OptimalMotionSolver(settings)

# Define detailed constraints
constraints = MotionConstraints(
    initial_position=0.0,
    initial_velocity=0.0,
    final_position=20.0,
    final_velocity=0.0,
    velocity_bounds=(-8.0, 8.0),
    acceleration_bounds=(-3.0, 3.0),
    jerk_bounds=(-2.0, 2.0)
)

# Solve minimum energy problem
solution = solver.solve_minimum_energy(
    constraints=constraints,
    distance=20.0,
    time_horizon=8.0,
    max_velocity=8.0,
    max_acceleration=3.0
)

# Visualize results
solver.plot_solution(solution, save_path="motion_law.png")
```

## Cam Motion Law Types

### 1. Minimum Jerk Motion (Default)
Minimize jerk for smooth, comfortable cam follower motion.

```python
solution = solve_cam_motion_law(
    stroke=20.0,
    upstroke_duration_percent=60.0,
    motion_type="minimum_jerk"
)
```

### 2. Minimum Energy Motion
Minimize energy consumption for cam follower motion.

```python
solution = solve_cam_motion_law(
    stroke=20.0,
    upstroke_duration_percent=60.0,
    motion_type="minimum_energy"
)
```

### 3. Minimum Time Motion
Minimize time to complete cam follower motion.

```python
solution = solve_cam_motion_law(
    stroke=20.0,
    upstroke_duration_percent=60.0,
    motion_type="minimum_time"
)
```

## Cam Constraint Parameters

### Core Parameters
- **`stroke`**: Total follower stroke (required)
- **`upstroke_duration_percent`**: Percentage of cycle for upstroke (0-100)
- **`zero_accel_duration_percent`**: Percentage of cycle with zero acceleration (optional)

### Optional Constraints
- **`max_velocity`**: Maximum allowed velocity
- **`max_acceleration`**: Maximum allowed acceleration  
- **`max_jerk`**: Maximum allowed jerk

### Boundary Conditions
- **`dwell_at_tdc`**: Whether to dwell (zero velocity) at TDC (0°)
- **`dwell_at_bdc`**: Whether to dwell (zero velocity) at BDC (180°)

## General Motion Law Types (Advanced)

### 1. Minimum Time Motion
Minimize the time to complete a motion while satisfying all constraints.

```python
solution = solve_minimum_time_motion(
    distance=10.0,
    max_velocity=5.0,
    max_acceleration=2.0,
    max_jerk=1.0
)
```

### 2. Minimum Energy Motion
Minimize energy consumption for a fixed time horizon.

```python
solution = solve_minimum_energy_motion(
    distance=10.0,
    time_horizon=5.0,
    max_velocity=5.0,
    max_acceleration=2.0
)
```

### 3. Minimum Jerk Motion
Minimize jerk for smooth, comfortable motion.

```python
solution = solve_minimum_jerk_motion(
    distance=10.0,
    time_horizon=5.0,
    max_velocity=5.0,
    max_acceleration=2.0
)
```

### 4. Custom Objectives
Define your own objective functions.

```python
import casadi as ca

def custom_objective(t, x, v, a, u):
    """Minimize energy + smoothness penalty."""
    return ca.integral(u**2 + 0.1*v**2)

solution = solver.solve_custom_objective(
    objective_function=custom_objective,
    constraints=constraints,
    distance=20.0,
    time_horizon=8.0
)
```

## Collocation Methods

The framework supports three collocation methods:

- **Legendre**: Legendre-Gauss-Radau collocation (default)
- **Radau**: Radau collocation
- **Lobatto**: Lobatto collocation

```python
settings = CollocationSettings(
    degree=3,           # Collocation degree (1-7)
    method="radau",     # Collocation method
    max_iterations=1000,
    tolerance=1e-6
)
```

## Testing

Run the comprehensive test suite:

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=CamPro_OptimalMotion --cov=campro

# Run specific test file
pytest tests/test_optimal_motion.py -v
```

## Code Quality

Ensure code quality with the provided tools:

```bash
# Format and lint code
ruff --fix

# Type checking
mypy --strict

# Run all quality checks
pytest -q
```

## Documentation

- [Installation Guide](docs/installation_guide.md): Comprehensive installation instructions
- [Design Document](docs/optimal_motion_design.md): Detailed technical documentation
- [Project Status](docs/project_status.md): Current implementation status and API reference

## Installation Methods

### Method 1: Conda (Recommended)

Conda provides the most reliable installation with ipopt solver support:

```bash
# Create environment from environment.yml
conda env create -f environment.yml
conda activate larrak

# Verify installation
python scripts/check_environment.py
```

### Method 2: Automated Setup

Use the provided setup script for automated installation:

```bash
# Run setup script (detects conda/mamba automatically)
python scripts/setup_environment.py

# Activate environment
conda activate larrak
```

### Method 3: Pip (Alternative)

Pip installation may not include ipopt solver support:

```bash
# Install dependencies
pip install -r requirements.txt

# Verify installation
python scripts/check_environment.py
```

## Verifying Installation

After installation, verify everything works:

```bash
# Check environment status
python scripts/check_environment.py

# Test basic functionality
python -c "import campro; print('✓ Installation successful!')"

# Run GUI
python cam_motion_gui.py
```

## Dependencies

- **CasADi** (>=3.6.0): Symbolic computation and optimal control
- **Ipopt**: Interior-point optimizer (included with conda installation)
- **NumPy** (>=1.24.0): Numerical computations
- **SciPy** (>=1.10.0): Additional numerical tools
- **Matplotlib** (>=3.7.0): Plotting and visualization

## Development Dependencies

- **pytest** (>=7.4.0): Testing framework
- **mypy** (>=1.5.0): Static type checking
- **ruff** (>=0.0.280): Code formatting and linting
- **hypothesis** (>=6.82.0): Property-based testing

## Examples

### Example 1: Basic Cam Motion

```python
from CamPro_OptimalMotion import solve_cam_motion_law

# Create smooth cam follower motion
solution = solve_cam_motion_law(
    stroke=20.0,                    # 20mm stroke
    upstroke_duration_percent=60.0, # 60% of cycle for upstroke
    motion_type="minimum_jerk",     # Smooth motion
    cycle_time=1.0                  # 1 second cycle
)

# The resulting motion will be very smooth with minimal jerk
```

### Example 2: High-Speed Cam with Constraints

```python
from CamPro_OptimalMotion import solve_cam_motion_law

# High-speed cam with velocity and acceleration limits
solution = solve_cam_motion_law(
    stroke=15.0,                    # 15mm stroke
    upstroke_duration_percent=40.0, # 40% of cycle for upstroke
    motion_type="minimum_time",     # Minimize time
    cycle_time=0.2,                 # 0.2 second cycle (fast)
    max_velocity=200.0,             # 200 mm/s max velocity
    max_acceleration=1000.0,        # 1000 mm/s² max acceleration
    dwell_at_tdc=True,              # Dwell at TDC
    dwell_at_bdc=False              # No dwell at BDC
)
```

### Example 3: Cam with Zero Acceleration Phase

```python
from CamPro_OptimalMotion import solve_cam_motion_law

# Cam with constant velocity phase during expansion
solution = solve_cam_motion_law(
    stroke=25.0,                    # 25mm stroke
    upstroke_duration_percent=70.0, # 70% of cycle for upstroke
    motion_type="minimum_energy",   # Minimize energy
    cycle_time=1.5,                 # 1.5 second cycle
    zero_accel_duration_percent=20.0, # 20% of cycle with zero acceleration
    max_velocity=50.0,              # 50 mm/s max velocity
    dwell_at_tdc=True,              # Dwell at TDC
    dwell_at_bdc=True               # Dwell at BDC
)
```

### Example 4: General Motion Law (Advanced)

```python
from CamPro_OptimalMotion import solve_minimum_jerk_motion

# General motion law for non-cam applications
solution = solve_minimum_jerk_motion(
    distance=15.0,      # 15 meters
    time_horizon=6.0,   # 6 seconds
    max_velocity=4.0,   # 4 m/s
    max_acceleration=2.0 # 2 m/s²
)
```

## Performance

Typical performance on modern hardware:

- **Simple problems** (< 100 collocation points): 0.1-1.0 seconds
- **Medium problems** (100-1000 points): 0.2-3.0 seconds
- **Complex problems** (> 1000 points): 1.0-10.0 seconds

Memory usage scales linearly with problem size, typically 100-500 MB for most problems.

## Contributing

1. Follow the project's cursor rules for code organization
2. Write tests for all new functionality
3. Ensure all tests pass before submitting
4. Update documentation for new features
5. Use type hints and follow PEP 8 style

## License

MIT License - see LICENSE file for details.

## Acknowledgments

- [CasADi](https://web.casadi.org/) for symbolic computation and optimal control
- [Ipopt](https://coin-or.github.io/Ipopt/) for interior-point optimization
- The optimal control and robotics communities for inspiration and algorithms
