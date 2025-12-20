# Cam Motion Law GUI

A simple Tkinter-based graphical interface for cam motion law optimization using CasADi and Ipopt.

## Features

- **Intuitive Input**: Simple form-based input for cam parameters
- **Real-time Visualization**: Live plotting of motion law curves
- **Multiple Motion Types**: Support for minimum jerk, energy, and time motion laws
- **Constraint Support**: Optional velocity, acceleration, and jerk limits
- **Zero Acceleration Phases**: Support for constant velocity phases
- **Export Capabilities**: Save plots in PNG, PDF, or SVG formats
- **Threaded Solving**: Non-blocking optimization with progress indication

## Quick Start

### Launch the GUI

```bash
# Launch the GUI directly
python cam_motion_gui.py

# Or use demo with preset parameters
python scripts/gui_demo.py
```

## GUI Components

### Input Parameters

#### Core Parameters
- **Stroke (mm)**: Total follower stroke (required)
- **Upstroke Duration (%)**: Percentage of cycle for upstroke (0-100)
- **Zero Accel Duration (%)**: Percentage of cycle with zero acceleration (optional, can be anywhere in cycle)
- **Cycle Time (s)**: Total cycle time for 360° rotation
- **Motion Type**: Choose from minimum_jerk, minimum_energy, or minimum_time

#### Optional Constraints
- **Max Velocity (mm/s)**: Maximum allowed velocity (0 = no limit)
- **Max Acceleration (mm/s²)**: Maximum allowed acceleration (0 = no limit)
- **Max Jerk (mm/s³)**: Maximum allowed jerk (0 = no limit)

#### Boundary Conditions
- **Dwell at TDC**: Whether to have zero velocity at Top Dead Center (0°)
- **Dwell at BDC**: Whether to have zero velocity at Bottom Dead Center (180°)

#### Solver Settings
- **Collocation Degree**: Accuracy level (1-7, higher = more accurate but slower)
- **Collocation Method**: Choose from legendre, radau, or lobatto

### Control Buttons

- **Solve Motion Law**: Run the optimization and generate curves
- **Save Plot**: Export the current plot to file
- **Clear Plot**: Clear the current plot
- **Reset Parameters**: Reset all inputs to default values

## Usage Examples

### Example 1: Basic Cam Motion

1. Set **Stroke** to `20.0` mm
2. Set **Upstroke Duration** to `60.0` %
3. Set **Cycle Time** to `1.0` s
4. Select **Motion Type** as `minimum_jerk`
5. Click **Solve Motion Law**

### Example 2: High-Speed Cam with Constraints

1. Set **Stroke** to `15.0` mm
2. Set **Upstroke Duration** to `40.0` %
3. Set **Cycle Time** to `0.2` s
4. Set **Max Velocity** to `200.0` mm/s
5. Set **Max Acceleration** to `1000.0` mm/s²
6. Select **Motion Type** as `minimum_time`
7. Uncheck **Dwell at BDC**
8. Click **Solve Motion Law**

### Example 3: Cam with Zero Acceleration Phase

1. Set **Stroke** to `25.0` mm
2. Set **Upstroke Duration** to `70.0` %
3. Set **Zero Accel Duration** to `20.0` %
4. Set **Cycle Time** to `1.5` s
5. Set **Max Velocity** to `50.0` mm/s
6. Select **Motion Type** as `minimum_energy`
7. Click **Solve Motion Law**

## Output Visualization

The GUI displays four curves on a single plot:

- **Displacement (blue)**: Follower position vs cam angle
- **Velocity (green)**: Follower velocity vs cam angle
- **Acceleration (red)**: Follower acceleration vs cam angle
- **Jerk (magenta)**: Follower jerk vs cam angle

### Plot Features

- **X-axis**: Cam angle in degrees (0° to 360°)
- **TDC/BDC Markers**: Vertical dashed lines at 0° (TDC) and 180° (BDC)
- **Grid**: Light grid for easy reading
- **Legend**: Shows all curve types
- **Interactive**: Zoom and pan capabilities

## Tips for Best Results

### Parameter Selection

1. **Stroke**: Should be positive and reasonable for your application
2. **Upstroke Duration**: Typically 40-70% of cycle
3. **Zero Accel Duration**: Must be ≤ upstroke duration
4. **Cycle Time**: Shorter times require higher constraints
5. **Constraints**: Set realistic limits based on your system capabilities

### Solver Settings

1. **Collocation Degree**: 
   - Use 3 for most applications (good balance of speed/accuracy)
   - Use 5-7 for high-precision requirements
   - Use 1-2 for quick prototyping

2. **Collocation Method**:
   - **Legendre**: Good general-purpose choice
   - **Radau**: Better for problems with constraints
   - **Lobatto**: Good for smooth solutions

### Troubleshooting

1. **Solver Fails**: Try reducing constraints or increasing cycle time
2. **Slow Solving**: Reduce collocation degree or use simpler constraints
3. **Inaccurate Results**: Increase collocation degree
4. **Constraints Not Met**: Check if constraints are physically achievable

## File Formats

The GUI can save plots in multiple formats:

- **PNG**: High-quality raster images (recommended for presentations)
- **PDF**: Vector format (recommended for publications)
- **SVG**: Scalable vector graphics (recommended for web use)

## Technical Details

### Threading

The optimization runs in a separate thread to prevent GUI freezing. The status bar shows progress, and the solve button is disabled during computation.

### Error Handling

The GUI includes comprehensive error handling for:
- Invalid input parameters
- Solver convergence failures
- File I/O errors
- Plot generation errors

### Performance

Typical solving times:
- **Simple problems**: 0.1-1.0 seconds
- **Complex problems**: 1.0-10.0 seconds
- **High-precision problems**: 10.0+ seconds

## Integration with Code

The GUI can be integrated into larger applications:

```python
from cam_motion_gui import CamMotionGUI
import tkinter as tk

# Create GUI programmatically
root = tk.Tk()
app = CamMotionGUI(root)

# Set parameters programmatically
app.variables['stroke'].set(25.0)
app.variables['upstroke_duration'].set(50.0)

# Run GUI
root.mainloop()
```

## Dependencies

- **tkinter**: GUI framework (included with Python)
- **matplotlib**: Plotting library
- **numpy**: Numerical computations
- **casadi**: Symbolic computation and optimization
- **threading**: For non-blocking optimization

## Support

For issues or questions:
1. Check the console output for error messages
2. Verify all input parameters are valid
3. Try reducing problem complexity
4. Check that constraints are physically achievable
