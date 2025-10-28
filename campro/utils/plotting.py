"""
Plotting utilities for motion law visualization.

This module provides functions for creating and customizing plots
of motion law solutions with smart scaling and professional styling.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from matplotlib.figure import Figure

from campro.logging import get_logger

log = get_logger(__name__)


def plot_solution(
    solution: dict[str, np.ndarray],
    save_path: str | Path | None = None,
    title: str = "Motion Law Solution",
    use_cam_angle: bool = False,
) -> Figure:
    """
    Create a comprehensive plot of motion law solution.

    Args:
        solution: Dictionary containing solution arrays
        save_path: Optional path to save the plot
        title: Plot title
        use_cam_angle: Whether to plot against cam angle instead of time

    Returns:
        matplotlib Figure object
    """
    fig = Figure(figsize=(12, 8), dpi=100)

    if use_cam_angle and "cam_angle" in solution:
        x_data = solution["cam_angle"]
        x_label = "Cam Angle (degrees)"
    else:
        x_data = solution.get("time", np.arange(len(solution.get("position", []))))
        x_label = "Time (s)"

    # Create subplots
    axes = fig.subplots(2, 2)
    fig.suptitle(title, fontsize=14, fontweight="bold")

    # Define curves to plot
    curves = [
        ("position", "b-", "Displacement", "mm", axes[0, 0]),
        ("velocity", "g-", "Velocity", "mm/s", axes[0, 1]),
        ("acceleration", "r-", "Acceleration", "mm/s²", axes[1, 0]),
        ("control", "m-", "Jerk", "mm/s³", axes[1, 1]),
    ]

    for data_key, color, title_text, unit, ax in curves:
        if data_key in solution:
            data = solution[data_key]
            ax.plot(x_data, data, color, linewidth=2)

            # Add reference lines
            if use_cam_angle:
                ax.axvline(x=0, color="black", linestyle="--", alpha=0.5)
                # Add BDC marker if cam angle data is available
                if "position" in solution:
                    max_pos_idx = np.argmax(solution["position"])
                    bdc_angle = x_data[max_pos_idx]
                    ax.axvline(x=bdc_angle, color="black", linestyle="--", alpha=0.5)
            else:
                ax.axhline(y=0, color="gray", linestyle="-", alpha=0.3, linewidth=0.5)

            # Apply smart scaling
            apply_smart_scaling(ax, data, x_data)

            # Formatting
            ax.set_xlabel(x_label)
            ax.set_ylabel(f"{title_text} ({unit})")
            ax.set_title(f"{title_text} vs {x_label.split()[0]}")
            ax.grid(True, alpha=0.3)

            # Add statistics box
            add_statistics_box(ax, data, unit)

    # Adjust layout
    fig.tight_layout()

    # Save if requested
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        log.info(f"Plot saved to {save_path}")

    return fig


def create_smart_scaled_plots(
    cam_angle: np.ndarray,
    position: np.ndarray,
    velocity: np.ndarray,
    acceleration: np.ndarray,
    jerk: np.ndarray,
    fig: Figure,
) -> None:
    """
    Create smart-scaled subplots for each motion law curve.

    Args:
        cam_angle: Cam angle array
        position: Position array
        velocity: Velocity array
        acceleration: Acceleration array
        jerk: Jerk array
        fig: matplotlib Figure object to populate
    """
    # Create 2x2 subplot layout
    axes = fig.subplots(2, 2)
    fig.suptitle("Cam Motion Law Curves", fontsize=14, fontweight="bold")

    # Define colors and labels
    curves = [
        (position, "b-", "Displacement", "mm", axes[0, 0]),
        (velocity, "g-", "Velocity", "mm/s", axes[0, 1]),
        (acceleration, "r-", "Acceleration", "mm/s²", axes[1, 0]),
        (jerk, "m-", "Jerk", "mm/s³", axes[1, 1]),
    ]

    for data, color, title, unit, ax in curves:
        # Plot the curve
        ax.plot(cam_angle, data, color, linewidth=2)

        # Add TDC/BDC markers
        ax.axvline(x=0, color="black", linestyle="--", alpha=0.5)

        # Calculate BDC position based on maximum position
        max_pos_idx = np.argmax(position)
        bdc_angle = cam_angle[max_pos_idx]
        ax.axvline(x=bdc_angle, color="black", linestyle="--", alpha=0.5)

        # Smart scaling
        apply_smart_scaling(ax, data, cam_angle)

        # Formatting
        ax.set_xlabel("Cam Angle (degrees)")
        ax.set_ylabel(f"{title} ({unit})")
        ax.set_title(f"{title} vs Cam Angle")
        ax.grid(True, alpha=0.3)

        # Add statistics text box
        add_statistics_box(ax, data, unit)

    # Adjust layout to prevent overlap
    fig.tight_layout()


def apply_smart_scaling(ax, data: np.ndarray, x_data: np.ndarray) -> None:
    """
    Apply smart scaling to a subplot.

    Args:
        ax: matplotlib axes object
        data: Data array for y-axis scaling
        x_data: Data array for x-axis scaling
    """
    # Calculate data range and add padding
    data_min, data_max = data.min(), data.max()
    data_range = data_max - data_min

    # Handle edge cases
    if data_range == 0:
        # All values are the same
        center = data_min
        y_min = center - abs(center) * 0.1 if center != 0 else -1
        y_max = center + abs(center) * 0.1 if center != 0 else 1
    else:
        # Add 10% padding on each side
        padding = data_range * 0.1
        y_min = data_min - padding
        y_max = data_max + padding

    # Set y-axis limits
    ax.set_ylim(y_min, y_max)

    # Set x-axis limits
    x_min, x_max = x_data.min(), x_data.max()
    if x_max - x_min > 0:
        x_padding = (x_max - x_min) * 0.02  # 2% padding
        ax.set_xlim(x_min - x_padding, x_max + x_padding)

    # Add horizontal line at zero if data crosses zero
    if y_min <= 0 <= y_max:
        ax.axhline(y=0, color="gray", linestyle="-", alpha=0.3, linewidth=0.5)


def add_statistics_box(ax, data: np.ndarray, unit: str) -> None:
    """
    Add a statistics text box to the subplot.

    Args:
        ax: matplotlib axes object
        data: Data array for statistics
        unit: Unit string for display
    """
    # Calculate statistics
    max_val = data.max()
    min_val = data.min()
    mean_val = data.mean()
    rms_val = np.sqrt(np.mean(data**2))

    # Create text box
    stats_text = f"Max: {max_val:.2f} {unit}\nMin: {min_val:.2f} {unit}\nMean: {mean_val:.2f} {unit}\nRMS: {rms_val:.2f} {unit}"

    # Position text box in upper right corner
    ax.text(
        0.98,
        0.98,
        stats_text,
        transform=ax.transAxes,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        fontsize=8,
        family="monospace",
    )


def create_single_plot(
    cam_angle: np.ndarray,
    position: np.ndarray,
    velocity: np.ndarray,
    acceleration: np.ndarray,
    jerk: np.ndarray,
    fig: Figure,
) -> None:
    """
    Create a single plot with all curves (traditional view).

    Args:
        cam_angle: Cam angle array
        position: Position array
        velocity: Velocity array
        acceleration: Acceleration array
        jerk: Jerk array
        fig: matplotlib Figure object to populate
    """
    # Create single subplot
    ax = fig.add_subplot(111)
    fig.suptitle("Cam Motion Law Curves", fontsize=14, fontweight="bold")

    # Plot all curves on the same axes
    ax.plot(cam_angle, position, "b-", linewidth=2, label="Displacement (mm)")
    ax.plot(cam_angle, velocity, "g-", linewidth=2, label="Velocity (mm/s)")
    ax.plot(cam_angle, acceleration, "r-", linewidth=2, label="Acceleration (mm/s²)")
    ax.plot(cam_angle, jerk, "m-", linewidth=2, label="Jerk (mm/s³)")

    # Add TDC/BDC markers
    ax.axvline(x=0, color="black", linestyle="--", alpha=0.5, label="TDC")

    # Calculate BDC position based on maximum position
    max_pos_idx = np.argmax(position)
    bdc_angle = cam_angle[max_pos_idx]
    ax.axvline(x=bdc_angle, color="black", linestyle="--", alpha=0.5, label="BDC")

    # Formatting
    ax.set_xlabel("Cam Angle (degrees)")
    ax.set_ylabel("Value")
    ax.set_title("All Motion Law Curves")
    ax.grid(True, alpha=0.3)
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

    # Set x-axis limits
    ax.set_xlim(0, 360)

    # Auto-scale y-axis to fit all data
    all_data = np.concatenate([position, velocity, acceleration, jerk])
    data_min, data_max = all_data.min(), all_data.max()
    data_range = data_max - data_min

    if data_range > 0:
        padding = data_range * 0.1
        ax.set_ylim(data_min - padding, data_max + padding)

    # Add horizontal line at zero if data crosses zero
    if data_min <= 0 <= data_max:
        ax.axhline(y=0, color="gray", linestyle="-", alpha=0.3, linewidth=0.5)

    # Adjust layout to prevent overlap
    fig.tight_layout()
