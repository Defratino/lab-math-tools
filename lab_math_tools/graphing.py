"""
Graphing module

This module provides graphing tools for experimental data (2D curves and measurements).

Conventions followed:
- `v_` : Abstract vector (numpy.ndarray with dynamic dimensions)
- `m_` : Abstract matrix (numpy.ndarray with dynamic dimensions)
- `l_` : Python list of numbers
- `lv_`: Python list of vectors
- `lm_`: Python list of matrices
(Note: Lists of strings, such as `line_labels` or `linestyles`, do not require the `l_` prefix.)
"""

from __future__ import annotations

import warnings
from typing import Any, Sequence

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np


def plot_measurements(
    lv_x: Any = None,
    lv_y: Any = None,
    x_label: str = "X",
    y_label: str = "Y",
    title: str = "",
    line_labels: list[str] | None = None,
    grid: bool = True,
    linestyles: str | list[str] | None = None,
    markers: str | list[str] | None = None,
    colors: str | list[str] | None = None,
    color_method: str = "gradient",
    save_file_name: str = "",
    ax: plt.Axes | None = None,
    show: bool = True,
    *,
    m_x: Any = None,
    m_y: Any = None,
    v_x: Any = None,
    v_y: Any = None,
) -> plt.Axes:
    """
    Plot multiple measurements on a 2D graph.

    Parameters:
    * lv_x (Sequence | np.ndarray): X-values of the measurements. Can be:
        - A single 1D vector (shared across all y-series)
        - A list/sequence of 1D vectors for each measurement
        - A 2D matrix where each row is a measurement
    * lv_y (Sequence | np.ndarray): Y-values of the measurements. Can be:
        - A single 1D vector (for a single measurement)
        - A list/sequence of 1D vectors for each measurement
        - A 2D matrix where each row is a measurement
    * x_label (str): Label for the x-axis. Default is "X".
    * y_label (str): Label for the y-axis. Default is "Y".
    * title (str): Title of the graph.
    * line_labels (list[str] | None): Labels for each measurement line in the legend.
    * grid (bool): Whether to display a grid. Default is True.
    * linestyles (str | list[str] | None): Style(s) of the lines (e.g. '-', '--').
    * markers (str | list[str] | None): Marker style(s) for data points (e.g. '.', 'o').
    * colors (str | list[str] | None): Colors of the lines or name of a colormap.
    * color_method (str): Method used to determine the line colors:
        - "gradient": Colors vary smoothly between 2 colors (default colors: ['blue', 'green']).
        - "specific": Specific colors supplied for each measurement in `colors`.
        - "colormap": Colors sampled from a Matplotlib colormap name passed in `colors`.
    * save_file_name (str): If provided, saves the figure to this file path.
    * ax (plt.Axes | None): Matplotlib Axes object to plot on. If None, creates a new figure and axes.
    * show (bool): Whether to call plt.show() at the end. Default is True.

    Returns:
    * plt.Axes: The Matplotlib Axes object containing the plot.

    Backward Compatibility Keyword Arguments:
    * m_x, m_y: Aliases for 2D matrix or vector inputs.
    * v_x, v_y: Aliases for single vector inputs.
    """
    # Backward compatibility resolution for x and y
    if lv_x is None:
        if m_x is not None:
            lv_x = m_x
        elif v_x is not None:
            lv_x = v_x
        else:
            raise ValueError("x-data must be provided via lv_x (or m_x / v_x).")

    if lv_y is None:
        if m_y is not None:
            lv_y = m_y
        elif v_y is not None:
            lv_y = v_y
        else:
            raise ValueError("y-data must be provided via lv_y (or m_y / v_y).")

    # Create figure and axes if not provided
    if ax is None:
        _, ax = plt.subplots()

    # Normalize inputs to paired lists of 1D numpy arrays
    lv_x_norm, lv_y_norm = _normalize_xy(lv_x, lv_y)
    n_measurements = len(lv_y_norm)

    if n_measurements == 0:
        return ax

    # Normalize linestyles
    if linestyles is None:
        linestyles = ["-"] * n_measurements
    elif isinstance(linestyles, str):
        linestyles = [linestyles] * n_measurements
    elif isinstance(linestyles, list):
        if len(linestyles) == 1:
            linestyles = [linestyles[0]] * n_measurements
        elif len(linestyles) != n_measurements:
            raise ValueError(
                f"linestyles list length ({len(linestyles)}) must match the number of measurements ({n_measurements})."
            )
    else:
        raise TypeError("linestyles must be a string or a list of strings.")

    # Normalize markers
    if markers is None:
        markers = ["."] * n_measurements
    elif isinstance(markers, str):
        markers = [markers] * n_measurements
    elif isinstance(markers, list):
        if len(markers) == 1:
            markers = [markers[0]] * n_measurements
        elif len(markers) != n_measurements:
            raise ValueError(
                f"markers list length ({len(markers)}) must match the number of measurements ({n_measurements})."
            )
    else:
        raise TypeError("markers must be a string or a list of strings.")

    # Validate line_labels
    if line_labels is not None:
        if len(line_labels) != n_measurements:
            raise ValueError(
                f"line_labels length ({len(line_labels)}) must match the number of measurements ({n_measurements})."
            )

    # Resolve colors
    resolved_colors = _resolve_colors(n_measurements, colors, color_method)

    # Plotting loop
    for i in range(n_measurements):
        label = line_labels[i] if line_labels is not None else None
        ax.plot(
            lv_x_norm[i],
            lv_y_norm[i],
            linestyle=linestyles[i],
            marker=markers[i],
            color=resolved_colors[i],
            label=label,
        )

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    if title:
        ax.set_title(title)
    if line_labels is not None:
        ax.legend()
    if grid:
        ax.grid(True)
    if save_file_name:
        ax.figure.savefig(save_file_name)

    if show:
        plt.show()

    return ax


def _normalize_xy(
    lv_x: Any,
    lv_y: Any,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """
    Normalize arbitrary x and y inputs into paired lists of 1D numpy arrays: (lv_x_final, lv_y_out).
    """
    # 1. Normalize Y to a list of 1D arrays
    if isinstance(lv_y, np.ndarray):
        if lv_y.ndim == 1:
            lv_y_out = [lv_y.astype(float)]
        elif lv_y.ndim == 2:
            lv_y_out = [row.astype(float) for row in lv_y]
        else:
            raise ValueError("y-data must be 1-dimensional or 2-dimensional.")
    elif isinstance(lv_y, (list, tuple)):
        if len(lv_y) == 0:
            return [], []
        # Check if it is a 1D list of scalar numbers
        if isinstance(lv_y[0], (int, float, np.number)):
            lv_y_out = [np.asarray(lv_y, dtype=float)]
        else:
            lv_y_out = [np.asarray(row, dtype=float) for row in lv_y]
    else:
        raise TypeError("y-data must be a numpy array, list of numbers, or list of vectors.")

    n_measurements = len(lv_y_out)
    if n_measurements == 0:
        return [], []

    # 2. Normalize X to a list of 1D arrays
    if isinstance(lv_x, np.ndarray):
        if lv_x.ndim == 1:
            lv_x_out = [lv_x.astype(float)]
        elif lv_x.ndim == 2:
            if lv_x.shape[0] == 1 and n_measurements > 1:
                lv_x_out = [lv_x[0].astype(float)]
            elif lv_x.shape[0] == n_measurements:
                lv_x_out = [row.astype(float) for row in lv_x]
            else:
                raise ValueError(
                    f"Number of rows in x matrix ({lv_x.shape[0]}) does not match y measurements ({n_measurements})."
                )
        else:
            raise ValueError("x-data must be 1-dimensional or 2-dimensional.")
    elif isinstance(lv_x, (list, tuple)):
        if len(lv_x) == 0:
            raise ValueError("x-data cannot be empty when y-data is provided.")
        # Check if it is a 1D list of scalar numbers
        if isinstance(lv_x[0], (int, float, np.number)):
            lv_x_out = [np.asarray(lv_x, dtype=float)]
        else:
            if len(lv_x) != n_measurements:
                raise ValueError(
                    f"Number of x measurements ({len(lv_x)}) does not match y measurements ({n_measurements})."
                )
            lv_x_out = [np.asarray(row, dtype=float) for row in lv_x]
    else:
        raise TypeError("x-data must be a numpy array, list of numbers, or list of vectors.")

    # 3. Handle shared single X vector across multiple Y measurements
    if len(lv_x_out) == 1 and n_measurements > 1:
        lv_x_out = [lv_x_out[0]] * n_measurements

    # 4. Validate point counts and trim extra x-values if necessary
    lv_x_final: list[np.ndarray] = []
    for i, (v_x, v_y) in enumerate(zip(lv_x_out, lv_y_out)):
        if len(v_y) > len(v_x):
            raise ValueError(
                f"Not all y-values have x-values: measurement {i} has {len(v_y)} y-points but only {len(v_x)} x-points."
            )
        if len(v_x) > len(v_y):
            warnings.warn(
                f"x-values for measurement {i} were trimmed from {len(v_x)} to {len(v_y)} to match y-values.",
                UserWarning,
                stacklevel=2,
            )
            v_x = v_x[: len(v_y)]
        lv_x_final.append(v_x)

    return lv_x_final, lv_y_out


def _resolve_colors(
    n: int,
    colors: str | list[str] | None,
    color_method: str,
) -> list[str]:
    """
    Resolve colors based on the chosen color_method.
    """
    if n <= 0:
        return []

    # If colors is a colormap name or method is 'colormap'
    if color_method == "colormap" or (isinstance(colors, str) and color_method != "specific"):
        cmap_name = colors if isinstance(colors, str) else "viridis"
        try:
            cmap = plt.get_cmap(cmap_name)
        except ValueError as err:
            raise ValueError(f"Unknown Matplotlib colormap: '{cmap_name}'.") from err

        if n == 1:
            return [mcolors.to_hex(cmap(0.5))]
        return [mcolors.to_hex(cmap(i / (n - 1))) for i in range(n)]

    if color_method == "gradient":
        if colors is None:
            colors = ["blue", "green"]
        if not isinstance(colors, list) or len(colors) != 2:
            raise ValueError("Colors must be a list of 2 colors when using the gradient method.")
        return _color_gradient(n, colors[0], colors[1])

    if color_method == "specific":
        if colors is None:
            raise ValueError("A list of colors must be provided when using color_method='specific'.")
        if not isinstance(colors, list) or len(colors) != n:
            raise ValueError(
                f"Colors must be a list of colors matching the number of measurements ({n})."
            )
        return colors

    raise ValueError(f"Invalid color method '{color_method}', use 'gradient', 'specific', or 'colormap'.")


def _color_gradient(n: int, color_1: str, color_2: str) -> list[str]:
    """
    Generate a list of colors that vary smoothly from one color to another.
    """
    if n <= 0:
        return []
    v_rgb_1 = np.array(mcolors.to_rgb(color_1))
    if n == 1:
        return [mcolors.to_hex(v_rgb_1)]
    v_rgb_2 = np.array(mcolors.to_rgb(color_2))
    return [mcolors.to_hex(v_rgb_1 + (v_rgb_2 - v_rgb_1) * i / (n - 1)) for i in range(n)]


def turn_list_of_vectors_to_matrix(lv_vectors: Sequence[Any]) -> np.ndarray:
    """
    Turn a list of vectors into a 2D matrix of measurements padded with np.nan.

    Parameters:
    * lv_vectors (Sequence): A list or sequence of 1D vectors/sequences.

    Returns:
    * np.ndarray: A 2D array where each row corresponds to a vector,
      padded with np.nan to match the length of the longest vector.
    """
    if not lv_vectors:
        return np.array([]).reshape(0, 0)

    # Convert each element to float array
    lv_converted = [np.asarray(v, dtype=float) for v in lv_vectors]

    # If it was a list of scalars (e.g. [1, 2, 3])
    if all(v.ndim == 0 for v in lv_converted):
        return np.array(lv_converted).reshape(1, -1)

    # Find max length vector in the sequence
    max_len = max(len(v_vec) for v_vec in lv_converted)

    # Pad shorter vectors with np.nan
    lv_padded = []
    for v_vec in lv_converted:
        v_padded = np.pad(
            v_vec,
            (0, max_len - len(v_vec)),
            "constant",
            constant_values=np.nan,
        )
        lv_padded.append(v_padded)

    return np.array(lv_padded)

    
    