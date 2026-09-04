"""
Geometric Shape Visualization Module

This module provides 2D and 3D visualization tools for implicit shape functions 
(s_shape: R^n -> {0, 1}) used in geometric analysis and Monte Carlo integration.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np


def _get_plt():
    """
    Lazy import helper for matplotlib to keep the core library lightweight.
    """
    try:
        import matplotlib.pyplot as plt
        return plt
    except ImportError as exc:
        raise ImportError(
            "Matplotlib is required for visualization functions. "
            "Install it using: pip install lab_math_tools[viz] or pip install matplotlib"
        ) from exc


def plot_shape_2d(
    s_shape: Callable[[np.ndarray], float | int | bool],
    am_bounds: np.ndarray,
    resolution: int = 200,
    title: str = "2D Shape Function",
    color: str = "#3b82f6",
    show: bool = True,
    ax: Any | None = None,
) -> Any:
    """
    Visualizes a 2D shape indicator function within the specified bounding box.

    Parameters:
    * s_shape (callable): Indicator function defining the shape (R^2 -> [0, 1]).
    * am_bounds (np.ndarray): A (2, 2) array containing [min, max] limits for x and y.
    * resolution (int): Grid resolution for evaluation.
    * title (str): Title for the plot.
    * color (str): Primary fill color for the shape.
    * show (bool): Whether to call plt.show() immediately.
    * ax (matplotlib.axes.Axes | None): Optional existing Matplotlib Axes to draw on.

    Returns:
    * matplotlib.axes.Axes: The Axes object containing the visualization.
    """
    am_bounds_arr = np.asarray(am_bounds, dtype=float)
    if am_bounds_arr.shape != (2, 2):
        raise ValueError("am_bounds must be a (2, 2) array of [min, max] limits for x and y.")
    if np.any(am_bounds_arr[:, 0] > am_bounds_arr[:, 1]):
        raise ValueError("Lower bounds must not exceed upper bounds.")
    if resolution <= 1:
        raise ValueError("Resolution must be strictly greater than 1.")

    plt = _get_plt()

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))

    xs = np.linspace(am_bounds_arr[0, 0], am_bounds_arr[0, 1], resolution)
    ys = np.linspace(am_bounds_arr[1, 0], am_bounds_arr[1, 1], resolution)
    X, Y = np.meshgrid(xs, ys)

    # Attempt vectorized evaluation first; fallback to grid looping
    try:
        pts = np.column_stack([X.ravel(), Y.ravel()])
        evaluated = np.asarray(s_shape(pts)).reshape(X.shape)
        Z = evaluated.astype(float)
    except Exception:
        Z = np.zeros_like(X, dtype=float)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                Z[i, j] = float(s_shape(np.array([X[i, j], Y[i, j]])))

    # Draw filled interior and boundary contour
    ax.contourf(X, Y, Z, levels=[0.5, 1.0], colors=[color], alpha=0.5)
    ax.contour(X, Y, Z, levels=[0.5], colors=[color], linewidths=2)

    # Draw bounding box outline
    bx = [am_bounds_arr[0, 0], am_bounds_arr[0, 1], am_bounds_arr[0, 1], am_bounds_arr[0, 0], am_bounds_arr[0, 0]]
    by = [am_bounds_arr[1, 0], am_bounds_arr[1, 0], am_bounds_arr[1, 1], am_bounds_arr[1, 1], am_bounds_arr[1, 0]]
    ax.plot(bx, by, "--", color="#94a3b8", label="Bounding Box")

    ax.set_aspect("equal")
    ax.set_xlim(am_bounds_arr[0, 0] - 0.05 * (am_bounds_arr[0, 1] - am_bounds_arr[0, 0]),
                am_bounds_arr[0, 1] + 0.05 * (am_bounds_arr[0, 1] - am_bounds_arr[0, 0]))
    ax.set_ylim(am_bounds_arr[1, 0] - 0.05 * (am_bounds_arr[1, 1] - am_bounds_arr[1, 0]),
                am_bounds_arr[1, 1] + 0.05 * (am_bounds_arr[1, 1] - am_bounds_arr[1, 0]))
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="upper right")

    if show:
        plt.show()

    return ax


def plot_shape_3d(
    s_shape: Callable[[np.ndarray], float | int | bool],
    am_bounds: np.ndarray,
    method: str = "voxels",
    resolution: int = 30,
    n_samples: int = 25000,
    title: str = "3D Shape Function",
    color: str = "#3b82f6",
    rng: np.random.Generator | int | None = None,
    show: bool = True,
    ax: Any | None = None,
) -> Any:
    """
    Visualizes a 3D shape indicator function within the specified bounding box.

    Parameters:
    * s_shape (callable): Indicator function defining the shape (R^3 -> [0, 1]).
    * am_bounds (np.ndarray): A (3, 2) array containing [min, max] limits for x, y, and z.
    * method (str): Rendering method:
        - "voxels": Discretizes volume into solid voxel cubes.
        - "scatter": Generates a Monte Carlo point cloud of interior points.
    * resolution (int): Grid resolution per dimension when method="voxels".
    * n_samples (int): Number of random points to sample when method="scatter".
    * title (str): Title for the 3D plot.
    * color (str): Hex color string for rendering.
    * rng (np.random.Generator | int | None): Optional random seed/generator for "scatter".
    * show (bool): Whether to call plt.show() immediately.
    * ax (matplotlib.axes.Axes | None): Optional existing 3D Axes.

    Returns:
    * matplotlib.axes.Axes: The 3D Axes object containing the visualization.
    """
    am_bounds_arr = np.asarray(am_bounds, dtype=float)
    if am_bounds_arr.shape != (3, 2):
        raise ValueError("am_bounds must be a (3, 2) array of [min, max] limits for x, y, and z.")
    if np.any(am_bounds_arr[:, 0] > am_bounds_arr[:, 1]):
        raise ValueError("Lower bounds must not exceed upper bounds.")

    plt = _get_plt()

    if ax is None:
        fig = plt.figure(figsize=(8, 7))
        ax = fig.add_subplot(111, projection="3d")

    if method == "voxels":
        if resolution <= 1:
            raise ValueError("Resolution must be strictly greater than 1.")

        xs = np.linspace(am_bounds_arr[0, 0], am_bounds_arr[0, 1], resolution)
        ys = np.linspace(am_bounds_arr[1, 0], am_bounds_arr[1, 1], resolution)
        zs = np.linspace(am_bounds_arr[2, 0], am_bounds_arr[2, 1], resolution)

        voxel_grid = np.zeros((resolution, resolution, resolution), dtype=bool)

        for i, x_val in enumerate(xs):
            for j, y_val in enumerate(ys):
                for k, z_val in enumerate(zs):
                    voxel_grid[i, j, k] = bool(s_shape(np.array([x_val, y_val, z_val])))

        # Render voxels with transparency
        facecolor = color + "80" if len(color) == 7 and color.startswith("#") else color
        ax.voxels(voxel_grid, facecolors=facecolor, edgecolors="#1e3a8a30")

    elif method == "scatter":
        if n_samples <= 0:
            raise ValueError("n_samples must be strictly positive.")

        generator = rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)
        random_points = generator.uniform(
            low=am_bounds_arr[:, 0],
            high=am_bounds_arr[:, 1],
            size=(n_samples, 3),
        )

        # Filter points inside shape
        try:
            evaluated = np.asarray(s_shape(random_points))
            if evaluated.shape == (n_samples,):
                inside_mask = evaluated > 0
            else:
                raise ValueError()
        except Exception:
            inside_mask = np.array([bool(s_shape(pt)) for pt in random_points])

        interior_points = random_points[inside_mask]

        if len(interior_points) > 0:
            ax.scatter(
                interior_points[:, 0],
                interior_points[:, 1],
                interior_points[:, 2],
                c=color,
                alpha=0.35,
                s=5,
                edgecolors="none",
            )

        ax.set_xlim(am_bounds_arr[0, 0], am_bounds_arr[0, 1])
        ax.set_ylim(am_bounds_arr[1, 0], am_bounds_arr[1, 1])
        ax.set_zlim(am_bounds_arr[2, 0], am_bounds_arr[2, 1])

    else:
        raise ValueError("method must be either 'voxels' or 'scatter'.")

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title(title)

    if show:
        plt.show()

    return ax
