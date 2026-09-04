import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless test environments

import numpy as np
import pytest
from lab_math_tools.visualization import plot_shape_2d, plot_shape_3d


def test_plot_shape_2d_basic():
    """Test 2D shape plotting returns a valid matplotlib Axes."""
    def circle_shape(v: np.ndarray) -> float:
        return float(v[0] ** 2 + v[1] ** 2 <= 1.0)

    bounds = np.array([[-1.5, 1.5], [-1.5, 1.5]])
    ax = plot_shape_2d(circle_shape, bounds, resolution=50, show=False)
    assert ax is not None
    assert ax.get_title() == "2D Shape Function"


def test_plot_shape_2d_invalid_bounds():
    """Test that invalid bounds raise ValueError in 2D plot."""
    def shape(v): return 1.0

    # 3D bounds passed to 2D plot
    with pytest.raises(ValueError, match="am_bounds must be a \\(2, 2\\) array"):
        plot_shape_2d(shape, np.zeros((3, 2)), show=False)

    # Lower > Upper
    with pytest.raises(ValueError, match="Lower bounds must not exceed upper bounds."):
        plot_shape_2d(shape, np.array([[2.0, 1.0], [0.0, 1.0]]), show=False)

    # Resolution <= 1
    with pytest.raises(ValueError, match="Resolution must be strictly greater than 1."):
        plot_shape_2d(shape, np.array([[0.0, 1.0], [0.0, 1.0]]), resolution=1, show=False)


def test_plot_shape_3d_voxels():
    """Test 3D voxel rendering returns a valid 3D Axes."""
    def sphere_shape(v: np.ndarray) -> bool:
        return bool(v[0] ** 2 + v[1] ** 2 + v[2] ** 2 <= 1.0)

    bounds = np.array([[-1.2, 1.2], [-1.2, 1.2], [-1.2, 1.2]])
    ax = plot_shape_3d(sphere_shape, bounds, method="voxels", resolution=15, show=False)
    assert ax is not None
    assert ax.name == "3d"


def test_plot_shape_3d_scatter():
    """Test 3D scatter point cloud rendering returns a valid 3D Axes."""
    def cylinder_shape(v: np.ndarray) -> bool:
        return bool(v[0] ** 2 + v[1] ** 2 <= 1.0 and -1.0 <= v[2] <= 1.0)

    bounds = np.array([[-1.5, 1.5], [-1.5, 1.5], [-1.5, 1.5]])
    ax = plot_shape_3d(cylinder_shape, bounds, method="scatter", n_samples=2000, rng=42, show=False)
    assert ax is not None
    assert ax.name == "3d"


def test_plot_shape_3d_invalid_arguments():
    """Test that invalid bounds or methods raise ValueError in 3D plot."""
    def shape(v): return True

    # 2D bounds passed to 3D plot
    with pytest.raises(ValueError, match="am_bounds must be a \\(3, 2\\) array"):
        plot_shape_3d(shape, np.zeros((2, 2)), show=False)

    # Lower > Upper
    with pytest.raises(ValueError, match="Lower bounds must not exceed upper bounds."):
        plot_shape_3d(shape, np.array([[0.0, 1.0], [2.0, 1.0], [0.0, 1.0]]), show=False)

    # Invalid method
    with pytest.raises(ValueError, match="method must be either 'voxels' or 'scatter'"):
        plot_shape_3d(shape, np.array([[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]]), method="raymarching", show=False)
