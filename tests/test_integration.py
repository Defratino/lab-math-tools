import numpy as np
import pytest
from lab_math_tools.integration import (
    integral_over_shape,
    integral_trapezoidal,
)


def test_integral_trapezoidal_scalar():
    """Test 1D definite integral using the composite Trapezoidal rule for scalar functions."""
    # Integrate f(x) = x^2 from 0 to 3
    # Exact value: [x^3 / 3] from 0 to 3 = 27 / 3 = 9.0
    def s_f(x: float | np.ndarray) -> float | np.ndarray:
        return x**2

    result = integral_trapezoidal(s_f, a=0.0, b=3.0, n_steps=1000)
    assert isinstance(result, float)
    np.testing.assert_allclose(result, 9.0, rtol=1e-3)


def test_integral_trapezoidal_vector():
    """Test 1D definite integral using the composite Trapezoidal rule for vector functions (R -> R^n)."""
    # Integrate f(x) = [x, 2*x, x**2] from 0 to 2
    # Exact: [x^2/2, x^2, x^3/3] from 0 to 2 = [2.0, 4.0, 8.0 / 3.0]
    def v3_f(x: float | np.ndarray) -> np.ndarray:
        return np.array([x, 2 * x, x**2])

    result = integral_trapezoidal(v3_f, a=0.0, b=2.0, n_steps=1000)
    assert isinstance(result, np.ndarray)
    expected = np.array([2.0, 4.0, 8.0 / 3.0])
    np.testing.assert_allclose(result, expected, rtol=1e-3)


def test_integral_trapezoidal_invalid_steps():
    """Test that n_steps <= 0 raises a ValueError."""
    def f(x): return x
    with pytest.raises(ValueError, match="Number of steps must be strictly positive."):
        integral_trapezoidal(f, a=0.0, b=1.0, n_steps=0)
    with pytest.raises(ValueError, match="Number of steps must be strictly positive."):
        integral_trapezoidal(f, a=0.0, b=1.0, n_steps=-10)


def test_integral_over_shape_monte_carlo():
    """Test n-dimensional Monte Carlo integration over a custom shape with a seeded RNG."""
    # Integrate f(x, y) = x + y over a 2D unit square [0, 1] x [0, 1]
    # Exact value: int_0^1 int_0^1 (x + y) dx dy = 1.0
    def s_f(v_x: np.ndarray) -> float:
        return v_x[0] + v_x[1]

    # Indicator function defining the unit square region (returns 1.0 inside, 0.0 outside)
    def s_shape(v_x: np.ndarray) -> float:
        return 1.0 if (0.0 <= v_x[0] <= 1.0 and 0.0 <= v_x[1] <= 1.0) else 0.0

    am_bounds = np.array([
        [0.0, 1.0],  # x bounds
        [0.0, 1.0],  # y bounds
    ])

    result = integral_over_shape(s_f, s_shape, am_bounds, n_samples=50000, rng=42)
    np.testing.assert_allclose(result, 1.0, rtol=0.03)


def test_integral_over_shape_vectorized():
    """Test that vectorized functions run through the fast path correctly."""
    def vf(pts: np.ndarray) -> np.ndarray:
        return pts[:, 0] + pts[:, 1]

    def vshape(pts: np.ndarray) -> np.ndarray:
        return ((pts[:, 0] >= 0) & (pts[:, 0] <= 1) & (pts[:, 1] >= 0) & (pts[:, 1] <= 1)).astype(float)

    am_bounds = np.array([[0.0, 1.0], [0.0, 1.0]])
    result = integral_over_shape(vf, vshape, am_bounds, n_samples=50000, rng=42)
    np.testing.assert_allclose(result, 1.0, rtol=0.03)


def test_integral_over_shape_invalid_bounds():
    """Test that invalid bounds raise ValueError."""
    def s_f(x): return 1.0
    def s_shape(x): return 1.0

    # 1D bounds array instead of 2D
    with pytest.raises(ValueError, match="am_bounds must be a 2D array"):
        integral_over_shape(s_f, s_shape, np.array([0.0, 1.0]))

    # Min > Max
    with pytest.raises(ValueError, match="Lower bounds must not exceed upper bounds."):
        integral_over_shape(s_f, s_shape, np.array([[2.0, 1.0]]))