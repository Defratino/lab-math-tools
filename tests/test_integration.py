import numpy as np
import pytest
from lab_math_tools.integration import (
    integral_trapezoidal, 
    integral_over_shape,
)


def test_integral_trapezoidal():
    """Test 1D definite integral using the composite Trapezoidal rule."""
    # Integrate f(x) = x^2 from 0 to 3
    # Exact value: [x^3 / 3] from 0 to 3 = 27 / 3 = 9.0
    def s_f(x: float | np.ndarray) -> float | np.ndarray:
        return x**2
    
    result = integral_trapezoidal(s_f, a=0.0, b=3.0, n_steps=1000)
    np.testing.assert_allclose(result, 9.0, rtol=1e-3)


def test_integral_over_shape_monte_carlo():
    """Test n-dimensional Monte Carlo integration over a custom shape."""
    # Integrate f(x, y) = x + y over a 2D unit square [0, 1] x [0, 1]
    # Exact value: int_0^1 int_0^1 (x + y) dx dy = 1.0
    def s_f(v_x: np.ndarray) -> float:
        return v_x[0] + v_x[1]
    
    # Indicator function defining the unit square region (returns 1.0 inside, 0.0 outside)
    def s_shape(v_x: np.ndarray) -> float:
        return 1.0 if (0.0 <= v_x[0] <= 1.0 and 0.0 <= v_x[1] <= 1.0) else 0.0
    
    am_bounds = np.array([
        [0.0, 1.0],  # x bounds
        [0.0, 1.0]   # y bounds
    ])
    
    # Using a moderate sample size for quick test execution while maintaining good statistical convergence
    result = integral_over_shape(s_f, s_shape, am_bounds, n_samples=50000)
    
    # Monte Carlo integration relies on random sampling, so we use a slightly relaxed tolerance (rtol=0.05)
    np.testing.assert_allclose(result, 1.0, rtol=0.05)