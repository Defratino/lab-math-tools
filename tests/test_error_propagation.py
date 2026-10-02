import numpy as np
import pytest
from lab_math_tools.error_propagation import (
    propagate_uncertainty,
    relative_uncertainty,
)

# --- propagate_uncertainty tests ---


def test_propagate_uncertainty_scalar():
    """Test scalar (R -> R) error propagation."""
    def f(x: float) -> float:
        return x**2

    # df/dx = 2x. At x=3, df/dx = 6. Error = |6 * 0.5| = 3.0
    result = propagate_uncertainty(f, x=3.0, dx=0.5)
    np.testing.assert_allclose(result, 3.0, rtol=1e-4)


def test_propagate_uncertainty_vector_statistical():
    """Test vector (R^n -> R) statistical error propagation."""
    def s_f(v2_x: np.ndarray) -> float:
        return v2_x[0] * v2_x[1]

    v_vals = np.array([10.0, 5.0])
    v_errs = np.array([0.5, 0.2])

    # df/dx0 = x1 = 5. Term: 5 * 0.5 = 2.5
    # df/dx1 = x0 = 10. Term: 10 * 0.2 = 2.0
    # Stat error = sqrt(2.5^2 + 2.0^2) = sqrt(6.25 + 4.0) = sqrt(10.25)
    expected = np.sqrt(10.25)

    result = propagate_uncertainty(s_f, v_vals, v_errs, method="statistical")
    np.testing.assert_allclose(result, expected, rtol=1e-4)


def test_propagate_uncertainty_vector_absolute():
    """Test vector (R^n -> R) absolute error propagation."""
    def s_f(v2_x: np.ndarray) -> float:
        return v2_x[0] * v2_x[1]

    v_vals = np.array([10.0, 5.0])
    v_errs = np.array([0.5, 0.2])

    # Abs error = |2.5| + |2.0| = 4.5
    result = propagate_uncertainty(s_f, v_vals, v_errs, method="absolute")
    np.testing.assert_allclose(result, 4.5, rtol=1e-4)


def test_propagate_uncertainty_invalid_method():
    """Test that an invalid method raises a ValueError."""
    def f(x: float) -> float:
        return x

    with pytest.raises(ValueError, match="Method must be 'statistical' or 'absolute'."):
        propagate_uncertainty(f, np.array([1.0]), np.array([0.1]), method="magic")


def test_propagate_uncertainty_dimension_mismatch():
    """Test that mismatched value and error dimensions raise ValueError."""
    def s_f(v_x): return v_x[0] + v_x[1]
    with pytest.raises(ValueError, match="dimension"):
        propagate_uncertainty(s_f, np.array([1.0, 2.0]), np.array([0.1]))


def test_propagate_uncertainty_scalar_vector_mismatch():
    """Test that passing scalar x and vector dx raises ValueError."""
    def f(x): return x
    with pytest.raises(ValueError, match="must both be scalars or both be vectors"):
        propagate_uncertainty(f, 2.0, np.array([0.1, 0.2]))




# --- relative_uncertainty tests ---


def test_relative_uncertainty():
    """Test relative uncertainty calculation."""
    def s_f(v2_x: np.ndarray) -> float:
        return v2_x[0] * v2_x[1]

    v_vals = np.array([10.0, 5.0])  # Nominal f = 50.0
    v_errs = np.array([0.5, 0.2])

    abs_err = np.sqrt(10.25)
    expected_rel_err = abs_err / 50.0

    result = relative_uncertainty(s_f, v_vals, v_errs)
    np.testing.assert_allclose(result, expected_rel_err, rtol=1e-4)


def test_relative_uncertainty_zero_division():
    """Test that a nominal value of zero raises a ZeroDivisionError."""
    def s_f(v2_x: np.ndarray) -> float:
        return v2_x[0] * v2_x[1]

    v_vals = np.array([0.0, 5.0])  # Nominal f = 0.0
    v_errs = np.array([0.5, 0.2])

    with pytest.raises(ZeroDivisionError):
        relative_uncertainty(s_f, v_vals, v_errs)


def test_error_propagation_invalid_h():
    """Test that non-positive step size h raises ValueError across error propagation routines."""
    def f(x): return x**2
    with pytest.raises(ValueError, match="Step size h must be strictly positive."):
        propagate_uncertainty(f, x=2.0, dx=0.1, h=0.0)