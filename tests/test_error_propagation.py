import numpy as np
import pytest
from lab_math_tools.error_propagation import (
    error_contribution_saap,
    propagate_covariance_saap,
    propagate_uncertainty_saap,
    relative_uncertainty_saap,
)

# --- propagate_uncertainty_saap tests ---


def test_propagate_uncertainty_scalar():
    """Test scalar (R -> R) error propagation."""
    def f(x: float) -> float:
        return x**2

    # df/dx = 2x. At x=3, df/dx = 6. Error = |6 * 0.5| = 3.0
    result = propagate_uncertainty_saap(f, x=3.0, dx=0.5)
    np.testing.assert_allclose(result, 3.0, rtol=1e-4)


def test_propagate_uncertainty_vector_statistical():
    """Test vector (R^n -> R) statistical error propagation."""
    def s_f(v2_x: np.ndarray) -> float:
        return v2_x[0] * v2_x[1]

    av_vals = np.array([10.0, 5.0])
    av_errs = np.array([0.5, 0.2])

    # df/dx0 = x1 = 5. Term: 5 * 0.5 = 2.5
    # df/dx1 = x0 = 10. Term: 10 * 0.2 = 2.0
    # Stat error = sqrt(2.5^2 + 2.0^2) = sqrt(6.25 + 4.0) = sqrt(10.25)
    expected = np.sqrt(10.25)

    result = propagate_uncertainty_saap(s_f, av_vals, av_errs, method="statistical")
    np.testing.assert_allclose(result, expected, rtol=1e-4)


def test_propagate_uncertainty_vector_absolute():
    """Test vector (R^n -> R) absolute error propagation."""
    def s_f(v2_x: np.ndarray) -> float:
        return v2_x[0] * v2_x[1]

    av_vals = np.array([10.0, 5.0])
    av_errs = np.array([0.5, 0.2])

    # Abs error = |2.5| + |2.0| = 4.5
    result = propagate_uncertainty_saap(s_f, av_vals, av_errs, method="absolute")
    np.testing.assert_allclose(result, 4.5, rtol=1e-4)


def test_propagate_uncertainty_invalid_method():
    """Test that an invalid method raises a ValueError."""
    def f(x: float) -> float:
        return x

    with pytest.raises(ValueError, match="Method must be 'statistical' or 'absolute'."):
        propagate_uncertainty_saap(f, np.array([1.0]), np.array([0.1]), method="magic")


def test_propagate_uncertainty_dimension_mismatch():
    """Test that mismatched value and error dimensions raise ValueError."""
    def s_f(v_x): return v_x[0] + v_x[1]
    with pytest.raises(ValueError, match="dimension"):
        propagate_uncertainty_saap(s_f, np.array([1.0, 2.0]), np.array([0.1]))


def test_propagate_uncertainty_scalar_vector_mismatch():
    """Test that passing scalar x and vector dx raises ValueError."""
    def f(x): return x
    with pytest.raises(ValueError, match="must both be scalars or both be vectors"):
        propagate_uncertainty_saap(f, 2.0, np.array([0.1, 0.2]))


# --- propagate_covariance_saap tests ---


def test_propagate_covariance_saap():
    """Test full covariance matrix propagation for R^n -> R^m."""
    def v2_f(v2_x: np.ndarray) -> np.ndarray:
        return np.array([v2_x[0] + v2_x[1], v2_x[0] * v2_x[1]])

    v2_x = np.array([2.0, 3.0])
    # J = [[1, 1], [3, 2]]
    am_vx = np.array([[0.1, 0.0], [0.0, 0.2]])

    # J * Vx * J.T = [[1, 1], [3, 2]] * [[0.1, 0], [0, 0.2]] * [[1, 3], [1, 2]]
    # = [[0.1, 0.2], [0.3, 0.4]] * [[1, 3], [1, 2]]
    # = [[0.3, 0.7], [0.7, 1.7]]
    expected_vy = np.array([[0.3, 0.7], [0.7, 1.7]])

    result = propagate_covariance_saap(v2_f, v2_x, am_vx)
    np.testing.assert_allclose(result, expected_vy, rtol=1e-4)


def test_propagate_covariance_mismatch():
    """Test that a non-matching covariance dimension raises ValueError."""
    def v2_f(v_x): return np.array([v_x[0], v_x[1]])
    v_x = np.array([1.0, 2.0])
    am_vx_bad = np.array([[0.1, 0.0, 0.0], [0.0, 0.2, 0.0]])  # not 2x2
    with pytest.raises(ValueError, match="Input covariance matrix must be square"):
        propagate_covariance_saap(v2_f, v_x, am_vx_bad)


# --- error_contribution_saap tests ---


def test_error_contribution_saap():
    """Test calculation of fractional error contributions."""
    def s_f(v2_x: np.ndarray) -> float:
        return v2_x[0] * v2_x[1]

    av_vals = np.array([10.0, 5.0])
    av_errs = np.array([0.5, 0.2])

    # Variances: 2.5^2 = 6.25, 2.0^2 = 4.0. Total = 10.25
    expected_contributions = np.array([6.25 / 10.25, 4.0 / 10.25])

    result = error_contribution_saap(s_f, av_vals, av_errs)
    np.testing.assert_allclose(result, expected_contributions, rtol=1e-4)


def test_error_contribution_zero_variance():
    """Test that zero total variance returns zeros without division by zero error."""
    def s_f(v_x): return v_x[0] + v_x[1]
    result = error_contribution_saap(s_f, np.array([1.0, 2.0]), np.array([0.0, 0.0]))
    np.testing.assert_allclose(result, np.array([0.0, 0.0]))


# --- relative_uncertainty_saap tests ---


def test_relative_uncertainty_saap():
    """Test relative uncertainty calculation."""
    def s_f(v2_x: np.ndarray) -> float:
        return v2_x[0] * v2_x[1]

    av_vals = np.array([10.0, 5.0])  # Nominal f = 50.0
    av_errs = np.array([0.5, 0.2])

    abs_err = np.sqrt(10.25)
    expected_rel_err = abs_err / 50.0

    result = relative_uncertainty_saap(s_f, av_vals, av_errs)
    np.testing.assert_allclose(result, expected_rel_err, rtol=1e-4)


def test_relative_uncertainty_zero_division():
    """Test that a nominal value of zero raises a ZeroDivisionError."""
    def s_f(v2_x: np.ndarray) -> float:
        return v2_x[0] * v2_x[1]

    av_vals = np.array([0.0, 5.0])  # Nominal f = 0.0
    av_errs = np.array([0.5, 0.2])

    with pytest.raises(ZeroDivisionError):
        relative_uncertainty_saap(s_f, av_vals, av_errs)