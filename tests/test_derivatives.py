import numpy as np
import pytest
from lab_math_tools.derivatives import (
    am_jacobian_saap,
    av_gradient_saap,
    derivative_saap,
    divergence_saap,
    partial_derivative_saap,
)


def test_derivative_saap_scalar():
    """Test R -> R mapping."""
    def f(x: float) -> float:
        return x**2

    result = derivative_saap(f, x=2.0)
    np.testing.assert_allclose(result, 4.0, rtol=1e-4)


def test_derivative_saap_vector():
    """Test R -> R^n mapping."""
    def v2_f(x: float) -> np.ndarray:
        return np.array([2 * x**2, x**3])

    result = derivative_saap(v2_f, x=2.0)
    np.testing.assert_allclose(result, np.array([8.0, 12.0]), rtol=1e-4)


def test_partial_derivative_saap_scalar():
    """Test R^n -> R mapping."""
    def s_f(v2_x: np.ndarray) -> float:
        return v2_x[0] ** 2 + v2_x[1] ** 3

    v2_x = np.array([2.0, 2.0])
    result = partial_derivative_saap(s_f, v2_x, idx=1)
    np.testing.assert_allclose(result, 12.0, rtol=1e-4)


def test_partial_derivative_saap_vector():
    """Test R^n -> R^m mapping."""
    def v2_f(v2_x: np.ndarray) -> np.ndarray:
        return np.array([v2_x[0] * v2_x[1], v2_x[1] ** 2])

    v2_x = np.array([3.0, 2.0])
    result = partial_derivative_saap(v2_f, v2_x, idx=0)
    np.testing.assert_allclose(result, np.array([2.0, 0.0]), rtol=1e-4)


def test_av_gradient_saap():
    """Test gradient on R^n -> R mapping."""
    def s_f(v2_x: np.ndarray) -> float:
        return v2_x[0] ** 2 * v2_x[1]

    v2_x = np.array([2.0, 3.0])
    result = av_gradient_saap(s_f, v2_x)
    np.testing.assert_allclose(result, np.array([12.0, 4.0]), rtol=1e-4)


def test_divergence_saap():
    """Test divergence on R^n -> R^n mapping."""
    def v2_f(v2_x: np.ndarray) -> np.ndarray:
        return np.array([v2_x[0] ** 2, v2_x[1] ** 3])

    v2_x = np.array([2.0, 2.0])
    result = divergence_saap(v2_f, v2_x)
    np.testing.assert_allclose(result, 16.0, rtol=1e-4)


def test_am_jacobian_saap():
    """Test Jacobian on R^n -> R^m mapping."""
    def v2_f(v2_x: np.ndarray) -> np.ndarray:
        return np.array([v2_x[0] ** 2, v2_x[0] * v2_x[1]])

    v2_x = np.array([2.0, 3.0])
    result = am_jacobian_saap(v2_f, v2_x)
    expected_m22_j = np.array([[4.0, 0.0], [3.0, 2.0]])
    np.testing.assert_allclose(result, expected_m22_j, rtol=1e-4)


def test_partial_derivative_saap_index_out_of_bounds():
    """Test that an invalid idx raises IndexError."""
    def s_f(v2_x: np.ndarray) -> float:
        return v2_x[0] ** 2 + v2_x[1] ** 2

    v2_x = np.array([2.0, 3.0])

    # Index >= length triggers IndexError
    with pytest.raises(IndexError, match="The index is out of bounds."):
        partial_derivative_saap(s_f, v2_x, idx=2)

    with pytest.raises(IndexError, match="The index is out of bounds."):
        partial_derivative_saap(s_f, v2_x, idx=5)

    # Negative index triggers IndexError
    with pytest.raises(IndexError, match="The index is out of bounds."):
        partial_derivative_saap(s_f, v2_x, idx=-1)


def test_divergence_saap_dimension_mismatch():
    """Test that a non-matching vector field (R^n -> R^m where n != m) raises a ValueError."""
    # R^2 -> R^3 mapping
    def v3_f(v2_x: np.ndarray) -> np.ndarray:
        return np.array([v2_x[0] ** 2, v2_x[1] ** 2, v2_x[0] * v2_x[1]])

    v2_x = np.array([1.0, 2.0])

    # Divergence requires R^n -> R^n. Passing R^2 -> R^3 should fail.
    with pytest.raises(ValueError, match="The function must return a vector of the same dimension as the input."):
        divergence_saap(v3_f, v2_x)


def test_derivative_saap_invalid_h():
    """Test that non-positive step size h raises ValueError."""
    def f(x: float) -> float: return x**2
    with pytest.raises(ValueError, match="Step size h must be strictly positive."):
        derivative_saap(f, x=2.0, h=0.0)
    with pytest.raises(ValueError, match="Step size h must be strictly positive."):
        derivative_saap(f, x=2.0, h=-1e-3)
    with pytest.raises(ValueError, match="Step size h must be strictly positive."):
        partial_derivative_saap(f, np.array([2.0]), idx=0, h=0.0)