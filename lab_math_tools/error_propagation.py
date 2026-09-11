"""
Statistical Error Propagation Module

This module provides tools for error propagation and statistical analysis of
functions using the Simple Approximation At Point (saap) method. Functions are 
designed to handle multidimensional mappings (R -> R, R -> R^n, R^n -> R, R^n -> R^m) 
seamlessly through numpy broadcasting.

Conventions followed:
- `v_` : Abstract vector (numpy.ndarray with dynamic dimensions)
- `m_` : Abstract matrix (numpy.ndarray with dynamic dimensions)
- `s_f` : Scalar-valued function
- `v_f`: Vector-valued function
- `f`   : Generic function (can output scalar or vector)
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from lab_math_tools.derivatives import m_jacobian_saap, v_gradient_saap, derivative_saap


def propagate_uncertainty_saap(
    f: Callable[..., float],
    x: float | np.ndarray,
    dx: float | np.ndarray,
    method: str = "statistical",
    h: float = 1e-5,
) -> float:
    """
    Calculates the propagated uncertainty of a function using numerical 
    derivative approximations (SAAP).

    Supports both scalar (R -> R) and vector (R^n -> R) mappings.

    This function automatically computes the partial derivatives (sensitivities)
    of the function with respect to each input variable, then scales them by 
    their respective uncertainties to find the total combined error.

    Parameters:
    * f (callable): The scalar-valued objective function.
    * x (float | np.ndarray): The nominal values of the independent variables.
    * dx (float | np.ndarray): The absolute uncertainties (errors) associated with each variable in x.
    * method (str): The combination rule for the errors.
        - "statistical": (Default) Assumes errors are independent and random, combining them via Root-Sum-Square (RSS).
        - "absolute": Assumes a worst-case scenario where all errors stack in the same direction.
    * h (float): The step size for the numerical derivative approximation.

    Returns:
    * float: The total propagated uncertainty (Δf).

    Mathematical Formulation:
    Statistical (Root-Sum-Square):
        Δf = sqrt( Σ ( (∂f/∂x_i) * Δx_i )^2 )
    Absolute (Worst-Case == Sum-Absolute):
        Δf = Σ | (∂f/∂x_i) * Δx_i |
    """
    if h <= 0:
        raise ValueError("Step size h must be strictly positive.")

    # 1. Handle pure scalar mapping (R -> R)
    if np.isscalar(x) and np.isscalar(dx):
        sensitivity = derivative_saap(f, float(x), h)
        return float(np.abs(sensitivity * dx))

    # Guard against scalar-vector mismatch
    if np.isscalar(x) != np.isscalar(dx):
        raise ValueError("x and dx must both be scalars or both be vectors with the same dimension.")

    # 2. Handle vector mapping (R^n -> R)
    x_arr = np.asarray(x, dtype=float)
    dx_arr = np.asarray(dx, dtype=float)

    if x_arr.ndim == 0 and dx_arr.ndim == 0:
        sensitivity = derivative_saap(f, float(x_arr), h)
        return float(np.abs(sensitivity * dx_arr))

    if x_arr.shape != dx_arr.shape:
        raise ValueError("Value and uncertainty vectors must have the same dimension.")

    v_sensitivities = v_gradient_saap(f, x_arr, h)
    v_terms = v_sensitivities * dx_arr

    if method == "statistical":
        return float(np.sqrt(np.sum(v_terms**2)))
    elif method == "absolute":
        return float(np.sum(np.abs(v_terms)))
    else:
        raise ValueError("Method must be 'statistical' or 'absolute'.")


def propagate_covariance_saap(
    v_f: Callable[[np.ndarray], np.ndarray],
    v_x: np.ndarray,
    m_vx: np.ndarray,
    h: float = 1e-5,
) -> np.ndarray:
    """
    Propagates a covariance matrix through a vector-valued function (R^n -> R^m).
    
    Parameters:
    * v_f (callable): The vector-valued objective function.
    * v_x (np.ndarray): The nominal values of the independent variables (length n).
    * m_vx (np.ndarray): The (n, n) covariance matrix of the inputs.
    * h (float): The step size for the numerical derivative approximation.
    
    Returns:
    * np.ndarray: The (m, m) output covariance matrix (m_vy).
    
    Mathematical Formulation:
    V_y = J * V_x * J.T
    """
    if h <= 0:
        raise ValueError("Step size h must be strictly positive.")

    v_x_arr = np.asarray(v_x, dtype=float)
    m_vx_arr = np.asarray(m_vx, dtype=float)
    n = len(v_x_arr)

    if m_vx_arr.shape != (n, n):
        raise ValueError("Input covariance matrix must be square with dimensions matching v_x.")

    m_j = m_jacobian_saap(v_f, v_x_arr, h)
    m_vy = m_j @ m_vx_arr @ m_j.T
    return m_vy


def error_contribution_saap(
    s_f: Callable[[np.ndarray], float],
    v_x: np.ndarray,
    v_dx: np.ndarray,
    h: float = 1e-5,
) -> np.ndarray:
    """
    Calculates the fractional contribution of each input variable to the total 
    statistical variance of a scalar function (R^n -> R).
    
    Parameters:
    * s_f (callable): The scalar-valued objective function.
    * v_x (np.ndarray): The nominal values of the independent variables.
    * v_dx (np.ndarray): The absolute uncertainties (errors) associated with v_x.
    * h (float): The step size for the numerical derivative approximation.
    
    Returns:
    * np.ndarray: An array of fractional weights summing to 1.0.
    
    Mathematical Formulation:
    * Weight_i = ((∂f/∂x_i) * Δx_i)^2 / (Δf)^2
    """
    if h <= 0:
        raise ValueError("Step size h must be strictly positive.")

    v_x_arr = np.asarray(v_x, dtype=float)
    v_dx_arr = np.asarray(v_dx, dtype=float)

    if v_x_arr.shape != v_dx_arr.shape:
        raise ValueError("Value and uncertainty vectors must have the same dimension.")

    v_sensitivities = v_gradient_saap(s_f, v_x_arr, h)
    v_variance_terms = (v_sensitivities * v_dx_arr) ** 2

    total_variance = np.sum(v_variance_terms)

    # Avoid division by zero if the total variance is entirely zero
    if total_variance == 0:
        return np.zeros_like(v_variance_terms)

    return v_variance_terms / total_variance


def relative_uncertainty_saap(
    f: Callable[..., float],
    x: float | np.ndarray,
    dx: float | np.ndarray,
    method: str = "statistical",
    h: float = 1e-5,
) -> float:
    """
    Calculates the relative (fractional) uncertainty of a function.
    
    Parameters:
    * f (callable): The objective function.
    * x (float | np.ndarray): The nominal values of the independent variables.
    * dx (float | np.ndarray): The absolute uncertainties of the input variables.
    * method (str): "statistical" or "absolute" combination rule.
    * h (float): The step size for the numerical derivative approximation.
    
    Returns:
    * float: The dimensionless relative uncertainty.
    
    Mathematical Formulation:
    * Relative Error = |Δf / f(x)|
    """
    absolute_uncertainty = propagate_uncertainty_saap(f, x, dx, method, h)
    nominal_value = float(np.abs(f(x)))

    if nominal_value == 0:
        raise ZeroDivisionError("Nominal function value is zero; relative uncertainty is undefined.")

    return absolute_uncertainty / nominal_value