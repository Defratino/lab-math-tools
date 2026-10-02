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
from lab_math_tools.derivatives import derivative_saap, v_gradient_saap


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