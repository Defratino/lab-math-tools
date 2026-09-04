"""
Statistical Error Propagation Module

This module provides tools for error propagation and statistical analysis of
functions using the Simple Approximation At Point (saap) method. Functions are 
designed to handle multidimensional mappings (R -> R, R -> R^n, R^n -> R, R^n -> R^m) 
seamlessly through numpy broadcasting.

Conventions followed:
- `av_` : Abstract vector (numpy.ndarray with dynamic dimensions)
- `am_` : Abstract matrix (numpy.ndarray with dynamic dimensions)
- `s_f` : Scalar-valued function
- `av_f`: Vector-valued function
- `f`   : Generic function (can output scalar or vector)
"""

import numpy as np
from lab_math_tools.derivatives import av_gradient_saap, derivative_saap, am_jacobian_saap

def propagate_uncertainty_saap(f, x: float | np.ndarray, dx: float | np.ndarray, method: str = "statistical", h: float = 1e-5) -> float:
    """
    Calculates the propagated uncertainty of a function using numerical 
    derivative approximations (SAAP).

    Supports both scalar (R -> R) and vector (R^n -> R) mappings.

    This function automatically computes the partial derivatives (sensitivities)
    of the function with respect to each input variable, then scales them by 
    their respective uncertainties to find the total combined error.

    Parameters:
    * s_f (callable): The scalar-valued objective function.
    * av_x (np.ndarray): The nominal values of the independent variables.
    * av_dx (np.ndarray): The absolute uncertainties (errors) associated with each variable in av_x.
    * method (str): The combination rule for the errors.
        - "statistical": (Default) Assumes errors are independent and random, combining them via Root-Sum-Square (RSS).
        - "absolute": Assumes a worst-case scenario where all errors stack in the same direction.
    * h (float): The step size for the numerical derivative approximation.

    Returns:
    * float: The total propagated uncertainty (Δf).

    Mathematical Formulation:
    Statistical (Root-Sum-Square):
    Absolute (Worst-Case == Sum-Absolute):
    """

    # 1. Handle pure scalar mapping (R -> R)
    if np.isscalar(x) and np.isscalar(dx):
        sensitivity = derivative_saap(f, x, h)
        return float(np.abs(sensitivity * dx))
        
    # 2. Handle vector mapping (R^n -> R)
    assert len(x) == len(dx), "Value and uncertainty vectors must have the same dimension."
    
    av_sensitivities = av_gradient_saap(f, x, h)
    av_terms = av_sensitivities * dx
    
    if method == "statistical":
        return float(np.sqrt(np.sum(av_terms**2)))
    elif method == "absolute":
        return float(np.sum(np.abs(av_terms)))
    else:
        raise ValueError("Method must be 'statistical' or 'absolute'.")

def propagate_covariance_saap(av_f, av_x: np.ndarray, am_vx: np.ndarray, h: float = 1e-5) -> np.ndarray:
    """
    Propagates a covariance matrix through a vector-valued function (R^n -> R^m).
    
    Parameters:
    * av_f (callable): The vector-valued objective function.
    * av_x (np.ndarray): The nominal values of the independent variables (length n).
    * am_vx (np.ndarray): The (n, n) covariance matrix of the inputs.
    * h (float): The step size for the numerical derivative approximation.
    
    Returns:
    * np.ndarray: The (m, m) output covariance matrix (am_vy).
    
    Mathematical Formulation:
    V_y = J * V_x * J.T
    """
    assert am_vx.shape == (len(av_x), len(av_x)), "Input covariance matrix must be square with dimensions matching av_x."
    
    am_j = am_jacobian_saap(av_f, av_x, h)
    
    # Compute J * Vx * J^T
    am_vy = am_j @ am_vx @ am_j.T
    return am_vy

def error_contribution_saap(s_f, av_x: np.ndarray, av_dx: np.ndarray, h: float = 1e-5) -> np.ndarray:
    """
    Calculates the fractional contribution of each input variable to the total 
    statistical variance of a scalar function (R^n -> R).
    
    Parameters:
    * s_f (callable): The scalar-valued objective function.
    * av_x (np.ndarray): The nominal values of the independent variables.
    * av_dx (np.ndarray): The absolute uncertainties (errors) associated with av_x.
    * h (float): The step size for the numerical derivative approximation.
    
    Returns:
    * np.ndarray: An array of fractional weights summing to 1.0.
    
    Mathematical Formulation:
    * Weight_i = ((∂f/∂x_i) * Δx_i)^2 / (Δf)^2
    """
    assert len(av_x) == len(av_dx), "Value and uncertainty vectors must have the same dimension."
    
    av_sensitivities = av_gradient_saap(s_f, av_x, h)
    av_variance_terms = (av_sensitivities * av_dx)**2
    
    total_variance = np.sum(av_variance_terms)
    
    # Avoid division by zero if the total variance is entirely zero
    if total_variance == 0:
        return np.zeros_like(av_variance_terms)
        
    return av_variance_terms / total_variance

def relative_uncertainty_saap(f, x: float | np.ndarray, dx: float | np.ndarray, method: str = "statistical", h: float = 1e-5) -> float:
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