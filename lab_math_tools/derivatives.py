"""
Calculus Simple Approximation At Point (saap) Module

This module provides numerical differentiation tools for scalar and vector-valued 
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

def derivative_saap(f, x: float, h: float = 1e-5) -> float | np.ndarray:
    """
    Calculates the derivative of a function at a given point using the saap method.
    Works for both scalar-valued (R -> R) and vector-valued (R -> R^n) functions.
    """
    return (f(x + h) - f(x-h)) / (2*h)

def partial_derivative_saap(f, av_x: np.ndarray, idx: int = 0, h: float = 1e-5) -> float | np.ndarray:
    """
    Calculates the partial derivative of a function with respect to the idx-th variable.
    Works for both scalar-valued (R^n -> R) and vector-valued (R^n -> R^m) functions.
    """
    assert idx < len(av_x), "The index is out of bounds."

    av_x_plus_h = np.array(av_x, dtype=float) 
    av_x_plus_h[idx] += h
    av_x_minus_h = np.array(av_x, dtype=float) 
    av_x_minus_h[idx] -= h

    return (f(av_x_plus_h) - f(av_x_minus_h)) / (2*h)

def av_gradient_saap(s_f, av_x: np.ndarray, h: float = 1e-5) -> np.ndarray:
    """
    Calculates the gradient of a scalar function (R^n -> R) at a given point.
    """
    return np.array([partial_derivative_saap(s_f, av_x, idx, h) for idx in range(len(av_x))])

def divergence_saap(av_f, av_x: np.ndarray, h: float = 1e-5) -> float:
    """
    Calculates the divergence of a vector field (R^n -> R^n) at a given point.
    """
    assert av_f(av_x).shape == av_x.shape, "The function must return a vector of the same dimension as the input."

    return sum(partial_derivative_saap(av_f, av_x, idx, h)[idx] for idx in range(len(av_x)))

def am_jacobian_saap(av_f, av_x: np.ndarray, h: float = 1e-5) -> np.ndarray:
    """
    Calculates the Jacobian matrix of a vector-valued function (R^n -> R^m) at a given point.
    The resulting matrix has dimensions (m, n).
    """
    return np.column_stack([partial_derivative_saap(av_f, av_x, idx, h) for idx in range(len(av_x))])