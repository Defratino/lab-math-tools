"""
Calculus Simple Approximation At Point (saap) Derivative Module

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

from __future__ import annotations

from typing import Any, Callable

import numpy as np


def derivative_saap(
    f: Callable[[float], float | np.ndarray],
    x: float,
    h: float = 1e-5,
) -> float | np.ndarray:
    """
    Calculates the derivative of a function at a given point using the saap method.
    Works for both scalar-valued (R -> R) and vector-valued (R -> R^n) functions.
    """
    if h <= 0:
        raise ValueError("Step size h must be strictly positive.")
    return (f(x + h) - f(x - h)) / (2 * h)


def partial_derivative_saap(
    f: Callable[[np.ndarray], float | np.ndarray],
    av_x: np.ndarray,
    idx: int = 0,
    h: float = 1e-5,
) -> float | np.ndarray:
    """
    Calculates the partial derivative of a function with respect to the idx-th variable.
    Works for both scalar-valued (R^n -> R) and vector-valued (R^n -> R^m) functions.
    """
    if h <= 0:
        raise ValueError("Step size h must be strictly positive.")

    av_x_arr = np.asarray(av_x, dtype=float)
    if idx < 0 or idx >= len(av_x_arr):
        raise IndexError("The index is out of bounds.")

    av_x_plus_h = av_x_arr.copy()
    av_x_plus_h[idx] += h
    av_x_minus_h = av_x_arr.copy()
    av_x_minus_h[idx] -= h

    return (f(av_x_plus_h) - f(av_x_minus_h)) / (2 * h)


def av_gradient_saap(
    s_f: Callable[[np.ndarray], float],
    av_x: np.ndarray,
    h: float = 1e-5,
) -> np.ndarray:
    """
    Calculates the gradient of a scalar function (R^n -> R) at a given point.
    """
    av_x_arr = np.asarray(av_x, dtype=float)
    return np.array([partial_derivative_saap(s_f, av_x_arr, idx, h) for idx in range(len(av_x_arr))])


def divergence_saap(
    av_f: Callable[[np.ndarray], np.ndarray],
    av_x: np.ndarray,
    h: float = 1e-5,
) -> float:
    """
    Calculates the divergence of a vector field (R^n -> R^n) at a given point.
    """
    av_x_arr = np.asarray(av_x, dtype=float)
    out = np.asarray(av_f(av_x_arr))
    if out.shape != av_x_arr.shape:
        raise ValueError("The function must return a vector of the same dimension as the input.")

    return float(sum(partial_derivative_saap(av_f, av_x_arr, idx, h)[idx] for idx in range(len(av_x_arr))))


def am_jacobian_saap(
    av_f: Callable[[np.ndarray], np.ndarray],
    av_x: np.ndarray,
    h: float = 1e-5,
) -> np.ndarray:
    """
    Calculates the Jacobian matrix of a vector-valued function (R^n -> R^m) at a given point.
    The resulting matrix has dimensions (m, n).
    """
    av_x_arr = np.asarray(av_x, dtype=float)
    return np.column_stack([partial_derivative_saap(av_f, av_x_arr, idx, h) for idx in range(len(av_x_arr))])