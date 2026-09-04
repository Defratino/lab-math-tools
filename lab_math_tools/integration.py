"""
Calculus Integration Module

This module provides numerical integration tools for scalar and vector-valued 
functions using the composite Trapezoidal rule and Monte Carlo integration. Functions 
are designed to handle multidimensional mappings (R -> R, R -> R^n, R^n -> R, R^n -> R^m) 
seamlessly through numpy broadcasting.

Conventions followed:
- `av_` : Abstract vector (numpy.ndarray with dynamic dimensions)
- `am_` : Abstract matrix (numpy.ndarray with dynamic dimensions)
- `s_f` : Scalar-valued function
- `av_f`: Vector-valued function
- `f`   : Generic function (can output scalar or vector)
"""

from __future__ import annotations

from typing import Callable

import numpy as np


def integral_trapezoidal(
    s_f: Callable[[float | np.ndarray], float | np.ndarray],
    a: float,
    b: float,
    n_steps: int = 1000,
) -> float | np.ndarray:
    """
    Calculates the definite integral of a function over the interval [a, b] 
    using the composite Trapezoidal rule.
    
    Works for both scalar-valued (R -> R) and vector-valued (R -> R^n) functions.

    Mathematical Formulation:
    ∫f(x)dx ≈ (Δx/2) * Σ(f(x(i-1)) + f(x(i)))
    """
    if n_steps <= 0:
        raise ValueError("Number of steps must be strictly positive.")

    x = np.linspace(a, b, n_steps + 1)
    try:
        y = np.asarray(s_f(x))
        if y.ndim == 1 and len(y) != len(x):
            # Function returned a single fixed-length vector instead of vectorized evaluation
            y = np.array([s_f(val) for val in x])
        elif y.ndim > 1:
            if y.shape[0] != len(x) and y.shape[-1] == len(x):
                # Shape (d, N) -> transpose to (N, d)
                y = np.moveaxis(y, -1, 0)
            elif y.shape[0] != len(x):
                y = np.array([s_f(val) for val in x])
    except Exception:
        y = np.array([s_f(val) for val in x])

    dx = (b - a) / n_steps

    # Sum along sample axis (axis 0), halving the first and last endpoints
    integral = dx * (np.sum(y, axis=0) - (y[0] + y[-1]) / 2.0)

    return float(integral) if np.ndim(integral) == 0 else np.asarray(integral)


def integral_over_shape(
    s_f: Callable[[np.ndarray], float],
    s_shape: Callable[[np.ndarray], float | int | bool],
    am_bounds: np.ndarray,
    n_samples: int = 100000,
    rng: np.random.Generator | int | None = None,
) -> float:
    """
    Calculates the n-dimensional integral of a scalar function over a custom shape 
    using Monte Carlo integration.
    
    Parameters:
    * s_f (callable): The scalar function to integrate (R^n -> R).
    * s_shape (callable): Indicator function defining the region (R^n -> [0, 1]).
    * am_bounds (np.ndarray): An (n, 2) matrix of [min, max] bounding box limits for each dimension.
    * n_samples (int): Number of random points for the Monte Carlo estimation.
    * rng (np.random.Generator | int | None): Optional random number generator or seed for reproducibility.
    
    Mathematical Formulation:
    V ≈ (V_box / N) * Σ(f(x(i)) * shape(x(i)))
    """
    am_bounds_arr = np.asarray(am_bounds, dtype=float)
    if am_bounds_arr.ndim != 2 or am_bounds_arr.shape[1] != 2:
        raise ValueError("am_bounds must be a 2D array of shape (n_dims, 2) with [min, max] limits.")
    if np.any(am_bounds_arr[:, 0] > am_bounds_arr[:, 1]):
        raise ValueError("Lower bounds must not exceed upper bounds.")
    if n_samples <= 0:
        raise ValueError("Number of samples must be strictly positive.")

    generator = rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)
    n_dims = am_bounds_arr.shape[0]

    # Generate random points within the bounding box
    am_random_points = generator.uniform(
        low=am_bounds_arr[:, 0],
        high=am_bounds_arr[:, 1],
        size=(n_samples, n_dims),
    )

    # Calculate bounding box volume
    box_volume = float(np.prod(am_bounds_arr[:, 1] - am_bounds_arr[:, 0]))

    # Attempt vectorized evaluation (fast path); fall back to sample-by-sample evaluation
    try:
        f_vals = np.asarray(s_f(am_random_points))
        shape_vals = np.asarray(s_shape(am_random_points))
        if f_vals.shape == (n_samples,) and shape_vals.shape == (n_samples,):
            total_sum = float(np.sum(f_vals * shape_vals))
        else:
            raise ValueError("Non-matching shapes for vectorized evaluation.")
    except Exception:
        total_sum = 0.0
        for i in range(n_samples):
            v_x = am_random_points[i]
            w = s_shape(v_x)
            if w:
                total_sum += s_f(v_x) * w

    return float(box_volume * total_sum / n_samples)