"""
Calculus Integration Module

This module provides numerical integration tools for scalar and vector-valued 
functions using the Trapezoidal rule and Monte Carlo integration. Functions are 
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
from lab_math_tools.derivatives import derivative_saap, partial_derivative_saap

def integral_trapezoidal(s_f, a: float, b: float, n_steps: int = 1000) -> float:
    """
    Calculates the definite integral of a function over the interval [a, b] 
    using the composite Trapezoidal rule.
    
    Works for both scalar-valued (R -> R) and vector-valued (R -> R^n) functions.

    Mathematical Formulation:
    ∫f(x)dx ≈ (Δx/2) * Σ(f(x(i-1)) + f(x(i)))
    """
    assert n_steps > 0, "Number of steps must be strictly positive."
    
    x = np.linspace(a, b, n_steps + 1)
    y = s_f(x)
    
    dx = (b - a) / n_steps
    
    # Sum all elements, but halve the first and last endpoints
    integral = dx * (np.sum(y) - (y[0] + y[-1]) / 2.0)
    
    return float(integral)

def integral_over_shape(s_f, s_shape, am_bounds: np.ndarray, n_samples: int = 100000) -> float:
    """
    Calculates the n-dimensional integral of a scalar function over a custom shape 
    using Monte Carlo integration.
    
    Parameters:
    * s_f (callable): The scalar function to integrate (R^n -> R).
    * s_shape (callable): Indicator function defining the region (R^n -> [0, 1]).
    * am_bounds (np.ndarray): An (n, 2) matrix of [min, max] bounding box limits for each dimension.
    * n_samples (int): Number of random points for the Monte Carlo estimation.
    
    Mathematical Formulation:
    V ≈ (V_box / N) * Σ(f(x(i)) * shape(x(i)))
    """
    n_dims = am_bounds.shape[0]
    
    # Generate random points within the bounding box
    am_random_points = np.random.uniform(
        low=am_bounds[:, 0], 
        high=am_bounds[:, 1], 
        size=(n_samples, n_dims)
    )
    
    # Calculate bounding box volume
    box_volume = np.prod(am_bounds[:, 1] - am_bounds[:, 0])
    
    total_sum = 0.0
    for i in range(n_samples):
        v_x = am_random_points[i]
        # Only add to the sum if the point is inside the shape (s_shape returns 1)
        total_sum += s_f(v_x) * s_shape(v_x)
        
    return float(box_volume * total_sum / n_samples)