"""
Lab Math Tools: A lightweight numerical calculus and error propagation library.
"""

from __future__ import annotations

from lab_math_tools.derivatives import (
    am_jacobian_saap,
    av_gradient_saap,
    derivative_saap,
    divergence_saap,
    partial_derivative_saap,
)
from lab_math_tools.error_propagation import (
    error_contribution_saap,
    propagate_covariance_saap,
    propagate_uncertainty_saap,
    relative_uncertainty_saap,
)
from lab_math_tools.integration import (
    integral_over_shape,
    integral_trapezoidal,
)

__version__ = "0.1.0"

__all__ = [
    # Derivatives
    "derivative_saap",
    "partial_derivative_saap",
    "av_gradient_saap",
    "divergence_saap",
    "am_jacobian_saap",
    # Integration
    "integral_trapezoidal",
    "integral_over_shape",
    # Error Propagation
    "propagate_uncertainty_saap",
    "propagate_covariance_saap",
    "error_contribution_saap",
    "relative_uncertainty_saap",
]