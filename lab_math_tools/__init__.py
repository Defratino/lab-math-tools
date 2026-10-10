"""
Lab Math Tools: A lightweight numerical calculus and error propagation library.
"""

from __future__ import annotations

from lab_math_tools.derivatives import (
    derivative,
    divergence,
    partial_derivative,
    v_gradient,
)
from lab_math_tools.error_propagation import (
    propagate_uncertainty,
    relative_uncertainty,
)
from lab_math_tools.curve_fitting import (
    FitResult,
    fit_custom,
    fit_linear,
    fit_metrics,
)
from lab_math_tools.graphing import (
    add_zoom_inset,
    plot_measurements,
    plot_with_errors,
    turn_list_of_vectors_to_matrix,
)

__version__ = "0.1.0"

__all__ = [
    # Derivatives
    "derivative",
    "partial_derivative",
    "v_gradient",
    "divergence",
    # Error Propagation
    "propagate_uncertainty",
    "relative_uncertainty",
    # Graphing
    "plot_measurements",
    "plot_with_errors",
    "turn_list_of_vectors_to_matrix",
    "add_zoom_inset",
    # Curve Fitting
    "FitResult",
    "fit_metrics",
    "fit_linear",
    "fit_custom",
]