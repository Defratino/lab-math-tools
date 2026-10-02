"""
Lab Math Tools: A lightweight numerical calculus and error propagation library.
"""

from __future__ import annotations

from lab_math_tools.derivatives import (
    derivative_saap,
    divergence_saap,
    partial_derivative_saap,
    v_gradient_saap,
)
from lab_math_tools.error_propagation import (
    propagate_uncertainty_saap,
    relative_uncertainty_saap,
)
from lab_math_tools.graphing import (
    add_zoom_inset,
    plot_measurements,
    turn_list_of_vectors_to_matrix,
)

__version__ = "0.1.0"

__all__ = [
    # Derivatives
    "derivative_saap",
    "partial_derivative_saap",
    "v_gradient_saap",
    "divergence_saap",
    # Error Propagation
    "propagate_uncertainty_saap",
    "relative_uncertainty_saap",
    # Graphing
    "plot_measurements",
    "turn_list_of_vectors_to_matrix",
    "add_zoom_inset",
]