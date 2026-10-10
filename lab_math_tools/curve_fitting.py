"""
Curve Fitting Module

Provides tools for fitting mathematical models to experimental data.
Supports linear regression and custom user-defined function fitting.

Conventions followed:
- `v_`  : Abstract vector (numpy.ndarray with dynamic dimensions)
- `m_`  : Abstract matrix (numpy.ndarray with dynamic dimensions)
- `s_f` : Scalar-valued function callback (R^n -> R)
- `f`   : Generic function callback
"""

from __future__ import annotations

from dataclasses import dataclass
import inspect
import re
from typing import Callable

import numpy as np
from scipy.optimize import curve_fit


# =============================================================================
# FitResult
# =============================================================================

@dataclass
class FitResult:
    """
    Container for curve fitting results.

    Attributes:
    * v_params (np.ndarray): Best-fit parameter values.
        - fit_linear: [intercept, slope]
        - fit_custom: fitted parameter values [p0, p1, ...]
    * m_covariance (np.ndarray): Estimated covariance matrix of v_params.
    * v_residuals (np.ndarray): Residuals (v_y - v_y_predicted) at each point.
    * metrics (dict[str, float]): Goodness-of-fit metrics.
        Always present: "mse", "rmse", "mae", "r2", "std_residuals".
        When applicable: "r2_adj".
    * model_name (str): Human-readable label for the fitted model.
    """

    v_params: np.ndarray
    m_covariance: np.ndarray
    v_residuals: np.ndarray
    metrics: dict[str, float]
    model_name: str

    def summary(self) -> str:
        """
        Returns a formatted multi-line summary of the fit results.

        Returns:
        * str: Human-readable summary with model name, parameters, and metrics.
        """
        lines = [f"Model: {self.model_name}"]
        lines.append("Parameters:")
        for i, p in enumerate(self.v_params):
            lines.append(f"  p[{i}] = {p:.6g}")
        lines.append("Metrics:")
        for key, val in self.metrics.items():
            lines.append(f"  {key} = {val:.6g}")
        return "\n".join(lines)

    def plot_residuals(self, ax=None):
        """
        Plots the residuals as a stem plot against their sample index.

        Parameters:
        * ax (matplotlib.axes.Axes | None): Axes to draw on. A new figure is
          created when None.

        Returns:
        * matplotlib.axes.Axes: The axes containing the residuals plot.
        """
        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots()

        indices = np.arange(len(self.v_residuals))
        ax.axhline(0.0, color="gray", linewidth=0.8, linestyle="--")
        ax.stem(indices, self.v_residuals,
                linefmt="C0-", markerfmt="C0o", basefmt="gray")
        ax.set_xlabel("Sample index")
        ax.set_ylabel("Residual")
        ax.set_title(f"Residuals — {self.model_name}")
        return ax


# =============================================================================
# Internal helpers
# =============================================================================

def _validate_1d_xy(
    v_x: np.ndarray,
    v_y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Validates and converts x/y inputs to 1-D float arrays of equal length."""
    v_x_arr = np.asarray(v_x, dtype=float)
    v_y_arr = np.asarray(v_y, dtype=float)
    if v_x_arr.ndim != 1 or v_y_arr.ndim != 1:
        raise ValueError("v_x and v_y must be 1-D arrays.")
    if len(v_x_arr) != len(v_y_arr):
        raise ValueError(
            "v_x and v_y must have the same length; "
            f"got {len(v_x_arr)} and {len(v_y_arr)}."
        )
    return v_x_arr, v_y_arr


# =============================================================================
# fit_metrics
# =============================================================================

def fit_metrics(
    v_y: np.ndarray,
    v_y_pred: np.ndarray,
    n_params: int = 0,
) -> dict[str, float]:
    """
    Computes goodness-of-fit metrics from true and predicted values.

    Parameters:
    * v_y (np.ndarray): True (observed) target values.
    * v_y_pred (np.ndarray): Model-predicted values.
    * n_params (int): Number of free model parameters. When > 0 and degrees
        of freedom allow, the adjusted R^2 ("r2_adj") is added to the output.

    Returns:
    * dict[str, float]: Metric name -> value.
        - "mse"          : Mean Squared Error
        - "rmse"         : Root Mean Squared Error
        - "mae"          : Mean Absolute Error
        - "r2"           : Coefficient of Determination
        - "std_residuals": Standard deviation of residuals
        - "r2_adj"       : Adjusted R^2 (only when n_params > 0)
    """
    v_y_arr = np.asarray(v_y, dtype=float)
    v_y_pred_arr = np.asarray(v_y_pred, dtype=float)
    n = len(v_y_arr)

    residuals = v_y_arr - v_y_pred_arr
    ss_res = float(np.sum(residuals**2))
    ss_tot = float(np.sum((v_y_arr - np.mean(v_y_arr))**2))

    mse = ss_res / n
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(residuals)))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot != 0.0 else 0.0
    std_res = float(np.std(residuals))

    metrics: dict[str, float] = {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "std_residuals": std_res,
    }

    dof = n - n_params - 1
    if n_params > 0 and dof > 0:
        r2_adj = float(1.0 - (1.0 - r2) * (n - 1) / dof)
        metrics["r2_adj"] = r2_adj

    return metrics


# =============================================================================
# fit_linear
# =============================================================================

def fit_linear(v_x: np.ndarray, v_y: np.ndarray) -> FitResult:
    """
    Fits a straight line y = a0 + a1*x using ordinary least squares.

    Parameters:
    * v_x (np.ndarray): 1-D array of independent variable values.
    * v_y (np.ndarray): 1-D array of observed dependent variable values.

    Returns:
    * FitResult:
        - v_params:     [intercept (a0), slope (a1)]
        - m_covariance: 2x2 parameter covariance matrix (sigma^2 * (X^T X)^{-1})
        - metrics:      includes "r2_adj"
        - model_name:   "linear"
    """
    v_x_arr, v_y_arr = _validate_1d_xy(v_x, v_y)
    n = len(v_x_arr)

    # Design matrix: columns = [1, x]
    m_X = np.column_stack([np.ones(n), v_x_arr])

    # Ordinary Least Squares
    v_params, _, _, _ = np.linalg.lstsq(m_X, v_y_arr, rcond=None)

    v_y_pred = m_X @ v_params
    v_residuals = v_y_arr - v_y_pred

    # Unbiased variance estimate (n - 2 degrees of freedom for a line)
    sigma2 = float(np.sum(v_residuals**2) / max(n - 2, 1))
    m_XtX_inv = np.linalg.inv(m_X.T @ m_X)
    m_covariance = sigma2 * m_XtX_inv

    metrics = fit_metrics(v_y_arr, v_y_pred, n_params=2)

    return FitResult(
        v_params=v_params,
        m_covariance=m_covariance,
        v_residuals=v_residuals,
        metrics=metrics,
        model_name="linear",
    )


# =============================================================================
# fit_custom
# =============================================================================

def _infer_n_params(s_f: Callable[..., float], x_sample: float = 1.0) -> int:
    """
    Infers the number of free parameters for a model function s_f(v_x).

    Checks in order:
    1. Parameter name prefix (e.g., v3_x implies dimension 3 -> 2 free parameters).
    2. Probing evaluation with vectors of dimension 2, 3, ... until one succeeds.
    """
    sig = inspect.signature(s_f)
    params = list(sig.parameters.values())
    if len(params) == 1:
        param_name = params[0].name
        match = re.match(r"^v(\d+)", param_name)
        if match:
            dim = int(match.group(1))
            if dim >= 2:
                return dim - 1
            raise ValueError(
                f"Parameter name '{param_name}' indicates dimension {dim}, "
                "which does not have at least one free parameter besides x (v[0])."
            )

    # Probing evaluation for abstract vectors v_ or lambdas
    for k in range(1, 31):
        for val in (0.0, 1.0, 0.5):
            test_vec = np.full(k + 1, val, dtype=float)
            test_vec[0] = x_sample
            try:
                s_f(test_vec)
                return k
            except (IndexError, ValueError):
                break
            except Exception:
                continue

    raise ValueError(
        "Could not automatically infer the number of free parameters for s_f. "
        "Please provide v_p0 explicitly."
    )


def fit_custom(
    v_x: np.ndarray,
    v_y: np.ndarray,
    s_f: Callable[[np.ndarray], float] | None = None,
    v_p0: np.ndarray | None = None,
    bounds: tuple = (-np.inf, np.inf),
    *,
    f: Callable[[np.ndarray], float] | None = None,
) -> FitResult:
    """
    Fits data to a user-defined function following the project naming conventions.

    The user-supplied function must follow the scalar-valued mapping convention:
        s_f(v_x) -> float    (R^n -> R)

    Within s_f, the input vector is assembled per sample as:
        v_x[0] -> x data for this sample
        v_x[1 : ...] -> free parameters (varied by optimizer)

    Parameters:
    * v_x (np.ndarray): 1-D array of independent variable values.
    * v_y (np.ndarray): 1-D array of observed dependent variable values.
    * s_f (Callable): Scalar-valued function callback (R^n -> R).
    * v_p0 (np.ndarray | None): Initial parameter guesses for the free parameters.
        If None, automatically defaults to an array of zeros.
    * bounds (tuple): Lower and upper bounds on parameters, forwarded to
      `scipy.optimize.curve_fit`.
    * f (Callable | None): Keyword-only alias for s_f.

    Returns:
    * FitResult:
        - v_params:     Fitted parameter values.
        - m_covariance: Covariance matrix of the fitted parameters.
        - v_residuals:  Residuals at each point.
        - metrics:      Goodness-of-fit metrics.
        - model_name:   "custom"
    """
    func = s_f if s_f is not None else f
    if func is None:
        raise ValueError("A model function s_f must be provided.")

    v_x_arr, v_y_arr = _validate_1d_xy(v_x, v_y)

    if v_p0 is None:
        n_params = _infer_n_params(func, x_sample=float(v_x_arr[0]))
        v_p0_arr = np.zeros(n_params, dtype=float)
    else:
        v_p0_arr = np.asarray(v_p0, dtype=float)

    def _wrapper(v_x_data: np.ndarray, *free_params: float) -> np.ndarray:
        v_free = np.array(free_params, dtype=float)
        return np.array([
            func(np.concatenate(([xi], v_free)))
            for xi in v_x_data
        ], dtype=float)

    v_free_params, m_covariance = curve_fit(
        _wrapper, v_x_arr, v_y_arr, p0=v_p0_arr, bounds=bounds, maxfev=10_000,
    )

    v_y_pred = _wrapper(v_x_arr, *v_free_params)
    v_residuals = v_y_arr - v_y_pred
    metrics = fit_metrics(v_y_arr, v_y_pred, n_params=len(v_free_params))

    return FitResult(
        v_params=v_free_params,
        m_covariance=m_covariance,
        v_residuals=v_residuals,
        metrics=metrics,
        model_name="custom",
    )
