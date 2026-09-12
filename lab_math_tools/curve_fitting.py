"""
Curve Fitting Module

Provides tools for fitting mathematical models to experimental data.
Supports linear regression, polynomial fitting, preset model fitting,
and custom user-defined function fitting.

Conventions followed:
- `v_`  : Abstract vector (numpy.ndarray with dynamic dimensions)
- `m_`  : Abstract matrix (numpy.ndarray with dynamic dimensions)
- `s_f` : Scalar-valued function callback (R^n -> R)
- `f`   : Generic function callback
"""

from __future__ import annotations

from dataclasses import dataclass
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
        - fit_linear:    [intercept, slope]
        - fit_polynomial: [a0, a1, ..., a_degree] ascending power order;
          pinned coefficients are exactly 0.0.
        - fit_preset:    model-dependent (see fit_preset docstring).
        - fit_custom:    only the FREE parameters that were fitted.
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
# fit_polynomial
# =============================================================================

def fit_polynomial(
    v_x: np.ndarray,
    v_y: np.ndarray,
    degree: int,
    zero_degrees: list[int] | None = None,
) -> FitResult:
    """
    Fits a polynomial y = a0 + a1*x + ... + a_n*x^n using least squares.

    Parameters:
    * v_x (np.ndarray): 1-D array of independent variable values.
    * v_y (np.ndarray): 1-D array of observed dependent variable values.
    * degree (int): Polynomial degree n. The full coefficient vector has
        length degree+1 in ascending power order [a0, a1, ..., a_degree].
    * zero_degrees (list[int] | None): Power indices to pin to exactly 0.
        For example, zero_degrees=[0, 2] on a degree-3 fit produces
        f(x) = a1*x + a3*x^3. Equivalent to bias=False when zero_degrees=[0].
        Raises ValueError for any index outside [0, degree].

    Returns:
    * FitResult:
        - v_params:     1-D float array of length degree+1. Pinned entries are 0.
        - m_covariance: (n_free, n_free) covariance for the FREE parameters only.
        - model_name:   "poly_{degree}" or "poly_{degree}_zero{zero_degrees}"
    """
    v_x_arr, v_y_arr = _validate_1d_xy(v_x, v_y)

    if zero_degrees is not None:
        bad = [d for d in zero_degrees if d < 0 or d > degree]
        if bad:
            raise ValueError(
                f"zero_degrees contains indices {bad} outside the valid "
                f"range [0, {degree}]."
            )

    all_powers = list(range(degree + 1))
    zero_set = set(zero_degrees) if zero_degrees else set()
    free_powers = [p for p in all_powers if p not in zero_set]

    # Design matrix for free powers only
    m_X_free = np.column_stack([v_x_arr**p for p in free_powers])

    # Least squares on the reduced system
    v_free_params, _, _, _ = np.linalg.lstsq(m_X_free, v_y_arr, rcond=None)

    # Reconstruct full coefficient vector (zeros at pinned positions)
    v_params = np.zeros(degree + 1)
    for i, p in enumerate(free_powers):
        v_params[p] = v_free_params[i]

    # Predictions using the full polynomial
    m_X_full = np.column_stack([v_x_arr**p for p in all_powers])
    v_y_pred = m_X_full @ v_params
    v_residuals = v_y_arr - v_y_pred

    n = len(v_x_arr)
    n_free = len(free_powers)
    sigma2 = float(np.sum(v_residuals**2) / max(n - n_free, 1))
    m_covariance = sigma2 * np.linalg.pinv(m_X_free.T @ m_X_free)

    metrics = fit_metrics(v_y_arr, v_y_pred, n_params=n_free)

    suffix = f"_zero{zero_degrees}" if zero_degrees else ""
    return FitResult(
        v_params=v_params,
        m_covariance=m_covariance,
        v_residuals=v_residuals,
        metrics=metrics,
        model_name=f"poly_{degree}{suffix}",
    )


# =============================================================================
# Preset model functions
# =============================================================================

def _model_sine(x: np.ndarray, A: float, omega: float, phi: float, C: float) -> np.ndarray:
    return A * np.sin(omega * x + phi) + C


def _model_sine_no_bias(x: np.ndarray, A: float, omega: float, phi: float) -> np.ndarray:
    return A * np.sin(omega * x + phi)


def _model_exponential(x: np.ndarray, A: float, b: float, C: float) -> np.ndarray:
    return A * np.exp(b * x) + C


def _model_exponential_no_bias(x: np.ndarray, A: float, b: float) -> np.ndarray:
    return A * np.exp(b * x)


def _model_logarithm(x: np.ndarray, A: float, b: float, C: float) -> np.ndarray:
    return A * np.log(b * x) + C


def _model_logarithm_no_bias(x: np.ndarray, A: float, b: float) -> np.ndarray:
    return A * np.log(b * x)


def _model_power(x: np.ndarray, A: float, b: float) -> np.ndarray:
    return A * x**b


def _model_gaussian(x: np.ndarray, A: float, mu: float, sigma: float) -> np.ndarray:
    return A * np.exp(-((x - mu)**2) / (2.0 * sigma**2))


# Preset registry: maps model name -> biased model, unbiased model, default p0 pairs
_PRESET_INFO: dict[str, dict] = {
    "sine": {
        "model": _model_sine,
        "model_no_bias": _model_sine_no_bias,
        "p0": [1.0, 1.0, 0.0, 0.0],
        "p0_no_bias": [1.0, 1.0, 0.0],
    },
    "exponential": {
        "model": _model_exponential,
        "model_no_bias": _model_exponential_no_bias,
        "p0": [1.0, 1.0, 0.0],
        "p0_no_bias": [1.0, 1.0],
    },
    "logarithm": {
        "model": _model_logarithm,
        "model_no_bias": _model_logarithm_no_bias,
        "p0": [1.0, 1.0, 0.0],
        "p0_no_bias": [1.0, 1.0],
    },
    "power": {
        # Power law A*x^b has no additive C; bias flag is ignored.
        "model": _model_power,
        "model_no_bias": _model_power,
        "p0": [1.0, 1.0],
        "p0_no_bias": [1.0, 1.0],
    },
    "gaussian": {
        # Gaussian A*exp(...) has no additive C; bias flag is ignored.
        "model": _model_gaussian,
        "model_no_bias": _model_gaussian,
        "p0": [1.0, 0.0, 1.0],
        "p0_no_bias": [1.0, 0.0, 1.0],
    },
}


# =============================================================================
# fit_preset
# =============================================================================

def fit_preset(
    v_x: np.ndarray,
    v_y: np.ndarray,
    model: str,
    bias: bool = True,
    v_p0: np.ndarray | None = None,
    bounds: tuple = (-np.inf, np.inf),
) -> FitResult:
    """
    Fits data to a pre-defined model using non-linear least squares
    (scipy.optimize.curve_fit).

    Parameters:
    * v_x (np.ndarray): 1-D array of independent variable values.
    * v_y (np.ndarray): 1-D array of observed dependent variable values.
    * model (str): Preset model identifier. One of:
        - "sine"        : A*sin(w*x + phi) + C,  v_params=[A, w, phi, C]
        - "exponential" : A*exp(b*x) + C,        v_params=[A, b, C]
        - "logarithm"   : A*ln(b*x) + C,         v_params=[A, b, C]
        - "power"       : A*x^b,                 v_params=[A, b]
        - "gaussian"    : A*exp(-(x-u)^2/2s^2),  v_params=[A, u, s]
    * bias (bool): If True (default), includes additive constant C in the
        model. If False, C is fixed to zero and excluded from v_params.
        Note: "power" and "gaussian" have no additive C; bias is ignored.
    * v_p0 (np.ndarray | None): Initial parameter guesses. Must match the
        number of free parameters for the chosen model/bias combination.
        Default guesses are used when None.
    * bounds: Parameter bounds forwarded to scipy.optimize.curve_fit.

    Returns:
    * FitResult:
        - model_name: "{model}" or "{model}_no_bias"
    """
    v_x_arr, v_y_arr = _validate_1d_xy(v_x, v_y)

    if model not in _PRESET_INFO:
        raise ValueError(
            f"Unknown model '{model}'. "
            f"Available models: {list(_PRESET_INFO.keys())}"
        )

    info = _PRESET_INFO[model]
    model_func = info["model"] if bias else info["model_no_bias"]
    default_p0 = info["p0"] if bias else info["p0_no_bias"]
    p0 = np.asarray(v_p0, dtype=float) if v_p0 is not None else np.array(default_p0)

    v_params, m_covariance = curve_fit(
        model_func, v_x_arr, v_y_arr, p0=p0, bounds=bounds, maxfev=10_000,
    )

    v_y_pred = model_func(v_x_arr, *v_params)
    v_residuals = v_y_arr - v_y_pred
    metrics = fit_metrics(v_y_arr, v_y_pred, n_params=len(v_params))

    name_suffix = "" if bias else "_no_bias"
    return FitResult(
        v_params=v_params,
        m_covariance=m_covariance,
        v_residuals=v_residuals,
        metrics=metrics,
        model_name=f"{model}{name_suffix}",
    )


# =============================================================================
# fit_custom
# =============================================================================

def fit_custom(
    v_x: np.ndarray,
    v_y: np.ndarray,
    f: Callable[[np.ndarray], float],
    v_p0: np.ndarray,
    v_fixed_params: np.ndarray | None = None,
    bounds: tuple = (-np.inf, np.inf),
) -> FitResult:
    """
    Fits data to a user-defined function following the project naming conventions.

    The user-supplied function must follow the scalar-valued mapping convention:
        f(v_x) -> float    (R^n -> R)

    Within f, the input vector is assembled per sample as:
        v_x[0]                       -> x data for this sample
        v_x[1 : 1 + n_fixed]         -> v_fixed_params (pre-set constants)
        v_x[1 + n_fixed : ...]       -> free parameters (varied by optimizer)

    Example
    -------
    Fitting the amplitude of a sine wave with frequency pre-fixed to 2.0:

        def s_f(v3_x: np.ndarray) -> float:
            # v3_x[0]=x_data, v3_x[1]=fixed_freq=2.0, v3_x[2]=amplitude (free)
            return v3_x[2] * np.sin(v3_x[1] * v3_x[0])

        result = fit_custom(v_x, v_y, s_f,
                            v_p0=np.array([1.0]),
                            v_fixed_params=np.array([2.0]))

    Parameters:
    * v_x (np.ndarray): 1-D array of x data values (one per sample).
    * v_y (np.ndarray): 1-D array of observed y values.
    * f (callable): User function f(v_x) -> float per the convention above.
    * v_p0 (np.ndarray): Initial guess for the FREE parameters only.
    * v_fixed_params (np.ndarray | None): Constant values placed at positions
        [1 : 1 + n_fixed] in the assembled vector. None means no fixed params.
    * bounds: Bounds for the free parameters, forwarded to curve_fit.

    Returns:
    * FitResult:
        - v_params:     FREE (fitted) parameters only.
        - m_covariance: (n_free, n_free) covariance for the free parameters.
        - model_name:   "custom"
    """
    v_x_arr, v_y_arr = _validate_1d_xy(v_x, v_y)
    v_p0_arr = np.asarray(v_p0, dtype=float)
    v_fixed = (
        np.asarray(v_fixed_params, dtype=float)
        if v_fixed_params is not None
        else np.array([], dtype=float)
    )

    def _wrapper(v_x_data: np.ndarray, *free_params: float) -> np.ndarray:
        v_free = np.array(free_params, dtype=float)
        return np.array([
            f(np.concatenate(([xi], v_fixed, v_free)))
            for xi in v_x_data
        ])

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
