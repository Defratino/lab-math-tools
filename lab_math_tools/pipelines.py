"""
Pipelines Module

Provides high-level, end-to-end scientific workflow pipelines combining
curve fitting, residual analysis, visualization, and Excel reporting.

Conventions followed:
- `v_`  : Abstract vector (numpy.ndarray with dynamic dimensions)
- `m_`  : Abstract matrix (numpy.ndarray with dynamic dimensions)
- `s_f` : Scalar-valued function callback (R^n -> R)
- `l_`  : Numerical list or collection
- `d_`  : Dictionary
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .curve_fitting import (
    FitResult,
    _validate_1d_xy,
    fit_custom,
    fit_linear,
)
from .graphing import plot_with_errors


# =============================================================================
# Pipeline Result Containers
# =============================================================================

@dataclass
class LinearPipelineResult:
    """
    Container for results produced by `linear_fit_pipeline`.

    Attributes:
    * fit_result (FitResult): Core linear fit output containing parameters,
      covariance matrix, residuals, and goodness-of-fit metrics.
    * v_stderr (np.ndarray): Standard errors of [intercept, slope], computed as
      sqrt(diag(m_covariance)).
    * fig (matplotlib.figure.Figure): Matplotlib Figure containing the plot.
    * ax (matplotlib.axes.Axes): Matplotlib Axes containing the data and fit line.
    """

    fit_result: FitResult
    v_stderr: np.ndarray
    fig: matplotlib.figure.Figure
    ax: matplotlib.axes.Axes


@dataclass
class CustomPipelineResult:
    """
    Container for results produced by `custom_fit_pipeline`.

    Attributes:
    * d_results (dict[str, FitResult]): Mapping of model name to FitResult.
    * d_stderr (dict[str, np.ndarray]): Mapping of model name to parameter
      standard errors (sqrt of covariance matrix diagonal).
    * df_metrics_comparison (pd.DataFrame): Comparison table of goodness-of-fit
      metrics across all candidate models.
    * fig (matplotlib.figure.Figure): Matplotlib Figure containing the 1x2 subplots.
    * axes (np.ndarray | tuple[matplotlib.axes.Axes, matplotlib.axes.Axes]):
      Tuple or array of (fit_ax, residuals_ax).
    """

    d_results: dict[str, FitResult]
    d_stderr: dict[str, np.ndarray]
    df_metrics_comparison: pd.DataFrame
    fig: matplotlib.figure.Figure
    axes: np.ndarray | tuple[matplotlib.axes.Axes, matplotlib.axes.Axes]


# =============================================================================
# Internal Excel Export Helper
# =============================================================================

def _export_to_excel(
    excel_path: str | Path,
    sheets: dict[str, pd.DataFrame],
) -> None:
    """
    Exports a dictionary of DataFrames to an Excel file (.xlsx) with multiple sheets.

    Creates parent directories automatically if they do not exist.
    """
    path = Path(excel_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for sheet_name, df in sheets.items():
            df.to_excel(writer, sheet_name=sheet_name, index=False)


# =============================================================================
# 1. linear_fit_pipeline
# =============================================================================

def linear_fit_pipeline(
    v_x: np.ndarray,
    v_y: np.ndarray,
    v_x_err: np.ndarray | None = None,
    v_y_err: np.ndarray | None = None,
    *,
    title: str = "Linear Fit",
    xlabel: str = "x",
    ylabel: str = "y",
    data_label: str = "Measurements",
    fit_label: str | None = None,
    excel_path: str | Path | None = None,
    ax: matplotlib.axes.Axes | None = None,
    show: bool = False,
    save_file_name: str | Path | None = None,
    legend_loc: str = "best",
    capsize: float = 3.0,
) -> LinearPipelineResult:
    """
    Automated pipeline for linear regression, plotting, and parameter reporting.

    Fits data to y = a0 + a1*x using ordinary least squares, plots the data
    (with error bars if provided) and the fitted line, and optionally saves the
    fitted parameters and goodness-of-fit metrics to an Excel workbook.

    Parameters:
    * v_x (np.ndarray): 1-D array of independent variable values.
    * v_y (np.ndarray): 1-D array of dependent variable values.
    * v_x_err (np.ndarray | None): Optional 1-D symmetric uncertainties on x.
    * v_y_err (np.ndarray | None): Optional 1-D symmetric uncertainties on y.
    * title (str): Title for the plot. Default "Linear Fit".
    * xlabel (str): Label for the x-axis. Default "x".
    * ylabel (str): Label for the y-axis. Default "y".
    * data_label (str): Legend label for the data points. Default "Measurements".
    * fit_label (str | None): Custom legend label for the fitted line. If None,
      an automatic equation string with parameter uncertainties and R^2 is used.
    * excel_path (str | Path | None): File path to save Excel report (.xlsx).
      If None, export is skipped.
    * ax (matplotlib.axes.Axes | None): Existing Axes to draw on. When None,
      a new Figure and Axes are created.
    * show (bool): Whether to display the plot interactively. Default False.
    * save_file_name (str | Path | None): File path to save the plot image.
    * legend_loc (str): Legend placement. Default "best".
    * capsize (float): Cap width for error bars. Default 3.0.

    Returns:
    * LinearPipelineResult: Result container with fields:
      - fit_result: FitResult object from fit_linear
      - v_stderr: Standard errors of [intercept, slope]
      - fig: Matplotlib Figure
      - ax: Matplotlib Axes
    """
    v_x_arr, v_y_arr = _validate_1d_xy(v_x, v_y)
    n = len(v_x_arr)

    # Validate error bar arrays
    v_x_err_arr: np.ndarray | None = None
    if v_x_err is not None:
        v_x_err_arr = np.asarray(v_x_err, dtype=float)
        if v_x_err_arr.ndim != 1 or len(v_x_err_arr) != n:
            raise ValueError(
                f"v_x_err must have the same length as v_x ({n}); got {len(v_x_err_arr)}."
            )

    v_y_err_arr: np.ndarray | None = None
    if v_y_err is not None:
        v_y_err_arr = np.asarray(v_y_err, dtype=float)
        if v_y_err_arr.ndim != 1 or len(v_y_err_arr) != n:
            raise ValueError(
                f"v_y_err must have the same length as v_y ({n}); got {len(v_y_err_arr)}."
            )

    # 1. Compute linear fit
    fit_res = fit_linear(v_x_arr, v_y_arr)
    v_params = fit_res.v_params
    intercept, slope = float(v_params[0]), float(v_params[1])

    # 2. Compute standard errors from covariance diagonal
    v_stderr = np.sqrt(np.diag(fit_res.m_covariance))
    se_intercept, se_slope = float(v_stderr[0]), float(v_stderr[1])

    # 3. Setup Figure and Axes
    if ax is None:
        fig, ax = plt.subplots(figsize=(7.5, 5))
    else:
        fig = ax.get_figure()

    # 4. Plot data measurements
    has_errors = (v_x_err_arr is not None) or (v_y_err_arr is not None)
    if has_errors:
        plot_with_errors(
            lv_x=v_x_arr,
            lv_y=v_y_arr,
            lv_x_err=v_x_err_arr,
            lv_y_err=v_y_err_arr,
            ax=ax,
            show=False,
            line_labels=[data_label],
            capsize=capsize,
            colors=["#1f77b4"],
            color_method="specific",
            linestyles=["none"],
            markers=["o"],
            legend_loc=legend_loc,
        )
    else:
        ax.scatter(
            v_x_arr,
            v_y_arr,
            color="#1f77b4",
            s=28,
            alpha=0.8,
            label=data_label,
            zorder=3,
        )

    # 5. Plot fitted line
    x_min, x_max = float(np.min(v_x_arr)), float(np.max(v_x_arr))
    x_pad = 0.02 * (x_max - x_min) if x_max > x_min else 0.1
    x_line = np.linspace(x_min - x_pad, x_max + x_pad, 200)
    y_line = intercept + slope * x_line

    r2_val = fit_res.metrics.get("r2", 0.0)
    if fit_label is None:
        fit_label = (
            f"Fit: y = ({slope:.4g} ± {se_slope:.2g})x + ({intercept:.4g} ± {se_intercept:.2g})\n"
            f"(R² = {r2_val:.4f})"
        )

    ax.plot(x_line, y_line, color="#d62728", lw=2.0, label=fit_label, zorder=4)

    # 6. Styling
    ax.set_title(title, fontsize=12)
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.legend(loc=legend_loc, framealpha=0.85)
    ax.grid(True, linestyle="--", alpha=0.5)

    # 7. Excel export
    if excel_path is not None:
        df_params = pd.DataFrame({
            "Parameter": ["intercept", "slope"],
            "Value": [intercept, slope],
            "Std_Error": [se_intercept, se_slope],
        })
        df_metrics = pd.DataFrame({
            "Metric": list(fit_res.metrics.keys()),
            "Value": list(fit_res.metrics.values()),
        })
        _export_to_excel(
            excel_path,
            {"Parameters": df_params, "Metrics": df_metrics},
        )

    # 8. Save image & show
    if save_file_name is not None:
        save_path = Path(save_file_name)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")

    if show:
        plt.show()

    return LinearPipelineResult(
        fit_result=fit_res,
        v_stderr=v_stderr,
        fig=fig,
        ax=ax,
    )


# =============================================================================
# 2. custom_fit_pipeline
# =============================================================================

def custom_fit_pipeline(
    v_x: np.ndarray,
    v_y: np.ndarray,
    s_f: Callable[[np.ndarray], float] | None = None,
    v_p0: np.ndarray | None = None,
    bounds: tuple = (-np.inf, np.inf),
    v_x_err: np.ndarray | None = None,
    v_y_err: np.ndarray | None = None,
    *,
    d_s_f: dict[str, Callable[[np.ndarray], float]] | None = None,
    l_s_f: list[Callable[[np.ndarray], float]] | None = None,
    d_v_p0: dict[str, np.ndarray] | None = None,
    d_bounds: dict[str, tuple] | None = None,
    title: str = "Curve Fitting & Residual Analysis",
    xlabel: str = "x",
    ylabel: str = "y",
    data_label: str = "Measurements",
    excel_path: str | Path | None = None,
    axes: tuple[matplotlib.axes.Axes, matplotlib.axes.Axes] | np.ndarray | None = None,
    show: bool = False,
    save_file_name: str | Path | None = None,
    legend_loc: str = "best",
    capsize: float = 3.0,
) -> CustomPipelineResult:
    """
    Automated pipeline for custom curve fitting, residual analysis, and model comparison.

    Fits data to one or more user-defined scalar functions s_f(v_x), visualizes the
    measurements and all fitted curves side-by-side with an aligned residuals plot,
    and optionally exports the parameter uncertainties and comparison report to Excel.

    Parameters:
    * v_x (np.ndarray): 1-D array of independent variable values.
    * v_y (np.ndarray): 1-D array of dependent variable values.
    * s_f (Callable | None): Single scalar model function s_f(v_x) -> float.
    * v_p0 (np.ndarray | None): Initial parameter guesses for single s_f model.
    * bounds (tuple): Parameter bounds for single s_f model.
    * v_x_err (np.ndarray | None): Optional 1-D symmetric uncertainties on x.
    * v_y_err (np.ndarray | None): Optional 1-D symmetric uncertainties on y.
    * d_s_f (dict[str, Callable] | None): Dictionary mapping model names to model functions.
    * l_s_f (list[Callable] | None): List of candidate model functions.
    * d_v_p0 (dict[str, np.ndarray] | None): Initial guesses per model name.
    * d_bounds (dict[str, tuple] | None): Parameter bounds per model name.
    * title (str): Overall title or fit axis title. Default "Curve Fitting & Residual Analysis".
    * xlabel (str): Label for the x-axis. Default "x".
    * ylabel (str): Label for the y-axis. Default "y".
    * data_label (str): Label for measurements in legend. Default "Measurements".
    * excel_path (str | Path | None): File path to save Excel report (.xlsx).
    * axes (tuple[Axes, Axes] | np.ndarray | None): Pair of Matplotlib Axes (fit_ax, res_ax).
      If None, a 1x2 subplot figure is created.
    * show (bool): Whether to display the plot interactively. Default False.
    * save_file_name (str | Path | None): File path to save the plot figure.
    * legend_loc (str): Legend position on fit axis. Default "best".
    * capsize (float): Cap width for error bars. Default 3.0.

    Returns:
    * CustomPipelineResult: Result container with fields:
      - d_results: dict[model_name, FitResult]
      - d_stderr: dict[model_name, v_stderr]
      - df_metrics_comparison: DataFrame comparing goodness-of-fit metrics
      - fig: Matplotlib Figure
      - axes: Pair of Axes (fit_ax, res_ax)
    """
    v_x_arr, v_y_arr = _validate_1d_xy(v_x, v_y)
    n = len(v_x_arr)

    # Validate error bar arrays
    v_x_err_arr: np.ndarray | None = None
    if v_x_err is not None:
        v_x_err_arr = np.asarray(v_x_err, dtype=float)
        if v_x_err_arr.ndim != 1 or len(v_x_err_arr) != n:
            raise ValueError(
                f"v_x_err must have the same length as v_x ({n}); got {len(v_x_err_arr)}."
            )

    v_y_err_arr: np.ndarray | None = None
    if v_y_err is not None:
        v_y_err_arr = np.asarray(v_y_err, dtype=float)
        if v_y_err_arr.ndim != 1 or len(v_y_err_arr) != n:
            raise ValueError(
                f"v_y_err must have the same length as v_y ({n}); got {len(v_y_err_arr)}."
            )

    # Normalize models into a dictionary: {model_name: s_f}
    models: dict[str, Callable[[np.ndarray], float]] = {}
    if d_s_f is not None:
        models = dict(d_s_f)
    elif l_s_f is not None:
        models = {f"Model_{i + 1}": fn for i, fn in enumerate(l_s_f)}
    elif s_f is not None:
        models = {"Custom_Model": s_f}
    else:
        raise ValueError(
            "At least one model function must be provided via s_f, l_s_f, or d_s_f."
        )

    # Fit each model and compute parameter standard errors
    d_results: dict[str, FitResult] = {}
    d_stderr: dict[str, np.ndarray] = {}

    for name, func in models.items():
        # Resolve initial guesses and bounds
        p0 = d_v_p0.get(name, v_p0) if d_v_p0 is not None else v_p0
        bnd = d_bounds.get(name, bounds) if d_bounds is not None else bounds

        res = fit_custom(v_x_arr, v_y_arr, s_f=func, v_p0=p0, bounds=bnd)
        res.model_name = name

        stderr = np.sqrt(np.diag(res.m_covariance))
        d_results[name] = res
        d_stderr[name] = stderr

    # Build metrics comparison DataFrame
    comparison_rows = []
    for name, res in d_results.items():
        row = {"Model": name}
        row.update(res.metrics)
        comparison_rows.append(row)

    df_metrics = pd.DataFrame(comparison_rows)

    # Setup 1x2 subplots
    if axes is None:
        fig, axes_pair = plt.subplots(1, 2, figsize=(12, 4.5))
    else:
        axes_pair = axes
        fig = axes_pair[0].get_figure()

    ax_fit, ax_res = axes_pair[0], axes_pair[1]

    # Colors for candidate models
    color_cycle = ["#d62728", "#2ca02c", "#ff7f0e", "#9467bd", "#8c564b", "#e377c2", "#17becf"]

    # 1. Plot measurements on ax_fit
    has_errors = (v_x_err_arr is not None) or (v_y_err_arr is not None)
    if has_errors:
        plot_with_errors(
            lv_x=v_x_arr,
            lv_y=v_y_arr,
            lv_x_err=v_x_err_arr,
            lv_y_err=v_y_err_arr,
            ax=ax_fit,
            show=False,
            line_labels=[data_label],
            capsize=capsize,
            colors=["#1f77b4"],
            color_method="specific",
            linestyles=["none"],
            markers=["o"],
            legend_loc=legend_loc,
        )
    else:
        ax_fit.scatter(
            v_x_arr,
            v_y_arr,
            color="#1f77b4",
            s=25,
            alpha=0.75,
            label=data_label,
            zorder=3,
        )

    # 2. Overlay fitted curves on ax_fit and residuals on ax_res
    x_min, x_max = float(np.min(v_x_arr)), float(np.max(v_x_arr))
    x_pad = 0.02 * (x_max - x_min) if x_max > x_min else 0.1
    x_dense = np.linspace(x_min - x_pad, x_max + x_pad, 300)

    # Baseline on residuals plot
    ax_res.axhline(0.0, color="gray", linewidth=0.8, linestyle="--", zorder=2)

    indices = np.arange(n)
    is_multi_model = len(d_results) > 1

    for idx, (name, fit_res) in enumerate(d_results.items()):
        color = color_cycle[idx % len(color_cycle)]
        func = models[name]
        v_free = fit_res.v_params

        # Predict along dense grid
        y_dense = np.array([
            func(np.concatenate(([xi], v_free)))
            for xi in x_dense
        ], dtype=float)

        r2_val = fit_res.metrics.get("r2", 0.0)
        curve_label = f"{name} (R²={r2_val:.4f})"
        ax_fit.plot(x_dense, y_dense, color=color, lw=2.0, label=curve_label, zorder=4)

        # Plot residuals on ax_res
        if not is_multi_model:
            # Single model: use stem plot with clean color styling
            markerline, stemlines, baseline = ax_res.stem(
                indices,
                fit_res.v_residuals,
            )
            plt.setp(markerline, color=color, markersize=5)
            plt.setp(stemlines, color=color, linewidth=1.5)
            if baseline is not None:
                plt.setp(baseline, visible=False)
        else:
            # Multiple models: stem / scatter comparison with jittered markers
            ax_res.plot(
                indices,
                fit_res.v_residuals,
                color=color,
                marker="o",
                markersize=4,
                linestyle="--",
                alpha=0.7,
                label=f"{name} (RMSE={fit_res.metrics.get('rmse', 0.0):.3g})",
            )

    # Fit Axis Styling
    ax_fit.set_title(title, fontsize=12)
    ax_fit.set_xlabel(xlabel, fontsize=11)
    ax_fit.set_ylabel(ylabel, fontsize=11)
    ax_fit.legend(loc=legend_loc, framealpha=0.85)
    ax_fit.grid(True, linestyle="--", alpha=0.5)

    # Residuals Axis Styling
    ax_res.set_title(f"Residuals ({'Comparison' if is_multi_model else 'y − ŷ'})", fontsize=12)
    ax_res.set_xlabel("Sample index", fontsize=11)
    ax_res.set_ylabel("Residual (y − ŷ)", fontsize=11)
    if is_multi_model:
        ax_res.legend(loc="best", framealpha=0.85)
    ax_res.grid(True, linestyle="--", alpha=0.5)

    # 3. Excel Export
    if excel_path is not None:
        param_rows = []
        for name, fit_res in d_results.items():
            stderrs = d_stderr[name]
            for i, (val, se) in enumerate(zip(fit_res.v_params, stderrs)):
                param_rows.append({
                    "Model": name,
                    "Parameter": f"p[{i}]",
                    "Value": float(val),
                    "Std_Error": float(se),
                })
        df_params = pd.DataFrame(param_rows)

        _export_to_excel(
            excel_path,
            {"Model_Comparison": df_metrics, "Parameters": df_params},
        )

    # 4. Save image & show
    if save_file_name is not None:
        save_path = Path(save_file_name)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")

    if show:
        plt.show()

    return CustomPipelineResult(
        d_results=d_results,
        d_stderr=d_stderr,
        df_metrics_comparison=df_metrics,
        fig=fig,
        axes=axes_pair,
    )
