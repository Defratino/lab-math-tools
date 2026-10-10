"""
Unit tests for lab_math_tools.pipelines module.

Tests cover:
- Input validation and shape checking
- linear_fit_pipeline (OLS fit, errors/uncertainties, plotting, Excel export)
- custom_fit_pipeline (single-model & multi-model comparison, residuals plot, Excel export)
- Plotting customizations (error bars, titles, labels, custom axes, figure saving)
- Excel export structure and data integrity
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for testing
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from lab_math_tools.curve_fitting import FitResult
from lab_math_tools.pipelines import (
    CustomPipelineResult,
    LinearPipelineResult,
    custom_fit_pipeline,
    linear_fit_pipeline,
)


@pytest.fixture(autouse=True)
def close_figures():
    """Ensure all Matplotlib figures are closed after each test to prevent memory leaks."""
    yield
    plt.close("all")


RNG = np.random.default_rng(42)
RTOL = 1e-2


# =============================================================================
# Suite 1: Input Validation & Error Handling
# =============================================================================

class TestPipelinesValidation:
    def test_linear_fit_mismatched_lengths(self):
        v_x = np.array([1.0, 2.0, 3.0])
        v_y = np.array([1.0, 2.0])
        with pytest.raises(ValueError, match="same length"):
            linear_fit_pipeline(v_x, v_y)

    def test_linear_fit_non_1d_input(self):
        v_x = np.ones((3, 2))
        v_y = np.ones((3, 2))
        with pytest.raises(ValueError, match="1-D"):
            linear_fit_pipeline(v_x, v_y)

    def test_linear_fit_mismatched_errors(self):
        v_x = np.array([1.0, 2.0, 3.0])
        v_y = np.array([2.0, 4.0, 6.0])
        v_y_err = np.array([0.1, 0.2])
        with pytest.raises(ValueError, match="v_y_err must have the same length"):
            linear_fit_pipeline(v_x, v_y, v_y_err=v_y_err)

    def test_linear_fit_mismatched_x_errors(self):
        v_x = np.array([1.0, 2.0, 3.0])
        v_y = np.array([2.0, 4.0, 6.0])
        v_x_err = np.array([0.1, 0.2])
        with pytest.raises(ValueError, match="v_x_err must have the same length"):
            linear_fit_pipeline(v_x, v_y, v_x_err=v_x_err)

    def test_custom_fit_missing_model(self):
        v_x = np.linspace(0, 5, 10)
        v_y = 2.0 * v_x
        with pytest.raises(ValueError, match="At least one model function"):
            custom_fit_pipeline(v_x, v_y)

    def test_custom_fit_mismatched_lengths(self):
        v_x = np.array([1.0, 2.0, 3.0])
        v_y = np.array([1.0, 2.0])
        with pytest.raises(ValueError, match="same length"):
            custom_fit_pipeline(v_x, v_y, s_f=lambda v: v[1] * v[0])


# =============================================================================
# Suite 2: linear_fit_pipeline Functionality
# =============================================================================

class TestLinearFitPipeline:
    def test_linear_fit_clean_recovery(self):
        v_x = np.linspace(0.0, 10.0, 50)
        # y = 3.5 * x - 2.0
        v_y = 3.5 * v_x - 2.0
        res = linear_fit_pipeline(v_x, v_y)

        assert isinstance(res, LinearPipelineResult)
        assert isinstance(res.fit_result, FitResult)
        intercept, slope = res.fit_result.v_params
        assert np.isclose(intercept, -2.0, rtol=RTOL)
        assert np.isclose(slope, 3.5, rtol=RTOL)
        assert np.isclose(res.fit_result.metrics["r2"], 1.0, atol=1e-5)
        assert res.ax is not None
        assert res.fig is not None

    def test_linear_fit_standard_errors(self):
        v_x = np.linspace(0.0, 10.0, 40)
        v_y = 2.0 * v_x + 1.0 + RNG.normal(0, 0.5, 40)
        res = linear_fit_pipeline(v_x, v_y)

        assert len(res.v_stderr) == 2
        assert res.v_stderr[0] > 0.0
        assert res.v_stderr[1] > 0.0
        expected_stderr = np.sqrt(np.diag(res.fit_result.m_covariance))
        assert np.allclose(res.v_stderr, expected_stderr)

    def test_linear_fit_with_y_errors(self):
        v_x = np.linspace(1.0, 5.0, 10)
        v_y = 1.5 * v_x + 0.5
        v_y_err = np.full_like(v_y, 0.2)

        res = linear_fit_pipeline(v_x, v_y, v_y_err=v_y_err, capsize=4.0)

        # Matplotlib creates an ErrorbarContainer when error bars are plotted
        assert len(res.ax.containers) > 0
        container = res.ax.containers[0]
        assert isinstance(container, matplotlib.container.ErrorbarContainer)
        assert container.has_yerr

    def test_linear_fit_with_xy_errors(self):
        v_x = np.linspace(1.0, 5.0, 10)
        v_y = 1.5 * v_x + 0.5
        v_x_err = np.full_like(v_x, 0.1)
        v_y_err = np.full_like(v_y, 0.2)

        res = linear_fit_pipeline(v_x, v_y, v_x_err=v_x_err, v_y_err=v_y_err)

        container = res.ax.containers[0]
        assert container.has_xerr
        assert container.has_yerr

    def test_linear_fit_plot_line_and_legend(self):
        v_x = np.linspace(0.0, 5.0, 15)
        v_y = 2.0 * v_x + 3.0
        res = linear_fit_pipeline(v_x, v_y, data_label="Raw Exp Data", fit_label="OLS Line")

        # Check line plotted on ax
        assert len(res.ax.lines) >= 1
        legend = res.ax.get_legend()
        assert legend is not None
        legend_texts = [t.get_text() for t in legend.get_texts()]
        assert any("OLS Line" in txt or "y =" in txt for txt in legend_texts)


# =============================================================================
# Suite 3: custom_fit_pipeline (Single Model)
# =============================================================================

class TestCustomFitPipelineSingleModel:
    def test_custom_fit_single_exponential_decay(self):
        # f(x) = a * exp(-b * x) -> v_x[1] * exp(-v_x[2] * v_x[0])
        def s_f_decay(v_x: np.ndarray) -> float:
            return float(v_x[1] * np.exp(-v_x[2] * v_x[0]))

        v_x = np.linspace(0.0, 4.0, 40)
        v_y = 5.0 * np.exp(-0.8 * v_x)

        res = custom_fit_pipeline(v_x, v_y, s_f=s_f_decay, v_p0=[4.0, 1.0])

        assert isinstance(res, CustomPipelineResult)
        assert len(res.d_results) == 1
        model_name = list(res.d_results.keys())[0]
        fit_res = res.d_results[model_name]

        # Parameters recovery
        assert np.isclose(fit_res.v_params[0], 5.0, rtol=RTOL)
        assert np.isclose(fit_res.v_params[1], 0.8, rtol=RTOL)
        assert np.isclose(fit_res.metrics["r2"], 1.0, atol=1e-4)

        # Check 1x2 subplots
        assert len(res.axes) == 2
        # axes[0] has data + fitted curve
        assert len(res.axes[0].lines) >= 1
        # axes[1] has residuals
        assert len(res.axes[1].lines) >= 1

    def test_custom_fit_single_residuals_plot(self):
        def s_f_quad(v_x: np.ndarray) -> float:
            return float(v_x[1] * (v_x[0] ** 2))

        v_x = np.linspace(1.0, 5.0, 20)
        v_y = 2.0 * (v_x ** 2) + RNG.normal(0, 0.1, 20)

        res = custom_fit_pipeline(v_x, v_y, s_f=s_f_quad, v_p0=[1.0])

        # axes[1] should have the reference line at y=0
        ax_res = res.axes[1]
        assert ax_res.get_title().lower().startswith("residuals")
        # Check that there's a horizontal reference line at y=0
        ref_lines = [line for line in ax_res.lines if np.all(line.get_ydata() == 0.0)]
        assert len(ref_lines) >= 1

    def test_custom_fit_bounds_respected(self):
        def s_f_bounded(v_x: np.ndarray) -> float:
            return float(v_x[1] * v_x[0])

        v_x = np.linspace(1.0, 5.0, 20)
        v_y = -3.0 * v_x  # slope would normally be negative

        # Enforce slope in [0, 10]
        res = custom_fit_pipeline(
            v_x, v_y, s_f=s_f_bounded, v_p0=[1.0], bounds=([0.0], [10.0])
        )
        model_res = list(res.d_results.values())[0]
        assert model_res.v_params[0] >= 0.0


# =============================================================================
# Suite 4: custom_fit_pipeline (Multi-Model Comparison)
# =============================================================================

class TestCustomFitPipelineMultiModel:
    def test_custom_fit_multi_model_dict_input(self):
        # Candidate 1: Linear f(x) = p0 + p1*x
        def s_f_lin(v_x: np.ndarray) -> float:
            return float(v_x[1] + v_x[2] * v_x[0])

        # Candidate 2: Quadratic f(x) = p0 + p1*x^2
        def s_f_quad(v_x: np.ndarray) -> float:
            return float(v_x[1] + v_x[2] * (v_x[0] ** 2))

        v_x = np.linspace(0.0, 5.0, 30)
        v_y = 1.0 + 2.0 * (v_x ** 2) + RNG.normal(0, 0.05, 30)

        d_models = {"Linear": s_f_lin, "Quadratic": s_f_quad}
        res = custom_fit_pipeline(v_x, v_y, d_s_f=d_models)

        assert len(res.d_results) == 2
        assert "Linear" in res.d_results
        assert "Quadratic" in res.d_results

        # Quadratic model should fit the data significantly better
        assert res.d_results["Quadratic"].metrics["r2"] > res.d_results["Linear"].metrics["r2"]
        assert res.d_results["Quadratic"].metrics["rmse"] < res.d_results["Linear"].metrics["rmse"]

        # Check comparison DataFrame
        assert isinstance(res.df_metrics_comparison, pd.DataFrame)
        assert len(res.df_metrics_comparison) == 2
        assert "r2" in res.df_metrics_comparison.columns
        assert "rmse" in res.df_metrics_comparison.columns

    def test_custom_fit_multi_model_list_input(self):
        def f1(v_x: np.ndarray) -> float:
            return float(v_x[1] * v_x[0])

        def f2(v_x: np.ndarray) -> float:
            return float(v_x[1] * (v_x[0] ** 2))

        v_x = np.linspace(1.0, 4.0, 15)
        v_y = 2.0 * v_x

        res = custom_fit_pipeline(v_x, v_y, l_s_f=[f1, f2])
        assert len(res.d_results) == 2

    def test_custom_fit_multi_model_overlays(self):
        def f_lin(v_x: np.ndarray) -> float:
            return float(v_x[1] * v_x[0])

        def f_quad(v_x: np.ndarray) -> float:
            return float(v_x[1] * (v_x[0] ** 2))

        v_x = np.linspace(1.0, 5.0, 20)
        v_y = 2.0 * v_x

        res = custom_fit_pipeline(v_x, v_y, d_s_f={"Lin": f_lin, "Quad": f_quad})

        # axes[0] must have lines for both models
        assert len(res.axes[0].lines) >= 2
        # axes[1] must have residual plots for both models
        assert len(res.axes[1].lines) >= 2


# =============================================================================
# Suite 5: Plotting, Styling & OOP Integration
# =============================================================================

class TestPlottingAndOOPIntegration:
    def test_linear_fit_custom_ax(self):
        fig, ax = plt.subplots(figsize=(6, 4))
        v_x = np.linspace(0, 10, 20)
        v_y = 2 * v_x + 1

        res = linear_fit_pipeline(v_x, v_y, ax=ax)
        assert res.ax is ax
        assert res.fig is fig

    def test_custom_fit_custom_axes(self):
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        v_x = np.linspace(0, 10, 20)
        v_y = 2 * v_x + 1

        res = custom_fit_pipeline(v_x, v_y, s_f=lambda v: v[1] * v[0] + v[2], axes=axes)
        assert res.axes[0] is axes[0]
        assert res.axes[1] is axes[1]

    def test_save_file_name_creates_image(self, tmp_path: Path):
        img_path = tmp_path / "linear_fit.png"
        v_x = np.linspace(0, 5, 10)
        v_y = 3 * v_x

        linear_fit_pipeline(v_x, v_y, save_file_name=str(img_path))
        assert img_path.exists()
        assert img_path.stat().st_size > 1000

    def test_labels_and_title_customization(self):
        v_x = np.linspace(0, 5, 10)
        v_y = 2 * v_x
        res = linear_fit_pipeline(
            v_x, v_y,
            title="Custom Title",
            xlabel="Voltage (V)",
            ylabel="Current (A)",
        )
        assert res.ax.get_title() == "Custom Title"
        assert res.ax.get_xlabel() == "Voltage (V)"
        assert res.ax.get_ylabel() == "Current (A)"


# =============================================================================
# Suite 6: Excel Export Validation
# =============================================================================

class TestExcelExportValidation:
    def test_linear_fit_excel_export_structure(self, tmp_path: Path):
        excel_file = tmp_path / "linear_report.xlsx"
        v_x = np.linspace(0.0, 10.0, 25)
        v_y = 2.5 * v_x + 1.2 + RNG.normal(0, 0.2, 25)

        res = linear_fit_pipeline(v_x, v_y, excel_path=str(excel_file))
        assert excel_file.exists()

        excel = pd.ExcelFile(excel_file)
        assert "Parameters" in excel.sheet_names
        assert "Metrics" in excel.sheet_names

        df_params = pd.read_excel(excel_file, sheet_name="Parameters")
        assert "Parameter" in df_params.columns
        assert "Value" in df_params.columns
        assert "Std_Error" in df_params.columns
        assert len(df_params) == 2  # intercept, slope
        assert np.isclose(df_params.loc[0, "Value"], res.fit_result.v_params[0])
        assert np.isclose(df_params.loc[1, "Value"], res.fit_result.v_params[1])

        df_metrics = pd.read_excel(excel_file, sheet_name="Metrics")
        assert "Metric" in df_metrics.columns
        assert "Value" in df_metrics.columns
        assert "r2" in df_metrics["Metric"].values

    def test_custom_fit_excel_export_structure(self, tmp_path: Path):
        excel_file = tmp_path / "custom_report.xlsx"

        def f_lin(v_x: np.ndarray) -> float:
            return float(v_x[1] * v_x[0] + v_x[2])

        def f_exp(v_x: np.ndarray) -> float:
            return float(v_x[1] * np.exp(v_x[2] * v_x[0]))

        v_x = np.linspace(0.0, 2.0, 20)
        v_y = 1.5 * np.exp(0.5 * v_x)

        d_models = {"Linear": f_lin, "Exponential": f_exp}
        custom_fit_pipeline(v_x, v_y, d_s_f=d_models, excel_path=str(excel_file))

        assert excel_file.exists()
        excel = pd.ExcelFile(excel_file)
        assert "Model_Comparison" in excel.sheet_names
        assert "Parameters" in excel.sheet_names

        df_comp = pd.read_excel(excel_file, sheet_name="Model_Comparison")
        assert "Model" in df_comp.columns
        assert "r2" in df_comp.columns
        assert len(df_comp) == 2

    def test_excel_export_creates_parent_dir(self, tmp_path: Path):
        deep_file = tmp_path / "deep" / "nested" / "folder" / "out.xlsx"
        v_x = np.linspace(0, 5, 10)
        v_y = 2 * v_x
        linear_fit_pipeline(v_x, v_y, excel_path=str(deep_file))
        assert deep_file.exists()
