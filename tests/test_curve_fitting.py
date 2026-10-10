"""
Tests for lab_math_tools.curve_fitting

Written BEFORE the implementation (TDD). Tests are derived from the
mathematical specification and API contract, not from any existing code.

Conventions used in tests follow conventions.cd:
  - v_ prefix for 1-D numpy array arguments
  - s_f naming for scalar-valued function arguments in fit_custom
"""

from __future__ import annotations

import numpy as np
import pytest

from lab_math_tools.curve_fitting import (
    FitResult,
    fit_custom,
    fit_linear,
    fit_metrics,
)

RNG = np.random.default_rng(42)
RTOL = 1e-2        # 1 % relative tolerance for parameter recovery on clean data
ATOL_NOISY = 0.1   # absolute tolerance for recovery on lightly-noisy data


# =============================================================================
# FitResult structure and helper methods
# =============================================================================

class TestFitResult:
    def _sample_result(self) -> FitResult:
        v_x = np.linspace(0.0, 5.0, 20)
        v_y = 2.0 * v_x + 1.0
        return fit_linear(v_x, v_y)

    def test_has_v_params(self):
        assert hasattr(self._sample_result(), "v_params")

    def test_has_m_covariance(self):
        assert hasattr(self._sample_result(), "m_covariance")

    def test_has_v_residuals(self):
        assert hasattr(self._sample_result(), "v_residuals")

    def test_has_metrics(self):
        assert hasattr(self._sample_result(), "metrics")

    def test_has_model_name(self):
        assert hasattr(self._sample_result(), "model_name")

    def test_v_params_is_ndarray(self):
        assert isinstance(self._sample_result().v_params, np.ndarray)

    def test_m_covariance_is_ndarray(self):
        assert isinstance(self._sample_result().m_covariance, np.ndarray)

    def test_v_residuals_is_ndarray(self):
        assert isinstance(self._sample_result().v_residuals, np.ndarray)

    def test_metrics_is_dict(self):
        assert isinstance(self._sample_result().metrics, dict)

    def test_model_name_is_str(self):
        assert isinstance(self._sample_result().model_name, str)

    def test_summary_returns_nonempty_string(self):
        s = self._sample_result().summary()
        assert isinstance(s, str) and len(s) > 0

    def test_summary_contains_model_name(self):
        result = self._sample_result()
        assert result.model_name in result.summary()

    def test_summary_contains_metric_keys(self):
        summary = self._sample_result().summary()
        assert "r2" in summary and "rmse" in summary

    def test_plot_residuals_returns_axes(self):
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.axes
        ax = self._sample_result().plot_residuals()
        assert isinstance(ax, matplotlib.axes.Axes)

    def test_plot_residuals_accepts_existing_ax(self):
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax_in = plt.subplots()
        ax_out = self._sample_result().plot_residuals(ax=ax_in)
        assert ax_out is ax_in
        plt.close("all")

    def test_plot_residuals_none_creates_new_figure(self):
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt, matplotlib.axes
        ax = self._sample_result().plot_residuals(ax=None)
        assert isinstance(ax, matplotlib.axes.Axes)
        plt.close("all")


# =============================================================================
# fit_metrics
# =============================================================================

class TestFitMetrics:
    def test_perfect_fit_mse_zero(self):
        v_y = np.array([1.0, 2.0, 3.0])
        assert fit_metrics(v_y, v_y.copy(), n_params=1)["mse"] == pytest.approx(0.0, abs=1e-12)

    def test_perfect_fit_rmse_zero(self):
        v_y = np.array([1.0, 2.0, 3.0])
        assert fit_metrics(v_y, v_y.copy(), n_params=1)["rmse"] == pytest.approx(0.0, abs=1e-12)

    def test_perfect_fit_mae_zero(self):
        v_y = np.array([1.0, 2.0, 3.0])
        assert fit_metrics(v_y, v_y.copy(), n_params=1)["mae"] == pytest.approx(0.0, abs=1e-12)

    def test_perfect_fit_r2_one(self):
        v_y = np.array([1.0, 2.0, 3.0, 4.0])
        assert fit_metrics(v_y, v_y.copy(), n_params=1)["r2"] == pytest.approx(1.0, abs=1e-12)

    def test_known_mse_value(self):
        v_y, v_y_pred = np.zeros(3), np.array([1.0, 2.0, 3.0])
        assert fit_metrics(v_y, v_y_pred, n_params=1)["mse"] == pytest.approx(14.0 / 3.0, rel=1e-6)

    def test_known_rmse_value(self):
        v_y, v_y_pred = np.zeros(3), np.array([1.0, 2.0, 3.0])
        assert fit_metrics(v_y, v_y_pred, n_params=1)["rmse"] == pytest.approx(np.sqrt(14.0 / 3.0), rel=1e-6)

    def test_known_mae_value(self):
        v_y, v_y_pred = np.zeros(3), np.array([1.0, 2.0, 3.0])
        assert fit_metrics(v_y, v_y_pred, n_params=1)["mae"] == pytest.approx(2.0, rel=1e-6)

    def test_r2_adj_present_when_n_params_positive(self):
        v_y = np.linspace(1.0, 5.0, 10)
        assert "r2_adj" in fit_metrics(v_y, v_y + 0.01, n_params=2)

    def test_r2_adj_absent_when_n_params_zero(self):
        v_y = np.linspace(1.0, 5.0, 10)
        assert "r2_adj" not in fit_metrics(v_y, v_y + 0.01, n_params=0)

    def test_required_keys_always_present(self):
        m = fit_metrics(np.array([1.0, 2.0, 3.0]), np.array([1.1, 2.1, 3.1]), n_params=0)
        for key in ("mse", "rmse", "mae", "r2", "std_residuals"):
            assert key in m

    def test_std_residuals_known_value(self):
        v_y = np.array([0.0, 2.0, 0.0, 2.0])
        v_y_pred = np.array([1.0, 1.0, 1.0, 1.0])
        assert fit_metrics(v_y, v_y_pred, n_params=0)["std_residuals"] == pytest.approx(1.0, rel=1e-6)


# =============================================================================
# fit_linear
# =============================================================================

class TestFitLinear:
    def test_perfect_line_intercept(self):
        v_x = np.linspace(0.0, 10.0, 50)
        assert fit_linear(v_x, 3.0 * v_x + 5.0).v_params[0] == pytest.approx(5.0, rel=RTOL)

    def test_perfect_line_slope(self):
        v_x = np.linspace(0.0, 10.0, 50)
        assert fit_linear(v_x, 3.0 * v_x + 5.0).v_params[1] == pytest.approx(3.0, rel=RTOL)

    def test_perfect_line_r2_one(self):
        v_x = np.linspace(0.0, 10.0, 50)
        assert fit_linear(v_x, 2.0 * v_x - 1.0).metrics["r2"] == pytest.approx(1.0, abs=1e-10)

    def test_noisy_line_intercept_recovery(self):
        v_x = np.linspace(0.0, 10.0, 200)
        v_y = 2.5 * v_x + 1.0 + RNG.normal(0.0, 0.1, 200)
        assert fit_linear(v_x, v_y).v_params[0] == pytest.approx(1.0, abs=ATOL_NOISY)

    def test_noisy_line_slope_recovery(self):
        v_x = np.linspace(0.0, 10.0, 200)
        v_y = 2.5 * v_x + 1.0 + RNG.normal(0.0, 0.1, 200)
        assert fit_linear(v_x, v_y).v_params[1] == pytest.approx(2.5, abs=0.05)

    def test_negative_slope(self):
        v_x = np.linspace(0.0, 5.0, 50)
        assert fit_linear(v_x, -4.0 * v_x + 10.0).v_params[1] == pytest.approx(-4.0, rel=RTOL)

    def test_model_name_is_linear(self):
        assert fit_linear(np.array([1.0, 2.0, 3.0]), np.array([2.0, 4.0, 6.0])).model_name == "linear"

    def test_all_metric_keys_present(self):
        v_x = np.linspace(0.0, 5.0, 30)
        result = fit_linear(v_x, v_x + RNG.normal(0.0, 0.05, 30))
        for key in ("mse", "rmse", "mae", "r2", "r2_adj", "std_residuals"):
            assert key in result.metrics

    def test_covariance_shape_is_2x2(self):
        v_x = np.linspace(0.0, 5.0, 20)
        assert fit_linear(v_x, 2.0 * v_x + 1.0).m_covariance.shape == (2, 2)

    def test_v_params_length_is_2(self):
        v_x = np.linspace(0.0, 5.0, 20)
        assert len(fit_linear(v_x, 2.0 * v_x + 1.0).v_params) == 2

    def test_residuals_shape_matches_data(self):
        v_x = np.linspace(0.0, 5.0, 25)
        assert fit_linear(v_x, 2.0 * v_x + RNG.normal(0.0, 0.05, 25)).v_residuals.shape == (25,)

    def test_mismatched_xy_length_raises(self):
        with pytest.raises(ValueError, match="same length"):
            fit_linear(np.array([1.0, 2.0]), np.array([1.0, 2.0, 3.0]))

    def test_2d_v_x_raises(self):
        with pytest.raises(ValueError):
            fit_linear(np.ones((3, 2)), np.ones(3))

    def test_2d_v_y_raises(self):
        with pytest.raises(ValueError):
            fit_linear(np.ones(3), np.ones((3, 2)))



# =============================================================================
# fit_custom
# =============================================================================

class TestFitCustom:
    """
    Tests for fit_custom following the project naming conventions.

    Function signature convention:
        s_f(v_x) -> float  (R^n -> R)
    where v_x[0] is the independent variable x data point, and v_x[1:]
    are the free parameters fitted to the data. If v_p0 is not provided,
    it defaults to zeros.
    """

    def test_simple_linear_with_p0(self):
        def s_f(v2_x: np.ndarray) -> float:
            return v2_x[1] * v2_x[0]

        v_x = np.linspace(0.0, 5.0, 100)
        result = fit_custom(v_x, 3.0 * v_x, s_f, v_p0=np.array([1.0]))
        assert result.v_params[0] == pytest.approx(3.0, rel=RTOL)

    def test_default_v_p0_zeros_named_dim(self):
        def s_f(v3_x: np.ndarray) -> float:
            return v3_x[1] * v3_x[0] + v3_x[2]

        v_x = np.linspace(0.0, 5.0, 100)
        v_y = 2.0 * v_x + 3.0
        # v_p0 omitted: defaults to zeros [0.0, 0.0]
        result = fit_custom(v_x, v_y, s_f)
        assert result.v_params[0] == pytest.approx(2.0, rel=RTOL)
        assert result.v_params[1] == pytest.approx(3.0, rel=RTOL)

    def test_default_v_p0_zeros_abstract_vector(self):
        def s_f(v_x: np.ndarray) -> float:
            return v_x[1] * v_x[0] + v_x[2]

        v_x = np.linspace(0.0, 5.0, 100)
        v_y = 2.0 * v_x + 3.0
        result = fit_custom(v_x, v_y, s_f)
        assert result.v_params[0] == pytest.approx(2.0, rel=RTOL)
        assert result.v_params[1] == pytest.approx(3.0, rel=RTOL)

    def test_simple_affine_with_p0(self):
        def s_f(v3_x: np.ndarray) -> float:
            return v3_x[1] * v3_x[0] + v3_x[2]

        v_x = np.linspace(0.0, 5.0, 100)
        v_y = 2.0 * v_x + 3.0
        result = fit_custom(v_x, v_y, s_f, v_p0=np.array([1.0, 0.0]))
        assert result.v_params[0] == pytest.approx(2.0, rel=RTOL)
        assert result.v_params[1] == pytest.approx(3.0, rel=RTOL)

    def test_sine_amplitude_with_p0(self):
        def s_f(v2_x: np.ndarray) -> float:
            return v2_x[1] * np.sin(v2_x[0])

        v_x = np.linspace(0.0, 4.0 * np.pi, 200)
        result = fit_custom(v_x, 2.5 * np.sin(v_x), s_f, v_p0=np.array([1.0]))
        assert result.v_params[0] == pytest.approx(2.5, rel=RTOL)

    def test_lambda_default_v_p0(self):
        v_x = np.linspace(0.0, 5.0, 100)
        result = fit_custom(v_x, 4.0 * v_x, lambda v: v[1] * v[0])
        assert result.v_params[0] == pytest.approx(4.0, rel=RTOL)

    def test_model_name_is_custom(self):
        result = fit_custom(
            np.array([1.0, 2.0, 3.0]),
            np.array([2.0, 4.0, 6.0]),
            lambda v: v[1] * v[0],
        )
        assert result.model_name == "custom"

    def test_metrics_present(self):
        v_x = np.linspace(0.0, 3.0, 50)
        result = fit_custom(v_x, 2.0 * v_x, lambda v: v[1] * v[0])
        for key in ("mse", "rmse", "mae", "r2"):
            assert key in result.metrics

    def test_residuals_shape_matches_data(self):
        v_x = np.linspace(0.0, 3.0, 50)
        result = fit_custom(v_x, 2.0 * v_x, lambda v: v[1] * v[0])
        assert result.v_residuals.shape == (50,)

    def test_mismatched_xy_raises(self):
        with pytest.raises(ValueError, match="same length"):
            fit_custom(
                np.array([1.0, 2.0]),
                np.array([1.0, 2.0, 3.0]),
                lambda v: v[1] * v[0],
            )

    def test_noisy_custom_fit_slope_recovery(self):
        v_x = np.linspace(0.0, 5.0, 200)
        v_y = 3.5 * v_x + RNG.normal(0.0, 0.1, 200)
        result = fit_custom(v_x, v_y, lambda v: v[1] * v[0], v_p0=np.array([1.0]))
        assert result.v_params[0] == pytest.approx(3.5, abs=ATOL_NOISY)

    def test_covariance_shape_matches_params(self):
        v_x = np.linspace(0.0, 5.0, 80)
        result = fit_custom(v_x, 2.0 * v_x + 1.0, lambda v: v[1] * v[0] + v[2])
        assert result.m_covariance.shape == (2, 2)

    def test_keyword_f_alias(self):
        def s_f(v2_x: np.ndarray) -> float:
            return v2_x[1] * v2_x[0]

        v_x = np.linspace(0.0, 5.0, 50)
        result = fit_custom(v_x, 3.0 * v_x, f=s_f, v_p0=np.array([1.0]))
        assert result.v_params[0] == pytest.approx(3.0, rel=RTOL)

    def test_default_v_p0_zeros_tuple_unpacking(self):
        def s_f(v: np.ndarray) -> float:
            x, a, b, c = v
            return a * x**2 + b * x + c

        v_x = np.linspace(-2.0, 2.0, 50)
        v_y = 2.0 * v_x**2 + 1.0 * v_x + 3.0
        result = fit_custom(v_x, v_y, s_f)
        assert result.v_params[0] == pytest.approx(2.0, rel=RTOL)
        assert result.v_params[1] == pytest.approx(1.0, rel=RTOL)
        assert result.v_params[2] == pytest.approx(3.0, rel=RTOL)

    def test_bounds_forwarded(self):
        def s_f(v2_x: np.ndarray) -> float:
            return v2_x[1] * v2_x[0]

        v_x = np.linspace(1.0, 5.0, 50)
        v_y = 5.0 * v_x
        # Cap slope at 3.0
        result = fit_custom(v_x, v_y, s_f, bounds=(-np.inf, 3.0))
        assert result.v_params[0] == pytest.approx(3.0, rel=RTOL)

    def test_missing_function_raises(self):
        with pytest.raises(ValueError, match="s_f must be provided"):
            fit_custom(np.array([1.0, 2.0]), np.array([1.0, 2.0]), None)

    def test_dimension_less_than_2_raises(self):
        def s_f(v1_x: np.ndarray) -> float:
            return v1_x[0]

        with pytest.raises(ValueError, match="at least one free parameter"):
            fit_custom(np.array([1.0, 2.0]), np.array([1.0, 2.0]), s_f)

