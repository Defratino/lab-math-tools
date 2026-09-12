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
    fit_polynomial,
    fit_preset,
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
# fit_polynomial
# =============================================================================

class TestFitPolynomial:
    def test_degree_1_matches_fit_linear(self):
        v_x = np.linspace(0.0, 5.0, 60)
        v_y = 2.0 * v_x + 1.5 + RNG.normal(0.0, 0.1, 60)
        np.testing.assert_allclose(fit_linear(v_x, v_y).v_params,
                                   fit_polynomial(v_x, v_y, degree=1).v_params, rtol=1e-6)

    def test_degree_2_recovers_a0(self):
        v_x = np.linspace(-3.0, 3.0, 300)
        assert fit_polynomial(v_x, 2.0*v_x**2 + 3.0*v_x + 1.0, degree=2).v_params[0] == pytest.approx(1.0, abs=1e-5)

    def test_degree_2_recovers_a1(self):
        v_x = np.linspace(-3.0, 3.0, 300)
        assert fit_polynomial(v_x, 2.0*v_x**2 + 3.0*v_x + 1.0, degree=2).v_params[1] == pytest.approx(3.0, abs=1e-5)

    def test_degree_2_recovers_a2(self):
        v_x = np.linspace(-3.0, 3.0, 300)
        assert fit_polynomial(v_x, 2.0*v_x**2 + 3.0*v_x + 1.0, degree=2).v_params[2] == pytest.approx(2.0, abs=1e-5)

    def test_degree_3_recovers_coefficients(self):
        v_x = np.linspace(-2.0, 2.0, 400)
        result = fit_polynomial(v_x, v_x**3 + 2.0*v_x, degree=3)
        np.testing.assert_allclose(result.v_params, [0.0, 2.0, 0.0, 1.0], atol=1e-5)

    def test_params_length_equals_degree_plus_one(self):
        v_x = np.linspace(0.0, 3.0, 50)
        assert len(fit_polynomial(v_x, v_x**4, degree=4).v_params) == 5

    def test_model_name_contains_degree(self):
        v_x = np.linspace(0.0, 3.0, 30)
        assert "3" in fit_polynomial(v_x, v_x**2, degree=3).model_name

    def test_zero_degrees_pins_constant_to_zero(self):
        v_x = np.linspace(1.0, 5.0, 100)
        result = fit_polynomial(v_x, 3.0*v_x, degree=1, zero_degrees=[0])
        assert result.v_params[0] == pytest.approx(0.0, abs=1e-10)

    def test_zero_degrees_recovers_free_coefficient(self):
        v_x = np.linspace(1.0, 5.0, 100)
        result = fit_polynomial(v_x, 3.0*v_x, degree=1, zero_degrees=[0])
        assert result.v_params[1] == pytest.approx(3.0, rel=RTOL)

    def test_zero_degrees_multiple_pins(self):
        v_x = np.linspace(-2.0, 2.0, 400)
        result = fit_polynomial(v_x, v_x**3 + 2.0*v_x, degree=3, zero_degrees=[0, 2])
        assert result.v_params[0] == pytest.approx(0.0, abs=1e-10)
        assert result.v_params[2] == pytest.approx(0.0, abs=1e-10)
        assert result.v_params[1] == pytest.approx(2.0, rel=RTOL)
        assert result.v_params[3] == pytest.approx(1.0, rel=RTOL)

    def test_zero_degrees_out_of_range_raises(self):
        v_x = np.linspace(0.0, 3.0, 30)
        with pytest.raises(ValueError, match="zero_degrees"):
            fit_polynomial(v_x, v_x**2, degree=2, zero_degrees=[5])

    def test_zero_degrees_negative_raises(self):
        v_x = np.linspace(0.0, 3.0, 30)
        with pytest.raises(ValueError, match="zero_degrees"):
            fit_polynomial(v_x, v_x**2, degree=2, zero_degrees=[-1])

    def test_metrics_keys_present(self):
        v_x = np.linspace(0.0, 3.0, 30)
        result = fit_polynomial(v_x, v_x**2 + RNG.normal(0.0, 0.05, 30), degree=2)
        for key in ("mse", "rmse", "mae", "r2", "std_residuals"):
            assert key in result.metrics

    def test_mismatched_xy_raises(self):
        with pytest.raises(ValueError, match="same length"):
            fit_polynomial(np.array([1.0, 2.0]), np.array([1.0, 2.0, 3.0]), degree=1)

    def test_r2_near_one_for_perfect_data(self):
        v_x = np.linspace(-2.0, 2.0, 100)
        result = fit_polynomial(v_x, v_x**3 - 2.0*v_x, degree=3)
        assert result.metrics["r2"] == pytest.approx(1.0, abs=1e-8)


# =============================================================================
# fit_preset
# =============================================================================

def _add_noise(v_y: np.ndarray, scale: float = 0.01) -> np.ndarray:
    return v_y + RNG.normal(0.0, scale, len(v_y))


class TestFitPresetSine:
    def test_recovers_amplitude(self):
        v_x = np.linspace(0.0, 4.0 * np.pi, 300)
        A, omega, phi, C = 2.5, 1.0, 0.3, 1.0
        v_y = _add_noise(A * np.sin(omega * v_x + phi) + C)
        result = fit_preset(v_x, v_y, model="sine", v_p0=np.array([2.5, 1.0, 0.3, 1.0]))
        assert abs(result.v_params[0]) == pytest.approx(A, rel=0.05)

    def test_recovers_offset(self):
        v_x = np.linspace(0.0, 4.0 * np.pi, 300)
        A, omega, phi, C = 2.5, 1.0, 0.3, 1.0
        v_y = _add_noise(A * np.sin(omega * v_x + phi) + C)
        result = fit_preset(v_x, v_y, model="sine", v_p0=np.array([2.5, 1.0, 0.3, 1.0]))
        assert result.v_params[3] == pytest.approx(C, abs=0.05)

    def test_bias_true_has_four_params(self):
        v_x = np.linspace(0.0, 2.0 * np.pi, 100)
        result = fit_preset(v_x, np.sin(v_x), model="sine", v_p0=np.array([1.0, 1.0, 0.0, 0.0]))
        assert len(result.v_params) == 4

    def test_bias_false_has_three_params(self):
        v_x = np.linspace(0.0, 2.0 * np.pi, 100)
        result = fit_preset(v_x, 2.0 * np.sin(v_x), model="sine", bias=False, v_p0=np.array([2.0, 1.0, 0.0]))
        assert len(result.v_params) == 3

    def test_model_name_contains_sine(self):
        v_x = np.linspace(0.0, 2.0 * np.pi, 50)
        assert "sine" in fit_preset(v_x, np.sin(v_x), model="sine", v_p0=np.array([1.0, 1.0, 0.0, 0.0])).model_name

    def test_no_bias_model_name_contains_no_bias(self):
        v_x = np.linspace(0.0, 2.0 * np.pi, 50)
        result = fit_preset(v_x, np.sin(v_x), model="sine", bias=False, v_p0=np.array([1.0, 1.0, 0.0]))
        assert "no_bias" in result.model_name


class TestFitPresetExponential:
    def test_recovers_amplitude(self):
        v_x = np.linspace(0.0, 3.0, 200)
        v_y = _add_noise(2.0 * np.exp(0.5 * v_x) + 1.0)
        result = fit_preset(v_x, v_y, model="exponential", v_p0=np.array([2.0, 0.5, 1.0]))
        assert result.v_params[0] == pytest.approx(2.0, rel=0.05)

    def test_recovers_decay_rate(self):
        v_x = np.linspace(0.0, 3.0, 200)
        v_y = _add_noise(2.0 * np.exp(0.5 * v_x) + 1.0)
        result = fit_preset(v_x, v_y, model="exponential", v_p0=np.array([2.0, 0.5, 1.0]))
        assert result.v_params[1] == pytest.approx(0.5, rel=0.05)

    def test_bias_true_has_three_params(self):
        v_x = np.linspace(0.0, 2.0, 80)
        result = fit_preset(v_x, np.exp(v_x) + 1.0, model="exponential", v_p0=np.array([1.0, 1.0, 1.0]))
        assert len(result.v_params) == 3

    def test_bias_false_has_two_params(self):
        v_x = np.linspace(0.0, 2.0, 80)
        result = fit_preset(v_x, 3.0 * np.exp(0.5 * v_x), model="exponential", bias=False, v_p0=np.array([3.0, 0.5]))
        assert len(result.v_params) == 2


class TestFitPresetLogarithm:
    def test_recovers_amplitude(self):
        v_x = np.linspace(0.5, 5.0, 200)
        v_y = _add_noise(3.0 * np.log(1.0 * v_x) + 0.5)
        result = fit_preset(v_x, v_y, model="logarithm", v_p0=np.array([3.0, 1.0, 0.5]))
        assert result.v_params[0] == pytest.approx(3.0, rel=0.05)

    def test_recovers_offset(self):
        v_x = np.linspace(0.5, 5.0, 200)
        v_y = _add_noise(3.0 * np.log(1.0 * v_x) + 0.5)
        result = fit_preset(v_x, v_y, model="logarithm", v_p0=np.array([3.0, 1.0, 0.5]))
        assert result.v_params[2] == pytest.approx(0.5, abs=0.1)

    def test_bias_false_has_two_params(self):
        v_x = np.linspace(0.5, 5.0, 100)
        v_y = 2.0 * np.log(1.5 * v_x)
        result = fit_preset(v_x, v_y, model="logarithm", bias=False, v_p0=np.array([2.0, 1.5]))
        assert len(result.v_params) == 2


class TestFitPresetPower:
    def test_recovers_amplitude(self):
        v_x = np.linspace(0.1, 5.0, 200)
        v_y = _add_noise(2.0 * v_x**1.5)
        result = fit_preset(v_x, v_y, model="power", v_p0=np.array([2.0, 1.5]))
        assert result.v_params[0] == pytest.approx(2.0, rel=0.05)

    def test_recovers_exponent(self):
        v_x = np.linspace(0.1, 5.0, 200)
        v_y = _add_noise(2.0 * v_x**1.5)
        result = fit_preset(v_x, v_y, model="power", v_p0=np.array([2.0, 1.5]))
        assert result.v_params[1] == pytest.approx(1.5, rel=0.05)

    def test_has_two_params(self):
        v_x = np.linspace(0.1, 5.0, 100)
        result = fit_preset(v_x, 3.0 * v_x**2.0, model="power", v_p0=np.array([3.0, 2.0]))
        assert len(result.v_params) == 2


class TestFitPresetGaussian:
    def test_recovers_amplitude(self):
        v_x = np.linspace(-5.0, 5.0, 300)
        A, mu, sigma = 3.0, 0.5, 1.2
        v_y = _add_noise(A * np.exp(-((v_x - mu)**2) / (2.0 * sigma**2)))
        result = fit_preset(v_x, v_y, model="gaussian", v_p0=np.array([3.0, 0.5, 1.2]))
        assert result.v_params[0] == pytest.approx(A, rel=0.05)

    def test_recovers_mean(self):
        v_x = np.linspace(-5.0, 5.0, 300)
        A, mu, sigma = 3.0, 0.5, 1.2
        v_y = _add_noise(A * np.exp(-((v_x - mu)**2) / (2.0 * sigma**2)))
        result = fit_preset(v_x, v_y, model="gaussian", v_p0=np.array([3.0, 0.5, 1.2]))
        assert result.v_params[1] == pytest.approx(mu, abs=0.05)

    def test_recovers_sigma(self):
        v_x = np.linspace(-5.0, 5.0, 300)
        A, mu, sigma = 3.0, 0.5, 1.2
        v_y = _add_noise(A * np.exp(-((v_x - mu)**2) / (2.0 * sigma**2)))
        result = fit_preset(v_x, v_y, model="gaussian", v_p0=np.array([3.0, 0.5, 1.2]))
        assert abs(result.v_params[2]) == pytest.approx(sigma, rel=0.05)

    def test_has_three_params(self):
        v_x = np.linspace(-3.0, 3.0, 100)
        result = fit_preset(v_x, np.exp(-(v_x**2) / 2.0), model="gaussian", v_p0=np.array([1.0, 0.0, 1.0]))
        assert len(result.v_params) == 3


class TestFitPresetErrors:
    def test_invalid_model_raises_value_error(self):
        with pytest.raises(ValueError, match="Unknown model"):
            fit_preset(np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 3.0]), model="not_a_model")

    def test_mismatched_xy_raises(self):
        with pytest.raises(ValueError, match="same length"):
            fit_preset(np.array([1.0, 2.0]), np.array([1.0, 2.0, 3.0]), model="sine")


class TestFitPresetMetrics:
    def test_all_required_metric_keys_present(self):
        v_x = np.linspace(0.0, 2.0 * np.pi, 100)
        result = fit_preset(v_x, np.sin(v_x), model="sine", v_p0=np.array([1.0, 1.0, 0.0, 0.0]))
        for key in ("mse", "rmse", "mae", "r2"):
            assert key in result.metrics

    def test_residuals_shape_matches_data(self):
        v_x = np.linspace(0.0, 2.0 * np.pi, 80)
        result = fit_preset(v_x, np.sin(v_x), model="sine", v_p0=np.array([1.0, 1.0, 0.0, 0.0]))
        assert result.v_residuals.shape == (80,)


# =============================================================================
# fit_custom
# =============================================================================

class TestFitCustom:
    """
    Custom user-defined functions following the project naming conventions.

    Vector layout inside f per sample:
        v_x[0]             -> x data value
        v_x[1..1+n_fixed]  -> v_fixed_params
        v_x[1+n_fixed..]   -> free parameters (fitted)
    """

    def test_simple_linear_no_fixed(self):
        def s_f(v2_x: np.ndarray) -> float:
            return v2_x[1] * v2_x[0]

        v_x = np.linspace(0.0, 5.0, 100)
        result = fit_custom(v_x, 3.0 * v_x, s_f, v_p0=np.array([1.0]))
        assert result.v_params[0] == pytest.approx(3.0, rel=RTOL)

    def test_simple_affine_no_fixed(self):
        def s_f(v3_x: np.ndarray) -> float:
            return v3_x[1] * v3_x[0] + v3_x[2]

        v_x = np.linspace(0.0, 5.0, 100)
        v_y = 2.0 * v_x + 3.0
        result = fit_custom(v_x, v_y, s_f, v_p0=np.array([1.0, 0.0]))
        assert result.v_params[0] == pytest.approx(2.0, rel=RTOL)
        assert result.v_params[1] == pytest.approx(3.0, rel=RTOL)

    def test_sine_amplitude_no_fixed(self):
        def s_f(v2_x: np.ndarray) -> float:
            return v2_x[1] * np.sin(v2_x[0])

        v_x = np.linspace(0.0, 4.0 * np.pi, 200)
        result = fit_custom(v_x, 2.5 * np.sin(v_x), s_f, v_p0=np.array([1.0]))
        assert result.v_params[0] == pytest.approx(2.5, rel=RTOL)

    def test_fixed_params_populate_first_slots(self):
        """slope * x + fixed[0] + fixed[1] = slope * x + 3.0"""
        def s_f(v4_x: np.ndarray) -> float:
            return v4_x[3] * v4_x[0] + v4_x[1] + v4_x[2]

        v_x = np.linspace(0.0, 5.0, 100)
        v_y = 4.0 * v_x + 3.0
        result = fit_custom(v_x, v_y, s_f, v_p0=np.array([1.0]),
                            v_fixed_params=np.array([1.0, 2.0]))
        assert result.v_params[0] == pytest.approx(4.0, rel=RTOL)

    def test_fixed_params_do_not_appear_in_v_params(self):
        def s_f(v4_x: np.ndarray) -> float:
            return v4_x[3] * v4_x[0] + v4_x[1] + v4_x[2]

        v_x = np.linspace(0.0, 5.0, 100)
        v_y = 4.0 * v_x + 3.0
        result = fit_custom(v_x, v_y, s_f, v_p0=np.array([1.0]),
                            v_fixed_params=np.array([1.0, 2.0]))
        assert len(result.v_params) == 1

    def test_model_name_is_custom(self):
        def s_f(v2_x: np.ndarray) -> float:
            return v2_x[1] * v2_x[0]

        result = fit_custom(np.array([1.0, 2.0, 3.0]), np.array([2.0, 4.0, 6.0]),
                            s_f, v_p0=np.array([1.0]))
        assert result.model_name == "custom"

    def test_metrics_present(self):
        def s_f(v2_x: np.ndarray) -> float:
            return v2_x[1] * v2_x[0]

        v_x = np.linspace(0.0, 3.0, 50)
        result = fit_custom(v_x, 2.0 * v_x, s_f, v_p0=np.array([1.0]))
        for key in ("mse", "rmse", "mae", "r2"):
            assert key in result.metrics

    def test_residuals_shape_matches_data(self):
        def s_f(v2_x: np.ndarray) -> float:
            return v2_x[1] * v2_x[0]

        v_x = np.linspace(0.0, 3.0, 50)
        result = fit_custom(v_x, 2.0 * v_x, s_f, v_p0=np.array([1.0]))
        assert result.v_residuals.shape == (50,)

    def test_mismatched_xy_raises(self):
        def s_f(v2_x: np.ndarray) -> float:
            return v2_x[1] * v2_x[0]

        with pytest.raises(ValueError, match="same length"):
            fit_custom(np.array([1.0, 2.0]), np.array([1.0, 2.0, 3.0]),
                       s_f, v_p0=np.array([1.0]))

    def test_noisy_custom_fit_slope_recovery(self):
        def s_f(v2_x: np.ndarray) -> float:
            return v2_x[1] * v2_x[0]

        v_x = np.linspace(0.0, 5.0, 200)
        v_y = 3.5 * v_x + RNG.normal(0.0, 0.1, 200)
        result = fit_custom(v_x, v_y, s_f, v_p0=np.array([1.0]))
        assert result.v_params[0] == pytest.approx(3.5, abs=ATOL_NOISY)

    def test_covariance_shape_matches_free_params(self):
        def s_f(v3_x: np.ndarray) -> float:
            return v3_x[1] * v3_x[0] + v3_x[2]

        v_x = np.linspace(0.0, 5.0, 80)
        result = fit_custom(v_x, 2.0 * v_x + 1.0, s_f, v_p0=np.array([1.0, 0.0]))
        assert result.m_covariance.shape == (2, 2)
