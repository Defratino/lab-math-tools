import lab_math_tools as lmt


def test_package_exports():
    expected_exports = [
        "derivative_saap",
        "partial_derivative_saap",
        "av_gradient_saap",
        "divergence_saap",
        "am_jacobian_saap",
        "integral_trapezoidal",
        "integral_over_shape",
        "propagate_uncertainty_saap",
        "propagate_covariance_saap",
        "error_contribution_saap",
        "relative_uncertainty_saap",
    ]

    for export_name in expected_exports:
        assert hasattr(lmt, export_name), f"Package missing export: {export_name}"
        assert callable(getattr(lmt, export_name)), f"Export {export_name} should be callable"

    assert hasattr(lmt, "__version__")
    assert hasattr(lmt, "__all__")
    assert set(expected_exports).issubset(set(lmt.__all__))
