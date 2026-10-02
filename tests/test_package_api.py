import lab_math_tools as lmt


def test_package_exports():
    expected_exports = [
        "derivative",
        "partial_derivative",
        "v_gradient",
        "divergence",
        "propagate_uncertainty",
        "relative_uncertainty",
        "plot_measurements",
        "turn_list_of_vectors_to_matrix",
        "add_zoom_inset",
    ]

    for export_name in expected_exports:
        assert hasattr(lmt, export_name), f"Package missing export: {export_name}"
        assert callable(getattr(lmt, export_name)), f"Export {export_name} should be callable"

    assert hasattr(lmt, "__version__")
    assert hasattr(lmt, "__all__")
    assert set(expected_exports).issubset(set(lmt.__all__))
