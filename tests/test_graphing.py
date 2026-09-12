"""
Unit tests for lab_math_tools.graphing module.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Use non-interactive backend for headless testing
import matplotlib.pyplot as plt
import numpy as np
import pytest

from lab_math_tools.graphing import (
    _INSET_PRESETS,
    _color_gradient,
    _resolve_colors,
    add_zoom_inset,
    plot_measurements,
    turn_list_of_vectors_to_matrix,
)


@pytest.fixture(autouse=True)
def close_figures():
    """Ensure all Matplotlib figures are closed after each test to prevent memory leaks."""
    yield
    plt.close("all")


def test_color_gradient_n_values():
    """
    Test _color_gradient across edge cases:
    - n = 0: returns empty list.
    - n = 1: returns single color hex without ZeroDivisionError.
    - n = 2: returns start and end colors.
    - n > 2: returns smoothly interpolated list of n colors.
    """
    assert _color_gradient(0, "blue", "green") == []
    
    # Critical bug verification: n=1 should not divide by (n-1)==0
    grad_1 = _color_gradient(1, "blue", "green")
    assert len(grad_1) == 1
    assert grad_1[0].startswith("#")

    grad_2 = _color_gradient(2, "black", "white")
    assert len(grad_2) == 2
    assert grad_2[0] == "#000000"
    assert grad_2[1] == "#ffffff"

    grad_5 = _color_gradient(5, "red", "blue")
    assert len(grad_5) == 5
    assert grad_5[0] == "#ff0000"
    assert grad_5[-1] == "#0000ff"


def test_plot_measurements_single_curve_default_gradient():
    """
    Test plot_measurements with a single 1D curve using the default gradient color method.
    Previously, this raised ZeroDivisionError due to (n - 1) division in _color_gradient.
    """
    v_x = np.array([1.0, 2.0, 3.0, 4.0])
    v_y = np.array([2.0, 4.0, 6.0, 8.0])
    
    # Should execute smoothly without error
    plot_measurements(v_x, v_y)


def test_plot_measurements_shared_x_multiple_y_ragged():
    """
    Test plot_measurements when a single shared x-vector is used for multiple y-series
    with differing lengths (ragged data).
    Verifies that extra x-points are trimmed with a UserWarning rather than crashing.
    """
    v_x = np.array([1.0, 2.0, 3.0, 4.0])
    v_y1 = np.array([10.0, 20.0, 30.0, 40.0])  # length 4
    v_y2 = np.array([15.0, 25.0, 35.0])        # length 3 (needs trimming from x)
    v_y3 = np.array([5.0, 10.0])              # length 2 (needs trimming from x)

    with pytest.warns(UserWarning, match="x-values for measurement .* were trimmed"):
        plot_measurements(v_x, [v_y1, v_y2, v_y3], line_labels=["Y1", "Y2", "Y3"])


def test_plot_measurements_list_of_x_and_y():
    """
    Test plot_measurements with independent x and y lists of vectors,
    where each measurement has its own distinct x coordinates.
    """
    lv_x = [np.array([0.0, 1.0, 2.0]), np.array([0.5, 1.5, 2.5, 3.5])]
    lv_y = [np.array([1.0, 3.0, 5.0]), np.array([2.0, 4.0, 6.0, 8.0])]

    plot_measurements(lv_x=lv_x, lv_y=lv_y, title="Independent X & Y Series")


def test_plot_measurements_matrix_inputs():
    """
    Test plot_measurements with 2D numpy matrix inputs.
    Verifies support for both primary lv_x/lv_y arguments and backward-compatible m_x/m_y arguments.
    """
    m_x = np.array([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]])
    m_y = np.array([[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]])

    # Primary parameters
    plot_measurements(m_x, m_y)

    # Keyword aliases
    plot_measurements(m_x=m_x, m_y=m_y)

    # 1-row m_x shared across multi-row m_y
    m_x_single = np.array([[1.0, 2.0, 3.0]])
    plot_measurements(m_x=m_x_single, m_y=m_y)


def test_plot_measurements_raw_python_lists():
    """
    Test plot_measurements with standard Python lists of numbers (e.g., [1, 2, 3]).
    Verifies that passing raw numerical lists (l_) does not trigger AttributeError or TypeError.
    """
    l_x = [1, 2, 3, 4]
    l_y = [10, 20, 30, 40]

    plot_measurements(l_x, l_y)


def test_plot_measurements_color_methods():
    """
    Test plot_measurements across various color methods:
    1. 'gradient' with 2 colors.
    2. 'specific' with an explicit list of matching length.
    3. 'colormap' sampling from a named Matplotlib colormap (e.g. 'viridis', 'plasma').
    """
    lv_x = np.array([1, 2, 3])
    lv_y = [np.array([1, 2, 3]), np.array([2, 4, 6]), np.array([3, 6, 9])]

    # Gradient method
    plot_measurements(lv_x, lv_y, colors=["red", "yellow"], color_method="gradient")

    # Specific method
    plot_measurements(lv_x, lv_y, colors=["red", "green", "blue"], color_method="specific")

    # Colormap method
    plot_measurements(lv_x, lv_y, colors="viridis", color_method="colormap")

    # Colormap string passed directly with default method
    plot_measurements(lv_x, lv_y, colors="plasma")


def test_plot_measurements_linestyles_markers_labels():
    """
    Test customization of linestyles, markers, labels, grid, and title.
    Verifies both scalar broadcasting (e.g., linestyles='--') and list matching.
    """
    lv_x = np.array([0, 1, 2])
    lv_y = [np.array([1, 2, 3]), np.array([4, 5, 6])]

    # Broadcast single scalar string
    plot_measurements(
        lv_x,
        lv_y,
        linestyles="--",
        markers="o",
        line_labels=["Series A", "Series B"],
        title="Styled Plot",
        grid=True,
    )

    # Per-curve list of styles
    plot_measurements(
        lv_x,
        lv_y,
        linestyles=["-", ":"],
        markers=[".", "s"],
        line_labels=["A", "B"],
    )


def test_plot_measurements_save_file(tmp_path: Path):
    """
    Test that plot_measurements saves the figure to disk when save_file_name is provided.
    """
    lv_x = np.array([1, 2, 3])
    lv_y = np.array([4, 5, 6])
    out_file = tmp_path / "test_plot.png"

    plot_measurements(lv_x, lv_y, save_file_name=str(out_file))

    assert out_file.exists()
    assert out_file.stat().st_size > 0


def test_plot_measurements_validation_errors():
    """
    Test that invalid inputs raise appropriate descriptive exceptions:
    - Missing x or y data
    - More y-points than x-points
    - Mismatched measurement counts between x and y
    - Mismatched line_labels or styles length
    - Invalid color method or color counts
    """
    # Missing data
    with pytest.raises(ValueError, match="x-data must be provided"):
        plot_measurements(lv_x=None, lv_y=[1, 2])

    with pytest.raises(ValueError, match="y-data must be provided"):
        plot_measurements(lv_x=[1, 2], lv_y=None)

    # More y-values than x-values
    with pytest.raises(ValueError, match="Not all y-values have x-values"):
        plot_measurements(lv_x=[1, 2], lv_y=[1, 2, 3, 4])

    # Mismatched measurement series count
    with pytest.raises(ValueError, match="Number of x measurements"):
        plot_measurements(
            lv_x=[[1, 2], [1, 2]],
            lv_y=[[1, 2], [1, 2], [1, 2]],
        )

    # Mismatched line_labels count
    with pytest.raises(ValueError, match="line_labels length"):
        plot_measurements(
            lv_x=[1, 2],
            lv_y=[[1, 2], [3, 4]],
            line_labels=["Only One"],
        )

    # Invalid color_method
    with pytest.raises(ValueError, match="Invalid color method"):
        plot_measurements(
            lv_x=[1, 2],
            lv_y=[1, 2],
            color_method="invalid_method",
        )

    # Gradient requires 2 colors
    with pytest.raises(ValueError, match="Colors must be a list of 2 colors"):
        plot_measurements(
            lv_x=[1, 2],
            lv_y=[[1, 2], [3, 4]],
            colors=["red"],
            color_method="gradient",
        )


def test_turn_list_of_vectors_to_matrix():
    """
    Test turn_list_of_vectors_to_matrix utility:
    - Pads ragged vectors with np.nan to match the max length.
    - Handles list of scalar numbers as a 1 x N matrix.
    - Handles empty list as a 0 x 0 matrix.
    """
    # Ragged vectors
    v_1 = np.array([1.0, 2.0, 3.0])
    v_2 = np.array([4.0, 5.0])
    m_res = turn_list_of_vectors_to_matrix([v_1, v_2])

    assert m_res.shape == (2, 3)
    np.testing.assert_array_equal(m_res[0], [1.0, 2.0, 3.0])
    assert np.isnan(m_res[1, 2])
    np.testing.assert_array_equal(m_res[1, :2], [4.0, 5.0])

    # List of scalar numbers
    m_scalar = turn_list_of_vectors_to_matrix([1, 2, 3, 4])
    assert m_scalar.shape == (1, 4)
    np.testing.assert_array_equal(m_scalar[0], [1.0, 2.0, 3.0, 4.0])

    # Empty list
    m_empty = turn_list_of_vectors_to_matrix([])
    assert m_empty.shape == (0, 0)


def test_plot_measurements_returns_axes():
    """
    Test that plot_measurements returns a valid plt.Axes instance with the plotted line(s).
    """
    lv_x = np.array([1.0, 2.0, 3.0])
    lv_y = np.array([4.0, 5.0, 6.0])

    ax = plot_measurements(lv_x, lv_y)

    assert isinstance(ax, plt.Axes)
    assert len(ax.lines) == 1


def test_plot_measurements_custom_ax():
    """
    Test that plot_measurements plots onto a user-supplied Axes object rather than creating a new one.
    """
    fig, custom_ax = plt.subplots()
    lv_x = np.array([1.0, 2.0])
    lv_y = np.array([3.0, 4.0])

    returned_ax = plot_measurements(lv_x, lv_y, ax=custom_ax, show=False)

    assert returned_ax is custom_ax
    assert len(custom_ax.lines) == 1


def test_plot_measurements_show_false(monkeypatch):
    """
    Test that show=False suppresses calling plt.show(), while show=True invokes it.
    """
    show_called = []
    monkeypatch.setattr(plt, "show", lambda: show_called.append(True))

    lv_x = np.array([1.0, 2.0])
    lv_y = np.array([3.0, 4.0])

    # With show=False, plt.show should not be called
    ax = plot_measurements(lv_x, lv_y, show=False)
    assert isinstance(ax, plt.Axes)
    assert len(show_called) == 0

    # With show=True (default), plt.show should be called
    plot_measurements(lv_x, lv_y, show=True)
    assert len(show_called) == 1


def test_plot_measurements_composition():
    """
    Test that multiple plot_measurements calls can share the same ax,
    enabling composite multi-source plots.
    """
    lv_x1 = np.array([1.0, 2.0, 3.0])
    lv_y1 = np.array([10.0, 20.0, 30.0])
    lv_x2 = np.array([1.0, 2.0, 3.0])
    lv_y2 = np.array([5.0, 15.0, 25.0])

    ax = plot_measurements(lv_x1, lv_y1, line_labels=["Series 1"], show=False)
    plot_measurements(lv_x2, lv_y2, line_labels=["Series 2"], ax=ax, show=False)

    assert len(ax.lines) == 2


# ===========================================================================
# Tests for add_zoom_inset
# ===========================================================================

def test_add_zoom_inset_returns_axes():
    """
    Test that add_zoom_inset returns a Matplotlib Axes instance corresponding to the inset.
    """
    v_x = np.linspace(0, 10, 100)
    v_y = np.sin(v_x)
    ax = plot_measurements(v_x, v_y, show=False)

    ax_inset = add_zoom_inset(ax, x_range=(2.0, 4.0), y_range=(0.0, 1.0), show=False)

    assert isinstance(ax_inset, plt.Axes)
    assert ax_inset is not ax
    assert ax_inset in ax.child_axes


def test_add_zoom_inset_limits():
    """
    Test that the returned inset Axes has its xlim and ylim set to the specified ranges.
    """
    v_x = np.linspace(0, 10, 100)
    v_y = np.sin(v_x)
    ax = plot_measurements(v_x, v_y, show=False)

    x_range = (3.0, 6.0)
    y_range = (-0.5, 0.5)
    ax_inset = add_zoom_inset(ax, x_range=x_range, y_range=y_range, show=False)

    assert ax_inset.get_xlim() == x_range
    assert ax_inset.get_ylim() == y_range


def test_add_zoom_inset_line_count():
    """
    Test that all lines plotted on the parent Axes are copied to the inset Axes with identical data.
    """
    v_x = np.array([1.0, 2.0, 3.0])
    v_y1 = np.array([10.0, 20.0, 30.0])
    v_y2 = np.array([1.0, 4.0, 9.0])
    ax = plot_measurements(v_x, [v_y1, v_y2], show=False)

    assert len(ax.lines) == 2

    ax_inset = add_zoom_inset(ax, x_range=(1.5, 2.5), y_range=(2.0, 22.0), show=False)

    assert len(ax_inset.lines) == 2
    np.testing.assert_array_equal(ax_inset.lines[0].get_xdata(), v_x)
    np.testing.assert_array_equal(ax_inset.lines[0].get_ydata(), v_y1)
    np.testing.assert_array_equal(ax_inset.lines[1].get_xdata(), v_x)
    np.testing.assert_array_equal(ax_inset.lines[1].get_ydata(), v_y2)


def test_add_zoom_inset_string_presets():
    """
    Test that all 9 named position presets work correctly without error.
    """
    v_x = np.linspace(0, 5, 20)
    v_y = v_x**2

    for preset_name in _INSET_PRESETS:
        ax = plot_measurements(v_x, v_y, show=False)
        ax_inset = add_zoom_inset(
            ax,
            x_range=(1.0, 2.0),
            y_range=(1.0, 4.0),
            inset_bounds=preset_name,
            show=False,
        )
        assert isinstance(ax_inset, plt.Axes)


def test_add_zoom_inset_custom_bounds_tuple():
    """
    Test that user-defined 4-tuple bounds (x0, y0, width, height) are supported.
    """
    v_x = np.linspace(0, 5, 20)
    v_y = v_x**2
    ax = plot_measurements(v_x, v_y, show=False)

    custom_bounds = (0.2, 0.2, 0.35, 0.35)
    ax_inset = add_zoom_inset(
        ax,
        x_range=(1.0, 2.0),
        y_range=(1.0, 4.0),
        inset_bounds=custom_bounds,
        show=False,
    )
    assert isinstance(ax_inset, plt.Axes)


def test_add_zoom_inset_indicator_settings():
    """
    Test that custom indicator_settings dictionary merges with defaults and applies styles.
    """
    v_x = np.linspace(0, 5, 20)
    v_y = v_x**2
    ax = plot_measurements(v_x, v_y, show=False)

    # Partial customization (only changing edgecolor and linestyle)
    settings = {"edgecolor": "red", "linestyle": "--"}
    ax_inset = add_zoom_inset(
        ax,
        x_range=(1.0, 2.0),
        y_range=(1.0, 4.0),
        indicator_settings=settings,
        show=False,
    )
    assert isinstance(ax_inset, plt.Axes)


def test_add_zoom_inset_zoom_labels():
    """
    Test that zoom_labels toggles tick labels on the inset axes.
    """
    v_x = np.linspace(0, 5, 20)
    v_y = v_x**2

    # zoom_labels=False (default)
    ax1 = plot_measurements(v_x, v_y, show=False)
    ax_inset_no_labels = add_zoom_inset(
        ax1, x_range=(1.0, 2.0), y_range=(1.0, 4.0), zoom_labels=False, show=False
    )
    assert not ax_inset_no_labels.yaxis.majorTicks[0].label1.get_visible() or not ax_inset_no_labels.yaxis.get_tick_params()["labelleft"]

    # zoom_labels=True
    ax2 = plot_measurements(v_x, v_y, show=False)
    ax_inset_labels = add_zoom_inset(
        ax2, x_range=(1.0, 2.0), y_range=(1.0, 4.0), zoom_labels=True, show=False
    )
    assert isinstance(ax_inset_labels, plt.Axes)


def test_add_zoom_inset_validation_errors():
    """
    Test validation checks: invalid x_range, y_range, preset names, tuple bounds, and indicator settings.
    """
    v_x = np.linspace(0, 5, 20)
    v_y = v_x**2
    ax = plot_measurements(v_x, v_y, show=False)

    # x_min >= x_max
    with pytest.raises(ValueError, match="x_range"):
        add_zoom_inset(ax, x_range=(4.0, 2.0), y_range=(1.0, 4.0), show=False)

    # y_min >= y_max
    with pytest.raises(ValueError, match="y_range"):
        add_zoom_inset(ax, x_range=(1.0, 2.0), y_range=(5.0, 3.0), show=False)

    # Unknown preset string
    with pytest.raises(ValueError, match="Unknown inset preset"):
        add_zoom_inset(ax, x_range=(1.0, 2.0), y_range=(1.0, 4.0), inset_bounds="top_somewhere", show=False)

    # Tuple bounds not length 4
    with pytest.raises(ValueError, match="inset_bounds tuple must have 4 elements"):
        add_zoom_inset(ax, x_range=(1.0, 2.0), y_range=(1.0, 4.0), inset_bounds=(0.1, 0.2), show=False)

    # Invalid inset_bounds type
    with pytest.raises(TypeError, match="inset_bounds must be a 4-tuple"):
        add_zoom_inset(ax, x_range=(1.0, 2.0), y_range=(1.0, 4.0), inset_bounds=123, show=False)

    # Unknown indicator setting key
    with pytest.raises(ValueError, match="Unknown indicator_settings key"):
        add_zoom_inset(
            ax,
            x_range=(1.0, 2.0),
            y_range=(1.0, 4.0),
            indicator_settings={"invalid_key": 42},
            show=False,
        )

    # Non-dict indicator settings
    with pytest.raises(TypeError, match="indicator_settings must be a dictionary"):
        add_zoom_inset(
            ax,
            x_range=(1.0, 2.0),
            y_range=(1.0, 4.0),
            indicator_settings="bad_type",
            show=False,
        )


def test_add_zoom_inset_show_behavior(monkeypatch):
    """
    Test that show=False suppresses plt.show(), while show=True invokes it.
    """
    show_called = []
    monkeypatch.setattr(plt, "show", lambda: show_called.append(True))

    v_x = np.linspace(0, 5, 20)
    v_y = v_x**2
    ax = plot_measurements(v_x, v_y, show=False)

    add_zoom_inset(ax, x_range=(1.0, 2.0), y_range=(1.0, 4.0), show=False)
    assert len(show_called) == 0

    add_zoom_inset(ax, x_range=(1.0, 2.0), y_range=(1.0, 4.0), show=True)
    assert len(show_called) == 1

