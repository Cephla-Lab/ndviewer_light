"""``tests/_ndv_private.py``: "not created yet" and "renamed" are different outcomes.

ndv 0.5 creates ``_lut_controllers[key]`` and its first image handle inside
``ArrayViewer._on_data_response_ready``, a main-thread callback queued after the
async fetch; ``_is_idle()`` is already True once the worker's future is done. So a
window smoke that polls right after idle legitimately looks before they exist.
That must read as "poll again" (``displayed_texture`` -> None, ``texture_mean_is``
-> False), never as a skip blaming an ndv rename; only a missing *attribute* (the
path itself is gone) skips, and a dims slider ndv never built is a failure.
Stand-in viewer objects; no window.
"""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from types import SimpleNamespace  # noqa: E402

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from tests._ndv_private import (  # noqa: E402
    dims_slider,
    displayed_texture,
    is_idle,
    texture_mean_is,
)


def viewer_with(controllers: dict) -> SimpleNamespace:
    """An ``ArrayViewer`` stand-in that is idle and holds ``controllers``."""
    return SimpleNamespace(_lut_controllers=controllers, _is_idle=lambda: True)


def controller_with(*pixel_arrays: np.ndarray) -> SimpleNamespace:
    """A ``ChannelController`` stand-in with one image handle per array."""
    return SimpleNamespace(
        handles=[SimpleNamespace(data=lambda a=a: a) for a in pixel_arrays]
    )


def test_displayed_texture_is_none_before_the_first_data_response():
    assert displayed_texture(viewer_with({}), 0) is None


def test_displayed_texture_is_none_while_the_controller_has_no_handle():
    assert displayed_texture(viewer_with({0: controller_with()}), 0) is None


def test_displayed_texture_reads_the_first_handle():
    pixels = np.full((8, 8), 7, dtype=np.uint16)
    viewer = viewer_with({0: controller_with(pixels, np.zeros((8, 8)))})
    np.testing.assert_array_equal(displayed_texture(viewer, 0), pixels)


def test_texture_mean_is_polls_on_before_the_first_data_response():
    assert texture_mean_is(viewer_with({}), 0.0)() is False


def test_texture_mean_is_polls_on_without_a_handle():
    assert texture_mean_is(viewer_with({0: controller_with()}), 0.0)() is False


def test_texture_mean_is_true_once_the_handle_holds_the_pixels():
    viewer = viewer_with({1: controller_with(np.full((4, 4), 3.0))})
    assert texture_mean_is(viewer, 3.0, key=1)() is True
    assert texture_mean_is(viewer, 2.0, key=1)() is False


def test_texture_mean_is_false_while_a_fetch_is_in_flight():
    viewer = viewer_with({0: controller_with(np.zeros((4, 4)))})
    viewer._is_idle = lambda: False
    assert texture_mean_is(viewer, 0.0)() is False


@pytest.mark.parametrize(
    "read",
    [
        lambda v: displayed_texture(v, 0),
        lambda v: dims_slider(v, "z_level"),
        is_idle,
    ],
)
def test_a_missing_attribute_is_an_ndv_rename_and_skips(read):
    with pytest.raises(pytest.skip.Exception, match="ndv private path .* changed"):
        read(SimpleNamespace())


def test_a_dims_slider_ndv_never_built_is_a_failure_not_a_skip():
    viewer = SimpleNamespace(
        _view=SimpleNamespace(
            _qwidget=SimpleNamespace(dims_sliders=SimpleNamespace(_sliders={}))
        )
    )
    with pytest.raises(pytest.fail.Exception, match="z_level"):
        dims_slider(viewer, "z_level")


def test_a_dims_slider_ndv_built_is_returned():
    slider = object()
    viewer = SimpleNamespace(
        _view=SimpleNamespace(
            _qwidget=SimpleNamespace(
                dims_sliders=SimpleNamespace(_sliders={"z_level": slider})
            )
        )
    )
    assert dims_slider(viewer, "z_level") is slider
