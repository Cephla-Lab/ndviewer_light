"""Texture-level smokes for the ndv 0.5 port; need a real window (``window`` marker).

The offscreen suite can show that the model and signals are right; only a real GL
context shows what reached the GPU. These three guard the paths the port touched:

* S1 push mode: frame N+1 (``register_image`` for a key not seen before, as an
  acquisition delivers them) reaches the GPU while the ``ArrayViewer`` instance
  survives (no rebuild, no flicker).
* S2 shape change: ``_try_inplace_ndv_update`` with a taller z-stack updates the
  z slider range AND the texture (``dims_changed`` alone leaves the old pixels).
* S3 navigation: ``set_current_index("z_level", 2)`` puts plane 2 on screen.

Skipped when the Qt platform is offscreen (``window_qapp`` in conftest). Locally:
``QT_QPA_PLATFORM=cocoa NDVL_REAL_GL=1 pytest tests/test_window_smokes.py``; in CI
the same under xvfb with ``QT_QPA_PLATFORM=xcb``. Waits poll the event loop and
ndv's idle state, never sleep.
"""

import time

import numpy as np
import pytest
import tifffile
import xarray as xr

from ndviewer_light import LightweightViewer
from tests._ndv_private import dims_slider, displayed_texture, is_idle

pytestmark = pytest.mark.window

CHANNELS = ["BF", "GFP"]


def wait_until(qapp, predicate, timeout_s=5.0) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        qapp.processEvents()
        if predicate():
            return True
    return predicate()


def texture_mean_is(ndv_viewer, expected: float, key: int = 0):
    return (
        lambda: is_idle(ndv_viewer)
        and displayed_texture(ndv_viewer, key).mean() == expected
    )


def z_stack(n_z: int, voxel) -> xr.DataArray:
    """(z_level, channel, 8, 8) uint16 array shaped like _update_ndv_data builds."""
    z = np.arange(n_z, dtype=np.uint16).reshape(n_z, 1, 1, 1)
    vals = np.broadcast_to(voxel(z), (n_z, len(CHANNELS), 8, 8)).copy()
    return xr.DataArray(
        vals,
        dims=["z_level", "channel", "y", "x"],
        coords={"z_level": list(range(n_z)), "channel": list(range(len(CHANNELS)))},
    )


@pytest.fixture
def shown_viewer(window_qapp):
    """Push-mode viewer (3 z-levels, two channels) shown as a real window."""
    v = LightweightViewer()
    v.start_acquisition(CHANNELS, 3, 8, 8, ["A1:0"])
    v.resize(480, 400)
    v.show()
    assert wait_until(window_qapp, texture_mean_is(v.ndv_viewer, 0.0)), "placeholder"
    yield v
    v.close()
    v.deleteLater()
    window_qapp.processEvents()


def test_s1_push_frames_change_texture_without_rebuild(
    shown_viewer, window_qapp, tmp_path
):
    """Each registered plane lands in its channel's texture of the one viewer.

    Keys are (t, fov, z, channel) and an acquisition registers each once (the
    plane cache is keyed the same way), so frame N+1 here is the next channel.
    """
    v = shown_viewer
    ndv_viewer = v.ndv_viewer

    for frame, channel in enumerate(CHANNELS, start=1):
        tif = tmp_path / f"{channel}.tif"
        tifffile.imwrite(tif, np.full((8, 8), frame, dtype=np.uint16))

        v.register_image(0, 0, 0, channel, str(tif))

        assert wait_until(
            window_qapp, texture_mean_is(ndv_viewer, float(frame), key=frame - 1)
        )
        assert v.ndv_viewer is ndv_viewer
    assert displayed_texture(ndv_viewer, 0).mean() == 1.0  # BF still on channel 0


def test_s2_shape_change_updates_slider_range_and_texture(shown_viewer, window_qapp):
    v = shown_viewer
    ndv_viewer = v.ndv_viewer
    assert dims_slider(ndv_viewer, "z_level").maximum() == 2

    assert v._try_inplace_ndv_update(z_stack(5, lambda z: 555 + 0 * z)) is True

    assert wait_until(window_qapp, texture_mean_is(ndv_viewer, 555.0))
    assert dims_slider(ndv_viewer, "z_level").maximum() == 4
    assert v.ndv_viewer is ndv_viewer


def test_s3_set_current_index_moves_texture(shown_viewer, window_qapp):
    v = shown_viewer
    ndv_viewer = v.ndv_viewer
    # plane z holds 100 + 100*z, so z=0 shows 100 and z=2 shows 300
    assert v._try_inplace_ndv_update(z_stack(3, lambda z: 100 + 100 * z)) is True
    assert wait_until(window_qapp, texture_mean_is(ndv_viewer, 100.0))

    assert v.set_current_index("z_level", 2) is True

    assert wait_until(window_qapp, texture_mean_is(ndv_viewer, 300.0))
    assert dims_slider(ndv_viewer, "z_level").value() == 2
