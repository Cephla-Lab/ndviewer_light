"""``_ndv_compat.install_draw_superseded_frames``: the image follows a dragged slider.

ndv 0.5.0 draws a fetched frame only if no newer request was made meanwhile, so
while a slider moves faster than a frame loads the image stays frozen until the
slider stops. The shim draws such superseded frames unless something newer is
already on screen or the data/shape/layout changed since they were requested.
"""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import time  # noqa: E402

import dask.array as da  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
import xarray as xr  # noqa: E402
from ndv.controllers import ArrayViewer  # noqa: E402

from ndviewer_light import LightweightViewer, _ndv_compat  # noqa: E402
from tests._ndv_private import displayed_texture, is_idle  # noqa: E402

N_Z = 12
LOAD_S = 0.04  # per plane: slower than the 10 ms between slider moves below


@pytest.fixture(scope="module")
def qapp():
    from qtpy.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


def slow_stack() -> xr.DataArray:
    """(z, channel, y, x) stack where plane z is filled with z and loads slowly."""

    def load(block, block_info=None):
        time.sleep(LOAD_S)
        z = block_info[None]["chunk-location"][0]
        return np.full(block.shape, z, dtype=np.uint16)

    base = da.zeros((N_Z, 1, 16, 16), chunks=(1, 1, 16, 16), dtype=np.uint16)
    data = xr.DataArray(
        base.map_blocks(load, dtype=np.uint16),
        dims=("z", "channel", "y", "x"),
        coords={"z": range(N_Z), "channel": [0]},
    )
    data.attrs.update(luts={0: {"name": "BF", "cmap": "gray"}}, channel_names=["BF"])
    return data


def spin(qapp, seconds):
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        qapp.processEvents()


def wait_idle(qapp, nv, timeout_s=5.0):
    end = time.monotonic() + timeout_s
    while time.monotonic() < end:
        qapp.processEvents()
        if is_idle(nv) and not nv._futures:
            qapp.processEvents()
            return
    pytest.fail("ndv did not go idle")


@pytest.fixture
def slow_viewer(qapp):
    v = LightweightViewer()
    v._set_ndv_data(slow_stack())
    wait_idle(qapp, v.ndv_viewer)
    yield v
    v.close()
    v.deleteLater()
    qapp.processEvents()


def test_installed_on_import():
    if hasattr(ArrayViewer, "_is_drawable"):
        pytest.skip("ndv has the upstream fix; the shim stands down")
    assert getattr(ArrayViewer._cancel_futures, _ndv_compat._SUPERSEDED_FLAG, False)


def test_install_is_idempotent():
    before = ArrayViewer._on_data_response_ready
    _ndv_compat.install_draw_superseded_frames()
    assert ArrayViewer._on_data_response_ready is before


def test_image_follows_a_fast_drag_and_never_goes_back(qapp, slow_viewer):
    nv = slow_viewer.ndv_viewer
    shown = []
    refresh = nv._canvas.refresh

    def record():
        texture = displayed_texture(nv, 0)
        if texture is not None:
            shown.append(int(texture.mean()))
        refresh()

    nv._canvas.refresh = record

    for z in range(1, N_Z):  # drag: a move every 10 ms, a frame takes 40 ms
        nv.display_model.current_index["z"] = z
        spin(qapp, 0.01)
    wait_idle(qapp, nv)

    assert shown, "nothing was drawn"
    assert shown[-1] == N_Z - 1, f"ended on plane {shown[-1]}"
    intermediate = [z for z in shown if 0 < z < N_Z - 1]
    assert intermediate, f"image froze during the drag; drawn planes: {shown}"
    assert shown == sorted(shown), f"an older frame replaced a newer one: {shown}"


def test_layout_change_makes_earlier_requests_undrawable(qapp, slow_viewer):
    nv = slow_viewer.ndv_viewer
    nv.display_model.current_index["z"] = 3
    spin(qapp, 0.005)
    nv.display_model.visible_axes = ("z", "y", "x")  # 2D -> 3D
    assert nv._ndvl_min_gen == nv._current_gen
    wait_idle(qapp, nv)


def test_index_change_keeps_earlier_requests_drawable(qapp, slow_viewer):
    nv = slow_viewer.ndv_viewer
    min_before = nv._ndvl_min_gen
    nv.display_model.current_index["z"] = 5
    assert nv._ndvl_min_gen == min_before
    wait_idle(qapp, nv)


def test_dims_change_makes_earlier_requests_undrawable(qapp, slow_viewer):
    nv = slow_viewer.ndv_viewer
    current = nv._current_gen
    nv.data_wrapper.dims_changed.emit()
    assert nv._ndvl_min_gen >= current + 1
    wait_idle(qapp, nv)
