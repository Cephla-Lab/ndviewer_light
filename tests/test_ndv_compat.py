"""``ndviewer_light._ndv_compat``: ndv's texture downsampling must hand vispy an
array that owns its memory.

ndv 0.5.0 strides an oversized texture to a *view* (``ndv.views._util.
downsample_data``) and vispy's ``VolumeVisual`` keeps whatever it was given as
``_last_data`` (CPU clim rescaling), so without the shim the full-resolution
(z, y, x) block of every displayed channel stays in RAM while a volume is shown.
These tests pin the shim through ndv's private path: if ndv renames it, the
integration test fails here instead of the retention coming back silently.
"""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import time  # noqa: E402

import dask.array as da  # noqa: E402
import ndv.views._vispy._array_canvas as canvas_mod  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
import xarray as xr  # noqa: E402

from ndviewer_light import LightweightViewer, _ndv_compat  # noqa: E402
from tests._ndv_private import displayed_texture, is_idle  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    from qtpy.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


def test_patched_downsample_owns_memory_only_when_strided():
    """Strided result = contiguous copy; nothing strided = the same object (no copy)."""
    big = np.zeros((32, 64, 64), np.uint16)
    out, factors = canvas_mod.downsample_data(big, 16)
    assert factors == (2, 4, 4)
    assert out.shape == (16, 16, 16)
    assert out.base is None and out.flags.owndata, "strided view retains `big`"

    small = np.zeros((8, 8, 8), np.uint16)
    same, factors = canvas_mod.downsample_data(small, 16)
    assert factors == (1, 1, 1)
    assert same is small


def test_install_is_idempotent():
    before = canvas_mod.downsample_data
    _ndv_compat.install_compact_downsample()
    assert canvas_mod.downsample_data is before


def _wait_for_volume(qapp, viewer, timeout_s=5.0) -> np.ndarray:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        qapp.processEvents()
        if is_idle(viewer.ndv_viewer):
            texture = displayed_texture(viewer.ndv_viewer, 0)
            if texture is not None and texture.ndim == 3:
                return texture
    pytest.fail("ndv showed no 3D texture within the timeout")


def test_displayed_volume_owns_its_memory(qapp, monkeypatch):
    """A lazy (32, 1, 64, 64) stack shown in 3D under a 16-voxel limit keeps 8 KiB,
    not the 256 KiB block ndv materialised to stride it."""
    monkeypatch.setattr(canvas_mod, "get_max_texture_sizes", lambda: (4096, 16))
    viewer = LightweightViewer()
    data = xr.DataArray(
        da.ones((32, 1, 64, 64), chunks=(1, 1, 64, 64), dtype=np.uint16),
        dims=("z", "channel", "y", "x"),
        coords={"z": range(32), "channel": [0]},
    )
    data.attrs.update(
        luts={0: {"name": "BF", "cmap": "gray"}},
        channel_names=["BF"],
        pixel_size_um=0.5,
        dz_um=1.5,
    )
    try:
        viewer._xarray_data = data
        viewer._set_ndv_data(data)
        viewer.ndv_viewer.display_model.visible_axes = ("z", "y", "x")
        texture = _wait_for_volume(qapp, viewer)
        assert texture.shape == (16, 16, 16)
        assert texture.base is None and texture.flags.owndata, (
            f"vispy holds a view onto a {texture.base.shape} block of "
            f"{texture.base.nbytes} bytes; {texture.nbytes} are displayed"
        )
    finally:
        viewer.close()
