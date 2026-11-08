"""``_ndv_compat.wrap_xarray_fast_coords``: slider moves must not iterate xarray coords.

ndv 0.5 re-resolves the display state on every slider move, on the main thread,
and walks every coordinate in Python. With the stock xarray wrapper each element
becomes a new DataArray (~8,000 per move for a 2048 x 2048 image, 100-450 ms of
frozen UI per step). The wrapper hands ndv numpy coordinates instead.
"""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np  # noqa: E402
import pytest  # noqa: E402
import xarray as xr  # noqa: E402
from ndv import DataWrapper  # noqa: E402
from ndv.models._resolve import resolve  # noqa: E402

from ndviewer_light import LightweightViewer  # noqa: E402
from ndviewer_light._ndv_compat import wrap_xarray_fast_coords  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    from qtpy.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


def array_6d():
    return xr.DataArray(
        np.zeros((2, 3, 4, 2, 64, 64), np.uint16),
        dims=["time", "fov", "z", "channel", "y", "x"],
        coords={
            "time": [0, 1],
            "fov": [0, 1, 2],
            "z": [0.0, 1.5, 3.0, 4.5],
            "channel": [0, 1],
            "y": np.arange(64),
            "x": np.arange(64),
        },
    )


def test_wrapper_matches_the_stock_xarray_wrapper():
    data = array_6d()
    fast = wrap_xarray_fast_coords(data)
    stock = DataWrapper.create(data)
    assert type(stock) is not type(fast)
    assert fast.dims == stock.dims
    assert dict(fast.sizes()) == dict(stock.sizes())
    assert fast.axis_scales() == stock.axis_scales()
    for dim in data.dims:
        assert isinstance(fast.coords[dim], np.ndarray)
        assert list(fast.coords[dim]) == list(stock.coords[dim].values)
    index = {0: 1, 1: 2, 2: 3, 3: 0}
    np.testing.assert_array_equal(fast.isel(index), stock.isel(index))


def test_dims_without_coordinates_fall_back_to_range():
    data = xr.DataArray(np.zeros((3, 8, 8), np.uint16), dims=["z", "y", "x"])
    coords = wrap_xarray_fast_coords(data).coords
    assert coords == {"z": range(3), "y": range(8), "x": range(8)}


def test_create_never_picks_the_fast_wrapper_on_its_own():
    """``supports`` is False: other xarray users keep ndv's stock wrapper."""
    stock = DataWrapper.create(array_6d())
    assert "FastCoords" not in type(stock).__name__


def test_resolve_does_not_iterate_dataarrays(qapp, monkeypatch):
    """The per-slider-move resolve stays off xarray's per-element iteration."""
    v = LightweightViewer()
    try:
        v._set_ndv_data(array_6d())
        wrapper = v.ndv_viewer.data_wrapper
        assert "FastCoords" in type(wrapper).__name__

        def boom(self):
            raise AssertionError("resolve iterated an xarray.DataArray")

        monkeypatch.setattr(xr.DataArray, "__iter__", boom)
        resolve(v.ndv_viewer.display_model, wrapper)
    finally:
        v.close()
        v.deleteLater()
