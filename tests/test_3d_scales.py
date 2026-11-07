"""3D voxel aspect on ndv 0.5: ``_set_ndv_data`` passes the Z/XY ratio as ``scales``.

The viewer used to carry the ratio dz/pixel_size through a module global read by
monkeypatches on vispy's ``VolumeVisual`` and ndv's private ``VispyArrayCanvas``;
on ndv 0.5 / vispy 0.17 that block was inert (frozen visual, camera reset by
ndv's own ``set_scales``). ndv 0.5 has a public per-axis ``scales`` mapping on
``ArrayDisplayModel`` (``ArrayViewer(..., scales={<dim>: factor})``) that applies
an ``STTransform`` to the visual and frames the camera to match. These tests pin
the model side offscreen (conftest pins the GL texture query); the rendering is
checked once by hand in a real window (see the PR).
"""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pytest  # noqa: E402
import xarray as xr  # noqa: E402
from qtpy.QtWidgets import QApplication  # noqa: E402

from ndviewer_light import LightweightViewer  # noqa: E402
from ndviewer_light import core  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def stack(zdim: str, attrs: dict) -> xr.DataArray:
    """(zdim, channel, y, x) uint16 array with the given physical-size attrs."""
    return xr.DataArray(
        np.zeros((3, 1, 8, 8), dtype=np.uint16),
        dims=[zdim, "channel", "y", "x"],
        coords={zdim: [0, 1, 2], "channel": [0]},
        attrs=attrs,
    )


def scales(viewer: LightweightViewer) -> dict:
    return dict(viewer.ndv_viewer.display_model.scales)


@pytest.mark.parametrize("zdim", ["z", "z_level"])
def test_set_ndv_data_passes_z_scale(qapp, zdim):
    """dz 1.5 um over 0.5 um pixels -> the z axis is 3x a pixel, keyed by dim name."""
    v = LightweightViewer()
    v._set_ndv_data(stack(zdim, {"pixel_size_um": 0.5, "dz_um": 1.5}))
    assert scales(v) == {zdim: 3.0}


def test_no_scale_without_physical_sizes(qapp):
    """No attrs, or only one of the pair, leaves the model's scales empty."""
    for attrs in ({}, {"pixel_size_um": 0.5}, {"dz_um": 1.5}):
        v = LightweightViewer()
        v._set_ndv_data(stack("z", attrs))
        assert scales(v) == {}, attrs


def test_core_imports_no_private_vispy_canvas():
    """The port removed the private ``ndv.views._vispy`` import with the patches."""
    assert "ndv.views._vispy" not in Path(core.__file__).read_text()
