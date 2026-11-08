"""OME-Zarr planes are read straight from each image's array, one dask layer in all.

Stacking one dask array per field gave a graph with thousands of layers on a
real plate (1772 fields), and dask then spent 0.5-1 s ordering and optimising it
for every plane shown. These tests pin that the array has a constant-size graph
and that every plane still comes from the right field, axis order included.
"""

import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np  # noqa: E402
import pytest  # noqa: E402
import tensorstore as ts  # noqa: E402
from qtpy.QtWidgets import QApplication  # noqa: E402

from ndviewer_light import LightweightViewer  # noqa: E402
from tests.ome_zarr_fixtures import (  # noqa: E402
    SQUID_AXES,
    write_image_group,
    write_plate,
)


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def load(path) -> LightweightViewer:
    v = LightweightViewer()
    v.load_dataset(str(path))
    assert v._xarray_data is not None, v.status_label.text()
    return v


def fill_unique(array_dir: Path, seed: int) -> np.ndarray:
    """Overwrite an array with values unique to it and to each position; return them."""
    store = ts.open(
        {"driver": "zarr3", "kvstore": {"driver": "file", "path": str(array_dir)}}
    ).result()
    shape = tuple(store.shape)
    data = (np.arange(np.prod(shape)) % 50000 + seed * 7).astype(np.uint16)
    data = data.reshape(shape)
    store[...].write(data).result()
    return data


def test_plate_planes_come_from_their_field(qapp, tmp_path):
    wells = {"B3": [0, 1, 2], "D5": [0, 1]}
    write_plate(tmp_path / "p.ome.zarr", wells, rows=["B", "D"], cols=[3, 5])
    expected = []  # in the viewer's fov order: B3:0..2, D5:0..1
    for seed, (row, col, field) in enumerate(
        [("B", 3, 0), ("B", 3, 1), ("B", 3, 2), ("D", 5, 0), ("D", 5, 1)]
    ):
        array_dir = tmp_path / "p.ome.zarr" / row / str(col) / str(field) / "0"
        expected.append(fill_unique(array_dir, seed))  # (t, c, z, y, x)

    v = load(tmp_path / "p.ome.zarr")
    data = v._xarray_data
    assert dict(data.sizes) == {
        "time": 1,
        "fov": 5,
        "z": 3,
        "channel": 2,
        "y": 8,
        "x": 8,
    }
    for fov, stored in enumerate(expected):
        for z in range(3):
            for c in range(2):
                np.testing.assert_array_equal(
                    data.isel(time=0, fov=fov, z=z, channel=c).values,
                    stored[0, c, z],
                    err_msg=f"fov={fov} z={z} c={c}",
                )


def test_graph_size_does_not_grow_with_fields(qapp, tmp_path):
    write_plate(tmp_path / "small", {"B3": [0]}, rows=["B"], cols=[3])
    write_plate(tmp_path / "big", {"B3": list(range(12))}, rows=["B"], cols=[3])
    small = load(tmp_path / "small")._xarray_data.data
    big = load(tmp_path / "big")._xarray_data.data
    assert len(big.dask.layers) == len(small.dask.layers) <= 2


def test_x_stored_before_y_is_transposed(qapp, tmp_path):
    axes = SQUID_AXES[:3] + [SQUID_AXES[4], SQUID_AXES[3]]  # t c z x y
    write_image_group(tmp_path / "i.ome.zarr", (1, 2, 3, 5, 7), axes)
    stored = fill_unique(tmp_path / "i.ome.zarr" / "0", seed=1)  # (t, c, z, x, y)

    data = load(tmp_path / "i.ome.zarr")._xarray_data
    assert (data.sizes["y"], data.sizes["x"]) == (7, 5)
    np.testing.assert_array_equal(
        data.isel(time=0, fov=0, z=2, channel=1).values, stored[0, 1, 2].T
    )


def test_6d_regions_map_fov_to_region_and_local_field(qapp, tmp_path):
    from tests.test_ome_zarr_05_viewer import _write_6d_region

    _write_6d_region(tmp_path / "run", "r1", (2, 1, 2, 3, 8, 8))
    _write_6d_region(tmp_path / "run", "r2", (3, 1, 2, 3, 8, 8))
    stored = {}
    for seed, region in enumerate(["r1", "r2"]):
        array_dir = tmp_path / "run" / "zarr" / region / "acquisition.zarr"
        stored[region] = fill_unique(array_dir, seed)  # (fov, t, c, z, y, x)

    data = load(tmp_path / "run")._xarray_data
    assert data.sizes["fov"] == 5
    order = [("r1", 0), ("r1", 1), ("r2", 0), ("r2", 1), ("r2", 2)]
    for fov, (region, local) in enumerate(order):
        np.testing.assert_array_equal(
            data.isel(time=0, fov=fov, z=2, channel=1).values,
            stored[region][local, 0, 1, 2],
            err_msg=f"fov={fov} ({region}:{local})",
        )
