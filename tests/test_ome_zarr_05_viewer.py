"""LightweightViewer loads OME-Zarr 0.5 stores by their metadata."""

import logging
import os
import shutil

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np  # noqa: E402
import pytest  # noqa: E402
import tifffile  # noqa: E402
from PyQt5.QtWidgets import QApplication  # noqa: E402

from ndviewer_light import LightweightViewer, detect_format  # noqa: E402
from tests.ome_zarr_fixtures import (  # noqa: E402
    SQUID_AXES,
    plane_value,
    write_image_group,
    write_legacy_image,
    write_plate,
)


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def load(qapp, path) -> LightweightViewer:
    v = LightweightViewer()
    v.load_dataset(str(path))
    return v


def test_plate_loads_with_labels_and_pixels(qapp, tmp_path):
    write_plate(
        tmp_path / "plate.ome.zarr",
        {"D5": [0], "B3": [0, 1]},
        rows=["B", "D"],
        cols=[3, 5],
    )
    v = load(qapp, tmp_path / "plate.ome.zarr")
    assert dict(v._xarray_data.sizes) == {
        "time": 1,
        "fov": 3,
        "z": 3,
        "channel": 2,
        "y": 8,
        "x": 8,
    }
    assert v._fov_labels == ["B3:0", "B3:1", "D5:0"]
    assert int(
        v._xarray_data.isel(time=0, fov=2, z=1, channel=1, y=0, x=0).values
    ) == plane_value(1, 1)
    assert v._xarray_data.attrs["pixel_size_um"] == pytest.approx(0.65)
    assert v._xarray_data.attrs["dz_um"] == pytest.approx(1.5)
    assert v.status_label.text() == "Loaded: plate.ome.zarr"
    assert v.get_fov_list() == [
        {"region": "B3", "fov": 0},
        {"region": "B3", "fov": 1},
        {"region": "D5", "fov": 0},
    ]


def test_non_default_dataset_path_and_axes_subset(qapp, tmp_path):
    axes = SQUID_AXES[1:]  # c z y x
    write_image_group(
        tmp_path / "i.ome.zarr", (2, 3, 8, 8), axes, dataset_path="s0", omero=False
    )
    v = load(qapp, tmp_path / "i.ome.zarr")
    assert dict(v._xarray_data.sizes) == {
        "time": 1,
        "fov": 1,
        "z": 3,
        "channel": 2,
        "y": 8,
        "x": 8,
    }
    assert v._xarray_data.attrs["channel_names"] == ["Ch0", "Ch1"]


def test_broken_fov_fails_the_load(qapp, tmp_path):
    write_plate(tmp_path / "p", {"B3": [0, 1]}, rows=["B"], cols=[3])
    shutil.rmtree(tmp_path / "p" / "B" / "3" / "1" / "0")  # group stays, array gone
    v = load(qapp, tmp_path / "p")
    assert v._xarray_data is None
    assert v.status_label.text().startswith("Error:")
    assert "B/3/1" in v.status_label.text()
    assert v._fov_labels == []


def test_incomplete_array_is_labelled(qapp, tmp_path):
    write_plate(
        tmp_path / "p",
        {"B3": [0]},
        rows=["B"],
        cols=[3],
        squid={"acquisition_complete": False},
    )
    v = load(qapp, tmp_path / "p")
    assert v.status_label.text() == "Loaded: p (incomplete acquisition)"


def test_recognised_plate_with_no_fields_does_not_fall_back(qapp, tmp_path):
    write_plate(tmp_path / "p", {"B3": [0]}, rows=["B"], cols=[3])
    shutil.rmtree(tmp_path / "p" / "B" / "3" / "0")  # listed field gone
    write_image_group(  # unlisted directory with data
        tmp_path / "p" / "B" / "3" / "7", (1, 2, 3, 8, 8), SQUID_AXES
    )
    v = load(qapp, tmp_path / "p")
    assert v._xarray_data is None
    assert v.status_label.text().startswith("Error:")
    assert "no acquired fields" in v.status_label.text()


def test_dimension_names_mismatch_warns_and_loads(qapp, tmp_path, caplog):
    write_image_group(
        tmp_path / "i",
        (1, 2, 3, 8, 8),
        SQUID_AXES,
        dimension_names=["a", "b", "c", "d", "e"],
    )
    with caplog.at_level(logging.WARNING):
        v = load(qapp, tmp_path / "i")
    assert v._xarray_data is not None
    assert "dimension_names" in caplog.text


def test_6d_layout_loads(qapp, tmp_path):
    axes = [{"name": "fov", "type": "fov"}] + SQUID_AXES
    write_image_group(
        tmp_path / "run" / "zarr" / "r" / "acquisition.zarr",
        (2, 1, 2, 3, 8, 8),
        axes,
        dataset_path=".",
        node_type="array",
        dimension_names=["fov", "t", "c", "z", "y", "x"],
    )
    v = load(qapp, tmp_path / "run")
    assert dict(v._xarray_data.sizes)["fov"] == 2
    assert v._fov_labels == ["r:0", "r:1"]
    assert v.get_fov_list() == [{"region": "r", "fov": 0}, {"region": "r", "fov": 1}]


def test_legacy_layout_still_loads(qapp, tmp_path):
    # pre-0.5 Squid: plate.zarr/B/3/0/acquisition.zarr, multiscales at the root
    group = tmp_path / "run" / "plate.zarr" / "B" / "3" / "0" / "acquisition.zarr"
    write_legacy_image(group, (1, 2, 3, 8, 8))
    v = load(qapp, tmp_path / "run")
    assert dict(v._xarray_data.sizes)["fov"] == 1
    assert v._fov_labels == ["B3:0"]
    assert v.status_label.text() == "Loaded: run"


def test_detect_format_accepts_a_bare_image_group(tmp_path):
    write_image_group(tmp_path / "anything", (1, 2, 3, 8, 8), SQUID_AXES)
    assert detect_format(tmp_path / "anything") == "zarr_v3"


def test_pull_load_is_not_push_mode(qapp, tmp_path):
    write_plate(tmp_path / "p", {"B3": [0, 1]}, rows=["B"], cols=[3])
    v = load(qapp, tmp_path / "p")
    assert v._fov_labels == ["B3:0", "B3:1"]
    assert v.is_push_mode_active() is False


def test_refresh_reports_loader_errors(qapp, tmp_path):
    write_plate(tmp_path / "p", {"B3": [0, 1]}, rows=["B"], cols=[3])
    v = load(qapp, tmp_path / "p")
    assert v.status_label.text() == "Loaded: p"
    shutil.rmtree(tmp_path / "p" / "B" / "3" / "1" / "0")
    v.refresh()  # must not raise
    assert v.status_label.text().startswith("Error:")


_FOV_AXES = [{"name": "fov", "type": "fov"}] + SQUID_AXES
_6D_NAMES = ["fov", "t", "c", "z", "y", "x"]


def _write_6d_region(run, region, shape):
    write_image_group(
        run / "zarr" / region / "acquisition.zarr",
        shape,
        _FOV_AXES,
        dataset_path=".",
        node_type="array",
        dimension_names=_6D_NAMES,
    )


def test_6d_two_regions_concatenate_along_fov(qapp, tmp_path):
    _write_6d_region(tmp_path / "run", "r1", (2, 1, 2, 3, 8, 8))
    _write_6d_region(tmp_path / "run", "r2", (3, 1, 2, 3, 8, 8))
    v = load(qapp, tmp_path / "run")
    assert dict(v._xarray_data.sizes)["fov"] == 5
    assert v._fov_labels == ["r1:0", "r1:1", "r2:0", "r2:1", "r2:2"]
    assert int(
        v._xarray_data.isel(time=0, fov=2, z=1, channel=1, y=0, x=0).values
    ) == plane_value(1, 1)
    assert [e["region"] for e in v.get_fov_list()] == ["r1"] * 2 + ["r2"] * 3


def test_6d_regions_with_different_shapes_fail(qapp, tmp_path):
    _write_6d_region(tmp_path / "run", "r1", (2, 1, 2, 3, 8, 8))
    _write_6d_region(tmp_path / "run", "r2", (2, 1, 2, 3, 4, 4))
    v = load(qapp, tmp_path / "run")
    assert v._xarray_data is None
    assert v.status_label.text().startswith("Error:")
    assert "r1" in v.status_label.text() and "r2" in v.status_label.text()


def test_go_to_well_fov_is_false_in_pull_mode(qapp, tmp_path):
    write_plate(tmp_path / "p", {"B3": [0, 1]}, rows=["B"], cols=[3])
    v = load(qapp, tmp_path / "p")
    assert v.go_to_well_fov("B3", 1) is False


def test_failed_refresh_keeps_previous_labels_and_data(qapp, tmp_path):
    write_plate(tmp_path / "p", {"B3": [0, 1]}, rows=["B"], cols=[3])
    v = load(qapp, tmp_path / "p")
    previous = v._xarray_data
    shutil.rmtree(tmp_path / "p" / "B" / "3" / "1" / "0")
    v.refresh()
    assert v.status_label.text().startswith("Error:")
    assert v._fov_labels == ["B3:0", "B3:1"]
    assert v._xarray_data is previous


def test_tiff_load_after_zarr_clears_labels(qapp, tmp_path):
    write_plate(tmp_path / "p", {"B3": [0, 1]}, rows=["B"], cols=[3])
    tiff_dir = tmp_path / "tiffs" / "0"
    tiff_dir.mkdir(parents=True)
    for channel in ("BF", "DAPI"):
        tifffile.imwrite(
            str(tiff_dir / f"A1_0_0_{channel}.tiff"), np.zeros((8, 8), np.uint16)
        )
    v = load(qapp, tmp_path / "p")
    assert v._fov_labels == ["B3:0", "B3:1"]
    v.load_dataset(str(tmp_path / "tiffs"))
    assert v._xarray_data is not None
    assert v._fov_labels == []
