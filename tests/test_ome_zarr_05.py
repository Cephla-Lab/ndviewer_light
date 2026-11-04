"""Tests for OME-Zarr 0.5 image metadata parsing."""

import json
import logging

import pytest

from ndviewer_light.ome_zarr import (
    OmeZarrError,
    array_dimension_names,
    normalise_axis_names,
    parse_image,
    read_node_attrs,
)
from tests.ome_zarr_fixtures import SQUID_AXES, write_image_group


def _space(name):
    return {"name": name, "type": "space", "unit": "micrometer"}


class TestParseImage:
    def test_dataset_path_is_read_not_assumed(self, tmp_path):
        write_image_group(
            tmp_path / "img.ome.zarr", (1, 2, 3, 8, 8), SQUID_AXES, dataset_path="s0"
        )
        image = parse_image(tmp_path / "img.ome.zarr")
        assert image.array_path == "s0"
        assert image.array_dir == tmp_path / "img.ome.zarr" / "s0"

    def test_axes_subset_and_nanometre_units(self, tmp_path):
        axes = [
            {"name": "c", "type": "channel"},
            {"name": "z", "type": "space", "unit": "nanometer"},
            {"name": "y", "type": "space", "unit": "nanometer"},
            {"name": "x", "type": "space", "unit": "nanometer"},
        ]
        write_image_group(
            tmp_path / "i",
            (2, 3, 8, 8),
            axes,
            scale={"z": 1500, "y": 650, "x": 650},
            omero=False,
        )
        image = parse_image(tmp_path / "i")
        assert image.dim_names == ["c", "z", "y", "x"]
        assert image.scale_um == pytest.approx({"z": 1.5, "y": 0.65, "x": 0.65})
        assert image.channel_names == [] and image.channel_colors == []

    def test_axes_are_mapped_by_type_then_name(self):
        axes = [
            {"name": "T", "type": "time"},
            {"name": "position", "type": "fov"},
            {"name": "C", "type": "channel"},
            _space("Z"),
            _space("Y"),
            _space("X"),
        ]
        assert normalise_axis_names(axes) == ["t", "fov", "c", "z", "y", "x"]

    def test_spatial_axes_with_arbitrary_names_map_by_order(self):
        axes = [
            {"name": "c", "type": "channel"},
            _space("depth"),
            _space("row"),
            _space("col"),
        ]
        assert normalise_axis_names(axes) == ["c", "z", "y", "x"]
        assert normalise_axis_names(axes[:1] + axes[2:]) == ["c", "y", "x"]
        with pytest.raises(OmeZarrError):
            normalise_axis_names(axes + [{"name": "w", "type": "space"}])

    def test_multiscale_transform_composes_with_dataset_transform(self, tmp_path):
        write_image_group(
            tmp_path / "i",
            (1, 2, 3, 8, 8),
            SQUID_AXES,
            multiscale_transforms=[{"type": "scale", "scale": [1, 1, 2, 2, 2]}],
        )
        assert parse_image(tmp_path / "i").scale_um == pytest.approx(
            {"z": 3.0, "y": 1.3, "x": 1.3}
        )

    def test_spec_scale_wins_over_squid_block(self, tmp_path):
        write_image_group(
            tmp_path / "i",
            (1, 2, 3, 8, 8),
            SQUID_AXES,
            squid={"pixel_size_um": 9.9, "z_step_um": 9.9},
        )
        image = parse_image(tmp_path / "i")
        assert image.pixel_size_um == pytest.approx(0.65)
        assert image.dz_um == pytest.approx(1.5)

    def test_squid_block_fills_axes_without_spec_scale(self, tmp_path):
        write_image_group(
            tmp_path / "i",
            (1, 2, 3, 8, 8),
            SQUID_AXES,
            squid={"pixel_size_um": 9.9, "z_step_um": 2.5},
        )
        zarr_json = tmp_path / "i" / "zarr.json"
        node = json.loads(zarr_json.read_text())
        multiscale = node["attributes"]["ome"]["multiscales"][0]
        multiscale["datasets"][0]["coordinateTransformations"] = []
        zarr_json.write_text(json.dumps(node))
        image = parse_image(tmp_path / "i")
        assert image.pixel_size_um == pytest.approx(9.9)
        assert image.dz_um == pytest.approx(2.5)

    def test_no_squid_block_means_complete(self, tmp_path):
        write_image_group(tmp_path / "i", (1, 2, 3, 8, 8), SQUID_AXES, squid=None)
        image = parse_image(tmp_path / "i")
        assert image.acquisition_complete is True and image.aborted is False

    def test_squid_incomplete_and_aborted_are_read(self, tmp_path):
        write_image_group(
            tmp_path / "i",
            (1, 2, 3, 8, 8),
            SQUID_AXES,
            squid={"acquisition_complete": False, "aborted": True},
        )
        image = parse_image(tmp_path / "i")
        assert image.acquisition_complete is False and image.aborted is True

    def test_squid_block_without_key_is_incomplete(self, tmp_path):
        write_image_group(tmp_path / "i", (1, 2, 3, 8, 8), SQUID_AXES, squid={})
        assert parse_image(tmp_path / "i").acquisition_complete is False

    def test_unknown_version_warns_and_reads(self, tmp_path, caplog):
        write_image_group(tmp_path / "i", (1, 2, 3, 8, 8), SQUID_AXES, version="0.4")
        with caplog.at_level(logging.WARNING, logger="ndviewer_light.ome_zarr"):
            image = parse_image(tmp_path / "i")
        assert image.version == "0.4" and "0.4" in caplog.text

    def test_channels_are_read(self, tmp_path):
        write_image_group(tmp_path / "i", (1, 2, 3, 8, 8), SQUID_AXES)
        image = parse_image(tmp_path / "i")
        assert image.channel_names == ["BF LED matrix full", "Fluorescence 488 nm Ex"]
        assert image.channel_colors == ["FFFFFF", "1FFF00"]

    def test_array_is_the_group(self, tmp_path):
        write_image_group(
            tmp_path / "a",
            (1, 2, 3, 8, 8),
            SQUID_AXES,
            dataset_path=".",
            node_type="array",
            dimension_names=["t", "c", "z", "y", "x"],
        )
        image = parse_image(tmp_path / "a")
        assert image.array_dir == tmp_path / "a"
        assert array_dimension_names(image) == ["t", "c", "z", "y", "x"]

    def test_missing_multiscales_is_an_error(self, tmp_path):
        (tmp_path / "g").mkdir()
        (tmp_path / "g" / "zarr.json").write_text(
            json.dumps({"zarr_format": 3, "node_type": "group", "attributes": {}})
        )
        with pytest.raises(OmeZarrError):
            parse_image(tmp_path / "g")

    def test_read_node_attrs_rejects_non_v3(self, tmp_path):
        (tmp_path / "g").mkdir()
        (tmp_path / "g" / "zarr.json").write_text(json.dumps({"zarr_format": 2}))
        with pytest.raises(OmeZarrError):
            read_node_attrs(tmp_path / "g")

    def test_read_node_attrs_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            read_node_attrs(tmp_path)

    def test_array_dimension_names(self, tmp_path):
        write_image_group(
            tmp_path / "i",
            (1, 2, 3, 8, 8),
            SQUID_AXES,
            dimension_names=["t", "c", "z", "y", "x"],
        )
        assert array_dimension_names(parse_image(tmp_path / "i")) == [
            "t",
            "c",
            "z",
            "y",
            "x",
        ]
        write_image_group(tmp_path / "j", (1, 2, 3, 8, 8), SQUID_AXES)
        assert array_dimension_names(parse_image(tmp_path / "j")) is None
