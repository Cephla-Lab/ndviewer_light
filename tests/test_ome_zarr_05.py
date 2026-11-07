"""Tests for OME-Zarr 0.5 image metadata parsing."""

import json
import logging
import shutil

import pytest

from ndviewer_light.ome_zarr import (
    OmeZarrError,
    array_dimension_names,
    discover_ome_zarr,
    image_from_legacy,
    is_ome_zarr_node,
    normalise_axis_names,
    parse_image,
    read_node_attrs,
)
from tests.ome_zarr_fixtures import (
    SQUID_AXES,
    write_image_group,
    write_legacy_image,
    write_plate,
)


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


class TestUnitsAndPlateFixture:
    def test_centimetre_axes_convert_to_micrometres(self, tmp_path):
        axes = [{"name": n, "type": "space", "unit": "centimeter"} for n in "zyx"]
        write_image_group(
            tmp_path / "i",
            (3, 8, 8),
            axes,
            scale={"z": 0.0001, "y": 0.0001, "x": 0.0001},
            omero=False,
        )
        image = parse_image(tmp_path / "i")
        assert image.scale_um == pytest.approx({"z": 1.0, "y": 1.0, "x": 1.0})

    def test_unknown_unit_is_an_error_naming_it(self, tmp_path):
        axes = [{"name": n, "type": "space", "unit": "furlong"} for n in "zyx"]
        write_image_group(tmp_path / "i", (3, 8, 8), axes, omero=False)
        with pytest.raises(OmeZarrError, match="furlong"):
            parse_image(tmp_path / "i")

    def test_write_plate_accepts_well_ids_and_integer_columns(self, tmp_path):
        write_plate(
            tmp_path / "p", {"D5": [0], "B3": [0, 1]}, rows=["B", "D"], cols=[3, 5]
        )
        plate = read_node_attrs(tmp_path / "p")["ome"]["plate"]
        assert sorted(plate["wells"], key=lambda w: w["path"]) == [
            {"path": "B/3", "rowIndex": 0, "columnIndex": 0},
            {"path": "D/5", "rowIndex": 1, "columnIndex": 1},
        ]
        assert plate["columns"] == [{"name": "3"}, {"name": "5"}]
        for image in ("B/3/0", "B/3/1", "D/5/0"):
            assert (tmp_path / "p" / image / "zarr.json").exists()


class TestDiscover:
    def test_plate_discovery_follows_wells_and_images(self, tmp_path):
        write_plate(
            tmp_path / "plate.ome.zarr",
            {"D5": [0], "B3": [0, 1]},
            rows=["B", "D"],
            cols=[3, 5],
        )
        fovs, structure = discover_ome_zarr(tmp_path / "plate.ome.zarr")
        assert structure == "hcs_plate"
        assert [(f["region"], f["fov"]) for f in fovs] == [
            ("B3", 0),
            ("B3", 1),
            ("D5", 0),
        ]
        expected = tmp_path / "plate.ome.zarr" / "B" / "3" / "0" / "0"
        assert fovs[0]["image"].array_dir == expected

    def test_experiment_folder_with_plate(self, tmp_path):
        write_plate(
            tmp_path / "run" / "plate.ome.zarr", {"B3": [0]}, rows=["B"], cols=[3]
        )
        (tmp_path / "run" / "acquisition parameters.json").write_text("{}")
        fovs, _ = discover_ome_zarr(tmp_path / "run")
        assert [(f["region"], f["fov"]) for f in fovs] == [("B3", 0)]

    def test_field_id_is_the_numeric_path(self, tmp_path):
        write_plate(tmp_path / "p", {"B3": [0, 1]}, rows=["B"], cols=[3])
        shutil.rmtree(tmp_path / "p" / "B" / "3" / "0")
        fovs, _ = discover_ome_zarr(tmp_path / "p")
        assert [(f["region"], f["fov"]) for f in fovs] == [("B3", 1)]

    def test_absent_listed_well_is_skipped_with_a_warning(self, tmp_path, caplog):
        write_plate(
            tmp_path / "p", {"B3": [0], "D5": [0]}, rows=["B", "D"], cols=[3, 5]
        )
        shutil.rmtree(tmp_path / "p" / "D")
        with caplog.at_level(logging.WARNING, logger="ndviewer_light.ome_zarr"):
            fovs, _ = discover_ome_zarr(tmp_path / "p")
        assert [f["region"] for f in fovs] == ["B3"] and "D/5" in caplog.text

    def test_existing_field_without_metadata_is_broken(self, tmp_path):
        write_plate(tmp_path / "p", {"B3": [0]}, rows=["B"], cols=[3])
        (tmp_path / "p" / "B" / "3" / "0" / "zarr.json").unlink()
        with pytest.raises(OmeZarrError, match="B/3/0"):
            discover_ome_zarr(tmp_path / "p")

    def test_well_group_opened_directly(self, tmp_path):
        write_plate(tmp_path / "p", {"B3": [0, 1]}, rows=["B"], cols=[3])
        fovs, structure = discover_ome_zarr(tmp_path / "p" / "B" / "3")
        assert structure == "hcs_plate" and [f["fov"] for f in fovs] == [0, 1]
        assert [f["region"] for f in fovs] == ["B3", "B3"]

    def test_well_row_index_out_of_range_is_an_omezarrerror(self, tmp_path):
        write_plate(tmp_path / "p", {"B3": [0]}, rows=["B", "D"], cols=[3])
        zarr_json = tmp_path / "p" / "zarr.json"
        node = json.loads(zarr_json.read_text())
        node["attributes"]["ome"]["plate"]["wells"][0]["rowIndex"] = 5
        zarr_json.write_text(json.dumps(node))
        with pytest.raises(OmeZarrError, match=r"p.*B/3"):
            discover_ome_zarr(tmp_path / "p")

    def test_single_image_group(self, tmp_path):
        write_image_group(tmp_path / "img.ome.zarr", (1, 2, 3, 8, 8), SQUID_AXES)
        fovs, structure = discover_ome_zarr(tmp_path / "img.ome.zarr")
        assert structure == "per_fov"
        assert (fovs[0]["region"], fovs[0]["fov"]) == ("default", 0)

    def test_experiment_folder_per_fov_layout(self, tmp_path):
        for n in (10, 2):
            write_image_group(
                tmp_path / "run" / "zarr" / "region_0" / f"fov_{n}.ome.zarr",
                (1, 2, 3, 8, 8),
                SQUID_AXES,
            )
        fovs, structure = discover_ome_zarr(tmp_path / "run")
        assert structure == "per_fov"
        assert [(f["region"], f["fov"]) for f in fovs] == [
            ("region_0", 2),
            ("region_0", 10),
        ]

    def test_6d_image_is_structure_6d(self, tmp_path):
        axes = [{"name": "fov", "type": "fov"}] + SQUID_AXES
        write_image_group(
            tmp_path / "run" / "zarr" / "r" / "acquisition.zarr",
            (2, 1, 2, 3, 8, 8),
            axes,
            dataset_path=".",
            node_type="array",
            dimension_names=["fov", "t", "c", "z", "y", "x"],
        )
        fovs, structure = discover_ome_zarr(tmp_path / "run")
        assert structure == "6d" and fovs[0]["image"].array_path == "."
        assert (fovs[0]["region"], fovs[0]["fov"]) == ("r", 0)

    def test_is_ome_zarr_node(self, tmp_path):
        write_image_group(tmp_path / "i", (1, 2, 3, 8, 8), SQUID_AXES)
        assert is_ome_zarr_node(tmp_path / "i") and not is_ome_zarr_node(tmp_path)

    def test_nothing_matching_is_unknown(self, tmp_path):
        assert discover_ome_zarr(tmp_path) == ([], "unknown")

    def test_recognised_plate_with_no_acquired_fields_is_not_unknown(self, tmp_path):
        write_plate(tmp_path / "p", {"B3": [0]}, rows=["B"], cols=[3])
        shutil.rmtree(tmp_path / "p" / "B" / "3" / "0")
        assert discover_ome_zarr(tmp_path / "p") == ([], "hcs_plate")

    def test_pre_05_plate_folder_is_unknown_not_an_error(self, tmp_path):
        plate = tmp_path / "run" / "plate.zarr"
        plate.mkdir(parents=True)
        (plate / ".zgroup").write_text("{}")
        assert discover_ome_zarr(tmp_path / "run") == ([], "unknown")
        (plate / "zarr.json").write_text(
            '{"zarr_format": 3, "node_type": "group", "attributes": {}}'
        )
        assert discover_ome_zarr(tmp_path / "run") == ([], "unknown")

    def test_legacy_fov_folder_without_ome_suffix_is_ignored(self, tmp_path):
        (tmp_path / "run" / "zarr" / "r" / "fov_0.zarr").mkdir(parents=True)
        assert discover_ome_zarr(tmp_path / "run") == ([], "unknown")

    def test_image_from_legacy(self, tmp_path):
        meta = {
            "axes": [],
            "pixel_size_um": 0.5,
            "dz_um": 2.0,
            "channel_names": ["a"],
            "channel_colors": ["FFFFFF"],
            "acquisition_complete": False,
        }
        image = image_from_legacy(tmp_path, meta)
        assert image.array_path == "0" and image.dim_names == list("tczyx")
        assert (image.pixel_size_um, image.dz_um) == (0.5, 2.0)
        assert image.scale_um["y"] == 0.5
        assert image.version is None and image.aborted is False
        assert image.acquisition_complete is False
        assert image.channel_names == ["a"]


class TestExistingButUnreadable:
    """A node that exists but cannot be read is broken, never silently dropped (review of PR #42)."""

    def test_unreadable_region_metadata_is_an_error(self, tmp_path):
        axes = [{"name": "fov", "type": "fov"}] + SQUID_AXES
        for region in ("r1", "r2"):
            write_image_group(
                tmp_path / "run" / "zarr" / region / "acquisition.zarr",
                (2, 1, 2, 3, 8, 8),
                axes,
                dataset_path=".",
                node_type="array",
                dimension_names=["fov", "t", "c", "z", "y", "x"],
            )
        (
            tmp_path / "run" / "zarr" / "r2" / "acquisition.zarr" / "zarr.json"
        ).write_text("{not json")
        with pytest.raises(OmeZarrError, match="r2"):
            discover_ome_zarr(tmp_path / "run")

    def test_unreadable_plate_metadata_is_an_error(self, tmp_path):
        write_plate(
            tmp_path / "run" / "plate.ome.zarr", {"B3": [0]}, rows=["B"], cols=[3]
        )
        (tmp_path / "run" / "plate.ome.zarr" / "zarr.json").write_text("{not json")
        with pytest.raises(OmeZarrError, match="plate.ome.zarr"):
            discover_ome_zarr(tmp_path / "run")


class TestLegacyRankInference:
    """Pre-0.5 stores without axes metadata: dims come from the array's rank, as the old loaders did."""

    @pytest.mark.parametrize(
        "shape, expected",
        [
            ((3, 8, 8), ["z", "y", "x"]),
            ((2, 3, 8, 8), ["c", "z", "y", "x"]),
            ((1, 2, 3, 8, 8), ["t", "c", "z", "y", "x"]),
            ((2, 1, 2, 3, 8, 8), ["fov", "t", "c", "z", "y", "x"]),
        ],
    )
    def test_dims_from_the_array_rank(self, tmp_path, shape, expected):
        write_legacy_image(tmp_path / "old.zarr", shape, axes=None)
        image = image_from_legacy(
            tmp_path / "old.zarr", {"axes": [], "pixel_size_um": 0.65}
        )
        assert image.dim_names == expected
