"""
Unit tests for channel name display feature in ndviewer_light.

Tests cover:
1. Channel name extraction and preservation logic in OME-TIFF path
2. Channel name extraction in single-TIFF path
3. Channel names reaching ndv through the public ``LUTModel.name`` field
   (``build_luts`` -> ``ArrayViewer(luts=...)``), with no private ndv access
"""

import os
import time
import xml.etree.ElementTree as ET
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest  # noqa: E402
from qtpy.QtWidgets import QApplication, QCheckBox  # noqa: E402

from ndviewer_light import LightweightViewer, core  # noqa: E402
from ndviewer_light.core import (  # noqa: E402
    extract_wavelength,
    hex_to_colormap,
    wavelength_to_colormap,
)
from tests.ome_zarr_fixtures import DEFAULT_CHANNELS, write_plate  # noqa: E402


# Test helper to create mock OME metadata XML
def create_ome_metadata(channel_names):
    """Create OME-TIFF metadata XML with channel names."""
    root = ET.Element("{http://www.openmicroscopy.org/Schemas/OME/2016-06}OME")
    image = ET.SubElement(
        root, "{http://www.openmicroscopy.org/Schemas/OME/2016-06}Image"
    )
    pixels = ET.SubElement(
        image, "{http://www.openmicroscopy.org/Schemas/OME/2016-06}Pixels"
    )

    for name in channel_names:
        channel_elem = ET.SubElement(
            pixels, "{http://www.openmicroscopy.org/Schemas/OME/2016-06}Channel"
        )
        channel_elem.set("Name", name)

    return ET.tostring(root, encoding="unicode")


class TestOMETiffChannelNamesLogic:
    """Test channel name extraction and preservation logic (isolated from Qt)."""

    def test_empty_channel_names_fallback_to_default(self):
        """When channel_names is empty, should fallback to Ch0, Ch1, etc."""
        n_c = 3
        channel_names = []

        # Simulate the channel name adjustment logic from _load_ome_tiff
        if not channel_names:
            channel_names = [f"Ch{i}" for i in range(n_c)]
        elif len(channel_names) < n_c:
            channel_names.extend(f"Ch{i}" for i in range(len(channel_names), n_c))
        elif len(channel_names) > n_c:
            channel_names = channel_names[:n_c]

        assert channel_names == ["Ch0", "Ch1", "Ch2"]
        assert len(channel_names) == n_c

    def test_fewer_channel_names_extends_with_fallbacks(self):
        """When channel_names has fewer than n_c, extend with fallbacks (Ch2, Ch3, etc.)."""
        n_c = 5
        channel_names = ["DAPI", "GFP"]

        # Simulate the channel name adjustment logic from _load_ome_tiff
        if not channel_names:
            channel_names = [f"Ch{i}" for i in range(n_c)]
        elif len(channel_names) < n_c:
            channel_names.extend(f"Ch{i}" for i in range(len(channel_names), n_c))
        elif len(channel_names) > n_c:
            channel_names = channel_names[:n_c]

        assert channel_names == ["DAPI", "GFP", "Ch2", "Ch3", "Ch4"]
        assert len(channel_names) == n_c

    def test_more_channel_names_truncates(self):
        """When channel_names has more than n_c, truncate to match n_c."""
        n_c = 2
        channel_names = ["DAPI", "GFP", "RFP", "Cy5", "Cy7"]

        # Simulate the channel name adjustment logic from _load_ome_tiff
        if not channel_names:
            channel_names = [f"Ch{i}" for i in range(n_c)]
        elif len(channel_names) < n_c:
            channel_names.extend(f"Ch{i}" for i in range(len(channel_names), n_c))
        elif len(channel_names) > n_c:
            channel_names = channel_names[:n_c]

        assert channel_names == ["DAPI", "GFP"]
        assert len(channel_names) == n_c

    def test_exact_match_preserves_all(self):
        """When channel_names matches n_c exactly, preserve all names."""
        n_c = 3
        channel_names = ["DAPI", "GFP", "RFP"]

        # Simulate the channel name adjustment logic from _load_ome_tiff
        if not channel_names:
            channel_names = [f"Ch{i}" for i in range(n_c)]
        elif len(channel_names) < n_c:
            channel_names.extend(f"Ch{i}" for i in range(len(channel_names), n_c))
        elif len(channel_names) > n_c:
            channel_names = channel_names[:n_c]

        assert channel_names == ["DAPI", "GFP", "RFP"]
        assert len(channel_names) == n_c

    def test_ome_metadata_parsing(self):
        """Test parsing channel names from OME metadata XML."""
        provided_names = ["DAPI", "GFP", "RFP"]
        ome_metadata = create_ome_metadata(provided_names)

        # Simulate the OME metadata parsing logic from _load_ome_tiff
        channel_names = []
        try:
            if ome_metadata:
                root = ET.fromstring(ome_metadata)
                ns = {"ome": "http://www.openmicroscopy.org/Schemas/OME/2016-06"}
                for ch in root.findall(".//ome:Channel", ns):
                    name = ch.get("Name") or ch.get("ID", "")
                    if name:
                        channel_names.append(name)
        except Exception:
            pass  # Parsing errors fall through to empty channel_names

        assert channel_names == provided_names

    def test_ome_metadata_parsing_with_invalid_xml(self):
        """Test fallback when OME metadata XML is invalid."""
        ome_metadata = "invalid xml <>"
        n_c = 3

        # Simulate the parsing logic with error handling
        channel_names = []
        try:
            if ome_metadata:
                root = ET.fromstring(ome_metadata)
                ns = {"ome": "http://www.openmicroscopy.org/Schemas/OME/2016-06"}
                for ch in root.findall(".//ome:Channel", ns):
                    name = ch.get("Name") or ch.get("ID", "")
                    if name:
                        channel_names.append(name)
        except Exception:
            pass  # Invalid XML expected - testing fallback behavior

        # Apply fallback logic
        if not channel_names:
            channel_names = [f"Ch{i}" for i in range(n_c)]

        assert channel_names == ["Ch0", "Ch1", "Ch2"]

    def test_ome_metadata_parsing_with_empty_names(self):
        """Test handling of channels with empty or missing Name attributes."""
        # Create XML with one channel missing Name attribute
        root = ET.Element("{http://www.openmicroscopy.org/Schemas/OME/2016-06}OME")
        image = ET.SubElement(
            root, "{http://www.openmicroscopy.org/Schemas/OME/2016-06}Image"
        )
        pixels = ET.SubElement(
            image, "{http://www.openmicroscopy.org/Schemas/OME/2016-06}Pixels"
        )

        # Add channels with various attributes
        ch1 = ET.SubElement(
            pixels, "{http://www.openmicroscopy.org/Schemas/OME/2016-06}Channel"
        )
        ch1.set("Name", "DAPI")

        # Channel without Name attribute (intentionally unnamed)
        _ = ET.SubElement(
            pixels, "{http://www.openmicroscopy.org/Schemas/OME/2016-06}Channel"
        )

        ch3 = ET.SubElement(
            pixels, "{http://www.openmicroscopy.org/Schemas/OME/2016-06}Channel"
        )
        ch3.set("Name", "GFP")

        ome_metadata = ET.tostring(root, encoding="unicode")

        # Parse channel names
        channel_names = []
        try:
            if ome_metadata:
                root = ET.fromstring(ome_metadata)
                ns = {"ome": "http://www.openmicroscopy.org/Schemas/OME/2016-06"}
                for ch in root.findall(".//ome:Channel", ns):
                    name = ch.get("Name") or ch.get("ID", "")
                    if name:  # Only append if name is not empty
                        channel_names.append(name)
        except Exception:
            pass  # Parsing errors fall through to partial channel_names

        # Only channels with names should be included
        assert channel_names == ["DAPI", "GFP"]


class TestSingleTiffChannelNamesLogic:
    """Test channel name extraction logic in single-TIFF path."""

    def test_channel_names_sorted_alphabetically(self):
        """Channel names from filenames should be sorted alphabetically."""
        # Simulate discovering channels from filenames in _load_single_tiff
        c_set = {"RFP", "DAPI", "GFP", "Cy5"}
        channel_names = sorted(c_set)

        expected = ["Cy5", "DAPI", "GFP", "RFP"]
        assert channel_names == expected

    def test_channel_names_extracted_from_set(self):
        """Channel names should be extracted into a sorted list."""
        # Simulate the logic from _load_single_tiff
        c_set = set()
        # Add channels as discovered from filenames
        c_set.add("DAPI")
        c_set.add("GFP")
        c_set.add("RFP")

        channel_names = sorted(c_set)

        assert channel_names == ["DAPI", "GFP", "RFP"]
        assert len(channel_names) == 3

    def test_channel_names_stored_in_xarray_attrs(self):
        """Channel names should be stored in xarray attrs."""
        # Tests channel_names storage in xarray attrs (in _load_single_tiff)
        channel_names = ["DAPI", "GFP", "RFP"]

        # Simulate creating xarray attrs
        attrs = {"luts": {}, "channel_names": channel_names}

        assert attrs["channel_names"] == channel_names
        assert len(attrs["channel_names"]) == 3


class TestChannelNamesIntegration:
    """Integration tests for channel name handling."""

    def test_channel_names_length_consistency(self):
        """Channel names length should match channel dimension after processing."""
        n_c = 4
        initial_names = ["DAPI", "GFP"]

        # Apply the adjustment logic
        channel_names = initial_names.copy()
        if len(channel_names) < n_c:
            channel_names.extend(f"Ch{i}" for i in range(len(channel_names), n_c))

        # Verify consistency
        assert len(channel_names) == n_c
        assert channel_names == ["DAPI", "GFP", "Ch2", "Ch3"]

    def test_channel_names_default_format(self):
        """Default channel names should follow Ch0, Ch1, Ch2... format."""
        n_c = 5
        channel_names = [f"Ch{i}" for i in range(n_c)]

        assert channel_names == ["Ch0", "Ch1", "Ch2", "Ch3", "Ch4"]

    def test_empty_list_becomes_defaults(self):
        """Empty channel names list should be replaced with defaults."""
        n_c = 3
        channel_names = []

        if not channel_names:
            channel_names = [f"Ch{i}" for i in range(n_c)]

        assert channel_names == ["Ch0", "Ch1", "Ch2"]
        assert len(channel_names) == n_c

    def test_none_becomes_defaults(self):
        """None channel names should be replaced with defaults."""
        n_c = 3
        channel_names = None

        if not channel_names:
            channel_names = [f"Ch{i}" for i in range(n_c)]

        assert channel_names == ["Ch0", "Ch1", "Ch2"]

    def test_preserves_custom_names_when_exact_match(self):
        """Custom channel names should be preserved when count matches."""
        n_c = 3
        channel_names = ["405nm_DAPI", "488nm_GFP", "561nm_RFP"]

        # Apply adjustment (should be no-op)
        if not channel_names:
            channel_names = [f"Ch{i}" for i in range(n_c)]
        elif len(channel_names) < n_c:
            channel_names.extend(f"Ch{i}" for i in range(len(channel_names), n_c))
        elif len(channel_names) > n_c:
            channel_names = channel_names[:n_c]

        assert channel_names == ["405nm_DAPI", "488nm_GFP", "561nm_RFP"]


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def model_lut_names(viewer: LightweightViewer, n_channels: int) -> list:
    """Display names ndv holds for channel positions 0..n-1 (public LUTModel.name)."""
    luts = viewer.ndv_viewer.display_model.luts
    return [luts[i].name for i in range(n_channels)]


class TestChannelNamesReachNdv:
    """Channel names flow into ndv through LUTModel.name, set at construction.

    ndv 0.5 labels each LUT row from the public ``LUTModel.name`` field and falls
    back to the channel coordinate ("0", "1", ...) when it is empty. ndviewer_light
    keeps integer channel coordinates, so the name must be in the ``luts`` passed to
    ``ArrayViewer``. The viewer tests run offscreen (conftest pins the GL texture
    query); only the checkbox test needs the event loop and polls for it.
    """

    def test_build_luts_pairs_name_with_cmap(self):
        from ndviewer_light.core import build_luts

        assert build_luts(["BF", "GFP 488"]) == {
            0: {"cmap": wavelength_to_colormap(None), "name": "BF"},
            1: {"cmap": wavelength_to_colormap(488), "name": "GFP 488"},
        }
        assert build_luts(["A"], ["#00ff00"])[0]["cmap"] == hex_to_colormap("#00ff00")
        # Missing or empty colour entries fall back to the wavelength colormap
        assert build_luts(["GFP", "Cy5"], [""])[0]["cmap"] == wavelength_to_colormap(
            extract_wavelength("GFP")
        )
        assert build_luts(["GFP", "Cy5"], [""])[1]["cmap"] == wavelength_to_colormap(
            extract_wavelength("Cy5")
        )

    def test_push_start_names_lut_models(self, qapp):
        v = LightweightViewer()
        v.start_acquisition(["BF", "GFP"], 1, 8, 8, ["A1:0"])
        assert model_lut_names(v, 2) == ["BF", "GFP"]

    def test_zarr_plate_names_lut_models(self, qapp, tmp_path):
        write_plate(tmp_path / "plate.ome.zarr", {"B3": [0]}, rows=["B"], cols=[3])
        v = LightweightViewer()
        v.load_dataset(str(tmp_path / "plate.ome.zarr"))
        assert model_lut_names(v, len(DEFAULT_CHANNELS)) == DEFAULT_CHANNELS

    def test_lut_checkbox_text_matches_names(self, qapp):
        """The LUT rows ndv creates on its first data response carry the names."""
        names = ["BF", "GFP"]
        v = LightweightViewer()
        v.start_acquisition(names, 1, 8, 8, ["A1:0"])
        deadline = time.monotonic() + 3.0
        texts = []
        while time.monotonic() < deadline:
            qapp.processEvents()
            texts = [
                cb.text()
                for cb in v.ndv_viewer.widget().findChildren(QCheckBox)
                if cb.text() in names
            ]
            if texts == names:
                break
        assert texts == names

    def test_core_has_no_private_lut_controller_access(self):
        assert "_lut_controllers" not in Path(core.__file__).read_text(encoding="utf-8")
