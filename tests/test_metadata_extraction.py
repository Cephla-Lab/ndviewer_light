"""Tests for acquisition metadata extraction functions.

Tests verify that physical pixel sizes are correctly extracted from:
- OME-TIFF metadata (PhysicalSizeX/Y/Z attributes)
- acquisition_parameters.json files
"""

import json
import tempfile
from pathlib import Path

import pytest

from ndviewer_light import extract_ome_physical_sizes, read_acquisition_parameters


class TestExtractOmePhysicalSizes:
    """Test suite for OME-TIFF physical size extraction."""

    def test_basic_ome_metadata(self):
        """Test extraction from standard OME-XML with 2016-06 namespace."""
        ome_xml = """<?xml version="1.0" encoding="UTF-8"?>
        <OME xmlns="http://www.openmicroscopy.org/Schemas/OME/2016-06">
            <Image ID="Image:0">
                <Pixels ID="Pixels:0" DimensionOrder="XYZCT" Type="uint16"
                        SizeX="1024" SizeY="1024" SizeZ="50" SizeC="2" SizeT="1"
                        PhysicalSizeX="0.325" PhysicalSizeY="0.325" PhysicalSizeZ="1.5"
                        PhysicalSizeXUnit="µm" PhysicalSizeYUnit="µm" PhysicalSizeZUnit="µm">
                </Pixels>
            </Image>
        </OME>"""

        px, py, pz = extract_ome_physical_sizes(ome_xml)

        assert px == pytest.approx(0.325)
        assert py == pytest.approx(0.325)
        assert pz == pytest.approx(1.5)

    def test_ome_metadata_with_nm_units(self):
        """Test extraction with nanometer units (converts to micrometers)."""
        ome_xml = """<?xml version="1.0" encoding="UTF-8"?>
        <OME xmlns="http://www.openmicroscopy.org/Schemas/OME/2016-06">
            <Image ID="Image:0">
                <Pixels ID="Pixels:0" DimensionOrder="XYZCT" Type="uint16"
                        SizeX="1024" SizeY="1024" SizeZ="50" SizeC="1" SizeT="1"
                        PhysicalSizeX="325" PhysicalSizeY="325" PhysicalSizeZ="1500"
                        PhysicalSizeXUnit="nm" PhysicalSizeYUnit="nm" PhysicalSizeZUnit="nm">
                </Pixels>
            </Image>
        </OME>"""

        px, py, pz = extract_ome_physical_sizes(ome_xml)

        assert px == pytest.approx(0.325)
        assert py == pytest.approx(0.325)
        assert pz == pytest.approx(1.5)

    def test_ome_metadata_default_units(self):
        """Test extraction without explicit units (assumes micrometers)."""
        ome_xml = """<?xml version="1.0" encoding="UTF-8"?>
        <OME xmlns="http://www.openmicroscopy.org/Schemas/OME/2016-06">
            <Image ID="Image:0">
                <Pixels ID="Pixels:0" DimensionOrder="XYZCT" Type="uint16"
                        SizeX="512" SizeY="512" SizeZ="20" SizeC="1" SizeT="1"
                        PhysicalSizeX="0.65" PhysicalSizeY="0.65" PhysicalSizeZ="2.0">
                </Pixels>
            </Image>
        </OME>"""

        px, py, pz = extract_ome_physical_sizes(ome_xml)

        assert px == pytest.approx(0.65)
        assert py == pytest.approx(0.65)
        assert pz == pytest.approx(2.0)

    def test_ome_metadata_partial_sizes(self):
        """Test extraction when only some physical sizes are present."""
        ome_xml = """<?xml version="1.0" encoding="UTF-8"?>
        <OME xmlns="http://www.openmicroscopy.org/Schemas/OME/2016-06">
            <Image ID="Image:0">
                <Pixels ID="Pixels:0" DimensionOrder="XYZCT" Type="uint16"
                        SizeX="1024" SizeY="1024" SizeZ="1" SizeC="1" SizeT="1"
                        PhysicalSizeX="0.5" PhysicalSizeY="0.5">
                </Pixels>
            </Image>
        </OME>"""

        px, py, pz = extract_ome_physical_sizes(ome_xml)

        assert px == pytest.approx(0.5)
        assert py == pytest.approx(0.5)
        assert pz is None

    def test_ome_metadata_no_physical_sizes(self):
        """Test extraction when no physical sizes are present."""
        ome_xml = """<?xml version="1.0" encoding="UTF-8"?>
        <OME xmlns="http://www.openmicroscopy.org/Schemas/OME/2016-06">
            <Image ID="Image:0">
                <Pixels ID="Pixels:0" DimensionOrder="XYZCT" Type="uint16"
                        SizeX="1024" SizeY="1024" SizeZ="1" SizeC="1" SizeT="1">
                </Pixels>
            </Image>
        </OME>"""

        px, py, pz = extract_ome_physical_sizes(ome_xml)

        assert px is None
        assert py is None
        assert pz is None

    def test_empty_metadata(self):
        """Test with empty or None metadata."""
        px, py, pz = extract_ome_physical_sizes("")
        assert px is None
        assert py is None
        assert pz is None

        px, py, pz = extract_ome_physical_sizes(None)
        assert px is None
        assert py is None
        assert pz is None

    def test_invalid_xml(self):
        """Test with invalid XML (should not raise, returns None)."""
        px, py, pz = extract_ome_physical_sizes("not valid xml <><>")
        assert px is None
        assert py is None
        assert pz is None


class TestReadAcquisitionParameters:
    """Test suite for acquisition_parameters.json reading."""

    def test_basic_parameters(self):
        """Test reading standard acquisition parameters."""
        with tempfile.TemporaryDirectory() as tmpdir:
            params = {"pixel_size_um": 0.325, "dz_um": 1.5}
            params_file = Path(tmpdir) / "acquisition_parameters.json"
            with open(params_file, "w") as f:
                json.dump(params, f)

            pixel_size, dz = read_acquisition_parameters(Path(tmpdir))

            assert pixel_size == pytest.approx(0.325)
            assert dz == pytest.approx(1.5)

    def test_alternative_key_names(self):
        """Test reading with alternative key names."""
        with tempfile.TemporaryDirectory() as tmpdir:
            params = {"pixel_size": 0.5, "z_step": 2.0}
            params_file = Path(tmpdir) / "acquisition_parameters.json"
            with open(params_file, "w") as f:
                json.dump(params, f)

            pixel_size, dz = read_acquisition_parameters(Path(tmpdir))

            assert pixel_size == pytest.approx(0.5)
            assert dz == pytest.approx(2.0)

    def test_partial_parameters(self):
        """Test when only some parameters are present."""
        with tempfile.TemporaryDirectory() as tmpdir:
            params = {"pixel_size_um": 0.65}
            params_file = Path(tmpdir) / "acquisition_parameters.json"
            with open(params_file, "w") as f:
                json.dump(params, f)

            pixel_size, dz = read_acquisition_parameters(Path(tmpdir))

            assert pixel_size == pytest.approx(0.65)
            assert dz is None

    def test_no_file(self):
        """Test when acquisition_parameters.json doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            pixel_size, dz = read_acquisition_parameters(Path(tmpdir))

            assert pixel_size is None
            assert dz is None

    def test_empty_file(self):
        """Test with empty JSON object."""
        with tempfile.TemporaryDirectory() as tmpdir:
            params_file = Path(tmpdir) / "acquisition_parameters.json"
            with open(params_file, "w") as f:
                json.dump({}, f)

            pixel_size, dz = read_acquisition_parameters(Path(tmpdir))

            assert pixel_size is None
            assert dz is None
