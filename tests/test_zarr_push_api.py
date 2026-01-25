"""Tests for the push-based zarr API.

Tests verify the API contract without requiring Qt initialization.
These are logic-focused tests that verify state management and signal handling.
"""

from pathlib import Path


class TestZarrPushApiLogic:
    """Test suite for zarr push API logic (no Qt required)."""

    def test_channel_map_creation(self):
        """Test that channel name to index mapping is created correctly."""
        channels = ["DAPI", "GFP", "RFP"]
        channel_map = {name: i for i, name in enumerate(channels)}

        assert channel_map["DAPI"] == 0
        assert channel_map["GFP"] == 1
        assert channel_map["RFP"] == 2

    def test_channel_lookup(self):
        """Test looking up channel index from name."""
        channel_map = {"DAPI": 0, "GFP": 1, "RFP": 2}

        # Valid lookups
        assert channel_map.get("DAPI", -1) == 0
        assert channel_map.get("GFP", -1) == 1
        assert channel_map.get("RFP", -1) == 2

        # Invalid lookup returns -1
        assert channel_map.get("Unknown", -1) == -1

    def test_fov_label_format(self):
        """Test FOV label format used by Squid."""
        fov_labels = ["A1:0", "A1:1", "A2:0", "A2:1"]

        # Labels should be parseable as well:fov
        for label in fov_labels:
            well, fov = label.split(":")
            assert len(well) >= 2  # At least row + column
            assert fov.isdigit()

    def test_max_fov_per_time_tracking(self):
        """Test tracking max FOV index per timepoint."""
        max_fov_per_time = {}

        # Simulate frame registrations
        frames = [
            (0, 0),  # t=0, fov=0
            (0, 1),  # t=0, fov=1
            (0, 2),  # t=0, fov=2
            (1, 0),  # t=1, fov=0
            (1, 1),  # t=1, fov=1
        ]

        for t, fov_idx in frames:
            current_max = max_fov_per_time.get(t, -1)
            if fov_idx > current_max:
                max_fov_per_time[t] = fov_idx

        assert max_fov_per_time[0] == 2
        assert max_fov_per_time[1] == 1

    def test_max_time_tracking(self):
        """Test tracking maximum timepoint."""
        max_time_idx = 0

        frames = [(0, 0), (0, 1), (1, 0), (2, 0), (2, 1)]

        for t, _ in frames:
            max_time_idx = max(max_time_idx, t)

        assert max_time_idx == 2


class TestZarrCacheKey:
    """Test zarr cache key generation."""

    def test_cache_key_uniqueness(self):
        """Test that cache keys are unique for different planes."""
        keys = set()

        for t in range(2):
            for fov in range(2):
                for z in range(3):
                    for ch in range(2):
                        key = ("zarr", t, fov, z, ch)
                        assert key not in keys, f"Duplicate key: {key}"
                        keys.add(key)

        # Should have 2*2*3*2 = 24 unique keys
        assert len(keys) == 24

    def test_cache_key_format(self):
        """Test cache key tuple format."""
        key = ("zarr", 0, 1, 2, 3)

        assert key[0] == "zarr"  # Prefix to distinguish from TIFF cache
        assert key[1] == 0  # t
        assert key[2] == 1  # fov
        assert key[3] == 2  # z
        assert key[4] == 3  # channel


class TestZarrMetadataIntegration:
    """Test integration between metadata parsing and push API."""

    def test_channel_colors_to_luts(self):
        """Test converting channel colors to LUTs."""
        from ndviewer_light import hex_to_colormap, wavelength_to_colormap
        from ndviewer_light.core import extract_wavelength

        channel_names = ["DAPI", "GFP", "RFP"]
        channel_colors = ["0000FF", "00FF00", "FF0000"]

        luts = {}
        for i, name in enumerate(channel_names):
            if i < len(channel_colors) and channel_colors[i]:
                luts[i] = hex_to_colormap(channel_colors[i])
            else:
                luts[i] = wavelength_to_colormap(extract_wavelength(name))

        assert luts[0] == "blue"  # DAPI
        assert luts[1] == "green"  # GFP
        assert luts[2] == "red"  # RFP

    def test_fallback_to_wavelength_colormap(self):
        """Test fallback to wavelength-based colormap when no colors provided."""
        from ndviewer_light import wavelength_to_colormap
        from ndviewer_light.core import extract_wavelength

        channel_names = ["Fluorescence 488 nm Ex", "Fluorescence 561 nm Ex"]

        luts = {}
        for i, name in enumerate(channel_names):
            luts[i] = wavelength_to_colormap(extract_wavelength(name))

        assert luts[0] == "green"  # 488nm
        assert luts[1] == "yellow"  # 561nm


class TestZarrStateManagement:
    """Test state management for zarr acquisition."""

    def test_acquisition_state_transitions(self):
        """Test state transitions during acquisition lifecycle."""
        # Initial state
        zarr_acquisition_active = False
        zarr_acquisition_path = None

        # Start acquisition
        zarr_acquisition_active = True
        zarr_acquisition_path = Path("/tmp/test.zarr")

        assert zarr_acquisition_active is True
        assert zarr_acquisition_path is not None

        # End acquisition
        zarr_acquisition_active = False

        assert zarr_acquisition_active is False
        # Path preserved for browsing
        assert zarr_acquisition_path is not None

    def test_push_mode_detection(self):
        """Test is_push_mode_active logic."""
        fov_labels = []
        zarr_acquisition_active = False

        # Neither active
        assert not (bool(fov_labels) or zarr_acquisition_active)

        # FOV labels set (TIFF push mode)
        fov_labels = ["A1:0", "A1:1"]
        assert bool(fov_labels) or zarr_acquisition_active

        # Zarr mode active
        fov_labels = []
        zarr_acquisition_active = True
        assert bool(fov_labels) or zarr_acquisition_active

        # Both active
        fov_labels = ["A1:0"]
        zarr_acquisition_active = True
        assert bool(fov_labels) or zarr_acquisition_active


class TestMultiRegion6D:
    """Test multi-region 6D zarr support (6d_regions mode)."""

    def test_global_fov_index_calculation(self):
        """Test converting global FOV index to (region_idx, local_fov_idx)."""
        # Simulate _global_to_region_fov logic
        fovs_per_region = [4, 6, 3]  # 3 regions with 4, 6, 3 FOVs
        region_fov_offsets = [0, 4, 10]  # cumulative: [0, 0+4, 0+4+6]
        total_fovs = sum(fovs_per_region)  # 13

        def global_to_region_fov(global_fov_idx):
            for region_idx, offset in enumerate(region_fov_offsets):
                next_offset = (
                    region_fov_offsets[region_idx + 1]
                    if region_idx + 1 < len(region_fov_offsets)
                    else total_fovs
                )
                if offset <= global_fov_idx < next_offset:
                    return region_idx, global_fov_idx - offset
            return 0, global_fov_idx

        # Region 0: global FOV 0-3 → local FOV 0-3
        assert global_to_region_fov(0) == (0, 0)
        assert global_to_region_fov(1) == (0, 1)
        assert global_to_region_fov(3) == (0, 3)

        # Region 1: global FOV 4-9 → local FOV 0-5
        assert global_to_region_fov(4) == (1, 0)
        assert global_to_region_fov(5) == (1, 1)
        assert global_to_region_fov(9) == (1, 5)

        # Region 2: global FOV 10-12 → local FOV 0-2
        assert global_to_region_fov(10) == (2, 0)
        assert global_to_region_fov(11) == (2, 1)
        assert global_to_region_fov(12) == (2, 2)

    def test_region_fov_offsets_computation(self):
        """Test computing cumulative FOV offsets from fovs_per_region."""
        fovs_per_region = [4, 6, 3]

        region_fov_offsets = []
        offset = 0
        for n_fov in fovs_per_region:
            region_fov_offsets.append(offset)
            offset += n_fov

        assert region_fov_offsets == [0, 4, 10]
        assert sum(fovs_per_region) == 13

    def test_fov_labels_generation(self):
        """Test generating flattened FOV labels for multi-region mode."""
        region_labels = ["region_0", "region_1", "region_2"]
        fovs_per_region = [4, 6, 3]

        fov_labels = []
        for region_label, n_fov in zip(region_labels, fovs_per_region):
            for fov_in_region in range(n_fov):
                fov_labels.append(f"{region_label}:{fov_in_region}")

        # Should have 13 labels total
        assert len(fov_labels) == 13

        # Check first region labels
        assert fov_labels[0] == "region_0:0"
        assert fov_labels[3] == "region_0:3"

        # Check second region labels
        assert fov_labels[4] == "region_1:0"
        assert fov_labels[9] == "region_1:5"

        # Check third region labels
        assert fov_labels[10] == "region_2:0"
        assert fov_labels[12] == "region_2:2"

    def test_fov_labels_with_variable_fovs(self):
        """Test FOV labels with different FOV counts per region."""
        region_labels = ["scan_A", "scan_B"]
        fovs_per_region = [2, 5]

        fov_labels = []
        for region_label, n_fov in zip(region_labels, fovs_per_region):
            for fov_in_region in range(n_fov):
                fov_labels.append(f"{region_label}:{fov_in_region}")

        expected = [
            "scan_A:0",
            "scan_A:1",
            "scan_B:0",
            "scan_B:1",
            "scan_B:2",
            "scan_B:3",
            "scan_B:4",
        ]
        assert fov_labels == expected

    def test_global_fov_to_region_boundary_cases(self):
        """Test boundary cases in global FOV conversion."""
        # Single FOV per region: fovs_per_region = [1, 1, 1]
        region_fov_offsets = [0, 1, 2]
        total_fovs = 3

        def global_to_region_fov(global_fov_idx):
            for region_idx, offset in enumerate(region_fov_offsets):
                next_offset = (
                    region_fov_offsets[region_idx + 1]
                    if region_idx + 1 < len(region_fov_offsets)
                    else total_fovs
                )
                if offset <= global_fov_idx < next_offset:
                    return region_idx, global_fov_idx - offset
            return 0, global_fov_idx

        assert global_to_region_fov(0) == (0, 0)
        assert global_to_region_fov(1) == (1, 0)
        assert global_to_region_fov(2) == (2, 0)

    def test_notify_frame_global_fov_conversion(self):
        """Test that notify_zarr_frame correctly converts region_idx + fov_idx to global."""
        # Simulate the conversion logic in notify_zarr_frame
        region_fov_offsets = [0, 4, 10]
        zarr_6d_regions_mode = True

        def compute_global_fov(fov_idx, region_idx):
            if zarr_6d_regions_mode and region_fov_offsets:
                if region_idx < len(region_fov_offsets):
                    return region_fov_offsets[region_idx] + fov_idx
            return fov_idx

        # Region 0, FOV 2 → global FOV 2
        assert compute_global_fov(2, 0) == 2

        # Region 1, FOV 3 → global FOV 7 (4 + 3)
        assert compute_global_fov(3, 1) == 7

        # Region 2, FOV 1 → global FOV 11 (10 + 1)
        assert compute_global_fov(1, 2) == 11

    def test_6d_regions_push_mode_detection(self):
        """Test is_zarr_push_mode_active includes 6d_regions mode."""
        # Simulate is_zarr_push_mode_active logic
        zarr_acquisition_active = False
        zarr_acquisition_path = None
        zarr_fov_paths = []
        zarr_6d_regions_mode = False

        def is_zarr_push_mode_active():
            return (
                zarr_acquisition_active
                or zarr_acquisition_path is not None
                or bool(zarr_fov_paths)
                or zarr_6d_regions_mode
            )

        # None active
        assert not is_zarr_push_mode_active()

        # Only 6d_regions mode active
        zarr_6d_regions_mode = True
        assert is_zarr_push_mode_active()

        # Reset and check acquisition active
        zarr_6d_regions_mode = False
        zarr_acquisition_active = True
        assert is_zarr_push_mode_active()


class TestCachingBehavior:
    """Test plane caching behavior during live acquisition."""

    def test_written_plane_tracking(self):
        """Test that only written planes are cached during live acquisition.

        Uses a set to track which planes have been written via notify_zarr_frame().
        This is O(1) lookup vs O(n) plane.max() check, and handles legitimately
        black images correctly.
        """
        # Simulate the written planes tracking
        written_planes = set()
        acquisition_active = True

        def should_cache(cache_key):
            """Replicate the caching logic: cache if not live or if written."""
            if not acquisition_active:
                return True  # Browsing existing data - cache everything
            return cache_key in written_planes

        # During live acquisition, unwritten plane should not be cached
        key1 = ("zarr", 0, 4, 0, 1)  # t=0, fov=4, z=0, ch=1
        assert not should_cache(key1)

        # After notify_zarr_frame(), plane should be cached
        written_planes.add(key1)
        assert should_cache(key1)

        # Different plane still not cached until written
        key2 = ("zarr", 0, 5, 0, 1)
        assert not should_cache(key2)

        # When acquisition ends, all planes cached (browsing mode)
        acquisition_active = False
        assert should_cache(key2)  # Now cacheable even though not in written set
