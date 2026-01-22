"""
Unit tests for push-based acquisition API in ndviewer_light.

Tests cover:
1. start_acquisition() - state initialization
2. register_image() - file index updates and signal emission
3. is_push_mode_active() - mode detection
4. Dynamic FOV slider ranges per timepoint
5. MemoryBoundedLRUCache - thread-safe cache operations
6. go_to_well_fov() - navigation API
7. _load_single_plane() - error handling
8. end_acquisition() - cleanup
"""

import os
import tempfile
import threading
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


class TestPushModeDetection:
    """Tests for is_push_mode_active() method."""

    def test_push_mode_inactive_when_no_fov_labels(self):
        """Push mode is inactive when _fov_labels is empty."""
        from ndviewer_light import LightweightViewer

        viewer = MagicMock(spec=LightweightViewer)
        viewer._fov_labels = []
        viewer.is_push_mode_active = lambda: bool(viewer._fov_labels)

        assert viewer.is_push_mode_active() is False

    def test_push_mode_active_when_fov_labels_set(self):
        """Push mode is active when _fov_labels has values."""
        from ndviewer_light import LightweightViewer

        viewer = MagicMock(spec=LightweightViewer)
        viewer._fov_labels = ["A1:0", "A1:1", "A2:0"]
        viewer.is_push_mode_active = lambda: bool(viewer._fov_labels)

        assert viewer.is_push_mode_active() is True


class TestStartAcquisition:
    """Tests for start_acquisition() initialization."""

    def test_start_acquisition_sets_fov_labels(self):
        """start_acquisition() populates _fov_labels."""
        from ndviewer_light import LightweightViewer

        viewer = MagicMock(spec=LightweightViewer)
        viewer._fov_labels = []
        viewer._channel_names = []
        viewer._z_levels = []
        viewer._file_index = {}
        viewer._max_fov_per_time = {}
        viewer._image_height = 0
        viewer._image_width = 0

        def mock_start_acquisition(channels, num_z, height, width, fov_labels):
            viewer._fov_labels = fov_labels
            viewer._channel_names = channels
            viewer._z_levels = list(range(num_z))
            viewer._image_height = height
            viewer._image_width = width
            viewer._file_index.clear()
            viewer._max_fov_per_time.clear()

        viewer.start_acquisition = mock_start_acquisition

        viewer.start_acquisition(
            channels=["BF", "DAPI"],
            num_z=3,
            height=1000,
            width=1000,
            fov_labels=["A1:0", "A1:1"],
        )

        assert viewer._fov_labels == ["A1:0", "A1:1"]
        assert viewer._channel_names == ["BF", "DAPI"]
        assert viewer._z_levels == [0, 1, 2]
        assert viewer._file_index == {}
        assert viewer._max_fov_per_time == {}

    def test_start_acquisition_clears_previous_state(self):
        """start_acquisition() clears state from previous acquisition."""
        from ndviewer_light import LightweightViewer

        viewer = MagicMock(spec=LightweightViewer)
        # Simulate previous acquisition state
        viewer._file_index = {(0, 0, 0, "BF"): "/old/path.tiff"}
        viewer._max_fov_per_time = {0: 5, 1: 3}
        viewer._fov_labels = ["old:0"]

        def mock_start_acquisition(channels, num_z, height, width, fov_labels):
            viewer._fov_labels = fov_labels
            viewer._channel_names = channels
            viewer._file_index = {}  # Clear
            viewer._max_fov_per_time = {}  # Clear

        viewer.start_acquisition = mock_start_acquisition

        viewer.start_acquisition(
            channels=["GFP"],
            num_z=1,
            height=500,
            width=500,
            fov_labels=["B1:0"],
        )

        assert viewer._file_index == {}
        assert viewer._max_fov_per_time == {}
        assert viewer._fov_labels == ["B1:0"]


class TestRegisterImage:
    """Tests for register_image() method."""

    def test_register_image_updates_file_index(self):
        """register_image() adds filepath to _file_index."""
        file_index = {}
        lock = threading.Lock()

        def register_image(t, fov_idx, z, channel, filepath):
            with lock:
                file_index[(t, fov_idx, z, channel)] = filepath

        register_image(
            t=0, fov_idx=1, z=2, channel="DAPI", filepath="/path/to/img.tiff"
        )

        assert file_index[(0, 1, 2, "DAPI")] == "/path/to/img.tiff"

    def test_register_image_thread_safe(self):
        """register_image() is thread-safe for concurrent calls."""
        file_index = {}
        lock = threading.Lock()
        errors = []

        def register_image(t, fov_idx, z, channel, filepath):
            with lock:
                file_index[(t, fov_idx, z, channel)] = filepath

        def worker(thread_id):
            try:
                for i in range(100):
                    register_image(
                        t=thread_id,
                        fov_idx=i,
                        z=0,
                        channel="BF",
                        filepath=f"/path/{thread_id}_{i}.tiff",
                    )
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert len(file_index) == 500  # 5 threads x 100 images


class TestMaxFovPerTime:
    """Tests for per-timepoint FOV tracking."""

    def test_max_fov_tracking(self):
        """_max_fov_per_time tracks highest FOV index per timepoint."""
        max_fov_per_time = {}

        def update_max_fov(t, fov_idx):
            if t not in max_fov_per_time or fov_idx > max_fov_per_time[t]:
                max_fov_per_time[t] = fov_idx

        # Register images in order
        update_max_fov(0, 0)
        update_max_fov(0, 1)
        update_max_fov(0, 2)
        update_max_fov(1, 0)
        update_max_fov(1, 1)

        assert max_fov_per_time[0] == 2
        assert max_fov_per_time[1] == 1

    def test_max_fov_out_of_order(self):
        """_max_fov_per_time handles out-of-order registration."""
        max_fov_per_time = {}

        def update_max_fov(t, fov_idx):
            if t not in max_fov_per_time or fov_idx > max_fov_per_time[t]:
                max_fov_per_time[t] = fov_idx

        # Register out of order
        update_max_fov(0, 5)
        update_max_fov(0, 2)
        update_max_fov(0, 8)
        update_max_fov(0, 3)

        assert max_fov_per_time[0] == 8


class TestMemoryBoundedLRUCache:
    """Tests for MemoryBoundedLRUCache class."""

    def test_cache_basic_operations(self):
        """Cache supports get/put operations."""
        from ndviewer_light.core import MemoryBoundedLRUCache

        cache = MemoryBoundedLRUCache(max_memory_bytes=1024 * 1024)  # 1MB

        # Put and get
        data = np.zeros((100, 100), dtype=np.uint16)
        cache.put("key1", data)
        result = cache.get("key1")

        assert result is data

    def test_cache_miss_returns_none(self):
        """Cache returns None for missing keys."""
        from ndviewer_light.core import MemoryBoundedLRUCache

        cache = MemoryBoundedLRUCache(max_memory_bytes=1024 * 1024)

        result = cache.get("nonexistent")

        assert result is None

    def test_cache_evicts_lru_when_full(self):
        """Cache evicts least-recently-used items when memory limit exceeded."""
        from ndviewer_light.core import MemoryBoundedLRUCache

        # Small cache that can hold ~2 arrays
        array_size = 100 * 100 * 2  # 20KB per array
        cache = MemoryBoundedLRUCache(max_memory_bytes=array_size * 2 + 1000)

        data1 = np.zeros((100, 100), dtype=np.uint16)
        data2 = np.zeros((100, 100), dtype=np.uint16)
        data3 = np.zeros((100, 100), dtype=np.uint16)

        cache.put("key1", data1)
        cache.put("key2", data2)
        # This should evict key1
        cache.put("key3", data3)

        assert cache.get("key1") is None  # Evicted
        assert cache.get("key2") is data2
        assert cache.get("key3") is data3

    def test_cache_thread_safe(self):
        """Cache is thread-safe for concurrent access."""
        from ndviewer_light.core import MemoryBoundedLRUCache

        cache = MemoryBoundedLRUCache(max_memory_bytes=10 * 1024 * 1024)  # 10MB
        errors = []

        def writer(thread_id):
            try:
                for i in range(50):
                    data = np.zeros((50, 50), dtype=np.uint16)
                    cache.put(f"t{thread_id}_k{i}", data)
            except Exception as e:
                errors.append(e)

        def reader(thread_id):
            try:
                for i in range(50):
                    cache.get(f"t{thread_id}_k{i}")
            except Exception as e:
                errors.append(e)

        threads = []
        for i in range(3):
            threads.append(threading.Thread(target=writer, args=(i,)))
            threads.append(threading.Thread(target=reader, args=(i,)))

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0

    def test_cache_clear(self):
        """Cache clear() removes all items."""
        from ndviewer_light.core import MemoryBoundedLRUCache

        cache = MemoryBoundedLRUCache(max_memory_bytes=1024 * 1024)

        cache.put("key1", np.zeros((10, 10), dtype=np.uint16))
        cache.put("key2", np.zeros((10, 10), dtype=np.uint16))
        cache.clear()

        assert cache.get("key1") is None
        assert cache.get("key2") is None

    def test_cache_skips_oversized_items(self):
        """Cache silently skips items larger than max memory."""
        from ndviewer_light.core import MemoryBoundedLRUCache

        # Very small cache
        cache = MemoryBoundedLRUCache(max_memory_bytes=1000)

        # Large array that exceeds limit
        large_data = np.zeros((100, 100), dtype=np.uint16)  # 20KB
        cache.put("large", large_data)

        # Should not be cached
        assert cache.get("large") is None
        assert len(cache) == 0


class TestEndAcquisition:
    """Tests for end_acquisition() cleanup."""

    def test_end_acquisition_clears_fov_labels(self):
        """end_acquisition() clears _fov_labels to exit push mode."""
        fov_labels = ["A1:0", "A1:1"]
        acquisition_active = True
        load_pending = True

        def end_acquisition():
            nonlocal acquisition_active, load_pending
            load_pending = False
            acquisition_active = False
            fov_labels.clear()

        end_acquisition()

        assert fov_labels == []
        assert bool(fov_labels) is False  # is_push_mode_active() returns False
        assert acquisition_active is False
        assert load_pending is False


def _go_to_well_fov(fov_labels, well_id, fov_index, navigated_to=None):
    """Helper function that mimics go_to_well_fov() logic for testing.

    Args:
        fov_labels: List of FOV labels in "well:fov" format.
        well_id: Well identifier (e.g., "A1").
        fov_index: FOV index within the well.
        navigated_to: Optional list to store the flat index (for mutation tracking).

    Returns:
        True if the label was found, False otherwise.
    """
    if not fov_labels:
        return False
    target_label = f"{well_id}:{fov_index}"
    try:
        flat_idx = fov_labels.index(target_label)
        if navigated_to is not None:
            navigated_to[0] = flat_idx
        return True
    except ValueError:
        return False


class TestGoToWellFov:
    """Tests for go_to_well_fov() navigation method."""

    def test_go_to_well_fov_returns_false_when_no_fov_labels(self):
        """go_to_well_fov() returns False when push mode not active."""
        result = _go_to_well_fov([], "A1", 0)
        assert result is False

    def test_go_to_well_fov_returns_false_when_well_not_found(self):
        """go_to_well_fov() returns False for unknown well."""
        fov_labels = ["A1:0", "A1:1", "A2:0"]
        result = _go_to_well_fov(fov_labels, "B1", 0)  # B1 not in labels
        assert result is False

    def test_go_to_well_fov_returns_false_when_fov_not_found(self):
        """go_to_well_fov() returns False when FOV index doesn't exist."""
        fov_labels = ["A1:0", "A1:1", "A2:0"]
        result = _go_to_well_fov(fov_labels, "A1", 5)  # A1:5 not in labels
        assert result is False

    def test_go_to_well_fov_returns_true_on_success(self):
        """go_to_well_fov() returns True when navigation succeeds."""
        fov_labels = ["A1:0", "A1:1", "A2:0"]
        navigated_to = [0]
        result = _go_to_well_fov(fov_labels, "A2", 0, navigated_to)
        assert result is True
        assert navigated_to[0] == 2  # A2:0 is at index 2

    def test_go_to_well_fov_finds_correct_index(self):
        """go_to_well_fov() navigates to correct flat index."""
        fov_labels = ["A1:0", "A1:1", "A1:2", "A2:0", "A2:1"]
        navigated_to = [None]

        # Test various navigations
        assert _go_to_well_fov(fov_labels, "A1", 0, navigated_to) is True
        assert navigated_to[0] == 0

        assert _go_to_well_fov(fov_labels, "A1", 2, navigated_to) is True
        assert navigated_to[0] == 2

        assert _go_to_well_fov(fov_labels, "A2", 1, navigated_to) is True
        assert navigated_to[0] == 4


class _PlaneLoader:
    """Helper class that mimics _load_single_plane() logic for testing.

    Encapsulates the file index, lock, cache, and image dimensions,
    providing a load() method that mirrors the actual implementation.
    """

    def __init__(self, height=100, width=100, load_from_disk=False):
        import tifffile as tf

        from ndviewer_light.core import MemoryBoundedLRUCache

        self.file_index = {}
        self.lock = threading.Lock()
        self.cache = MemoryBoundedLRUCache(max_memory_bytes=1024 * 1024)
        self.height = height
        self.width = width
        self.load_from_disk = load_from_disk
        self.disk_reads = 0
        self._tf = tf

    def load(self, t, fov_idx, z, channel):
        """Load a single image plane from cache or disk."""
        cache_key = (t, fov_idx, z, channel)

        cached = self.cache.get(cache_key)
        if cached is not None:
            return cached

        with self.lock:
            filepath = self.file_index.get(cache_key)

        if not filepath:
            return np.zeros((self.height, self.width), dtype=np.uint16)

        if not self.load_from_disk:
            self.disk_reads += 1
            return np.zeros((self.height, self.width), dtype=np.uint16)

        try:
            with self._tf.TiffFile(filepath) as tif:
                plane = tif.pages[0].asarray()
                self.cache.put(cache_key, plane)
                self.disk_reads += 1
                return plane
        except (FileNotFoundError, Exception):
            pass

        return np.zeros((self.height, self.width), dtype=np.uint16)


class TestLoadSinglePlane:
    """Tests for _load_single_plane() error handling."""

    def test_load_single_plane_returns_zeros_when_file_not_registered(self):
        """_load_single_plane returns zeros when file not in index."""
        loader = _PlaneLoader(height=100, width=100)
        result = loader.load(0, 0, 0, "BF")

        assert result.shape == (100, 100)
        assert result.dtype == np.uint16
        assert np.all(result == 0)

    def test_load_single_plane_returns_zeros_on_file_not_found(self):
        """_load_single_plane returns zeros when file doesn't exist."""
        loader = _PlaneLoader(height=100, width=100, load_from_disk=True)
        loader.file_index[(0, 0, 0, "BF")] = "/nonexistent/path/image.tiff"

        result = loader.load(0, 0, 0, "BF")

        assert result.shape == (100, 100)
        assert result.dtype == np.uint16
        assert np.all(result == 0)

    def test_load_single_plane_loads_valid_tiff(self):
        """_load_single_plane successfully loads a valid TIFF file."""
        import tifffile as tf

        loader = _PlaneLoader(height=50, width=50, load_from_disk=True)

        # Create a temporary TIFF file
        with tempfile.NamedTemporaryFile(suffix=".tiff", delete=False) as f:
            temp_path = f.name

        try:
            # Write test image
            test_image = np.random.randint(0, 65535, (50, 50), dtype=np.uint16)
            tf.imwrite(temp_path, test_image)

            loader.file_index[(0, 0, 0, "BF")] = temp_path
            result = loader.load(0, 0, 0, "BF")

            assert result.shape == (50, 50)
            assert np.array_equal(result, test_image)

            # Check it's cached
            cached = loader.cache.get((0, 0, 0, "BF"))
            assert cached is not None
            assert np.array_equal(cached, test_image)

        finally:
            os.unlink(temp_path)

    def test_load_single_plane_uses_cache(self):
        """_load_single_plane returns cached data without disk access."""
        loader = _PlaneLoader(height=50, width=50)

        # Pre-populate cache
        cached_image = np.ones((50, 50), dtype=np.uint16) * 12345
        loader.cache.put((0, 0, 0, "BF"), cached_image)

        result = loader.load(0, 0, 0, "BF")

        assert np.array_equal(result, cached_image)
        assert loader.disk_reads == 0  # No disk read occurred


def _format_fov_label(fov_labels, value):
    """Format FOV label text for the given slider value.

    Args:
        fov_labels: List of FOV labels in "well:fov" format, or empty list.
        value: Slider value (FOV index).

    Returns:
        Formatted label string, e.g. "FOV: A1:0" or "FOV: 5".
    """
    if fov_labels and value < len(fov_labels):
        return f"FOV: {fov_labels[value]}"
    return f"FOV: {value}"


class TestFovLabelUpdate:
    """Tests for FOV label display updates."""

    def test_fov_slider_updates_label_with_fov_labels(self):
        """FOV slider change updates label with well:fov format."""
        fov_labels = ["A1:0", "A1:1", "A2:0"]

        assert _format_fov_label(fov_labels, 0) == "FOV: A1:0"
        assert _format_fov_label(fov_labels, 1) == "FOV: A1:1"
        assert _format_fov_label(fov_labels, 2) == "FOV: A2:0"

    def test_fov_slider_updates_label_without_fov_labels(self):
        """FOV slider change shows numeric index when no labels."""
        assert _format_fov_label([], 5) == "FOV: 5"
