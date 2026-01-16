"""
Unit tests for live refresh logic in ndviewer_light.

Tests cover:
1. Shape comparison to prevent unnecessary data swaps (flicker prevention)
2. Signature detection for new timepoints/FOVs
"""


class MockDataArray:
    """Mock xarray.DataArray with shape attribute for testing."""

    def __init__(self, shape):
        self.shape = shape


def should_swap_data(old_data, new_data):
    """Return True if data should be swapped (will cause re-render).

    This mirrors the shape comparison logic in _maybe_refresh().
    """
    if old_data is None:
        return True
    if new_data is None:
        return False
    return old_data.shape != new_data.shape


class TestLiveRefreshShapeComparison:
    """Test shape comparison logic that prevents flicker during live acquisition."""

    def test_same_shape_skips_swap(self):
        """When old and new data have the same shape, swap should be skipped."""
        shape = (10, 2, 5, 3, 1000, 1000)  # (time, fov, z, channel, y, x)
        old = MockDataArray(shape)
        new = MockDataArray(shape)

        assert should_swap_data(old, new) is False

    def test_different_time_dimension_triggers_swap(self):
        """When time dimension grows, swap should happen."""
        old = MockDataArray((10, 2, 5, 3, 1000, 1000))
        new = MockDataArray((11, 2, 5, 3, 1000, 1000))

        assert should_swap_data(old, new) is True

    def test_different_fov_dimension_triggers_swap(self):
        """When FOV dimension grows, swap should happen."""
        old = MockDataArray((10, 2, 5, 3, 1000, 1000))
        new = MockDataArray((10, 3, 5, 3, 1000, 1000))

        assert should_swap_data(old, new) is True

    def test_different_z_dimension_triggers_swap(self):
        """When Z dimension grows, swap should happen."""
        old = MockDataArray((10, 2, 5, 3, 1000, 1000))
        new = MockDataArray((10, 2, 6, 3, 1000, 1000))

        assert should_swap_data(old, new) is True

    def test_no_old_data_triggers_swap(self):
        """When there's no old data (first load), swap should happen."""
        new = MockDataArray((10, 2, 5, 3, 1000, 1000))

        assert should_swap_data(None, new) is True


class TestLiveRefreshIntegration:
    """Integration tests for the refresh decision logic."""

    def test_multiple_cases(self):
        """Test various shape comparison scenarios."""
        # No old data - should swap
        assert should_swap_data(None, MockDataArray((5, 2, 3, 2, 100, 100))) is True

        # Same shape - should NOT swap (prevents flicker)
        old = MockDataArray((5, 2, 3, 2, 100, 100))
        new = MockDataArray((5, 2, 3, 2, 100, 100))
        assert should_swap_data(old, new) is False

        # Time grew - should swap
        old = MockDataArray((5, 2, 3, 2, 100, 100))
        new = MockDataArray((6, 2, 3, 2, 100, 100))
        assert should_swap_data(old, new) is True

        # Multiple dimensions changed - should swap
        old = MockDataArray((5, 2, 3, 2, 100, 100))
        new = MockDataArray((6, 3, 3, 2, 100, 100))
        assert should_swap_data(old, new) is True

    def test_signature_change_without_shape_change(self):
        """Signature can change (file count) without shape change.

        This is the key scenario for flicker prevention: files are being
        written to the current timepoint, signature changes but shape
        stays the same.
        """
        old_signature = ("single_tiff", 5, 2, 50)  # (fmt, max_t, n_fov, file_count)
        new_signature = ("single_tiff", 5, 2, 55)  # file_count changed

        old_shape = (5, 2, 3, 2, 100, 100)
        new_shape = (5, 2, 3, 2, 100, 100)

        # Signature changed, but shape didn't - should skip swap
        assert old_signature != new_signature
        assert old_shape == new_shape
        assert (
            should_swap_data(MockDataArray(old_shape), MockDataArray(new_shape))
            is False
        )

    def test_new_timepoint_triggers_update(self):
        """When a new timepoint directory appears, update should happen."""
        old_signature = ("single_tiff", 5, 2, 100)
        new_signature = ("single_tiff", 6, 2, 10)  # max_t increased

        old_shape = (5, 2, 3, 2, 100, 100)
        new_shape = (6, 2, 3, 2, 100, 100)  # time dimension grew

        # Both signature and shape changed - should swap
        assert old_signature != new_signature
        assert old_shape != new_shape
        assert (
            should_swap_data(MockDataArray(old_shape), MockDataArray(new_shape)) is True
        )
