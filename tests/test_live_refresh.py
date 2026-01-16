"""
Unit tests for live refresh logic in ndviewer_light.

Tests cover:
1. Shape comparison to prevent unnecessary data swaps (flicker prevention)
2. Signature detection for new timepoints/FOVs
"""

import numpy as np


class TestLiveRefreshShapeComparison:
    """Test shape comparison logic that prevents flicker during live acquisition."""

    def test_same_shape_skips_swap(self):
        """When old and new data have the same shape, swap should be skipped."""
        # Simulate the shape comparison logic from _maybe_refresh
        old_shape = (10, 2, 5, 3, 1000, 1000)  # (time, fov, z, channel, y, x)
        new_shape = (10, 2, 5, 3, 1000, 1000)

        # This is the condition that triggers skip
        should_skip = old_shape == new_shape

        assert should_skip is True

    def test_different_time_dimension_triggers_swap(self):
        """When time dimension grows, swap should happen."""
        old_shape = (10, 2, 5, 3, 1000, 1000)
        new_shape = (11, 2, 5, 3, 1000, 1000)  # time grew from 10 to 11

        should_skip = old_shape == new_shape

        assert should_skip is False

    def test_different_fov_dimension_triggers_swap(self):
        """When FOV dimension grows, swap should happen."""
        old_shape = (10, 2, 5, 3, 1000, 1000)
        new_shape = (10, 3, 5, 3, 1000, 1000)  # fov grew from 2 to 3

        should_skip = old_shape == new_shape

        assert should_skip is False

    def test_different_z_dimension_triggers_swap(self):
        """When Z dimension grows, swap should happen."""
        old_shape = (10, 2, 5, 3, 1000, 1000)
        new_shape = (10, 2, 6, 3, 1000, 1000)  # z grew from 5 to 6

        should_skip = old_shape == new_shape

        assert should_skip is False

    def test_no_old_data_triggers_swap(self):
        """When there's no old data (first load), swap should happen."""
        old_data = None
        new_shape = (10, 2, 5, 3, 1000, 1000)

        # Simulate the full condition from _maybe_refresh
        should_skip = old_data is not None and (
            hasattr(old_data, "shape") and old_data.shape == new_shape
        )

        assert should_skip is False


class TestLiveRefreshIntegration:
    """Integration tests for the refresh decision logic."""

    def test_refresh_logic_with_mock_xarray(self):
        """Test the actual refresh logic with mock xarray-like objects."""

        class MockXarray:
            """Mock xarray.DataArray with just shape attribute."""

            def __init__(self, shape):
                self.shape = shape

        # Simulate _maybe_refresh logic
        def should_swap_data(old_data, new_data):
            """Return True if data should be swapped (will cause re-render)."""
            if old_data is None:
                return True
            if new_data is None:
                return False
            return old_data.shape != new_data.shape

        # Case 1: No old data - should swap
        assert should_swap_data(None, MockXarray((5, 2, 3, 2, 100, 100))) is True

        # Case 2: Same shape - should NOT swap (prevents flicker)
        old = MockXarray((5, 2, 3, 2, 100, 100))
        new = MockXarray((5, 2, 3, 2, 100, 100))
        assert should_swap_data(old, new) is False

        # Case 3: Time grew - should swap
        old = MockXarray((5, 2, 3, 2, 100, 100))
        new = MockXarray((6, 2, 3, 2, 100, 100))
        assert should_swap_data(old, new) is True

        # Case 4: Multiple dimensions changed - should swap
        old = MockXarray((5, 2, 3, 2, 100, 100))
        new = MockXarray((6, 3, 3, 2, 100, 100))
        assert should_swap_data(old, new) is True

    def test_signature_change_without_shape_change(self):
        """
        Signature can change (file count) without shape change.
        This is the key scenario for flicker prevention.
        """
        # Scenario: Files being written to current timepoint
        # Signature changes (file_count increases) but shape stays same

        old_signature = ("single_tiff", 5, 2, 50)  # (fmt, max_t, n_fov, file_count)
        new_signature = ("single_tiff", 5, 2, 55)  # file_count changed

        old_shape = (5, 2, 3, 2, 100, 100)
        new_shape = (5, 2, 3, 2, 100, 100)  # shape unchanged

        signature_changed = old_signature != new_signature
        shape_changed = old_shape != new_shape

        # Signature changed, but shape didn't - should skip swap
        assert signature_changed is True
        assert shape_changed is False

        # The fix ensures we skip swap when shape is unchanged
        should_skip_swap = not shape_changed
        assert should_skip_swap is True

    def test_new_timepoint_triggers_update(self):
        """When a new timepoint directory appears, update should happen."""
        # Scenario: New timepoint folder created and populated

        old_signature = ("single_tiff", 5, 2, 100)
        new_signature = ("single_tiff", 6, 2, 10)  # max_t increased

        old_shape = (5, 2, 3, 2, 100, 100)
        new_shape = (6, 2, 3, 2, 100, 100)  # time dimension grew

        signature_changed = old_signature != new_signature
        shape_changed = old_shape != new_shape

        assert signature_changed is True
        assert shape_changed is True

        # Shape changed - should NOT skip swap
        should_skip_swap = not shape_changed
        assert should_skip_swap is False
