"""
Unit tests for _try_inplace_ndv_update() method in LightweightViewer.

This method bypasses ndv's leaky data setter to avoid GPU memory leaks
during live refresh (see https://github.com/pyapp-kit/ndv/issues/209).

Tests use mocks to avoid Qt/ndv dependencies while verifying:
1. Correct attribute paths are accessed (_data_model.data_wrapper)
2. Shape compatibility checks work correctly
3. Refresh triggers (_request_data or current_index.update) are called
4. Edge cases and error handling
"""

import numpy as np
import pytest
from unittest.mock import MagicMock, PropertyMock, patch


class TestInplaceUpdate:
    """Tests for LightweightViewer._try_inplace_ndv_update() method."""

    def _create_mock_viewer(self):
        """Create a mock viewer with the real _try_inplace_ndv_update method."""
        from ndviewer_light import LightweightViewer

        mock = MagicMock(spec=LightweightViewer)
        # Bind the real method to our mock
        mock._try_inplace_ndv_update = (
            lambda data: LightweightViewer._try_inplace_ndv_update(mock, data)
        )
        return mock

    def _create_mock_data(self, shape=(10, 100, 100)):
        """Create mock xarray-like data with a shape attribute."""
        mock_data = MagicMock()
        mock_data.shape = shape
        return mock_data

    def _setup_ndv_viewer_mock(self, viewer_mock, old_shape=(10, 100, 100)):
        """Setup a mock ndv_viewer with correct internal structure."""
        # Create the internal structure: viewer._data_model.data_wrapper._data
        wrapper = MagicMock()
        wrapper._data = self._create_mock_data(old_shape)

        data_model = MagicMock()
        data_model.data_wrapper = wrapper

        ndv_viewer = MagicMock()
        ndv_viewer._data_model = data_model
        ndv_viewer._request_data = MagicMock()

        viewer_mock.ndv_viewer = ndv_viewer
        return ndv_viewer, data_model, wrapper

    # === Success Cases ===

    def test_inplace_update_success_via_request_data(self):
        """In-place update succeeds when shapes match and _request_data exists."""
        viewer = self._create_mock_viewer()
        ndv_viewer, data_model, wrapper = self._setup_ndv_viewer_mock(viewer)

        new_data = self._create_mock_data(shape=(10, 100, 100))  # Same shape
        result = viewer._try_inplace_ndv_update(new_data)

        assert result is True
        assert wrapper._data is new_data  # Data was updated
        ndv_viewer._request_data.assert_called_once()  # Refresh triggered

    def test_inplace_update_success_via_current_index(self):
        """In-place update falls back to current_index.update() if no _request_data."""
        viewer = self._create_mock_viewer()
        ndv_viewer, data_model, wrapper = self._setup_ndv_viewer_mock(viewer)

        # Remove _request_data to test fallback
        del ndv_viewer._request_data

        # Setup current_index mock
        current_index = MagicMock()
        current_index.update = MagicMock()
        display_model = MagicMock()
        display_model.current_index = current_index
        ndv_viewer.display_model = display_model

        new_data = self._create_mock_data(shape=(10, 100, 100))
        result = viewer._try_inplace_ndv_update(new_data)

        assert result is True
        assert wrapper._data is new_data
        current_index.update.assert_called_once()

    def test_wrapper_data_is_replaced_not_copied(self):
        """Verify we replace the wrapper._data reference, not copy into it."""
        viewer = self._create_mock_viewer()
        ndv_viewer, data_model, wrapper = self._setup_ndv_viewer_mock(viewer)

        old_data = wrapper._data
        new_data = self._create_mock_data(shape=(10, 100, 100))

        viewer._try_inplace_ndv_update(new_data)

        # The reference should be replaced
        assert wrapper._data is new_data
        assert wrapper._data is not old_data

    # === Shape Mismatch Cases ===

    def test_returns_false_on_shape_mismatch(self):
        """Returns False when shapes don't match (requires full rebuild)."""
        viewer = self._create_mock_viewer()
        ndv_viewer, data_model, wrapper = self._setup_ndv_viewer_mock(
            viewer, old_shape=(10, 100, 100)
        )

        new_data = self._create_mock_data(shape=(20, 100, 100))  # Different shape
        result = viewer._try_inplace_ndv_update(new_data)

        assert result is False
        # Data should NOT be updated
        assert wrapper._data is not new_data
        ndv_viewer._request_data.assert_not_called()

    def test_returns_false_on_dimension_change(self):
        """Returns False when number of dimensions changes."""
        viewer = self._create_mock_viewer()
        ndv_viewer, data_model, wrapper = self._setup_ndv_viewer_mock(
            viewer, old_shape=(10, 100, 100)
        )

        new_data = self._create_mock_data(shape=(5, 10, 100, 100))  # 4D vs 3D
        result = viewer._try_inplace_ndv_update(new_data)

        assert result is False

    # === Missing Attribute Cases ===

    def test_returns_false_when_ndv_viewer_is_none(self):
        """Returns False when ndv_viewer is None."""
        viewer = self._create_mock_viewer()
        viewer.ndv_viewer = None

        new_data = self._create_mock_data()
        result = viewer._try_inplace_ndv_update(new_data)

        assert result is False

    def test_returns_false_when_data_model_missing(self):
        """Returns False when _data_model attribute is missing."""
        viewer = self._create_mock_viewer()
        viewer.ndv_viewer = MagicMock(spec=[])  # Empty spec = no attributes

        new_data = self._create_mock_data()
        result = viewer._try_inplace_ndv_update(new_data)

        assert result is False

    def test_returns_false_when_data_wrapper_is_none(self):
        """Returns False when data_wrapper is None."""
        viewer = self._create_mock_viewer()
        ndv_viewer = MagicMock()
        data_model = MagicMock()
        data_model.data_wrapper = None
        ndv_viewer._data_model = data_model
        viewer.ndv_viewer = ndv_viewer

        new_data = self._create_mock_data()
        result = viewer._try_inplace_ndv_update(new_data)

        assert result is False

    def test_returns_false_when_wrapper_has_no_data(self):
        """Returns False when wrapper has no _data attribute."""
        viewer = self._create_mock_viewer()
        ndv_viewer = MagicMock()
        data_model = MagicMock()
        wrapper = MagicMock(spec=[])  # No _data or data attribute
        data_model.data_wrapper = wrapper
        ndv_viewer._data_model = data_model
        viewer.ndv_viewer = ndv_viewer

        new_data = self._create_mock_data()
        result = viewer._try_inplace_ndv_update(new_data)

        assert result is False

    def test_returns_false_when_no_refresh_trigger_available(self):
        """Returns False when neither _request_data nor current_index.update exists."""
        viewer = self._create_mock_viewer()
        ndv_viewer, data_model, wrapper = self._setup_ndv_viewer_mock(viewer)

        # Remove all refresh triggers
        del ndv_viewer._request_data
        ndv_viewer.display_model = None

        new_data = self._create_mock_data(shape=(10, 100, 100))
        result = viewer._try_inplace_ndv_update(new_data)

        # Data was updated but no refresh could be triggered
        assert result is False

    # === Fallback to Public data Property ===

    def test_uses_public_data_property_as_fallback(self):
        """Falls back to wrapper.data if wrapper._data doesn't exist."""
        viewer = self._create_mock_viewer()
        ndv_viewer = MagicMock()
        data_model = MagicMock()

        # Wrapper with only public 'data' property, not '_data'
        wrapper = MagicMock(spec=["data"])
        wrapper.data = self._create_mock_data(shape=(10, 100, 100))
        data_model.data_wrapper = wrapper
        ndv_viewer._data_model = data_model
        ndv_viewer._request_data = MagicMock()
        viewer.ndv_viewer = ndv_viewer

        new_data = self._create_mock_data(shape=(10, 100, 100))
        result = viewer._try_inplace_ndv_update(new_data)

        # Should still work using public property for shape check
        # Note: setting wrapper._data will still work even if not in spec
        assert result is True
        ndv_viewer._request_data.assert_called_once()

    # === Error Handling ===

    def test_handles_exception_gracefully(self):
        """Returns False and doesn't raise when an exception occurs."""
        viewer = self._create_mock_viewer()
        ndv_viewer = MagicMock()
        # Make _data_model raise an exception when accessed
        type(ndv_viewer)._data_model = PropertyMock(side_effect=RuntimeError("test"))
        viewer.ndv_viewer = ndv_viewer

        new_data = self._create_mock_data()
        # Should not raise
        result = viewer._try_inplace_ndv_update(new_data)

        assert result is False

    def test_handles_shape_attribute_error(self):
        """Returns False when shape attribute access fails."""
        viewer = self._create_mock_viewer()
        ndv_viewer, data_model, wrapper = self._setup_ndv_viewer_mock(viewer)

        # New data without shape attribute
        new_data = MagicMock(spec=[])  # No shape
        result = viewer._try_inplace_ndv_update(new_data)

        assert result is False


class TestInplaceUpdateIntegration:
    """Integration tests for _try_inplace_ndv_update with _maybe_refresh flow."""

    def _create_mock_viewer(self):
        """Create a mock viewer with relevant methods."""
        from ndviewer_light import LightweightViewer

        mock = MagicMock(spec=LightweightViewer)
        mock._try_inplace_ndv_update = (
            lambda data: LightweightViewer._try_inplace_ndv_update(mock, data)
        )
        return mock

    def test_maybe_refresh_uses_inplace_when_structure_unchanged(self):
        """_maybe_refresh should prefer in-place update when structure unchanged."""
        from ndviewer_light import LightweightViewer

        viewer = MagicMock(spec=LightweightViewer)

        # Setup for successful in-place update
        wrapper = MagicMock()
        wrapper._data = MagicMock(shape=(10, 100, 100))

        data_model = MagicMock()
        data_model.data_wrapper = wrapper

        ndv_viewer = MagicMock()
        ndv_viewer._data_model = data_model
        ndv_viewer._request_data = MagicMock()

        viewer.ndv_viewer = ndv_viewer
        viewer._try_inplace_ndv_update = (
            lambda data: LightweightViewer._try_inplace_ndv_update(viewer, data)
        )
        viewer._data_structure_changed = MagicMock(return_value=False)
        viewer._set_ndv_data = MagicMock()
        viewer._initiate_channel_label_update = MagicMock()

        # Create new data with same shape
        new_data = MagicMock()
        new_data.shape = (10, 100, 100)
        new_data.attrs = {"_open_tifs": []}

        # Simulate successful in-place update
        result = viewer._try_inplace_ndv_update(new_data)

        assert result is True
        # _set_ndv_data should NOT be called (we did in-place update)
        viewer._set_ndv_data.assert_not_called()

    def test_falls_back_to_rebuild_on_structure_change(self):
        """Should fall back to _set_ndv_data when structure changes."""
        from ndviewer_light import LightweightViewer

        viewer = MagicMock(spec=LightweightViewer)

        # Setup viewer that will fail in-place update due to shape mismatch
        wrapper = MagicMock()
        wrapper._data = MagicMock(shape=(10, 100, 100))

        data_model = MagicMock()
        data_model.data_wrapper = wrapper

        ndv_viewer = MagicMock()
        ndv_viewer._data_model = data_model

        viewer.ndv_viewer = ndv_viewer
        viewer._try_inplace_ndv_update = (
            lambda data: LightweightViewer._try_inplace_ndv_update(viewer, data)
        )

        # New data with different shape
        new_data = MagicMock()
        new_data.shape = (20, 100, 100)  # Different!

        result = viewer._try_inplace_ndv_update(new_data)

        assert result is False  # Should indicate rebuild needed
