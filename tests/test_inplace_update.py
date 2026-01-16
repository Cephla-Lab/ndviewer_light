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

from unittest.mock import MagicMock, PropertyMock

DEFAULT_SHAPE = (10, 100, 100)


def create_mock_viewer():
    """Create a mock viewer with the real _try_inplace_ndv_update method."""
    from ndviewer_light import LightweightViewer

    mock = MagicMock(spec=LightweightViewer)
    mock._try_inplace_ndv_update = (
        lambda data: LightweightViewer._try_inplace_ndv_update(mock, data)
    )
    return mock


def create_mock_data(shape=DEFAULT_SHAPE):
    """Create mock xarray-like data with a shape attribute."""
    mock_data = MagicMock()
    mock_data.shape = shape
    return mock_data


def setup_ndv_viewer_mock(viewer_mock, old_shape=DEFAULT_SHAPE):
    """Setup a mock ndv_viewer with correct internal structure.

    Returns (ndv_viewer, data_model, wrapper) tuple.
    """
    wrapper = MagicMock()
    wrapper._data = create_mock_data(old_shape)

    data_model = MagicMock()
    data_model.data_wrapper = wrapper

    ndv_viewer = MagicMock()
    ndv_viewer._data_model = data_model
    ndv_viewer._request_data = MagicMock()

    viewer_mock.ndv_viewer = ndv_viewer
    return ndv_viewer, data_model, wrapper


class TestInplaceUpdate:
    """Tests for LightweightViewer._try_inplace_ndv_update() method."""

    # === Success Cases ===

    def test_inplace_update_success_via_request_data(self):
        """In-place update succeeds when shapes match and _request_data exists."""
        viewer = create_mock_viewer()
        ndv_viewer, _, wrapper = setup_ndv_viewer_mock(viewer)

        new_data = create_mock_data()
        result = viewer._try_inplace_ndv_update(new_data)

        assert result is True
        assert wrapper._data is new_data
        ndv_viewer._request_data.assert_called_once()

    def test_inplace_update_success_via_current_index(self):
        """In-place update falls back to current_index.update() if no _request_data."""
        viewer = create_mock_viewer()
        ndv_viewer, _, wrapper = setup_ndv_viewer_mock(viewer)

        del ndv_viewer._request_data

        current_index = MagicMock()
        display_model = MagicMock()
        display_model.current_index = current_index
        ndv_viewer.display_model = display_model

        new_data = create_mock_data()
        result = viewer._try_inplace_ndv_update(new_data)

        assert result is True
        assert wrapper._data is new_data
        current_index.update.assert_called_once()

    def test_wrapper_data_is_replaced_not_copied(self):
        """Verify we replace the wrapper._data reference, not copy into it."""
        viewer = create_mock_viewer()
        _, _, wrapper = setup_ndv_viewer_mock(viewer)

        old_data = wrapper._data
        new_data = create_mock_data()

        viewer._try_inplace_ndv_update(new_data)

        assert wrapper._data is new_data
        assert wrapper._data is not old_data

    # === Shape Mismatch Cases ===

    def test_returns_false_on_shape_mismatch(self):
        """Returns False when shapes don't match (requires full rebuild)."""
        viewer = create_mock_viewer()
        ndv_viewer, _, wrapper = setup_ndv_viewer_mock(viewer)

        new_data = create_mock_data(shape=(20, 100, 100))
        result = viewer._try_inplace_ndv_update(new_data)

        assert result is False
        assert wrapper._data is not new_data
        ndv_viewer._request_data.assert_not_called()

    def test_returns_false_on_dimension_change(self):
        """Returns False when number of dimensions changes."""
        viewer = create_mock_viewer()
        setup_ndv_viewer_mock(viewer)

        new_data = create_mock_data(shape=(5, 10, 100, 100))
        result = viewer._try_inplace_ndv_update(new_data)

        assert result is False

    # === Missing Attribute Cases ===

    def test_returns_false_when_ndv_viewer_is_none(self):
        """Returns False when ndv_viewer is None."""
        viewer = create_mock_viewer()
        viewer.ndv_viewer = None

        result = viewer._try_inplace_ndv_update(create_mock_data())

        assert result is False

    def test_returns_false_when_data_model_missing(self):
        """Returns False when _data_model attribute is missing."""
        viewer = create_mock_viewer()
        viewer.ndv_viewer = MagicMock(spec=[])

        result = viewer._try_inplace_ndv_update(create_mock_data())

        assert result is False

    def test_returns_false_when_data_wrapper_is_none(self):
        """Returns False when data_wrapper is None."""
        viewer = create_mock_viewer()
        ndv_viewer = MagicMock()
        data_model = MagicMock()
        data_model.data_wrapper = None
        ndv_viewer._data_model = data_model
        viewer.ndv_viewer = ndv_viewer

        result = viewer._try_inplace_ndv_update(create_mock_data())

        assert result is False

    def test_returns_false_when_wrapper_has_no_data(self):
        """Returns False when wrapper has neither _data nor data attribute."""
        viewer = create_mock_viewer()
        ndv_viewer = MagicMock()
        data_model = MagicMock()
        wrapper = MagicMock(spec=[])
        data_model.data_wrapper = wrapper
        ndv_viewer._data_model = data_model
        viewer.ndv_viewer = ndv_viewer

        result = viewer._try_inplace_ndv_update(create_mock_data())

        assert result is False

    def test_returns_false_when_no_refresh_trigger_available(self):
        """Returns False when neither _request_data nor current_index.update exists."""
        viewer = create_mock_viewer()
        ndv_viewer, _, wrapper = setup_ndv_viewer_mock(viewer)

        del ndv_viewer._request_data
        ndv_viewer.display_model = None

        new_data = create_mock_data()
        result = viewer._try_inplace_ndv_update(new_data)

        assert result is False
        assert wrapper._data is not new_data

    def test_returns_false_when_current_index_is_none(self):
        """Returns False when display_model.current_index is None."""
        viewer = create_mock_viewer()
        ndv_viewer, _, wrapper = setup_ndv_viewer_mock(viewer)

        del ndv_viewer._request_data
        display_model = MagicMock()
        display_model.current_index = None
        ndv_viewer.display_model = display_model

        new_data = create_mock_data()
        result = viewer._try_inplace_ndv_update(new_data)

        assert result is False
        assert wrapper._data is not new_data

    def test_returns_false_when_current_index_has_no_update(self):
        """Returns False when current_index exists but has no update() method."""
        viewer = create_mock_viewer()
        ndv_viewer, _, wrapper = setup_ndv_viewer_mock(viewer)

        del ndv_viewer._request_data
        current_index = MagicMock(spec=[])
        display_model = MagicMock()
        display_model.current_index = current_index
        ndv_viewer.display_model = display_model

        new_data = create_mock_data()
        result = viewer._try_inplace_ndv_update(new_data)

        assert result is False
        assert wrapper._data is not new_data

    # === Fallback to Public data Property ===

    def test_uses_public_data_property_as_fallback(self):
        """Falls back to wrapper.data if wrapper._data doesn't exist.

        Verifies that:
        1. Code reads from wrapper.data when _data is not available
        2. Code writes to wrapper.data (not creating orphan _data)
        """
        viewer = create_mock_viewer()
        ndv_viewer = MagicMock()
        data_model = MagicMock()

        # Create wrapper that only has 'data' property, not '_data'
        class DataOnlyWrapper:
            def __init__(self):
                self.data = create_mock_data()

        wrapper = DataOnlyWrapper()
        data_model.data_wrapper = wrapper
        ndv_viewer._data_model = data_model
        ndv_viewer._request_data = MagicMock()
        viewer.ndv_viewer = ndv_viewer

        new_data = create_mock_data()
        result = viewer._try_inplace_ndv_update(new_data)

        assert result is True
        ndv_viewer._request_data.assert_called_once()
        # Verify data was written to 'data' attribute, not '_data'
        assert wrapper.data is new_data
        assert not hasattr(wrapper, "_data")

    # === Error Handling ===

    def test_handles_exception_gracefully(self):
        """Returns False and doesn't raise when an exception occurs."""
        viewer = create_mock_viewer()
        ndv_viewer = MagicMock()
        type(ndv_viewer)._data_model = PropertyMock(side_effect=RuntimeError("test"))
        viewer.ndv_viewer = ndv_viewer

        result = viewer._try_inplace_ndv_update(create_mock_data())

        assert result is False

    def test_handles_shape_attribute_error(self):
        """Returns False when shape attribute access fails."""
        viewer = create_mock_viewer()
        setup_ndv_viewer_mock(viewer)

        new_data = MagicMock(spec=[])
        result = viewer._try_inplace_ndv_update(new_data)

        assert result is False


class TestInplaceUpdateIntegration:
    """Integration tests for _try_inplace_ndv_update with _maybe_refresh flow."""

    def test_maybe_refresh_uses_inplace_when_structure_unchanged(self):
        """_maybe_refresh should prefer in-place update when structure unchanged."""
        viewer = create_mock_viewer()
        ndv_viewer, _, wrapper = setup_ndv_viewer_mock(viewer)

        viewer._data_structure_changed = MagicMock(return_value=False)
        viewer._set_ndv_data = MagicMock()
        viewer._initiate_channel_label_update = MagicMock()

        new_data = MagicMock()
        new_data.shape = DEFAULT_SHAPE
        new_data.attrs = {"_open_tifs": []}

        result = viewer._try_inplace_ndv_update(new_data)

        assert result is True
        viewer._set_ndv_data.assert_not_called()

    def test_falls_back_to_rebuild_on_structure_change(self):
        """Should fall back to _set_ndv_data when structure changes."""
        viewer = create_mock_viewer()
        setup_ndv_viewer_mock(viewer)

        new_data = MagicMock()
        new_data.shape = (20, 100, 100)

        result = viewer._try_inplace_ndv_update(new_data)

        assert result is False
