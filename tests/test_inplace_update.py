"""
Unit tests for _try_inplace_ndv_update() method in LightweightViewer.

This method bypasses ndv's leaky data setter to avoid GPU memory leaks
during live refresh (see https://github.com/pyapp-kit/ndv/issues/209).
"""

from unittest.mock import MagicMock

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
    """Create mock data with a shape attribute."""
    mock_data = MagicMock()
    mock_data.shape = shape
    return mock_data


def setup_ndv_viewer(viewer_mock, old_shape=DEFAULT_SHAPE):
    """Setup a mock ndv_viewer with correct internal structure."""
    wrapper = MagicMock()
    wrapper._data = create_mock_data(old_shape)

    data_model = MagicMock()
    data_model.data_wrapper = wrapper

    ndv_viewer = MagicMock()
    ndv_viewer._data_model = data_model

    viewer_mock.ndv_viewer = ndv_viewer
    return ndv_viewer, wrapper


class TestInplaceUpdate:
    """Tests for _try_inplace_ndv_update()."""

    def test_success(self):
        """Updates data and calls _request_data when shapes match."""
        viewer = create_mock_viewer()
        ndv_viewer, wrapper = setup_ndv_viewer(viewer)

        new_data = create_mock_data()
        result = viewer._try_inplace_ndv_update(new_data)

        assert result is True
        assert wrapper._data is new_data
        ndv_viewer._request_data.assert_called_once()

    def test_emits_dims_changed_on_shape_change(self):
        """Emits dims_changed signal when shapes differ to update sliders.

        When data shape changes (e.g., new timepoint), we update the data
        in-place and emit dims_changed to trigger slider updates without
        rebuilding the entire viewer.
        """
        viewer = create_mock_viewer()
        ndv_viewer, wrapper = setup_ndv_viewer(viewer)

        new_data = create_mock_data(shape=(20, 100, 100))  # Different shape
        result = viewer._try_inplace_ndv_update(new_data)

        assert result is True
        assert wrapper._data is new_data  # Data swapped
        wrapper.dims_changed.emit.assert_called_once()
        ndv_viewer._request_data.assert_not_called()  # dims_changed triggers sync instead

    def test_returns_false_when_ndv_viewer_is_none(self):
        """Returns False when ndv_viewer is None."""
        viewer = create_mock_viewer()
        viewer.ndv_viewer = None

        result = viewer._try_inplace_ndv_update(create_mock_data())

        assert result is False

    def test_returns_false_when_data_is_none(self):
        """Returns False when wrapper._data is None."""
        viewer = create_mock_viewer()
        ndv_viewer, wrapper = setup_ndv_viewer(viewer)
        wrapper._data = None

        result = viewer._try_inplace_ndv_update(create_mock_data())

        assert result is False

    def test_returns_false_on_missing_attributes(self):
        """Returns False when internal attributes are missing."""
        viewer = create_mock_viewer()
        viewer.ndv_viewer = MagicMock(spec=[])  # no _data_model

        result = viewer._try_inplace_ndv_update(create_mock_data())

        assert result is False

    def test_returns_false_on_exception(self):
        """Returns False when an exception occurs."""
        viewer = create_mock_viewer()
        ndv_viewer = MagicMock()
        ndv_viewer._data_model = MagicMock()
        ndv_viewer._data_model.data_wrapper = MagicMock()
        ndv_viewer._data_model.data_wrapper._data = create_mock_data()
        ndv_viewer._request_data.side_effect = RuntimeError("test")
        viewer.ndv_viewer = ndv_viewer

        result = viewer._try_inplace_ndv_update(create_mock_data())

        assert result is False
