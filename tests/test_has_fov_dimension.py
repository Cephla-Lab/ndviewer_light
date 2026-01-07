"""
Unit tests for has_fov_dimension() method in LightweightViewer.

Tests use mocks to avoid Qt dependencies while testing the actual method.
"""

from unittest.mock import MagicMock


class TestHasFovDimension:
    """Tests for LightweightViewer.has_fov_dimension() method."""

    def _create_mock_viewer(self, xarray_data=None):
        """Create a mock viewer with has_fov_dimension method from real class."""
        from ndviewer_light import LightweightViewer

        mock = MagicMock(spec=LightweightViewer)
        mock._xarray_data = xarray_data
        # Bind the real method to our mock
        mock.has_fov_dimension = lambda: LightweightViewer.has_fov_dimension(mock)
        return mock

    def test_returns_false_when_no_xarray_data(self):
        """When _xarray_data is None, should return False."""
        viewer = self._create_mock_viewer(xarray_data=None)
        result = viewer.has_fov_dimension()
        assert result is False

    def test_returns_false_when_xarray_data_has_no_fov_dim(self):
        """When _xarray_data exists but has no 'fov' dimension, should return False."""
        # Create a mock xarray with dims that don't include 'fov'
        mock_xarray = MagicMock()
        mock_xarray.dims = ("time", "z", "channel", "y", "x")

        viewer = self._create_mock_viewer(xarray_data=mock_xarray)
        result = viewer.has_fov_dimension()
        assert result is False

    def test_returns_true_when_xarray_data_has_fov_dim(self):
        """When _xarray_data exists and has 'fov' dimension, should return True."""
        # Create a mock xarray with 'fov' in dims
        mock_xarray = MagicMock()
        mock_xarray.dims = ("time", "fov", "z", "channel", "y", "x")

        viewer = self._create_mock_viewer(xarray_data=mock_xarray)
        result = viewer.has_fov_dimension()
        assert result is True

    def test_returns_true_with_minimal_dims_including_fov(self):
        """Should return True even with minimal dimensions if 'fov' is present."""
        mock_xarray = MagicMock()
        mock_xarray.dims = ("fov", "y", "x")

        viewer = self._create_mock_viewer(xarray_data=mock_xarray)
        result = viewer.has_fov_dimension()
        assert result is True
