"""
Unit tests for refresh() method in LightweightViewer.

Tests use mocks to avoid Qt dependencies while testing the actual method.
"""

from unittest.mock import MagicMock


class TestRefresh:
    """Tests for LightweightViewer.refresh() method."""

    def _create_mock_viewer(self):
        """Create a mock viewer with refresh method from real class."""
        from ndviewer_light import LightweightViewer

        mock = MagicMock(spec=LightweightViewer)
        mock._force_refresh = MagicMock()
        # Bind the real method to our mock
        mock.refresh = lambda: LightweightViewer.refresh(mock)
        return mock

    def test_refresh_calls_force_refresh(self):
        """refresh() should delegate to _force_refresh()."""
        viewer = self._create_mock_viewer()
        viewer.refresh()
        viewer._force_refresh.assert_called_once()
