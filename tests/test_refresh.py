"""
Unit tests for refresh() method in LightweightViewer.

Tests use mocks to avoid Qt dependencies while testing the actual method.
"""

from unittest.mock import MagicMock, patch


class TestRefresh:
    """Tests for LightweightViewer.refresh() method."""

    def test_refresh_calls_force_refresh(self):
        """refresh() should delegate to _force_refresh()."""
        from ndviewer_light import LightweightViewer

        mock = MagicMock(spec=LightweightViewer)
        mock._force_refresh = MagicMock()

        # Bind the real method to our mock
        LightweightViewer.refresh(mock)

        mock._force_refresh.assert_called_once()
