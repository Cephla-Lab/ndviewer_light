"""Programmatic navigation: ``LightweightViewer.set_current_index`` on ndv 0.5.

ndv 0.5 keeps ``display_model.current_index`` empty until a user moves a slider
(index resolution became pure in ndv PR #233), so guarding the write on
``dim in current_index`` made ``set_current_index`` a silent no-op. The port
validates ``dim`` and ``value`` against the public ``data_wrapper`` and writes
the key unconditionally; an unknown dim or an out-of-range value is rejected
without touching the model. Model-level tests, offscreen (conftest pins the GL
texture query), no event loop.
"""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest  # noqa: E402
from qtpy.QtWidgets import QApplication  # noqa: E402

from ndviewer_light import LightweightViewer  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def push_viewer(qapp):
    """Push-mode viewer over a (z_level, channel, y, x) array with 3 z-levels."""
    v = LightweightViewer()
    v.start_acquisition(["BF"], 3, 8, 8, ["A1:0"])
    return v


def current_index(viewer: LightweightViewer) -> dict:
    return dict(viewer.ndv_viewer.display_model.current_index)


def test_model_index_is_empty_after_construction_on_ndv05(push_viewer):
    """The motivating fact: nothing fills current_index until a slider moves."""
    assert current_index(push_viewer) == {}


def test_set_current_index_writes_absent_key(push_viewer):
    assert push_viewer.set_current_index("z_level", 2) is True
    assert current_index(push_viewer)["z_level"] == 2


def test_unknown_dim_returns_false_and_leaves_model_clean(push_viewer):
    # Push mode has no "fov" dim (Squid's dataset-mode plate double-click uses it);
    # writing an unknown key would make ndv warn on every later write.
    assert push_viewer.set_current_index("fov", 0) is False
    assert "fov" not in current_index(push_viewer)


def test_out_of_range_returns_false(push_viewer):
    assert push_viewer.set_current_index("z_level", 3) is False
    assert push_viewer.set_current_index("z_level", -1) is False
    assert "z_level" not in current_index(push_viewer)
