"""``LightweightViewer._try_inplace_ndv_update`` on ndv 0.5.

Push-mode frames and same-structure refreshes swap the displayed array inside the
existing ndv viewer instead of rebuilding it (a rebuild flickers and, on ndv 0.4,
leaked GPU handles: https://github.com/pyapp-kit/ndv/issues/209). On ndv 0.5 the
swap goes through the public ``ArrayViewer.data_wrapper`` and the wrapper's
documented ``dims_changed`` / ``data_changed`` signals; the wrapped array has no
public setter, so ``wrapper._data`` stays the one private write.

Signal contract (verified against ndv 0.5): ``dims_changed`` re-creates the sliders
but leaves the old texture; only ``data_changed`` requests a re-fetch. So a shape
change emits ``dims_changed`` then ``data_changed``; a same-shape swap emits
``data_changed`` only.

The mock tests pin that sequence. The real-viewer test runs offscreen (conftest pins
the GL texture query) and drives the push API through the event loop.
"""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import time  # noqa: E402
from unittest.mock import MagicMock, call  # noqa: E402

import numpy as np  # noqa: E402
import pytest  # noqa: E402
import tifffile  # noqa: E402
from qtpy.QtWidgets import QApplication  # noqa: E402

from ndviewer_light import LightweightViewer  # noqa: E402

DEFAULT_SHAPE = (10, 100, 100)


def create_mock_viewer():
    """LightweightViewer stand-in running the real _try_inplace_ndv_update."""
    mock = MagicMock(spec=LightweightViewer)
    mock._try_inplace_ndv_update = (
        lambda data: LightweightViewer._try_inplace_ndv_update(mock, data)
    )
    return mock


def create_mock_data(shape=DEFAULT_SHAPE):
    mock_data = MagicMock()
    mock_data.shape = shape
    return mock_data


class FakeDataWrapper:
    """The ndv DataWrapper surface the swap uses.

    ``data`` is a read-only property over ``_data`` like the real wrapper; the two
    signals hang off one parent mock so ``emits.mock_calls`` records their order.
    """

    def __init__(self, data):
        self._data = data
        self.emits = MagicMock()
        self.dims_changed = self.emits.dims_changed
        self.data_changed = self.emits.data_changed

    @property
    def data(self):
        return self._data


def setup_ndv_viewer(viewer_mock, old_shape=DEFAULT_SHAPE):
    """Give the viewer mock an ndv_viewer whose public data_wrapper wraps old_shape."""
    wrapper = FakeDataWrapper(create_mock_data(old_shape))
    ndv_viewer = MagicMock()
    ndv_viewer.data_wrapper = wrapper
    viewer_mock.ndv_viewer = ndv_viewer
    return ndv_viewer, wrapper


class TestInplaceUpdate:
    def test_same_shape_emits_data_changed_only(self):
        viewer = create_mock_viewer()
        _, wrapper = setup_ndv_viewer(viewer)
        new_data = create_mock_data()

        assert viewer._try_inplace_ndv_update(new_data) is True

        assert wrapper.data is new_data
        wrapper.data_changed.emit.assert_called_once_with()
        wrapper.dims_changed.emit.assert_not_called()

    def test_shape_change_emits_dims_then_data(self):
        """dims_changed alone leaves the old texture: data_changed must follow it."""
        viewer = create_mock_viewer()
        _, wrapper = setup_ndv_viewer(viewer)
        new_data = create_mock_data(shape=(20, 100, 100))

        assert viewer._try_inplace_ndv_update(new_data) is True

        assert wrapper.data is new_data
        assert wrapper.emits.mock_calls == [
            call.dims_changed.emit(),
            call.data_changed.emit(),
        ]

    def test_returns_false_when_ndv_viewer_is_none(self):
        viewer = create_mock_viewer()
        viewer.ndv_viewer = None

        assert viewer._try_inplace_ndv_update(create_mock_data()) is False

    def test_returns_false_without_data_wrapper(self):
        viewer = create_mock_viewer()
        viewer.ndv_viewer = MagicMock(spec=[])  # no data_wrapper attribute at all

        assert viewer._try_inplace_ndv_update(create_mock_data()) is False

    def test_returns_false_when_data_wrapper_is_none(self):
        """ndv 0.5: ArrayViewer.data_wrapper is None until data is set."""
        viewer = create_mock_viewer()
        viewer.ndv_viewer = MagicMock()
        viewer.ndv_viewer.data_wrapper = None

        assert viewer._try_inplace_ndv_update(create_mock_data()) is False

    def test_returns_false_on_data_changed_emit_exception(self):
        viewer = create_mock_viewer()
        _, wrapper = setup_ndv_viewer(viewer)
        wrapper.data_changed.emit.side_effect = RuntimeError("signal error")

        assert viewer._try_inplace_ndv_update(create_mock_data()) is False

    def test_returns_false_on_dims_changed_emit_exception(self):
        viewer = create_mock_viewer()
        _, wrapper = setup_ndv_viewer(viewer)
        wrapper.dims_changed.emit.side_effect = RuntimeError("signal error")

        new_data = create_mock_data(shape=(20, 100, 100))
        assert viewer._try_inplace_ndv_update(new_data) is False
        wrapper.data_changed.emit.assert_not_called()


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def wait_until(qapp, predicate, timeout_s=3.0) -> bool:
    """Pump the event loop until predicate() holds (register_image loads are debounced)."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        qapp.processEvents()
        if predicate():
            return True
    return predicate()


def test_push_register_image_keeps_viewer_instance(qapp, tmp_path):
    """Two push frames land in the one ArrayViewer built at start_acquisition.

    Each register_image schedules a debounced load that swaps the viewer's array
    for a fresh lazy stack; a failed swap would fall back to _set_ndv_data and
    replace the ArrayViewer instance.
    """
    v = LightweightViewer()
    v.start_acquisition(["BF", "GFP"], 1, 8, 8, ["A1:0"])
    before = v.ndv_viewer

    for frame, channel in enumerate(["BF", "GFP"], start=1):
        tif = tmp_path / f"{channel}.tif"
        tifffile.imwrite(tif, np.full((8, 8), frame, dtype=np.uint16))
        shown = v.ndv_viewer.data

        v.register_image(0, 0, 0, channel, str(tif))

        assert wait_until(qapp, lambda: v.ndv_viewer.data is not shown)
        assert v.ndv_viewer is before
        assert v.ndv_viewer.data is v._xarray_data
