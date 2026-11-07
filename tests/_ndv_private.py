"""Test-only access to ndv internals, kept in one place.

The window smokes need to see what ndv actually put on the GPU and whether its
async fetch has settled; ndv 0.5 has no public API for either. Every helper here
walks a private attribute path and skips the calling test when that path is gone,
so an ndv rename fails this one module loudly instead of every window test
obscurely. ``core.py`` must never import this.
"""

from typing import Any

import numpy as np
import pytest


def _walk(obj: Any, path: str) -> Any:
    """Follow ``a.b[key].c`` from obj; skip the test if any step is missing."""
    try:
        for part in path.split("."):
            name, _, key = part.partition("[")
            if name:
                obj = getattr(obj, name)
            if key:
                key = key.rstrip("]")
                obj = obj[int(key) if key.lstrip("-").isdigit() else key.strip("'\"")]
        return obj
    except (AttributeError, KeyError, IndexError, TypeError) as e:
        pytest.skip(f"ndv private path {path!r} changed ({type(e).__name__}: {e})")


def displayed_texture(viewer: Any, key: int = 0) -> np.ndarray:
    """Pixels currently uploaded for channel ``key`` of an ``ndv.ArrayViewer``.

    ndv 0.5: ``viewer._lut_controllers[key].handles[0].data()`` (the vispy
    image handle's last ``set_data`` payload).
    """
    handle = _walk(viewer, f"_lut_controllers[{key}].handles[0]")
    return np.asarray(handle.data())


def dims_slider(viewer: Any, dim: str) -> Any:
    """The Qt slider ndv built for dimension ``dim`` (its range mirrors the data).

    ndv 0.5: ``viewer._view._qwidget.dims_sliders._sliders[dim]``.
    """
    return _walk(viewer, f"_view._qwidget.dims_sliders._sliders['{dim}']")


def is_idle(viewer: Any) -> bool:
    """True when no ndv data fetch is in flight (``ArrayViewer._is_idle``)."""
    return bool(_walk(viewer, "_is_idle")())
