"""Test-only access to ndv internals, kept in one place.

The window smokes need to see what ndv actually put on the GPU and whether its
async fetch has settled; ndv 0.5 has no public API for either. Every helper here
walks a private attribute path, and the two ways a step can be missing mean
different things:

* an attribute is gone -> ndv renamed the path: ``pytest.skip`` naming the path,
  so an ndv rename fails this one module loudly instead of every window test
  obscurely (CI's real-GL step fails when a smoke skipped);
* a key or index is absent -> ndv has not created that object yet: ndv 0.5 builds
  ``_lut_controllers[key]`` and its image handle in ``_on_data_response_ready``, a
  main-thread callback queued after the async fetch, while ``_is_idle()`` already
  reports True once the worker's future is done. ``displayed_texture`` returns
  ``None`` there and the smokes keep polling. A dims slider ndv never built is a
  failure, not a skip.

``core.py`` must never import this.
"""

from typing import Any, Optional

import numpy as np
import pytest

_MISSING = object()  # a key or index not present (yet); not an ndv rename


def _walk(obj: Any, path: str) -> Any:
    """Follow ``a.b[key].c`` from obj.

    Skips the test when an attribute is gone or a step has the wrong type (ndv
    changed the path); returns ``_MISSING`` when a key or index is absent (ndv has
    not created that object yet).
    """
    try:
        for part in path.split("."):
            name, _, key = part.partition("[")
            if name:
                obj = getattr(obj, name)
            if key:
                key = key.rstrip("]")
                index = int(key) if key.lstrip("-").isdigit() else key.strip("'\"")
                try:
                    obj = obj[index]
                except (KeyError, IndexError):
                    return _MISSING
        return obj
    except (AttributeError, TypeError) as e:
        pytest.skip(f"ndv private path {path!r} changed ({type(e).__name__}: {e})")


def displayed_texture(viewer: Any, key: int = 0) -> Optional[np.ndarray]:
    """Pixels currently uploaded for channel ``key`` of an ``ndv.ArrayViewer``.

    ndv 0.5: ``viewer._lut_controllers[key].handles[0].data()`` (the vispy
    image handle's last ``set_data`` payload). ``None`` until the first data
    response for ``key`` has created the controller and its handle: callers poll.
    """
    handle = _walk(viewer, f"_lut_controllers[{key}].handles[0]")
    if handle is _MISSING:
        return None
    return np.asarray(handle.data())


def dims_slider(viewer: Any, dim: str) -> Any:
    """The Qt slider ndv built for dimension ``dim`` (its range mirrors the data).

    ndv 0.5: ``viewer._view._qwidget.dims_sliders._sliders[dim]``. ndv builds the
    sliders synchronously when it takes data, so a missing one is a failure.
    """
    sliders = _walk(viewer, "_view._qwidget.dims_sliders._sliders")
    if dim not in sliders:
        pytest.fail(f"ndv built no dims slider for {dim!r}; it has {list(sliders)}")
    return sliders[dim]


def is_idle(viewer: Any) -> bool:
    """True when no ndv data fetch is in flight (``ArrayViewer._is_idle``).

    True as soon as the worker's future is done; the main-thread callback that
    creates the texture handle may still be queued (see ``displayed_texture``).
    """
    return bool(_walk(viewer, "_is_idle")())


def texture_mean_is(ndv_viewer, expected: float, key: int = 0):
    """Predicate: ndv is idle and channel ``key``'s uploaded pixels average ``expected``.

    False (poll on) until the fetch has settled *and* its main-thread callback has
    created the channel's controller and handle: ``_is_idle()`` is True as soon as
    the worker's future is done, so right after idle the texture can still be
    absent (``displayed_texture`` -> None). Never a skip.
    """

    def settled() -> bool:
        if not is_idle(ndv_viewer):
            return False
        texture = displayed_texture(ndv_viewer, key)
        return texture is not None and bool(texture.mean() == expected)

    return settled
