"""Shims over private ndv internals, each with the condition for removing it.

Everything here is a deliberate exception to "public ndv APIs only"
(docs/DESIGN.md, "Private points that remain") and is pinned by
tests/test_ndv_compat.py, so an ndv rename fails CI instead of silently undoing
the shim. ``core.py`` imports only this module's public function.
"""

import logging

import numpy as np

logger = logging.getLogger(__name__)

_INSTALLED_FLAG = "_ndviewer_light_compact"


def install_compact_downsample() -> None:
    """Make ndv's texture downsampling hand vispy an array that owns its memory.

    ndv 0.5.0 strides an oversized texture to a *view* (``ndv.views._util.
    downsample_data``, bound by name in ``ndv.views._vispy._array_canvas`` at
    import), and vispy's ``VolumeVisual`` keeps whatever it is given as
    ``_last_data`` for CPU clim rescaling. So while a volume is shown, the full
    resolution (z, y, x) block of every displayed channel stays in RAM: a
    50 x 3000 x 3000 uint16 stack keeps 900 MB where 225 MB is on screen. The
    former ``Downsampling3DXarrayWrapper`` released that block because it
    materialised a zoomed copy. This wrapper copies the strided result into
    contiguous memory (one extra pass over the *reduced* volume on the main
    thread) and returns the same object when nothing was strided.

    Idempotent. If ndv's private path is gone it logs a warning and returns: the
    viewer keeps working with the extra retention, and the pinning test fails.

    Remove when ndv's ``downsample_data`` returns a contiguous copy itself and the
    pin moves past that release.
    """
    try:
        import ndv.views._vispy._array_canvas as canvas_mod

        original = canvas_mod.downsample_data
    except (ImportError, AttributeError) as e:
        logger.warning(
            "ndv's private downsample path changed (%s): a 3D volume larger than "
            "the GL texture limit keeps its full-resolution block in RAM",
            e,
        )
        return
    if getattr(original, _INSTALLED_FLAG, False):
        return

    def compact_downsample_data(data, max_size, *, warn=True):
        data, factors = original(data, max_size, warn=warn)
        if any(f > 1 for f in factors):
            data = np.ascontiguousarray(data)
        return data, factors

    setattr(compact_downsample_data, _INSTALLED_FLAG, True)
    compact_downsample_data.__wrapped__ = original
    canvas_mod.downsample_data = compact_downsample_data


def _make_fast_coords_wrapper_class():
    import ndv

    class FastCoordsXarrayWrapper(ndv.DataWrapper):
        """xarray wrapper whose ``coords`` are plain numpy arrays.

        ndv 0.5 re-resolves the display state on every slider move, on the main
        thread, and that walks every coordinate in Python: ``tuple(coords[d])``
        (``_resolve._norm_data_coords``) and ``[float(v) for v in coord]``
        (``DataWrapper.axis_scales``). The stock wrapper returns xarray's own
        coords, and iterating one of those builds a new DataArray per element:
        about 8,000 per move for a 2048 x 2048 image, 100-450 ms of frozen UI
        per slider step. Iterating the underlying numpy array costs ~1 ms.

        Public ndv API only (a ``DataWrapper`` subclass, passed in directly).
        ``supports`` returns False so ``DataWrapper.create`` never picks this
        class for other xarray objects. Remove when ndv iterates coordinate
        values instead of DataArrays.
        """

        @classmethod
        def supports(cls, obj) -> bool:
            return False

        @property
        def coords(self):
            data = self._data
            return {
                dim: (data.coords[dim].values if dim in data.coords else range(n))
                for dim, n in zip(data.dims, data.shape)
            }

    return FastCoordsXarrayWrapper


_fast_coords_cls = None


def wrap_xarray_fast_coords(data):
    """Wrap an ``xr.DataArray`` for ``ndv.ArrayViewer`` with fast coordinate access."""
    global _fast_coords_cls
    if _fast_coords_cls is None:
        _fast_coords_cls = _make_fast_coords_wrapper_class()
    return _fast_coords_cls(data)
