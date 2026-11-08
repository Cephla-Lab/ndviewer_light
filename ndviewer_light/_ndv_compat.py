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


_SUPERSEDED_FLAG = "_ndviewer_light_draw_superseded"


def install_draw_superseded_frames() -> None:
    """Let ndv draw a late frame while only the slider position changed.

    ndv 0.5.0 tags each data request with a generation and draws a response only
    if it is from the latest one. Every slider move starts a new request, so
    while a slider is dragged faster than a frame loads (~25-30 ms for three
    2048 x 2048 channels), every response arrives superseded and the image stays
    frozen until the slider stops. ndv 0.4 drew those frames.

    This keeps ndv's guarantee that nothing older than what is on screen, or
    requested before a data, shape or layout change, is ever drawn, and draws a
    superseded response otherwise. Same logic as the upstream fix proposed to
    ndv (``ArrayViewer._is_drawable``); a no-op once ndv has it.

    Wraps private ``ArrayViewer`` methods: ``_cancel_futures`` (started fetches
    stay tracked so their generation is known), ``_apply_changes`` and
    ``_request_data`` (index-only requests keep earlier ones drawable),
    ``_on_dims_changed`` (shape change: earlier requests are not) and
    ``_on_data_response_ready`` (the drawing decision). Pinned by
    tests/test_ndv_compat.py. Remove when the ndv pin includes the upstream fix.
    """
    try:
        from ndv.controllers import ArrayViewer
        from ndv.views import _app

        cls = ArrayViewer
        orig_cancel = cls._cancel_futures
        orig_request = cls._request_data
        orig_apply = cls._apply_changes
        orig_dims = cls._on_dims_changed
        orig_ready = cls._on_data_response_ready.__wrapped__
        ensure_main_thread = _app.ensure_main_thread
    except (ImportError, AttributeError) as e:
        logger.warning(
            "ndv's private request path changed (%s): the image may not update "
            "while a slider is dragged",
            e,
        )
        return
    if hasattr(cls, "_is_drawable"):
        return  # ndv has the upstream fix
    if getattr(orig_cancel, _SUPERSEDED_FLAG, False):
        return

    def _cancel_futures(self) -> None:
        for f in list(self._futures):
            if f.cancel():
                # cancel() may already have run the done callback, which pops it
                self._futures.pop(f, None)
        self._viewer_model.show_progress_spinner = False

    def _request_data(self, *, index_only: bool = False) -> None:
        index_only = index_only or getattr(self, "_ndvl_index_only", False)
        orig_request(self)
        if not index_only:
            self._ndvl_min_gen = self._current_gen

    def _apply_changes(self, old, new) -> None:
        self._ndvl_index_only = (
            old.visible_axes == new.visible_axes
            and old.channel_axis == new.channel_axis
            and old.channel_mode == new.channel_mode
        )
        try:
            orig_apply(self, old, new)
        finally:
            self._ndvl_index_only = False

    def _on_dims_changed(self) -> None:
        self._ndvl_min_gen = self._current_gen + 1
        orig_dims(self)

    def _on_data_response_ready(self, future) -> None:
        gen = self._futures.get(future, -1)
        current = self._current_gen
        drawable = not future.cancelled() and (
            gen == current
            or 0 <= gen
            and getattr(self, "_ndvl_min_gen", 0) <= gen
            and getattr(self, "_ndvl_shown_gen", -1) <= gen
        )
        if drawable and gen != current:
            # ndv draws only the current generation: present this one as current
            self._current_gen = gen
            try:
                orig_ready(self, future)
            finally:
                if self._current_gen == gen:  # unless drawing started a new request
                    self._current_gen = current
        else:
            orig_ready(self, future)
        if drawable and future.exception() is None:
            self._ndvl_shown_gen = max(getattr(self, "_ndvl_shown_gen", -1), gen)

    for fn in (_cancel_futures, _request_data, _apply_changes, _on_dims_changed):
        setattr(fn, _SUPERSEDED_FLAG, True)
    cls._cancel_futures = _cancel_futures
    cls._request_data = _request_data
    cls._apply_changes = _apply_changes
    cls._on_dims_changed = _on_dims_changed
    cls._on_data_response_ready = ensure_main_thread(_on_data_response_ready)


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
