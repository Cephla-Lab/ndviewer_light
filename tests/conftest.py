"""Test-session setup shared by all tests."""

import os

import pytest


def _pin_gl_texture_limits() -> None:
    """Answer ndv's GL texture-size query with fixed limits before any viewer exists.

    ndv 0.5 queries GL_MAX_TEXTURE_SIZE on the first add_image/add_volume
    (ndv.views._vispy._array_canvas._downcast_and_downsample ->
    ndv.views._vispy._util.get_max_texture_sizes). With QT_QPA_PLATFORM=offscreen
    there is no GL context and the raw glGetIntegerv segfaults (rc 139).
    NDVL_REAL_GL=1 keeps the real query (xvfb/cocoa runs).
    """
    if os.environ.get("NDVL_REAL_GL") == "1":
        return
    import ndv.views._vispy._array_canvas as canvas_mod
    import ndv.views._vispy._util as util_mod

    def fixed_limits() -> tuple[int, int]:
        # (2D, 3D) limits; 3D = 2048, the conservative GL_MAX_3D_TEXTURE_SIZE
        return (4096, 2048)

    util_mod.get_max_texture_sizes = fixed_limits
    # the canvas module binds the name at import time
    canvas_mod.get_max_texture_sizes = fixed_limits


_pin_gl_texture_limits()


@pytest.fixture
def window_qapp():
    """QApplication for ``@pytest.mark.window`` tests; skips on the offscreen platform.

    Decided from the live application, not the environment: most test modules
    ``setdefault`` QT_QPA_PLATFORM=offscreen at import, so with the variable unset a
    full-suite run is offscreen and the window smokes skip. Set
    ``QT_QPA_PLATFORM=cocoa`` (``xcb`` under xvfb) and ``NDVL_REAL_GL=1`` to run them.
    """
    from qtpy.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    if app.platformName() == "offscreen":
        pytest.skip("window test: the Qt platform is offscreen")
    return app
