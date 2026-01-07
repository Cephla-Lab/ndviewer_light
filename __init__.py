# Shim for submodule usage in Squid
# Re-exports from the nested ndviewer_light package
from .ndviewer_light import *
from .ndviewer_light import (
    LightweightViewer,
    LightweightMainWindow,
    data_structure_changed,
    detect_format,
)
