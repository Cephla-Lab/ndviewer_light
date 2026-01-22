"""ndviewer_light - Lightweight NDV-based viewer for microscopy data."""

try:
    from importlib.metadata import version

    __version__ = version("ndviewer_light")
except Exception:
    __version__ = "unknown"

from .core import (
    FPATTERN,
    FPATTERN_OME,
    MAX_3D_TEXTURE_SIZE,
    LightweightMainWindow,
    LightweightViewer,
    MemoryBoundedLRUCache,
    data_structure_changed,
    detect_format,
    extract_ome_physical_sizes,
    read_acquisition_parameters,
    read_tiff_pixel_size,
    wavelength_to_colormap,
)

__all__ = [
    "__version__",
    "FPATTERN",
    "FPATTERN_OME",
    "MAX_3D_TEXTURE_SIZE",
    "LightweightMainWindow",
    "LightweightViewer",
    "MemoryBoundedLRUCache",
    "data_structure_changed",
    "detect_format",
    "extract_ome_physical_sizes",
    "read_acquisition_parameters",
    "read_tiff_pixel_size",
    "wavelength_to_colormap",
]
