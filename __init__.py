# Shim for submodule usage in Squid
# Re-exports from the nested ndviewer_light package
from .ndviewer_light import (
    __version__,
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
