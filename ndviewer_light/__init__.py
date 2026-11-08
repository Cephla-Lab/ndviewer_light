"""ndviewer_light - Lightweight NDV-based viewer for microscopy data."""

try:
    from importlib.metadata import version

    __version__ = version("ndviewer_light")
except Exception:
    __version__ = "unknown"

from .core import (
    FPATTERN,
    FPATTERN_OME,
    LightweightMainWindow,
    LightweightViewer,
    MemoryBoundedLRUCache,
    data_structure_changed,
    detect_format,
    discover_zarr_v3_fovs,
    extract_ome_physical_sizes,
    hex_to_colormap,
    open_zarr_tensorstore,
    parse_zarr_v3_metadata,
    read_acquisition_parameters,
    read_tiff_pixel_size,
    wavelength_to_colormap,
)
from .ome_zarr import OmeZarrImage, discover_ome_zarr, parse_image

__all__ = [
    "__version__",
    "FPATTERN",
    "FPATTERN_OME",
    "LightweightMainWindow",
    "LightweightViewer",
    "MemoryBoundedLRUCache",
    "OmeZarrImage",
    "data_structure_changed",
    "detect_format",
    "discover_ome_zarr",
    "discover_zarr_v3_fovs",
    "extract_ome_physical_sizes",
    "hex_to_colormap",
    "open_zarr_tensorstore",
    "parse_image",
    "parse_zarr_v3_metadata",
    "read_acquisition_parameters",
    "read_tiff_pixel_size",
    "wavelength_to_colormap",
]
