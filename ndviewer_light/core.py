"""
Lightweight NDV-based viewer

Supports: OME-TIFF and single-TIFF acquisitions with lazy loading via dask.
Lazy loading enables fast initial display by only reading image planes on-demand.
"""

import json
import logging
import re
import sys
import threading
from collections import OrderedDict
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import numpy as np
from PyQt5.QtCore import QSize, Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QColor, QPalette
from PyQt5.QtWidgets import (
    QApplication,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QStyleFactory,
    QVBoxLayout,
    QWidget,
)

# Try to import QIconifyIcon for NDV-style play buttons
try:
    from superqt.iconify import QIconifyIcon

    ICONIFY_AVAILABLE = True
except ImportError:
    ICONIFY_AVAILABLE = False

# Try to import QLabeledSlider from superqt (same as NDV uses)
try:
    from superqt import QLabeledSlider

    SUPERQT_AVAILABLE = True
except ImportError:
    from PyQt5.QtWidgets import QSlider

    SUPERQT_AVAILABLE = False

# NDV slider style (matches NDV's internal sliders)
NDV_SLIDER_STYLE = """
QSlider::groove:horizontal {
    height: 15px;
    background: qlineargradient(
        x1:0, y1:0, x2:0, y2:1,
        stop:0 rgba(128, 128, 128, 0.25),
        stop:1 rgba(128, 128, 128, 0.1)
    );
    border-radius: 3px;
}
QSlider::handle:horizontal {
    width: 38px;
    background: #999999;
    border-radius: 3px;
}
QSlider::sub-page:horizontal {
    background: qlineargradient(
        x1:0, y1:0, x2:0, y2:1,
        stop:0 rgba(100, 100, 100, 0.25),
        stop:1 rgba(100, 100, 100, 0.1)
    );
}
QLabel { font-size: 12px; }
SliderLabel { font-size: 10px; }
"""

if TYPE_CHECKING:
    import xarray as xr

# Constants
TIFF_EXTENSIONS = {".tif", ".tiff"}
LIVE_REFRESH_INTERVAL_MS = 750
SLIDER_PLAY_INTERVAL_MS = 100  # Animation interval for play buttons
PLANE_CACHE_MAX_MEMORY_BYTES = 256 * 1024 * 1024  # 256MB for z-stack plane cache

# Play button style (matches NDV's PlayButton)
PLAY_BUTTON_STYLE = "QPushButton {border: none; padding: 0; margin: 0;}"


def _create_play_button(parent=None) -> QPushButton:
    """Create a play button matching NDV's style."""
    if ICONIFY_AVAILABLE:
        icn = QIconifyIcon("bi:play-fill", color="#888888")
        icn.addKey("bi:pause-fill", state=QIconifyIcon.State.On, color="#4580DD")
        btn = QPushButton(icn, "", parent)
        btn.setIconSize(QSize(16, 16))
    else:
        btn = QPushButton("▶", parent)
    btn.setCheckable(True)
    btn.setFixedSize(18, 18)
    btn.setStyleSheet(PLAY_BUTTON_STYLE)
    return btn


logger = logging.getLogger(__name__)


class MemoryBoundedLRUCache:
    """Thread-safe LRU cache with memory-based size limit.

    Evicts least-recently-used entries when memory limit is exceeded.
    Designed for caching large numpy arrays (image planes).

    Thread safety is required because dask workers may load planes concurrently.
    """

    def __init__(self, max_memory_bytes: int):
        self._max_memory = max_memory_bytes
        self._current_memory = 0
        self._cache: OrderedDict = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key: tuple) -> Optional[np.ndarray]:
        """Get item from cache, marking it as recently used."""
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                return self._cache[key]
            return None

    def put(self, key: tuple, value: np.ndarray) -> None:
        """Add item to cache, evicting LRU entries if needed."""
        item_size = value.nbytes

        # Don't cache if single item exceeds limit
        if item_size > self._max_memory:
            logger.debug(
                "Cannot cache item (size %d bytes exceeds max %d bytes): key=%s",
                item_size,
                self._max_memory,
                key,
            )
            return

        with self._lock:
            # Remove existing entry if present
            if key in self._cache:
                self._current_memory -= self._cache[key].nbytes
                del self._cache[key]

            # Evict LRU entries until we have room
            while self._current_memory + item_size > self._max_memory and self._cache:
                oldest_key, oldest_value = self._cache.popitem(last=False)
                self._current_memory -= oldest_value.nbytes

            self._cache[key] = value
            self._current_memory += item_size

    def clear(self) -> None:
        """Clear all cached entries."""
        with self._lock:
            self._cache.clear()
            self._current_memory = 0

    def __contains__(self, key: tuple) -> bool:
        with self._lock:
            return key in self._cache

    def __len__(self) -> int:
        with self._lock:
            return len(self._cache)


# Module-level variable for voxel scale (used by monkey-patched add_volume)
_current_voxel_scale: Optional[Tuple[float, float, float]] = None

# NDV viewer
try:
    import ndv

    NDV_AVAILABLE = True
    # Monkeypatch superqt slider to respect full ranges (self-contained)
    try:
        from superqt.sliders import QLabeledSlider

        if not getattr(QLabeledSlider, "_ndv_range_patch", False):
            _orig_setRange = QLabeledSlider.setRange

            def _patched_setRange(self, a, b):
                _orig_setRange(self, a, b)
                if hasattr(self, "_slider"):
                    self._slider.setMinimum(a)
                    self._slider.setMaximum(b)
                if hasattr(self, "_label"):
                    try:
                        self._label.setRange(a, b)
                    except Exception as e:
                        logger.debug("Failed to set label range: %s", e)

            QLabeledSlider.setRange = _patched_setRange
            QLabeledSlider._ndv_range_patch = True
    except ImportError:
        pass  # superqt not available

    # Monkeypatch vispy VolumeVisual to support anisotropic voxels
    # This allows correct 3D rendering when Z step differs from XY pixel size
    try:
        from ndv.views._vispy._array_canvas import VispyArrayCanvas
        from vispy.visuals.volume import VolumeVisual

        if not getattr(VolumeVisual, "_voxel_scale_patch", False):
            _orig_init = VolumeVisual.__init__
            _orig_create_vertex_data = VolumeVisual._create_vertex_data

            def _patched_init(self, *args, **kwargs):
                """Initialize VolumeVisual and capture the current voxel scale.

                Storing the scale as an instance attribute ensures thread safety
                when multiple volumes are created concurrently - each volume
                captures the scale that was active at its construction time.
                """
                _orig_init(self, *args, **kwargs)
                # Capture the voxel scale active at construction time
                # Use try/except to handle frozen vispy objects (e.g., napari's Volume)
                global _current_voxel_scale
                try:
                    self._voxel_scale = _current_voxel_scale
                except AttributeError:
                    # Object is frozen (e.g., napari), skip the patch
                    pass

            VolumeVisual.__init__ = _patched_init

            def _patched_create_vertex_data(self):
                """Create vertices with Z scaling for anisotropic voxels.

                Uses the instance's _voxel_scale attribute (set at construction)
                rather than the global to ensure correct scaling even when
                multiple volumes exist with different scales.

                Falls back to original implementation when no scale is set.
                """
                # If no scale set, use original implementation
                scale = getattr(self, "_voxel_scale", None)
                if scale is None:
                    return _orig_create_vertex_data(self)

                shape = self._vol_shape

                # Get corner coordinates with Z scaling
                x0, x1 = -0.5, shape[2] - 0.5
                y0, y1 = -0.5, shape[1] - 0.5

                # Apply Z scale from instance attribute
                sz = scale[2]
                z0, z1 = -0.5 * sz, (shape[0] - 0.5) * sz

                pos = np.array(
                    [
                        [x0, y0, z0],
                        [x1, y0, z0],
                        [x0, y1, z0],
                        [x1, y1, z0],
                        [x0, y0, z1],
                        [x1, y0, z1],
                        [x0, y1, z1],
                        [x1, y1, z1],
                    ],
                    dtype=np.float32,
                )

                indices = np.array(
                    [2, 6, 0, 4, 5, 6, 7, 2, 3, 0, 1, 5, 3, 7], dtype=np.uint32
                )

                self._vertices.set_data(pos)
                self._index_buffer.set_data(indices)

            VolumeVisual._create_vertex_data = _patched_create_vertex_data
            VolumeVisual._voxel_scale_patch = True
            logger.info("Voxel scale patch applied to VolumeVisual")

        # Also patch add_volume to update camera range
        if not getattr(VispyArrayCanvas, "_camera_scale_patch", False):
            _orig_add_volume = VispyArrayCanvas.add_volume

            def _patched_add_volume(self, data=None):
                global _current_voxel_scale
                handle = _orig_add_volume(self, data)
                # Update camera to account for scaled Z dimension
                if _current_voxel_scale is not None and data is not None:
                    # Ensure data has at least 3 dimensions
                    shape = getattr(data, "shape", None)
                    if shape is None or len(shape) < 3:
                        return handle
                    try:
                        sz = _current_voxel_scale[2]
                        if abs(sz - 1.0) > 0.01:
                            z_size = shape[0] * sz
                            max_size = max(shape[1], shape[2], z_size)
                            # Add margin to scale_factor for comfortable viewing distance
                            self._camera.scale_factor = max_size + 6
                            self._view.camera.set_range(
                                x=(0, shape[2]),
                                y=(0, shape[1]),
                                z=(0, z_size),
                                margin=0.01,
                            )
                    except Exception as e:
                        logger.warning(
                            "Failed to adjust camera for anisotropic voxels: %s", e
                        )
                return handle

            VispyArrayCanvas.add_volume = _patched_add_volume
            VispyArrayCanvas._camera_scale_patch = True
    except ImportError:
        pass  # vispy not available

except ImportError:
    NDV_AVAILABLE = False

# Lazy loading
try:
    from functools import lru_cache

    import dask.array as da
    import tifffile as tf
    import xarray as xr
    from scipy.ndimage import zoom as ndimage_zoom

    LAZY_LOADING_AVAILABLE = True
except ImportError:
    LAZY_LOADING_AVAILABLE = False

# OpenGL 3D texture size limit (conservative estimate for most GPUs)
MAX_3D_TEXTURE_SIZE = 2048

# Channel label update retry configuration
CHANNEL_LABEL_UPDATE_MAX_RETRIES = 20
CHANNEL_LABEL_UPDATE_RETRY_DELAY_MS = 100

# Register custom DataWrapper for automatic 3D downsampling
if NDV_AVAILABLE and LAZY_LOADING_AVAILABLE:
    from collections.abc import Mapping

    from ndv.models._data_wrapper import XarrayWrapper

    class Downsampling3DXarrayWrapper(XarrayWrapper):
        """XarrayWrapper that automatically downsamples 3D volumes for OpenGL.

        This wrapper extends NDV's XarrayWrapper to detect when a 3D volume
        request would exceed OpenGL texture limits and automatically downsamples
        the data. 2D slice requests remain at full resolution.
        """

        # Higher priority than default XarrayWrapper (50)
        PRIORITY = 40

        # Class-level cache for OpenGL texture limit (queried once, shared by all instances)
        _cached_max_texture_size: Optional[int] = None

        def __init__(self, data: xr.DataArray):
            super().__init__(data)

        @classmethod
        def _get_max_texture_size(cls) -> int:
            """Query and cache the GPU's GL_MAX_3D_TEXTURE_SIZE.

            This is queried lazily on first 3D request when OpenGL context exists.
            Falls back to conservative default if query fails or no context available.
            """
            if cls._cached_max_texture_size is None:
                try:
                    # Check if vispy has an active GL context before querying
                    # Calling OpenGL without a context causes segfault
                    from vispy import app

                    # Check for active vispy application - use _backend_module which
                    # is set when a backend is actually loaded and initialized
                    backend = getattr(app, "_backend_module", None)
                    if backend is None:
                        logger.debug(
                            "No vispy backend loaded - using fallback texture size"
                        )
                        cls._cached_max_texture_size = MAX_3D_TEXTURE_SIZE
                        return cls._cached_max_texture_size

                    from OpenGL.GL import GL_MAX_3D_TEXTURE_SIZE, glGetIntegerv

                    limit = glGetIntegerv(GL_MAX_3D_TEXTURE_SIZE)
                    cls._cached_max_texture_size = int(limit)
                    logger.info(f"Detected GL_MAX_3D_TEXTURE_SIZE: {limit}")
                except Exception as e:
                    logger.debug(f"Failed to query GL_MAX_3D_TEXTURE_SIZE: {e}")
                    cls._cached_max_texture_size = MAX_3D_TEXTURE_SIZE  # Fallback
            return cls._cached_max_texture_size

        @classmethod
        def supports(cls, obj) -> bool:
            """Check if this wrapper supports the given object."""
            # Note: LAZY_LOADING_AVAILABLE check is unnecessary here since this
            # class is only defined when LAZY_LOADING_AVAILABLE is True (line 85)
            return isinstance(obj, xr.DataArray)

        def isel(self, index: Mapping[int, int | slice]) -> np.ndarray:
            """Return a slice of the data, with automatic 3D downsampling.

            For 2D slices (viewing a single plane), returns full resolution.
            For 3D volumes (viewing a stack), downsamples if needed to fit
            within OpenGL texture limits.

            Downsampling strategy:
            - If physical pixel sizes are known (pixel_size_um, dz_um in attrs),
              scale to maintain correct physical aspect ratio
            - Otherwise: z scaled independently, x/y scaled uniformly
            - channel/time/fov: never scaled
            """
            # Get the data using parent's implementation
            data = super().isel(index)

            # Determine which original dimensions are non-singleton
            dims = self._data.dims
            non_singleton_dims = []
            for i, dim in enumerate(dims):
                idx = index.get(i, slice(None))
                if isinstance(idx, slice):
                    dim_size = self._data.shape[i]
                    start = idx.start or 0
                    stop = idx.stop or dim_size
                    if stop - start > 1:
                        non_singleton_dims.append(str(dim).lower())

            # Check if we have spatial z dimension (indicates 3D volume)
            spatial_z_names = {"z", "z_level", "depth", "focus"}
            has_z = any(d in spatial_z_names for d in non_singleton_dims)
            if not has_z:
                return data  # Not a 3D volume request

            # Check if any spatial dimension exceeds the texture limit
            max_texture_size = self._get_max_texture_size()

            # First pass: find dimensions and their sizes in output data
            dim_info = []  # [(dim_name, size), ...]
            for i, dim in enumerate(dims):
                idx = index.get(i, slice(None))
                if isinstance(idx, int):
                    continue  # Dropped dimension
                dim_info.append((str(dim).lower(), data.shape[len(dim_info)]))

            # Compute zoom factors for downsampling
            zoom_factors, needs_downsampling = self._compute_simple_zoom_factors(
                dim_info, max_texture_size, spatial_z_names
            )

            if needs_downsampling:
                logger.info(
                    f"Downsampling 3D volume from {data.shape} "
                    f"(factors={[f'{z:.3f}' for z in zoom_factors]}) for OpenGL rendering"
                )

                # Use order=0 (nearest neighbor) for speed - much faster than bilinear
                try:
                    downsampled = ndimage_zoom(data, zoom_factors, order=0)
                    return downsampled.astype(data.dtype)
                except Exception as e:
                    logger.warning(f"Downsampling failed: {e}, returning original data")
                    return data

            return data

        def _compute_simple_zoom_factors(
            self, dim_info: list, max_texture_size: int, spatial_z_names: set
        ) -> tuple:
            """Compute zoom factors for 3D volume downsampling.

            Strategy:
            - XY: scaled uniformly (same factor for X and Y) to preserve XY aspect ratio
            - Z: scaled independently only if it exceeds the texture limit
            - Non-spatial dims (channel, time, fov): never scaled

            Note: Physical aspect ratio correction is handled via vertex scaling
            in the vispy VolumeVisual patch.
            """
            # Calculate xy scale factor (uniform for x and y to preserve XY aspect)
            xy_sizes = [size for name, size in dim_info if name in {"y", "x"}]
            xy_max = max(xy_sizes) if xy_sizes else 0
            xy_scale = max_texture_size / xy_max if xy_max > max_texture_size else 1.0

            # Build zoom factors
            zoom_factors = []
            needs_downsampling = False
            for dim_name, dim_size in dim_info:
                if dim_name in {"y", "x"}:
                    zoom_factors.append(xy_scale)
                    if xy_scale < 1.0:
                        needs_downsampling = True
                elif dim_name in spatial_z_names and dim_size > max_texture_size:
                    z_scale = max_texture_size / dim_size
                    zoom_factors.append(z_scale)
                    needs_downsampling = True
                else:
                    zoom_factors.append(1.0)

            return zoom_factors, needs_downsampling


# Filename patterns (from common.py)
FPATTERN = re.compile(
    r"(?P<r>[^_]+)_(?P<f>\d+)_(?P<z>\d+)_(?P<c>.+)\.tiff?", re.IGNORECASE
)
FPATTERN_OME = re.compile(r"(?P<r>[^_]+)_(?P<f>\d+)\.ome\.tiff?", re.IGNORECASE)


# Helper functions
def extract_wavelength(channel_str: str):
    """Extract wavelength (nm) from channel string; None if unknown."""
    if not channel_str:
        return None
    lower = channel_str.lower()
    if re.fullmatch(r"ch\d+", lower):
        return None
    # Direct wavelength pattern
    if m := re.search(r"(\d{3,4})[ _]*nm", channel_str, re.IGNORECASE):
        return int(m.group(1))

    # Common fluorophores
    fluor_map = {
        "dapi": 405,
        "hoechst": 405,
        "gfp": 488,
        "fitc": 488,
        "alexa488": 488,
        "tritc": 561,
        "cy3": 561,
        "mcherry": 561,
        "cy5": 640,
        "alexa647": 640,
        "cy7": 730,
    }
    channel_lower = channel_str.lower()
    for fluor, wl in fluor_map.items():
        if fluor in channel_lower:
            return wl

    # Fallback
    numbers = re.findall(r"\d{3,4}", channel_str)
    if numbers:
        # Prefer the last 3-4 digit group (likely wavelength)
        val = int(numbers[-1])
        return val if val > 0 else None
    return None


def extract_ome_physical_sizes(
    ome_metadata: str,
) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """Extract physical pixel sizes from OME-XML metadata.

    Returns:
        Tuple of (pixel_size_x_um, pixel_size_y_um, pixel_size_z_um).
        Values are in micrometers. None if not found or unable to parse.
    """
    if not ome_metadata:
        return None, None, None

    try:
        import xml.etree.ElementTree as ET

        root = ET.fromstring(ome_metadata)
        # Try multiple OME namespace versions
        namespaces = [
            {"ome": "http://www.openmicroscopy.org/Schemas/OME/2016-06"},
            {"ome": "http://www.openmicroscopy.org/Schemas/OME/2015-01"},
            {"ome": "http://www.openmicroscopy.org/Schemas/OME/2013-06"},
            {},  # No namespace fallback
        ]

        for ns in namespaces:
            # Find Pixels element which contains physical size attributes
            if ns:
                pixels = root.find(".//ome:Pixels", ns)
            else:
                # Try without or with any namespace (.//{*} matches any namespace)
                pixels = root.find(".//{*}Pixels")

            if pixels is not None:
                size_x = pixels.get("PhysicalSizeX")
                size_y = pixels.get("PhysicalSizeY")
                size_z = pixels.get("PhysicalSizeZ")
                unit_x = pixels.get("PhysicalSizeXUnit", "µm")
                unit_y = pixels.get("PhysicalSizeYUnit", "µm")
                unit_z = pixels.get("PhysicalSizeZUnit", "µm")

                def to_micrometers(value: Optional[str], unit: str) -> Optional[float]:
                    if value is None:
                        return None
                    try:
                        val = float(value)
                        # Convert to micrometers based on unit
                        unit_lower = unit.lower()
                        if unit_lower in ("nm", "nanometer", "nanometers"):
                            result = val / 1000.0
                        elif unit_lower in ("mm", "millimeter", "millimeters"):
                            result = val * 1000.0
                        elif unit_lower in ("m", "meter", "meters"):
                            result = val * 1e6
                        else:
                            # Default assumes micrometers (µm, um, micron, etc.)
                            result = val
                        # Physical sizes must be strictly positive
                        if result <= 0:
                            return None
                        return result
                    except (ValueError, TypeError):
                        return None

                px = to_micrometers(size_x, unit_x)
                py = to_micrometers(size_y, unit_y)
                pz = to_micrometers(size_z, unit_z)

                if px is not None or py is not None or pz is not None:
                    return px, py, pz

    except Exception as e:
        logger.debug("Failed to extract physical sizes from OME metadata: %s", e)

    return None, None, None


def read_acquisition_parameters(
    base_path: Path,
) -> Tuple[Optional[float], Optional[float]]:
    """Read pixel size and dz from acquisition parameters JSON file.

    Supports both "acquisition_parameters.json" and "acquisition parameters.json".
    Can compute pixel size from sensor_pixel_size_um and objective magnification.

    Returns:
        Tuple of (pixel_size_um, dz_um). None if not found.
    """
    # Try both filename variants
    params_file = base_path / "acquisition_parameters.json"
    if not params_file.exists():
        params_file = base_path / "acquisition parameters.json"
    if not params_file.exists():
        return None, None

    try:
        with open(params_file, "r") as f:
            params = json.load(f)

        pixel_size = None
        dz = None

        # Try direct pixel size keys first
        for key in ["pixel_size_um", "pixel_size", "pixelSize", "pixel_size_xy"]:
            if key in params:
                try:
                    candidate_pixel = float(params[key])
                except (TypeError, ValueError):
                    continue
                # Sanity check: typical microscopy pixel sizes are 0.1-10 µm
                # Range 0.01-100 µm covers most use cases including low-mag imaging
                if 0.01 < candidate_pixel <= 100:
                    pixel_size = candidate_pixel
                    break

        # If not found, compute from sensor pixel size and magnification
        # Account for tube lens ratio: actual_mag = nominal_mag × (tube_lens / obj_tube_lens)
        if pixel_size is None:
            sensor_pixel = params.get("sensor_pixel_size_um")
            objective = params.get("objective", {})
            if isinstance(objective, dict):
                nominal_mag = objective.get("magnification")
                obj_tube_lens = objective.get("tube_lens_f_mm")
            else:
                nominal_mag = None
                obj_tube_lens = None
            tube_lens = params.get("tube_lens_mm")

            if sensor_pixel is not None and nominal_mag is not None and nominal_mag > 0:
                # Compute actual magnification with tube lens correction
                if (
                    tube_lens is not None
                    and obj_tube_lens is not None
                    and obj_tube_lens > 0
                ):
                    actual_mag = float(nominal_mag) * (
                        float(tube_lens) / float(obj_tube_lens)
                    )
                else:
                    actual_mag = float(nominal_mag)
                computed = float(sensor_pixel) / actual_mag
                # Sanity check: typical microscopy pixel sizes are 0.1-10 µm
                # Range 0.01-100 µm covers most use cases including low-mag imaging
                if 0.01 < computed <= 100:
                    pixel_size = computed

        # Try common key names for z spacing
        for key in [
            "dz_um",
            "dz",
            "z_step",
            "zStep",
            "z_spacing",
            "pixel_size_z",
            "dz(um)",
        ]:
            if key in params:
                try:
                    candidate_dz = float(params[key])
                except (TypeError, ValueError):
                    continue
                # dz must be strictly positive to be physically meaningful
                if candidate_dz > 0:
                    dz = candidate_dz
                    break

        return pixel_size, dz

    except Exception as e:
        logger.debug("Failed to read acquisition parameters: %s", e)
        return None, None


def read_tiff_pixel_size(tiff_path: str) -> Optional[float]:
    """Read pixel size from TIFF metadata tags.

    Attempts to extract pixel size from (in priority order):
    1. ImageDescription tag (JSON metadata from some microscopy software)
    2. XResolution/YResolution tags with ResolutionUnit

    Note on ResolutionUnit: Only inch (2) and centimeter (3) units are supported.
    Unit value 1 ("no absolute unit") is explicitly rejected because it cannot
    be reliably converted to physical units. Many image editors set resolution
    tags without meaningful physical units, so we require explicit inch/cm units.

    Returns:
        Pixel size in micrometers, or None if not found.
    """
    if not LAZY_LOADING_AVAILABLE:
        return None

    try:
        with tf.TiffFile(tiff_path) as tif:
            page = tif.pages[0]

            # Try ImageDescription tag FIRST for JSON metadata
            # (more reliable for microscopy data)
            desc = page.tags.get("ImageDescription")
            if desc is not None:
                desc_str = desc.value
                if isinstance(desc_str, bytes):
                    desc_str = desc_str.decode("utf-8", errors="ignore")

                # Try to parse as JSON
                try:
                    metadata = json.loads(desc_str)
                    for key in [
                        "pixel_size_um",
                        "pixel_size",
                        "PixelSize",
                        "pixelSize",
                    ]:
                        if key in metadata:
                            val = float(metadata[key])
                            # Require strictly positive value
                            if val <= 0:
                                continue
                            # Sanity check: typical microscopy pixel sizes are 0.1-10 µm
                            # Range 0.01-100 µm covers most use cases including low-mag imaging
                            if 0.01 < val <= 100:
                                return val
                except (json.JSONDecodeError, ValueError, TypeError):
                    # JSON parsing failed; fall through to resolution tags below
                    pass

            # Try XResolution/YResolution tags with proper unit
            x_res = page.tags.get("XResolution")
            res_unit = page.tags.get("ResolutionUnit")

            # Only use resolution tags if we have a proper unit (inch=2 or cm=3)
            unit_value = res_unit.value if res_unit else 1
            if unit_value not in (2, 3):
                return None  # No unit or unknown unit - can't reliably convert

            if x_res is not None:
                # XResolution is stored as a fraction (numerator, denominator)
                x_res_value = x_res.value
                if isinstance(x_res_value, tuple) and len(x_res_value) == 2:
                    pixels_per_unit = x_res_value[0] / x_res_value[1]
                else:
                    pixels_per_unit = float(x_res_value)

                # Skip default/invalid values (must be > 1 to be meaningful)
                if pixels_per_unit <= 1:
                    return None

                # Convert to micrometers based on unit
                if unit_value == 2:  # inch
                    # pixels/inch -> um/pixel: 25400 um/inch / pixels_per_inch
                    pixel_size_um = 25400.0 / pixels_per_unit
                else:  # centimeter (unit_value == 3)
                    # pixels/cm -> um/pixel: 10000 um/cm / pixels_per_cm
                    pixel_size_um = 10000.0 / pixels_per_unit

                # Sanity check: typical microscopy pixel sizes are 0.1-10 µm
                # Range 0.01-100 µm covers most use cases including low-mag imaging
                if 0.01 < pixel_size_um <= 100:
                    return pixel_size_um

    except Exception as e:
        logger.debug("Failed to read pixel size from TIFF tags: %s", e)

    return None


def detect_format(base_path: Path) -> str:
    """Detect OME-TIFF vs single-TIFF format."""
    ome_dir = base_path / "ome_tiff"
    if ome_dir.exists():
        if any(".ome" in f.name for f in ome_dir.glob("*.tif*")):
            return "ome_tiff"

    first_tp = next(
        (d for d in base_path.iterdir() if d.is_dir() and d.name.isdigit()), None
    )
    if first_tp:
        if any(".ome" in f.name for f in first_tp.glob("*.tif*")):
            return "ome_tiff"
    return "single_tiff"


def wavelength_to_colormap(wavelength: Optional[int]) -> str:
    """Map wavelength to NDV colormap."""
    if wavelength is None or wavelength == 0:
        return "gray"
    if wavelength <= 420:
        return "blue"
    elif 470 <= wavelength <= 510:
        return "green"
    elif 540 <= wavelength <= 590:
        return "yellow"
    elif 620 <= wavelength <= 660:
        return "red"
    elif wavelength >= 700:
        return "magenta"
    return "gray"


def data_structure_changed(
    old_data: Optional["xr.DataArray"], new_data: "xr.DataArray"
) -> bool:
    """Check if data structure changed significantly (requiring full viewer rebuild).

    This is a module-level utility function that detects changes in dimensions,
    dtype, channel count, channel names, or LUT configuration that would require
    rebuilding the NDV viewer rather than just swapping data in-place.

    This function is used by both the LightweightViewer class and unit tests,
    ensuring a single source of truth for the comparison logic.

    Args:
        old_data: Previous dataset state. May be ``None`` if no prior dataset
            exists; when ``None``, the structure is treated as changed.
        new_data: Newly loaded dataset to compare against ``old_data``.

    Returns:
        True if structure changed and viewer needs full rebuild.

    Raises:
        Any exception from xarray attribute access is propagated to the caller.
    """
    if old_data is None:
        return True

    # Check if dims changed
    if old_data.dims != new_data.dims:
        return True

    # Check if dtype changed (may need different contrast limits)
    if old_data.dtype != new_data.dtype:
        return True

    # Check if channel count changed; treat missing "channel" dim as having 0 channels
    if old_data.sizes.get("channel", 0) != new_data.sizes.get("channel", 0):
        return True

    # Check if channel names changed
    old_names = old_data.attrs.get("channel_names", [])
    new_names = new_data.attrs.get("channel_names", [])
    if old_names != new_names:
        return True

    # Check if LUTs changed
    old_luts = old_data.attrs.get("luts", {})
    new_luts = new_data.attrs.get("luts", {})
    if old_luts != new_luts:
        return True

    return False


def _apply_dark_theme(widget: QWidget) -> None:
    """Apply dark Fusion theme to a widget."""
    widget.setStyle(QStyleFactory.create("Fusion"))

    p = widget.palette()
    p.setColor(QPalette.Window, QColor(53, 53, 53))
    p.setColor(QPalette.WindowText, QColor(255, 255, 255))
    p.setColor(QPalette.Base, QColor(35, 35, 35))
    p.setColor(QPalette.AlternateBase, QColor(53, 53, 53))
    p.setColor(QPalette.Text, QColor(255, 255, 255))
    p.setColor(QPalette.Button, QColor(53, 53, 53))
    p.setColor(QPalette.ButtonText, QColor(255, 255, 255))
    p.setColor(QPalette.Highlight, QColor(42, 130, 218))
    p.setColor(QPalette.HighlightedText, QColor(35, 35, 35))
    widget.setPalette(p)


class LauncherWindow(QMainWindow):
    """Separate launcher window with dropbox for dataset selection."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("NDViewer Lightweight - Open Dataset")
        self.setGeometry(100, 100, 400, 300)  # 4:3 aspect, narrower
        self._set_dark_theme()

        central = QWidget()
        layout = QVBoxLayout()

        # Drop zone / Open button
        self.drop_label = QLabel("Drop folder here\nor click to open")
        self.drop_label.setAlignment(Qt.AlignCenter)
        self.drop_label.setStyleSheet("""
            QLabel {
                border: 2px dashed #666;
                border-radius: 10px;
                padding: 40px;
                background: #2a2a2a;
                color: #aaa;
                font-size: 14px;
            }
            QLabel:hover {
                border-color: #888;
                background: #333;
            }
        """)
        self.drop_label.setMinimumHeight(150)
        self.drop_label.mousePressEvent = lambda e: self._open_folder_dialog()
        layout.addWidget(self.drop_label)

        # Status
        self.status_label = QLabel("No dataset loaded")
        self.status_label.setStyleSheet("color: #888; padding: 5px;")
        self.status_label.setAlignment(Qt.AlignCenter)
        # layout.addWidget(self.status_label) # hide status label

        central.setLayout(layout)
        self.setCentralWidget(central)
        self.setAcceptDrops(True)

        self.viewer_window = None

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event):
        urls = event.mimeData().urls()
        if urls:
            path = urls[0].toLocalFile()
            self._launch_viewer(path)

    def _open_folder_dialog(self):
        path = QFileDialog.getExistingDirectory(self, "Select Dataset Folder")
        if path:
            self._launch_viewer(path)

    def _launch_viewer(self, path: str):
        """Launch main viewer window with dataset."""
        self.status_label.setText(f"Opening: {Path(path).name}...")
        QApplication.processEvents()

        # Keep launcher open; allow multiple drops without restarting.
        if self.viewer_window:
            try:
                self.viewer_window.close()
            except Exception as e:
                logger.debug("Failed to close previous viewer: %s", e)
        self.viewer_window = LightweightMainWindow(path)
        self.viewer_window.show()

    def _set_dark_theme(self):
        _apply_dark_theme(self)


class LightweightViewer(QWidget):
    """Minimal NDV-based viewer with external FOV/Time navigation.

    For live acquisition, use the push-based API:
    - start_acquisition() to configure channels, z-levels, dimensions
    - register_image() to register each saved image
    - load_fov() to navigate to a specific position

    For viewing existing datasets, use load_dataset().
    """

    # Signal for thread-safe UI updates from register_image()
    # Signature: (t, fov_idx, _unused1, _unused2) - last two reserved for future use
    _image_registered = pyqtSignal(int, int, int, int)

    dataset_path: str
    ndv_viewer: Optional["ndv.ArrayViewer"]
    _xarray_data: Optional["xr.DataArray"]
    _open_handles: List
    _last_sig: Optional[tuple]
    _refresh_timer: Optional[QTimer]

    def __init__(self, dataset_path: str = ""):
        super().__init__()
        self.dataset_path = dataset_path
        self.ndv_viewer = None
        self._xarray_data = None  # Store for external access
        self._open_handles = []  # Keep tif handles alive when mmap is used
        self._last_sig = None
        self._refresh_timer = None
        self._channel_label_generation = 0  # Generation counter for retry cancellation
        self._pending_channel_label_retries = (
            0  # Retry counter for channel label updates
        )

        # External navigation state (push-based API for live acquisition)
        # _file_index is accessed from both main thread and dask workers, needs lock
        self._file_index: Dict[tuple, str] = {}  # (t, fov_idx, z, channel) -> filepath
        self._file_index_lock = threading.Lock()
        self._fov_labels: List[str] = []  # ["A1:0", "A1:1", ...]
        self._channel_names: List[str] = []
        self._z_levels: List[int] = []
        self._luts: Dict[int, Any] = {}  # channel_idx -> colormap
        self._current_fov_idx: int = 0
        self._current_time_idx: int = 0
        self._max_time_idx: int = 0  # Highest t seen (for slider range)
        self._max_fov_per_time: Dict[int, int] = {}  # timepoint -> max FOV index seen
        self._image_height: int = 0
        self._image_width: int = 0
        self._pixel_size_um: Optional[float] = None  # XY pixel size in micrometers
        self._dz_um: Optional[float] = None  # Z step size in micrometers
        self._is_ome_format: bool = False  # True if dataset is OME-TIFF format
        self._ome_file_index: Dict[int, str] = {}  # flat_fov_idx -> OME filepath
        self._pull_mode: bool = (
            False  # True if using pre-built 6D array (fast navigation)
        )
        self._plane_cache = MemoryBoundedLRUCache(PLANE_CACHE_MAX_MEMORY_BYTES)
        self._updating_sliders: bool = False  # Prevent recursive updates
        self._acquisition_active: bool = False  # True during live acquisition
        self._time_play_timer: Optional[QTimer] = None  # Timer for T slider animation
        self._fov_play_timer: Optional[QTimer] = None  # Timer for FOV slider animation
        self._load_debounce_timer: Optional[QTimer] = (
            None  # Debounce for _load_current_fov
        )
        self._load_pending: bool = False  # True if load is scheduled

        # Connect signal for thread-safe updates
        self._image_registered.connect(self._on_image_registered)

        self._setup_ui()
        if dataset_path:
            self.load_dataset(dataset_path)
        # Note: _setup_live_refresh() removed - using push-based API instead

    def _setup_ui(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        # Status
        self.status_label = QLabel("Loading dataset...")
        self.status_label.setStyleSheet("color: #888; padding: 5px;")
        # layout.addWidget(self.status_label)

        # NDV placeholder
        if NDV_AVAILABLE:
            dummy = np.zeros((1, 100, 100), dtype=np.uint16)
            self.ndv_viewer = ndv.ArrayViewer(
                dummy,
                channel_axis=0,
                channel_mode="composite",
                visible_axes=(-2, -1),
            )
            layout.addWidget(self.ndv_viewer.widget(), 1)
        else:
            placeholder = QLabel("NDV not available.\npip install ndv[vispy,pyqt]")
            placeholder.setAlignment(Qt.AlignCenter)
            layout.addWidget(placeholder, 1)

        # Create slider container with NDV style
        slider_container = QWidget()
        slider_container.setStyleSheet(NDV_SLIDER_STYLE)
        slider_layout = QVBoxLayout(slider_container)
        slider_layout.setContentsMargins(5, 2, 5, 2)
        slider_layout.setSpacing(2)

        # Time slider (hidden if only 1 timepoint)
        self._time_container = QWidget()
        t_layout = QHBoxLayout(self._time_container)
        t_layout.setContentsMargins(0, 0, 0, 0)
        t_layout.setSpacing(5)
        self._time_label = QLabel("T")
        self._time_label.setFixedWidth(30)
        self._time_play_btn = _create_play_button(self)
        self._time_play_btn.clicked.connect(self._on_time_play_clicked)
        if SUPERQT_AVAILABLE:
            self._time_slider = QLabeledSlider(Qt.Horizontal)
        else:
            self._time_slider = QSlider(Qt.Horizontal)
        self._time_slider.setMinimum(0)
        self._time_slider.setMaximum(0)
        self._time_slider.valueChanged.connect(self._on_time_slider_changed)
        t_layout.addWidget(self._time_play_btn)
        t_layout.addWidget(self._time_label)
        t_layout.addWidget(self._time_slider)
        self._time_container.setVisible(False)  # Hidden by default until max > 0
        slider_layout.addWidget(self._time_container)

        # FOV slider
        self._fov_container = QWidget()
        fov_layout = QHBoxLayout(self._fov_container)
        fov_layout.setContentsMargins(0, 0, 0, 0)
        fov_layout.setSpacing(5)
        self._fov_label = QLabel("FOV")
        self._fov_label.setFixedWidth(30)
        self._fov_play_btn = _create_play_button(self)
        self._fov_play_btn.clicked.connect(self._on_fov_play_clicked)
        if SUPERQT_AVAILABLE:
            self._fov_slider = QLabeledSlider(Qt.Horizontal)
        else:
            self._fov_slider = QSlider(Qt.Horizontal)
        self._fov_slider.setMinimum(0)
        self._fov_slider.setMaximum(0)
        self._fov_slider.valueChanged.connect(self._on_fov_slider_changed)
        fov_layout.addWidget(self._fov_play_btn)
        fov_layout.addWidget(self._fov_label)
        fov_layout.addWidget(self._fov_slider)
        slider_layout.addWidget(self._fov_container)

        # Store slider container reference (will be moved into NDV's layout later)
        self._slider_container = slider_container

        # Initially add to our layout (will be repositioned for pull mode)
        layout.addWidget(slider_container)

        self.setLayout(layout)

    def _on_time_slider_changed(self, value: int):
        """Handle time slider change."""
        if self._updating_sliders:
            return
        if value != self._current_time_idx:
            self._current_time_idx = value
            self._time_label.setText(f"T: {value}")

            if self._pull_mode:
                # Pull mode: navigate NDV directly (fast, no array rebuild)
                self._navigate_ndv("time", value)
            else:
                # Push mode: update FOV slider range and rebuild array
                self._updating_sliders = True
                try:
                    available_fov_max = self._max_fov_per_time.get(value, 0)
                    self._fov_slider.setMaximum(available_fov_max)

                    # Clamp current FOV if it exceeds available range
                    if self._current_fov_idx > available_fov_max:
                        self._current_fov_idx = available_fov_max
                        self._fov_slider.setValue(available_fov_max)

                    # Update FOV label to reflect current FOV after any clamping
                    if self._fov_labels and self._current_fov_idx < len(
                        self._fov_labels
                    ):
                        self._fov_label.setText(
                            f"FOV: {self._fov_labels[self._current_fov_idx]}"
                        )
                    else:
                        self._fov_label.setText(f"FOV: {self._current_fov_idx}")
                finally:
                    self._updating_sliders = False

                self._load_current_fov()

    def _on_fov_slider_changed(self, value: int):
        """Handle FOV slider change."""
        if self._updating_sliders:
            return
        if value != self._current_fov_idx:
            self._current_fov_idx = value
            # Update FOV label with well:fov format if available
            if self._fov_labels and value < len(self._fov_labels):
                self._fov_label.setText(f"FOV: {self._fov_labels[value]}")
            else:
                self._fov_label.setText(f"FOV: {value}")

            if self._pull_mode:
                # Pull mode: navigate NDV directly (fast, no array rebuild)
                self._navigate_ndv("fov", value)
            else:
                # Push mode: rebuild array for new FOV
                self._load_current_fov()

    def _navigate_ndv(self, dim: str, value: int):
        """Navigate NDV viewer to a specific dimension index.

        Used in pull mode for fast navigation without array rebuilds.

        Args:
            dim: Dimension name ("time", "fov", "z", "channel")
            value: Index value to navigate to
        """
        if not self.ndv_viewer:
            return

        try:
            # NDV ArrayViewer uses display_model.current_index
            if hasattr(self.ndv_viewer, "display_model"):
                dm = self.ndv_viewer.display_model
                if hasattr(dm, "current_index") and dim in dm.current_index:
                    dm.current_index[dim] = value
                    return

            # Fallback for older NDV versions
            if hasattr(self.ndv_viewer, "dims"):
                dims = self.ndv_viewer.dims
                if hasattr(dims, "current_step"):
                    current = dict(dims.current_step)
                    if dim in current:
                        current[dim] = value
                        dims.current_step = current
        except Exception as e:
            logger.debug("Failed to navigate NDV: %s", e)

    def _on_time_play_clicked(self, checked: bool):
        """Handle time play button click."""
        if checked:
            # Update text for fallback (iconify handles icon automatically)
            if not ICONIFY_AVAILABLE:
                self._time_play_btn.setText("⏸")
            if self._time_play_timer is None:
                self._time_play_timer = QTimer(self)
                self._time_play_timer.timeout.connect(self._time_play_step)
            self._time_play_timer.start(SLIDER_PLAY_INTERVAL_MS)
        else:
            if not ICONIFY_AVAILABLE:
                self._time_play_btn.setText("▶")
            if self._time_play_timer:
                self._time_play_timer.stop()

    def _time_play_step(self):
        """Advance time slider by one step (looping)."""
        max_t = self._time_slider.maximum()
        if max_t <= 0:
            return
        current = self._time_slider.value()
        next_val = (current + 1) % (max_t + 1)
        self._time_slider.setValue(next_val)

    def _on_fov_play_clicked(self, checked: bool):
        """Handle FOV play button click."""
        if checked:
            if not ICONIFY_AVAILABLE:
                self._fov_play_btn.setText("⏸")
            if self._fov_play_timer is None:
                self._fov_play_timer = QTimer(self)
                self._fov_play_timer.timeout.connect(self._fov_play_step)
            self._fov_play_timer.start(SLIDER_PLAY_INTERVAL_MS)
        else:
            if not ICONIFY_AVAILABLE:
                self._fov_play_btn.setText("▶")
            if self._fov_play_timer:
                self._fov_play_timer.stop()

    def _fov_play_step(self):
        """Advance FOV slider by one step (looping)."""
        max_fov = self._fov_slider.maximum()
        if max_fov <= 0:
            return
        current = self._fov_slider.value()
        next_val = (current + 1) % (max_fov + 1)
        self._fov_slider.setValue(next_val)

    def _stop_play_animation(
        self, timer: Optional[QTimer], button: QPushButton
    ) -> None:
        """Stop a play animation and reset the button state."""
        if timer and timer.isActive():
            timer.stop()
            button.setChecked(False)
            if not ICONIFY_AVAILABLE:
                button.setText("▶")

    # ─────────────────────────────────────────────────────────────────────────
    # Push-based API for live acquisition
    # ─────────────────────────────────────────────────────────────────────────

    def start_acquisition(
        self,
        channels: List[str],
        num_z: int,
        height: int,
        width: int,
        fov_labels: List[str],
    ):
        """Configure viewer for a new acquisition.

        Call this at acquisition start before any register_image() calls.
        Sets up LUTs based on channel wavelengths and configures sliders.

        Args:
            channels: Channel names, e.g. ["BF LED matrix full", "Fluorescence 488 nm Ex"]
            num_z: Number of z-levels
            height: Image height in pixels
            width: Image width in pixels
            fov_labels: FOV labels, e.g. ["A1:0", "A1:1", "A2:0"]
        """
        # Stop any running play animations and pending loads
        self._stop_play_animation(self._time_play_timer, self._time_play_btn)
        self._stop_play_animation(self._fov_play_timer, self._fov_play_btn)
        if self._load_debounce_timer and self._load_debounce_timer.isActive():
            self._load_debounce_timer.stop()
        self._load_pending = False

        # Clear previous state
        with self._file_index_lock:
            self._file_index.clear()
        self._plane_cache.clear()
        self._max_fov_per_time.clear()
        self._pull_mode = False  # Push mode: rebuild arrays on navigation

        # Store configuration
        self._channel_names = list(channels)
        self._z_levels = list(range(num_z))
        self._image_height = height
        self._image_width = width
        self._fov_labels = list(fov_labels)

        # Set up LUTs based on channel wavelengths
        self._luts = {
            i: wavelength_to_colormap(extract_wavelength(c))
            for i, c in enumerate(self._channel_names)
        }

        # Reset navigation state
        self._current_fov_idx = 0
        self._current_time_idx = 0
        self._max_time_idx = 0
        self._acquisition_active = True

        # Update sliders
        self._updating_sliders = True
        try:
            self._time_slider.setMaximum(0)
            self._time_slider.setValue(0)
            self._time_label.setText("T: 0")

            self._fov_slider.setMaximum(0)  # Start at 0, grows as FOVs are acquired
            self._fov_slider.setValue(0)
            if fov_labels:
                self._fov_label.setText(f"FOV: {fov_labels[0]}")
            else:
                self._fov_label.setText("FOV: -")
        finally:
            self._updating_sliders = False

        # Rebuild NDV viewer with channel configuration
        self._rebuild_viewer_for_acquisition()

        logger.info(
            f"NDViewer: Started acquisition with {len(channels)} channels, "
            f"{num_z} z-levels, {len(fov_labels)} FOVs"
        )

    def _rebuild_viewer_for_acquisition(self):
        """Rebuild the NDV viewer for the current acquisition configuration."""
        if not NDV_AVAILABLE or not self.ndv_viewer:
            return

        # Create placeholder array with correct shape: z_level × channel × y × x
        n_z = len(self._z_levels) if self._z_levels else 1
        n_c = len(self._channel_names) if self._channel_names else 1
        h = self._image_height if self._image_height > 0 else 100
        w = self._image_width if self._image_width > 0 else 100

        placeholder = np.zeros((n_z, n_c, h, w), dtype=np.uint16)

        import xarray as xr

        xarr = xr.DataArray(
            placeholder,
            dims=["z_level", "channel", "y", "x"],
            coords={
                "z_level": self._z_levels if self._z_levels else [0],
                "channel": list(range(n_c)),
            },
        )
        xarr.attrs["luts"] = self._luts
        xarr.attrs["channel_names"] = self._channel_names

        # Include pixel size metadata if available (for scale display and 3D rendering)
        if self._pixel_size_um is not None:
            xarr.attrs["pixel_size_um"] = self._pixel_size_um
        if self._dz_um is not None:
            xarr.attrs["dz_um"] = self._dz_um

        self._xarray_data = xarr
        self._set_ndv_data(xarr)

    def register_image(self, t: int, fov_idx: int, z: int, channel: str, filepath: str):
        """Register a newly saved image file.

        Thread-safe: can be called from worker thread.
        Updates file index and emits signal for GUI update.

        Args:
            t: Timepoint index
            fov_idx: FOV index (0-based)
            z: Z-level index
            channel: Channel name
            filepath: Path to the saved TIFF file
        """
        # Update file index (protected by lock for dask worker thread safety)
        with self._file_index_lock:
            self._file_index[(t, fov_idx, z, channel)] = filepath

        # Emit signal with raw indices - main thread computes max values
        # to avoid race condition on _max_time_idx
        try:
            self._image_registered.emit(t, fov_idx, 0, 0)
        except RuntimeError as e:
            # Qt object deleted - viewer was closed during acquisition
            logger.warning(
                "Could not emit image_registered signal (viewer may be closed): %s", e
            )

    def _on_image_registered(self, t: int, fov_idx: int, _unused1: int, _unused2: int):
        """Handle image registration signal (runs on main thread).

        Updates slider ranges and schedules debounced FOV load if needed.

        Note: _unused1/_unused2 are placeholder parameters kept for signal
        compatibility; max values are computed here from the current tracking
        state (_max_time_idx, _max_fov_per_time), not passed via signal.
        """
        try:
            # Update per-timepoint max FOV tracking
            current_max_for_t = self._max_fov_per_time.get(t, -1)
            if fov_idx > current_max_for_t:
                self._max_fov_per_time[t] = fov_idx

            # Compute max time
            new_max_t = max(self._max_time_idx, t)

            self._updating_sliders = True
            try:
                # Update T slider if needed
                if new_max_t > self._max_time_idx:
                    self._max_time_idx = new_max_t
                    self._time_slider.setMaximum(new_max_t)

                # Show T slider if we have multiple timepoints
                if new_max_t > 0:
                    self._time_container.setVisible(True)

                # Update FOV slider max for CURRENT timepoint only
                if t == self._current_time_idx:
                    current_fov_max = self._fov_slider.maximum()
                    available_fov_max = self._max_fov_per_time.get(t, 0)
                    if available_fov_max > current_fov_max:
                        self._fov_slider.setMaximum(available_fov_max)
            finally:
                self._updating_sliders = False

            # Schedule debounced load if this image is for the current FOV
            if t == self._current_time_idx and fov_idx == self._current_fov_idx:
                self._schedule_debounced_load()
        except Exception as e:
            logger.error("Error in _on_image_registered: %s", e, exc_info=True)

    def _schedule_debounced_load(self):
        """Schedule a debounced load of the current FOV.

        Coalesces rapid image registrations into a single load every 200ms.
        This prevents overwhelming the main thread during fast acquisitions.
        """
        # Mark that a load is pending
        self._load_pending = True

        # Create timer if needed
        if self._load_debounce_timer is None:
            self._load_debounce_timer = QTimer(self)
            self._load_debounce_timer.setSingleShot(True)
            self._load_debounce_timer.timeout.connect(self._execute_debounced_load)

        # If timer not running, start it; otherwise the existing timer will handle it
        if not self._load_debounce_timer.isActive():
            self._load_debounce_timer.start(200)  # 200ms debounce

    def _execute_debounced_load(self):
        """Execute the debounced FOV load."""
        if self._load_pending:
            self._load_pending = False
            self._load_current_fov()

    def load_fov(self, fov: int, t: Optional[int] = None, z: Optional[int] = None):
        """Load and display a specific FOV.

        Args:
            fov: FOV index to display
            t: Timepoint index (None = use current)
            z: Z-level index (None = use current, not used for NDV internal z)

        Only updates data, LUTs remain unchanged.
        """
        if t is not None:
            self._current_time_idx = t
        if fov != self._current_fov_idx:
            self._current_fov_idx = fov

        # Update sliders to reflect new position
        self._updating_sliders = True
        try:
            self._time_slider.setValue(self._current_time_idx)
            self._time_label.setText(f"T: {self._current_time_idx}")
            self._fov_slider.setValue(self._current_fov_idx)
            if self._fov_labels and self._current_fov_idx < len(self._fov_labels):
                self._fov_label.setText(
                    f"FOV: {self._fov_labels[self._current_fov_idx]}"
                )
            else:
                self._fov_label.setText(f"FOV: {self._current_fov_idx}")
        finally:
            self._updating_sliders = False

        self._load_current_fov()

    def go_to_well_fov(self, well_id: str, fov_index: int) -> bool:
        """Navigate to a specific well and FOV (push-based API).

        Maps (well_id, fov_index) to flat FOV index using _fov_labels.
        Labels are in format "A1:0", "A1:1", "A2:0", etc.

        Args:
            well_id: Well identifier (e.g., "A1", "B2")
            fov_index: FOV index within the well

        Returns:
            True if navigation succeeded, False if FOV not found.
        """
        if not self._fov_labels:
            logger.debug("go_to_well_fov: no FOV labels available")
            return False

        # Find the flat index for this well:fov combination
        target_label = f"{well_id}:{fov_index}"
        try:
            flat_idx = self._fov_labels.index(target_label)
        except ValueError:
            logger.debug(
                f"go_to_well_fov: label '{target_label}' not found in {self._fov_labels}"
            )
            return False

        self.load_fov(flat_idx)
        logger.info(
            f"go_to_well_fov: navigated to {target_label} (flat_idx={flat_idx})"
        )
        return True

    def is_push_mode_active(self) -> bool:
        """Check if push-based mode is active (has FOV labels configured)."""
        return bool(self._fov_labels)

    def _load_single_plane(
        self, t: int, fov_idx: int, z: int, channel: str
    ) -> np.ndarray:
        """Load a single image plane from cache or disk.

        Handles both single-TIFF (one plane per file) and OME-TIFF (all planes
        in one file per FOV) formats.

        Args:
            t: Timepoint index
            fov_idx: FOV index
            z: Z-level value (index into _z_levels for single-TIFF, direct index for OME)
            channel: Channel name

        Returns:
            Image plane as numpy array, or zeros if not available.
        """
        cache_key = (t, fov_idx, z, channel)

        # Check cache first
        cached_plane = self._plane_cache.get(cache_key)
        if cached_plane is not None:
            return cached_plane

        # Load from file (lock protects concurrent access from dask workers)
        with self._file_index_lock:
            filepath = self._file_index.get(cache_key)

        if not filepath:
            # File not yet registered - expected during acquisition, not an error
            return np.zeros((self._image_height, self._image_width), dtype=np.uint16)

        if not LAZY_LOADING_AVAILABLE:
            logger.error("tifffile not available for loading image planes")
            return np.zeros((self._image_height, self._image_width), dtype=np.uint16)

        try:
            # Check if this is OME-TIFF format (multi-plane file)
            is_ome = getattr(self, "_is_ome_format", False)

            if is_ome:
                # OME-TIFF: read specific plane from multi-dimensional file
                plane = self._load_ome_plane(filepath, t, z, channel)
            else:
                # Single-TIFF: one plane per file
                with tf.TiffFile(filepath) as tif:
                    plane = tif.pages[0].asarray()

            self._plane_cache.put(cache_key, plane)
            return plane
        except FileNotFoundError:
            logger.warning("Image file not found (may have been deleted): %s", filepath)
        except PermissionError as e:
            logger.error("Permission denied reading image %s: %s", filepath, e)
        except Exception as e:
            logger.error(
                "Failed to load image plane %s (t=%d, fov=%d, z=%d, ch=%s): %s",
                filepath,
                t,
                fov_idx,
                z,
                channel,
                e,
                exc_info=True,
            )

        # Return zeros on error - user sees black image
        return np.zeros((self._image_height, self._image_width), dtype=np.uint16)

    def _load_ome_plane(
        self, filepath: str, t: int, z: int, channel: str
    ) -> np.ndarray:
        """Load a single plane from an OME-TIFF file.

        Args:
            filepath: Path to OME-TIFF file
            t: Timepoint index
            z: Z-level index
            channel: Channel name

        Returns:
            Image plane as numpy array
        """
        # Get channel index from name
        try:
            c_idx = self._channel_names.index(channel)
        except ValueError:
            logger.warning("Channel '%s' not found in channel list", channel)
            return np.zeros((self._image_height, self._image_width), dtype=np.uint16)

        with tf.TiffFile(filepath) as tif:
            series = tif.series[0]
            axes = series.axes
            shape = series.shape

            # Build index based on axes order (commonly TZCYX or TCYX)
            idx = []
            for ax in axes:
                if ax == "T":
                    idx.append(t)
                elif ax == "Z":
                    idx.append(z)
                elif ax == "C":
                    idx.append(c_idx)
                elif ax in ("Y", "X"):
                    idx.append(slice(None))
                else:
                    # Unknown axis, take first element
                    idx.append(0)

            # Read the specific plane
            data = series.asarray()[tuple(idx)]
            return data

    def _load_current_fov(self):
        """Load and display data for the current FOV position.

        Creates a lazy dask array that only loads planes when NDV requests them.
        This avoids loading all z-planes when only one is displayed.
        """
        # Check if we have data configuration (set by start_acquisition)
        if not self._channel_names or not self._z_levels:
            return
        if self._image_height == 0 or self._image_width == 0:
            return
        # Check if we have any registered files
        with self._file_index_lock:
            if not self._file_index:
                return

        t = self._current_time_idx
        fov_idx = self._current_fov_idx
        h, w = self._image_height, self._image_width

        # Create lazy dask array - planes only load when accessed
        import dask
        import dask.array as da

        delayed_planes = []
        for z in self._z_levels:
            channel_planes = []
            for channel in self._channel_names:
                # Create delayed load - no disk I/O happens here
                delayed_load = dask.delayed(self._load_single_plane)(
                    t, fov_idx, z, channel
                )
                da_plane = da.from_delayed(delayed_load, shape=(h, w), dtype=np.uint16)
                channel_planes.append(da_plane)
            # Stack channels: (n_c, h, w)
            delayed_planes.append(da.stack(channel_planes))
        # Stack z-levels: (n_z, n_c, h, w)
        data = da.stack(delayed_planes)

        # Update NDV viewer data without rebuilding (preserves LUTs)
        self._update_ndv_data(data)

    def _update_ndv_data(self, data):
        """Update NDV viewer with new data array, preserving LUTs.

        Args:
            data: numpy or dask array of shape (z_level, channel, y, x).
                  Dask arrays enable lazy loading - planes only load when displayed.
        """
        if not NDV_AVAILABLE or not self.ndv_viewer:
            return

        import xarray as xr

        xarr = xr.DataArray(
            data,
            dims=["z_level", "channel", "y", "x"],
            coords={
                "z_level": self._z_levels,
                "channel": list(range(len(self._channel_names))),
            },
        )
        xarr.attrs["luts"] = self._luts
        xarr.attrs["channel_names"] = self._channel_names

        # Include pixel size metadata if available (for scale display and 3D rendering)
        if self._pixel_size_um is not None:
            xarr.attrs["pixel_size_um"] = self._pixel_size_um
        if self._dz_um is not None:
            xarr.attrs["dz_um"] = self._dz_um

        self._xarray_data = xarr

        # Try in-place update to avoid flickering
        if not self._try_inplace_ndv_update(xarr):
            # Fallback: full rebuild (shouldn't happen often)
            self._set_ndv_data(xarr)

    def end_acquisition(self):
        """Mark acquisition as ended.

        Call this when acquisition completes. The viewer remains in push mode
        (is_push_mode_active() returns True) so navigation via go_to_well_fov()
        continues to work for browsing the acquired data.

        FOV labels are preserved to enable navigation. They are only cleared
        when a new acquisition starts via start_acquisition().
        """
        # Stop any pending debounced load from previous acquisition
        if self._load_debounce_timer and self._load_debounce_timer.isActive():
            self._load_debounce_timer.stop()
        self._load_pending = False

        self._acquisition_active = False
        # NOTE: _fov_labels is NOT cleared here - navigation must still work
        # after acquisition ends. Labels are cleared in start_acquisition().
        logger.info("NDViewer: Acquisition ended")

    # ─────────────────────────────────────────────────────────────────────────
    # Legacy live refresh (kept for existing dataset viewing)
    # ─────────────────────────────────────────────────────────────────────────

    def _setup_live_refresh(self):
        """Poll the dataset folder periodically to pick up new timepoints during acquisition."""
        # Only enable when lazy loading + NDV are available; otherwise refresh does nothing useful.
        if not (LAZY_LOADING_AVAILABLE and NDV_AVAILABLE and self.ndv_viewer):
            return
        self._refresh_timer = QTimer(self)
        self._refresh_timer.setInterval(LIVE_REFRESH_INTERVAL_MS)
        self._refresh_timer.timeout.connect(self._maybe_refresh)
        self._refresh_timer.start()

    def _close_tiff_handles(self, handles):
        """Close a list of TiffFile handles, logging any errors."""
        for h in handles or []:
            try:
                h.close()
            except Exception as e:
                logger.debug("Failed to close TiffFile handle: %s", e)

    def _close_open_handles(self):
        """Close mmap TiffFile handles (OME path) from the previously loaded dataset."""
        self._close_tiff_handles(getattr(self, "_open_handles", []))
        self._open_handles = []

    def closeEvent(self, event):
        """Clean up resources when the widget is closed."""
        if self._refresh_timer:
            self._refresh_timer.stop()
        if self._time_play_timer:
            self._time_play_timer.stop()
        if self._fov_play_timer:
            self._fov_play_timer.stop()
        if self._load_debounce_timer:
            self._load_debounce_timer.stop()
        self._close_open_handles()
        super().closeEvent(event)

    def _force_refresh(self):
        self._last_sig = None
        self._maybe_refresh()

    def _dataset_signature(self) -> tuple:
        """Return a cheap signature that changes when new data likely arrived."""
        base = Path(self.dataset_path)
        fmt = detect_format(base)

        if fmt == "single_tiff":
            tp_dirs = [d for d in base.iterdir() if d.is_dir() and d.name.isdigit()]
            if not tp_dirs:
                return (fmt, -1, 0, 0)

            t_vals = sorted(int(d.name) for d in tp_dirs)
            first_tp = base / str(t_vals[0])
            latest_tp = base / str(t_vals[-1])

            # FOVs are assumed to only appear in the first timepoint during acquisition.
            fov_set = set()
            try:
                if first_tp.exists():
                    for f in first_tp.iterdir():
                        if f.suffix.lower() not in TIFF_EXTENSIONS:
                            continue
                        m = FPATTERN.search(f.name)
                        if m:
                            fov_set.add((m.group("r"), int(m.group("f"))))
            except Exception as e:
                logger.debug("Error scanning FOVs: %s", e)

            # Count files in latest timepoint to detect when files are actually written
            # (not just when the folder is created)
            latest_file_count = 0
            try:
                if latest_tp.exists():
                    latest_file_count = sum(
                        1
                        for f in latest_tp.iterdir()
                        if f.suffix.lower() in TIFF_EXTENSIONS
                    )
            except Exception as e:
                logger.debug("Error counting files in latest timepoint: %s", e)

            return (fmt, max(t_vals), len(fov_set), latest_file_count)

        # ome_tiff
        ome_dir = base / "ome_tiff"
        if not ome_dir.exists():
            ome_dir = next(
                (d for d in base.iterdir() if d.is_dir() and d.name.isdigit()), base
            )

        ome_files = sorted(ome_dir.glob("*.ome.tif*"))
        n_ome = len(ome_files)
        t_len = -1
        st = None
        if ome_files:
            try:
                st = ome_files[0].stat()
            except Exception as e:
                logger.debug("Failed to stat OME file: %s", e)
                st = None
            try:
                with tf.TiffFile(str(ome_files[0])) as tif:
                    series = tif.series[0]
                    axes = series.axes
                    shape = series.shape
                    if "T" in axes:
                        t_len = int(shape[axes.index("T")])
                    else:
                        t_len = 1
            except Exception as e:
                # File may be mid-write; fall back on size/mtime if available
                logger.debug("Failed to read OME series (may be mid-write): %s", e)

        if st is None:
            return (fmt, n_ome, t_len)
        return (
            fmt,
            n_ome,
            t_len,
            st.st_size,
            getattr(st, "st_mtime_ns", int(st.st_mtime * 1e9)),
        )

    def _try_inplace_ndv_update(self, data: "xr.DataArray") -> bool:
        """Update ndv data in-place to avoid memory leak (ndv#209).

        Bypasses ndv's data setter which leaks GPU handles. When the data
        shape changes, emits dims_changed to trigger slider updates without
        rebuilding the entire viewer.

        Args:
            data: The new xarray DataArray to display.

        Returns:
            True if in-place update succeeded, False if caller should
            fall back to _set_ndv_data() for a full viewer rebuild.

        Note:
            Relies on ndv internal APIs (_data_model.data_wrapper._data).
            Tested with ndv 0.4.0. May need updating if ndv internals change.
        """
        v = self.ndv_viewer
        if v is None:
            return False

        try:
            wrapper = v._data_model.data_wrapper
            if wrapper._data is None:
                return False

            shape_changed = wrapper._data.shape != data.shape
            wrapper._data = data

            if shape_changed:
                # Emit dims_changed signal to update slider ranges without full rebuild.
                # In ndv, this signal triggers _fully_synchronize_view() which recreates
                # sliders based on the new data shape.
                wrapper.dims_changed.emit()
            else:
                v._request_data()

            return True
        except AttributeError as e:
            # Expected when ndv version doesn't have the expected internal structure
            logger.debug("In-place update unavailable (ndv API mismatch): %s", e)
            return False
        except Exception as e:
            # Unexpected error - log for debugging but allow fallback
            logger.warning(
                "In-place ndv update failed unexpectedly: %s", e, exc_info=True
            )
            return False

    def _maybe_refresh(self):
        if not LAZY_LOADING_AVAILABLE:
            return

        try:
            sig = self._dataset_signature()
        except Exception as e:
            logger.debug("Failed to compute dataset signature: %s", e)
            return
        if sig == self._last_sig:
            return

        data = self._create_lazy_array(Path(self.dataset_path))
        if data is None:
            return

        # Update signature only after we've confirmed we'll swap data
        self._last_sig = sig

        # Swap dataset, keeping OME handles alive for the new data
        old_data = self._xarray_data
        old_handles = getattr(self, "_open_handles", [])
        self._xarray_data = data
        self._open_handles = data.attrs.get("_open_tifs", [])

        # Check if data structure changed (dims, channels, channel names, or LUTs) - if so, force full rebuild
        structure_changed = self._data_structure_changed(old_data, data)

        # Prefer in-place update to avoid visible refresh, but only if structure unchanged.
        # When structure changes (e.g., different channels), we must rebuild the viewer
        # to avoid stale channel controls persisting from the previous dataset.
        if not structure_changed and self._try_inplace_ndv_update(data):
            # Update channel labels for the new data
            self._initiate_channel_label_update()
            # Close old handles after successful swap.
            self._close_tiff_handles(old_handles)
            return

        # Fallback: rebuild widget (may be visible on some platforms). Reduce flicker a bit.
        reason = (
            "Data structure changed" if structure_changed else "In-place update failed"
        )
        logger.debug("%s, performing full viewer rebuild", reason)

        try:
            self.setUpdatesEnabled(False)
            self._set_ndv_data(data)
        finally:
            self.setUpdatesEnabled(True)
            # Close old handles regardless.
            self._close_tiff_handles(old_handles)

    def _data_structure_changed(
        self, old_data: Optional["xr.DataArray"], new_data: "xr.DataArray"
    ) -> bool:
        """Check if data structure changed significantly (requiring full viewer rebuild).

        Delegates to the module-level :func:`data_structure_changed` function,
        wrapping it with exception handling for safety in the viewer context.

        Args:
            old_data: Previous dataset state (or None for first load).
            new_data: Newly loaded dataset to compare.

        Returns:
            True if structure changed and viewer needs full rebuild.
        """
        try:
            return data_structure_changed(old_data, new_data)
        except Exception as e:
            # On any error, assume structure changed to be safe
            logger.debug("Error checking data structure change: %s", e)
            return True

    def load_dataset(self, path: str):
        """Load dataset with pre-built 6D array for fast navigation.

        Uses a hybrid approach:
        - Builds 6D array once (like original implementation) for fast slicing
        - Custom T/FOV sliders navigate via NDV's API (no array rebuilds)
        - NDV's built-in time/fov sliders are hidden to avoid duplicates

        This provides the unified slider UI while maintaining performance.
        """
        # Close any previously open file handles before loading new dataset
        self._close_open_handles()

        # Stop any running play animations and pending loads
        self._stop_play_animation(self._time_play_timer, self._time_play_btn)
        self._stop_play_animation(self._fov_play_timer, self._fov_play_btn)
        if self._load_debounce_timer and self._load_debounce_timer.isActive():
            self._load_debounce_timer.stop()
        self._load_pending = False

        # Reset state
        self._last_sig = None
        self._xarray_data = None
        self._pull_mode = True  # Enable fast navigation mode
        self.dataset_path = path
        self.status_label.setText(f"Loading: {Path(path).name}...")
        QApplication.processEvents()

        try:
            # Build 6D array using the optimized lazy loading path
            data = self._create_lazy_array(Path(path))
            if data is not None:
                self._xarray_data = data
                self._open_handles = data.attrs.get("_open_tifs", [])

                # Extract metadata for slider configuration
                self._channel_names = data.attrs.get("channel_names", [])
                self._luts = data.attrs.get("luts", {})
                self._pixel_size_um = data.attrs.get("pixel_size_um")
                self._dz_um = data.attrs.get("dz_um")

                # Get dimension sizes for slider ranges
                n_time = data.sizes.get("time", 1)
                n_fov = data.sizes.get("fov", 1)

                # Build FOV labels from discovered FOVs
                fmt = detect_format(Path(path))
                fovs = self._discover_fovs(Path(path), fmt)
                self._fov_labels = [f"{f['region']}:{f['fov']}" for f in fovs]

                # Configure custom sliders
                self._max_time_idx = n_time - 1
                self._updating_sliders = True
                try:
                    # Time slider
                    self._time_slider.setMaximum(self._max_time_idx)
                    self._time_slider.setValue(0)
                    self._time_label.setText("T: 0")
                    self._time_container.setVisible(self._max_time_idx > 0)

                    # FOV slider
                    max_fov = n_fov - 1
                    self._fov_slider.setMaximum(max_fov)
                    self._fov_slider.setValue(0)
                    if self._fov_labels:
                        self._fov_label.setText(f"FOV: {self._fov_labels[0]}")
                    else:
                        self._fov_label.setText("FOV: 0")
                finally:
                    self._updating_sliders = False

                # Reset navigation state
                self._current_time_idx = 0
                self._current_fov_idx = 0

                # Display the data (builds NDV viewer with 6D array)
                self._set_ndv_data(data)

                # Hide NDV's time/fov sliders since we use custom ones
                self._hide_ndv_dimension_sliders(["time", "fov"])

                # Move custom sliders into NDV's layout for better visual grouping
                self._insert_sliders_into_ndv_layout()

                self.status_label.setText(f"Loaded: {Path(path).name}")
            else:
                self.status_label.setText("Failed to load dataset")
        except Exception as e:
            self.status_label.setText(f"Error: {e}")
            import traceback

            traceback.print_exc()

    def _hide_ndv_dimension_sliders(self, dims_to_hide: List[str]):
        """Hide NDV's built-in sliders for specific dimensions.

        Used in pull mode to avoid duplicate sliders - we use custom T/FOV
        sliders while NDV handles z/channel.

        Args:
            dims_to_hide: List of dimension names to hide (e.g., ["time", "fov"])
        """
        if not self.ndv_viewer:
            return

        try:
            # Use NDV's official hide_sliders API
            # show_remainder=False prevents showing sliders for visible axes (x, y)
            if hasattr(self.ndv_viewer, "_view") and hasattr(
                self.ndv_viewer._view, "hide_sliders"
            ):
                self.ndv_viewer._view.hide_sliders(dims_to_hide, show_remainder=False)
        except Exception as e:
            logger.debug("Could not hide NDV sliders: %s", e)

    def _insert_sliders_into_ndv_layout(self):
        """Move custom T/FOV sliders into NDV's internal layout.

        This places our sliders right after NDV's dimension sliders (z, channel)
        for a cohesive visual grouping, instead of at the bottom of the window.

        NDV's layout structure:
        - _view.frontend_widget() -> QWidget with QVBoxLayout
          - [0] QSplitter
            - widget(0) -> QWidget with QVBoxLayout
              - [0] QWidget (toolbar)
              - [1] CanvasBackendDesktop
              - [2] _QDimsSliders  <- insert after this
              - [3] _UpCollapsible (LUT controls)
              - [4] QWidget (footer)
        """
        if not self.ndv_viewer or not hasattr(self, "_slider_container"):
            return

        try:
            # Navigate NDV's internal structure
            if not hasattr(self.ndv_viewer, "_view"):
                return

            frontend = self.ndv_viewer._view.frontend_widget()
            if not frontend or not frontend.layout():
                return

            # Get the QSplitter from frontend's layout
            splitter_item = frontend.layout().itemAt(0)
            if not splitter_item or not splitter_item.widget():
                return

            splitter = splitter_item.widget()
            if splitter.count() == 0:
                return

            # Get the main content widget (first child of splitter)
            main_widget = splitter.widget(0)
            if not main_widget or not main_widget.layout():
                return

            main_layout = main_widget.layout()

            # Find the index of _QDimsSliders in the layout
            dims_slider_idx = -1
            for i in range(main_layout.count()):
                item = main_layout.itemAt(i)
                if item and item.widget():
                    widget_class = item.widget().__class__.__name__
                    if "DimSliders" in widget_class or "Dims" in widget_class:
                        dims_slider_idx = i
                        break

            # Remove slider container from our main layout
            our_layout = self.layout()
            if our_layout:
                our_layout.removeWidget(self._slider_container)

            # Insert into NDV's layout right after the dims sliders
            if dims_slider_idx >= 0:
                main_layout.insertWidget(dims_slider_idx + 1, self._slider_container)
            else:
                # Fallback: add after canvas (index 2)
                insert_pos = min(2, main_layout.count())
                main_layout.insertWidget(insert_pos, self._slider_container)

            logger.debug(
                "Inserted custom sliders into NDV layout at position %d",
                dims_slider_idx + 1,
            )
        except Exception as e:
            logger.debug("Could not insert sliders into NDV layout: %s", e)

    def _on_ndv_ndims_requested(self, ndims: int):
        """Handle NDV's nDimsRequested signal (fired when 2D/3D toggle is clicked).

        When NDV switches between 2D and 3D modes, it recreates its dimension sliders.
        We need to re-hide the time/fov sliders to prevent duplicates with our custom ones.

        Args:
            ndims: Number of dimensions requested (2 or 3)
        """
        if self._pull_mode:
            # Use QTimer to defer hiding until after NDV finishes recreating sliders
            from PyQt5.QtCore import QTimer

            QTimer.singleShot(
                50, lambda: self._hide_ndv_dimension_sliders(["time", "fov"])
            )

    def _scan_dataset_to_internal_state(self, base_path: Path) -> bool:
        """Scan filesystem and populate internal state for push-mode architecture.

        This method discovers all files in the dataset and sets up:
        - _file_index: maps (t, fov_idx, z, channel) to filepath
        - _fov_labels: list of FOV labels like ["A1:0", "A1:1", ...]
        - _channel_names: sorted list of channel names
        - _z_levels: sorted list of z-level indices
        - _image_height, _image_width: image dimensions
        - _luts: channel colormaps based on wavelengths
        - _max_time_idx: highest timepoint index
        - _max_fov_per_time: maps timepoint to max FOV index for that timepoint

        Args:
            base_path: Path to the dataset directory

        Returns:
            True if successful, False otherwise
        """
        if not LAZY_LOADING_AVAILABLE:
            return False

        fmt = detect_format(base_path)
        fovs = self._discover_fovs(base_path, fmt)

        if not fovs:
            logger.warning("No FOVs found in dataset")
            return False

        # Clear previous state
        with self._file_index_lock:
            self._file_index.clear()
        self._plane_cache.clear()
        self._max_fov_per_time.clear()

        # Build FOV label list and reverse lookup
        self._fov_labels = [f"{f['region']}:{f['fov']}" for f in fovs]
        fov_to_flat = {(f["region"], f["fov"]): i for i, f in enumerate(fovs)}

        # Scan files based on format
        if fmt == "ome_tiff":
            return self._scan_ome_tiff_to_state(base_path, fov_to_flat)
        else:
            return self._scan_single_tiff_to_state(base_path, fov_to_flat)

    def _scan_single_tiff_to_state(
        self, base_path: Path, fov_to_flat: Dict[tuple, int]
    ) -> bool:
        """Scan single-TIFF format dataset into internal state.

        Args:
            base_path: Path to dataset directory
            fov_to_flat: Maps (region, fov) to flat FOV index

        Returns:
            True if successful, False otherwise
        """
        channels_seen: set = set()
        z_levels_seen: set = set()
        times_seen: set = set()
        height, width = 0, 0

        # Scan all timepoint directories
        for tp_dir in sorted(base_path.iterdir()):
            if not (tp_dir.is_dir() and tp_dir.name.isdigit()):
                continue
            t = int(tp_dir.name)
            has_files = False

            for f in tp_dir.iterdir():
                if f.suffix.lower() not in TIFF_EXTENSIONS:
                    continue
                m = FPATTERN.search(f.name)
                if not m:
                    continue

                region = m.group("r")
                fov = int(m.group("f"))
                z = int(m.group("z"))
                channel = m.group("c")

                # Convert (region, fov) to flat index
                flat_fov = fov_to_flat.get((region, fov))
                if flat_fov is None:
                    continue

                # Populate file index
                with self._file_index_lock:
                    self._file_index[(t, flat_fov, z, channel)] = str(f)

                channels_seen.add(channel)
                z_levels_seen.add(z)
                has_files = True

                # Get image dimensions from first file
                if height == 0:
                    try:
                        with tf.TiffFile(str(f)) as tif:
                            height, width = tif.pages[0].shape[-2:]
                    except Exception as e:
                        logger.debug("Failed to read image dimensions: %s", e)

            if has_files:
                times_seen.add(t)

        if not self._file_index:
            return False

        # Store discovered metadata
        self._channel_names = sorted(channels_seen)
        self._z_levels = sorted(z_levels_seen)
        self._image_height = height
        self._image_width = width
        self._max_time_idx = max(times_seen) if times_seen else 0

        # Set up LUTs based on channel wavelengths
        self._luts = {
            i: wavelength_to_colormap(extract_wavelength(c))
            for i, c in enumerate(self._channel_names)
        }

        # Build max FOV per timepoint mapping
        for t in times_seen:
            fovs_for_t = set()
            with self._file_index_lock:
                for ft, fov_idx, z, ch in self._file_index.keys():
                    if ft == t:
                        fovs_for_t.add(fov_idx)
            if fovs_for_t:
                self._max_fov_per_time[t] = max(fovs_for_t)

        # Read acquisition parameters for pixel size (stored for later use)
        pixel_size_um, dz_um = read_acquisition_parameters(base_path)
        self._pixel_size_um = pixel_size_um
        self._dz_um = dz_um

        # Mark as non-OME format
        self._is_ome_format = False
        self._ome_file_index.clear()

        return True

    def _scan_ome_tiff_to_state(
        self, base_path: Path, fov_to_flat: Dict[tuple, int]
    ) -> bool:
        """Scan OME-TIFF format dataset into internal state.

        Args:
            base_path: Path to dataset directory
            fov_to_flat: Maps (region, fov) to flat FOV index

        Returns:
            True if successful, False otherwise
        """
        ome_dir = base_path / "ome_tiff"
        if not ome_dir.exists():
            ome_dir = next(
                (d for d in base_path.iterdir() if d.is_dir() and d.name.isdigit()),
                base_path,
            )

        # Find all OME files and map to FOVs
        ome_files: Dict[int, str] = {}  # flat_fov_idx -> filepath
        for f in ome_dir.glob("*.ome.tif*"):
            m = FPATTERN_OME.search(f.name)
            if m:
                region, fov = m.group("r"), int(m.group("f"))
                flat_fov = fov_to_flat.get((region, fov))
                if flat_fov is not None:
                    ome_files[flat_fov] = str(f)

        if not ome_files:
            return False

        # Read metadata from first OME file
        first_file = next(iter(ome_files.values()))
        try:
            with tf.TiffFile(first_file) as tif:
                series = tif.series[0]
                axes = series.axes
                shape = series.shape
                shape_dict = dict(zip(axes, shape))

                n_t = shape_dict.get("T", 1)
                n_c = shape_dict.get("C", 1)
                n_z = shape_dict.get("Z", 1)
                height = shape_dict.get("Y", shape[-2])
                width = shape_dict.get("X", shape[-1])

                # Extract channel names from OME metadata
                channel_names = []
                pixel_size_x, pixel_size_y, pixel_size_z = None, None, None
                if tif.ome_metadata:
                    try:
                        import xml.etree.ElementTree as ET

                        root = ET.fromstring(tif.ome_metadata)
                        ns = {
                            "ome": "http://www.openmicroscopy.org/Schemas/OME/2016-06"
                        }
                        for ch in root.findall(".//ome:Channel", ns):
                            name = ch.get("Name") or ch.get("ID", "")
                            if name:
                                channel_names.append(name)

                        pixel_size_x, pixel_size_y, pixel_size_z = (
                            extract_ome_physical_sizes(tif.ome_metadata)
                        )
                    except Exception as e:
                        logger.debug("Failed to parse OME metadata: %s", e)

                # Fallback channel names if not found in metadata
                if not channel_names:
                    channel_names = [f"Ch{i}" for i in range(n_c)]

        except Exception as e:
            logger.error("Failed to read OME file metadata: %s", e)
            return False

        # Store metadata
        self._channel_names = channel_names
        self._z_levels = list(range(n_z))
        self._image_height = height
        self._image_width = width
        self._max_time_idx = n_t - 1
        self._pixel_size_um = pixel_size_x
        self._dz_um = pixel_size_z

        # Set up LUTs
        self._luts = {
            i: wavelength_to_colormap(extract_wavelength(c))
            for i, c in enumerate(self._channel_names)
        }

        # For OME-TIFF, we store the file path per FOV (not per plane)
        # The _load_single_plane method needs to handle this differently
        # Store OME file paths in a separate attribute for OME-TIFF loading
        self._ome_file_index = ome_files

        # Build file index entries for all (t, fov, z, channel) combinations
        # For OME-TIFF, the filepath is the same for all planes in a FOV
        for flat_fov, filepath in ome_files.items():
            for t in range(n_t):
                for z in range(n_z):
                    for ch_idx, ch_name in enumerate(channel_names):
                        with self._file_index_lock:
                            # Store as (t, fov, z, channel_name) -> filepath
                            # Also store channel index for OME reading
                            self._file_index[(t, flat_fov, z, ch_name)] = filepath

        # All FOVs available at all timepoints for OME-TIFF
        for t in range(n_t):
            self._max_fov_per_time[t] = len(ome_files) - 1

        # Store OME-specific info for plane loading
        self._is_ome_format = True
        self._ome_axes = axes
        self._ome_shape = shape

        return True

    def _configure_sliders_for_dataset(self):
        """Configure T and FOV sliders based on discovered dataset structure."""
        self._updating_sliders = True
        try:
            # Time slider
            self._time_slider.setMaximum(self._max_time_idx)
            self._time_slider.setValue(0)
            self._time_label.setText("T: 0")
            self._time_container.setVisible(self._max_time_idx > 0)

            # FOV slider
            max_fov = len(self._fov_labels) - 1 if self._fov_labels else 0
            self._fov_slider.setMaximum(max_fov)
            self._fov_slider.setValue(0)
            if self._fov_labels:
                self._fov_label.setText(f"FOV: {self._fov_labels[0]}")
            else:
                self._fov_label.setText("FOV: 0")
        finally:
            self._updating_sliders = False

    def set_current_index(self, dim: str, value: int) -> bool:
        """Set the current index for a dimension in the viewer.

        Programmatically navigate the viewer to a specific position along
        a dimension (e.g., 'fov', 'time', 'z', 'channel').

        Args:
            dim: Dimension name (must exist in the loaded data).
            value: Index value to set.

        Returns:
            True if successful, False otherwise.
        """
        if self.ndv_viewer is None:
            logger.debug("set_current_index: no viewer available")
            return False

        try:
            # NDV ArrayViewer uses display_model.current_index
            if hasattr(self.ndv_viewer, "display_model"):
                dm = self.ndv_viewer.display_model
                if hasattr(dm, "current_index") and dim in dm.current_index:
                    dm.current_index[dim] = value
                    logger.debug(f"set_current_index: {dim}={value}")
                    return True

            # Fallback for older NDV versions using dims API
            if hasattr(self.ndv_viewer, "dims"):
                dims = self.ndv_viewer.dims
                if hasattr(dims, "current_step"):
                    current = dict(dims.current_step)
                    if dim in current:
                        current[dim] = value
                        dims.current_step = current
                        logger.debug(f"set_current_index (fallback): {dim}={value}")
                        return True

            logger.debug(
                f"set_current_index: dimension '{dim}' not found or API unavailable"
            )
            return False
        except Exception as e:
            logger.debug(f"set_current_index error: {e}")
            return False

    def get_fov_list(self) -> List[Dict]:
        """Get the list of FOVs for the currently loaded dataset.

        Returns a list of dicts with 'region' (well ID) and 'fov' (FOV index)
        keys, sorted by region then FOV. This can be used to map between
        (well_id, fov_index) pairs and flat xarray FOV dimension indices.

        Returns:
            List of {"region": str, "fov": int} dicts, or empty list if no
            dataset is loaded.
        """
        if not getattr(self, "dataset_path", ""):
            return []

        try:
            base_path = Path(self.dataset_path)
            fmt = detect_format(base_path)
            return self._discover_fovs(base_path, fmt)
        except Exception as e:
            logger.debug(f"get_fov_list error: {e}")
            return []

    def has_fov_dimension(self) -> bool:
        """Check if loaded data has an FOV dimension.

        Returns:
            True if data is loaded and has 'fov' dimension, False otherwise.
        """
        xarray_data = getattr(self, "_xarray_data", None)
        if xarray_data is None:
            return False
        return "fov" in xarray_data.dims

    def refresh(self) -> None:
        """Force an immediate refresh of the viewer display.

        Useful after loading a new dataset or when you want to update
        the display without waiting for the automatic refresh timer.
        """
        self._force_refresh()

    def _create_lazy_array(self, base_path: Path) -> Optional[xr.DataArray]:
        """Create lazy xarray from dataset - auto-detects format."""
        if not LAZY_LOADING_AVAILABLE:
            return None

        fmt = detect_format(base_path)
        fovs = self._discover_fovs(base_path, fmt)

        if not fovs:
            print("No FOVs found")
            return None

        # print(f"Format: {fmt}, FOVs: {len(fovs)}")  # Disabled for profiling

        if fmt == "ome_tiff":
            return self._load_ome_tiff(base_path, fovs)
        else:
            return self._load_single_tiff(base_path, fovs)

    def _discover_fovs(self, base_path: Path, fmt: str) -> List[Dict]:
        """Discover all FOVs (region, fov) pairs."""
        fov_set = set()

        if fmt == "ome_tiff":
            ome_dir = base_path / "ome_tiff"
            if not ome_dir.exists():
                ome_dir = next(
                    (d for d in base_path.iterdir() if d.is_dir() and d.name.isdigit()),
                    base_path,
                )
            for f in ome_dir.glob("*.ome.tif*"):
                if m := FPATTERN_OME.search(f.name):
                    fov_set.add((m.group("r"), int(m.group("f"))))
        else:
            first_tp = next(
                (d for d in base_path.iterdir() if d.is_dir() and d.name.isdigit()),
                None,
            )
            if first_tp:
                for f in first_tp.glob("*.tiff"):
                    if m := FPATTERN.search(f.name):
                        fov_set.add((m.group("r"), int(m.group("f"))))

        return [{"region": r, "fov": f} for r, f in sorted(fov_set)]

    def _load_ome_tiff(
        self, base_path: Path, fovs: List[Dict]
    ) -> Optional[xr.DataArray]:
        """Fast OME-TIFF: mmap via tifffile.aszarr, small chunks, no big graphs."""
        try:
            ome_dir = base_path / "ome_tiff"
            if not ome_dir.exists():
                ome_dir = next(
                    (d for d in base_path.iterdir() if d.is_dir() and d.name.isdigit()),
                    base_path,
                )

            file_index = {}
            for f in ome_dir.glob("*.ome.tif*"):
                if m := FPATTERN_OME.search(f.name):
                    file_index[(m.group("r"), int(m.group("f")))] = str(f)
            if not file_index:
                return None

            first_file = next(iter(file_index.values()))
            with tf.TiffFile(first_file) as tif:
                series = tif.series[0]
                axes = series.axes
                shape = series.shape
                shape_dict = dict(zip(axes, shape))
                n_t = shape_dict.get("T", 1)
                n_c = shape_dict.get("C", 1)
                n_z = shape_dict.get("Z", 1)
                height = shape_dict.get("Y", shape[-2])
                width = shape_dict.get("X", shape[-1])
                channel_names = []
                pixel_size_x, pixel_size_y, pixel_size_z = None, None, None
                try:
                    if tif.ome_metadata:
                        import xml.etree.ElementTree as ET

                        root = ET.fromstring(tif.ome_metadata)
                        ns = {
                            "ome": "http://www.openmicroscopy.org/Schemas/OME/2016-06"
                        }
                        for ch in root.findall(".//ome:Channel", ns):
                            name = ch.get("Name") or ch.get("ID", "")
                            if name:
                                channel_names.append(name)

                        # Extract physical pixel sizes
                        pixel_size_x, pixel_size_y, pixel_size_z = (
                            extract_ome_physical_sizes(tif.ome_metadata)
                        )
                except Exception as e:
                    logger.debug("Failed to parse OME metadata: %s", e)

            axis_map = {"T": "time", "Z": "z", "C": "channel", "Y": "y", "X": "x"}
            dims_base = [axis_map.get(ax, f"ax_{ax}") for ax in axes]
            # Build channel name list - prefer extracted names, fill/truncate to match n_c
            if not channel_names:
                channel_names = [f"Ch{i}" for i in range(n_c)]
            elif len(channel_names) < n_c:
                channel_names.extend(f"Ch{i}" for i in range(len(channel_names), n_c))
            elif len(channel_names) > n_c:
                channel_names = channel_names[:n_c]
            # Keep coordinates numeric (indices) for all axes, including "channel";
            # channel names are stored in attrs and applied via _lut_controllers.
            # This convention is used consistently for both OME-TIFF and single-TIFF paths.
            coords_base = {
                axis_map.get(ax, f"ax_{ax}"): list(range(dim))
                for ax, dim in zip(axes, shape)
            }

            # Per-axis chunking: 1 for non-spatial, full for spatial
            chunks = []
            for ax, dim in zip(axes, shape):
                if ax in ("X", "Y"):
                    chunks.append(dim)
                else:
                    chunks.append(1)

            luts = {
                i: wavelength_to_colormap(extract_wavelength(name))
                for i, name in enumerate(channel_names)
            }
            n_fov = len(fovs)

            def open_zarr(path: str):
                tif = tf.TiffFile(path)
                zarr_store = tif.series[0].aszarr()
                return tif, zarr_store

            # One dask array per FOV, chunked per plane for fast single-slice reads
            fov_arrays = []
            tifs_kept = []
            for fov_idx in range(n_fov):
                region, fov = fovs[fov_idx]["region"], fovs[fov_idx]["fov"]
                filepath = file_index.get((region, fov))
                if not filepath:
                    fov_arrays.append(
                        da.zeros((n_t, n_z, n_c, height, width), dtype=np.uint16)
                    )
                    continue
                tif, zarr_store = open_zarr(filepath)
                tifs_kept.append(tif)
                arr = da.from_zarr(zarr_store, chunks=tuple(chunks))
                # keep tif open to support mmap; rely on Python GC after viewer closes
                fov_arrays.append(arr)

            # Insert fov axis immediately after time if present, else at front
            if "time" in dims_base:
                fov_axis = dims_base.index("time") + 1
            else:
                fov_axis = 0
            full_array = da.stack(fov_arrays, axis=fov_axis)

            dims_full = dims_base[:fov_axis] + ["fov"] + dims_base[fov_axis:]
            coords_full = coords_base.copy()
            coords_full["fov"] = list(range(n_fov))

            xarr = xr.DataArray(full_array, dims=dims_full, coords=coords_full)
            # Ensure standard dims exist with singleton axes if missing
            for ax in ["time", "fov", "z", "channel", "y", "x"]:
                if ax not in xarr.dims:
                    xarr = xarr.expand_dims({ax: [0]})
            xarr = xarr.transpose("time", "fov", "z", "channel", "y", "x")
            xarr.attrs["luts"] = luts
            xarr.attrs["channel_names"] = channel_names
            xarr.attrs["_open_tifs"] = tifs_kept

            # Store physical pixel sizes (in micrometers)
            if pixel_size_x is not None:
                xarr.attrs["pixel_size_x_um"] = pixel_size_x
            if pixel_size_y is not None:
                xarr.attrs["pixel_size_y_um"] = pixel_size_y
            if pixel_size_z is not None:
                xarr.attrs["pixel_size_z_um"] = pixel_size_z
            # Also store commonly used aliases
            if pixel_size_x is not None and pixel_size_y is not None:
                # Use average for isotropic XY pixel size
                xarr.attrs["pixel_size_um"] = (pixel_size_x + pixel_size_y) / 2
            if pixel_size_z is not None:
                xarr.attrs["dz_um"] = pixel_size_z

            return xarr
        except Exception as e:
            print(f"OME-TIFF load error: {e}")
            import traceback

            traceback.print_exc()
            return None

    def _load_single_tiff(
        self, base_path: Path, fovs: List[Dict]
    ) -> Optional[xr.DataArray]:
        """Fast single-TIFF: per-plane on-demand loads with tiny LRU cache."""
        try:
            file_index = {}  # (t, region, fov, z, channel) -> filepath
            t_set, z_set, c_set = set(), set(), set()

            for tp_dir in sorted(base_path.iterdir()):
                if not (tp_dir.is_dir() and tp_dir.name.isdigit()):
                    continue
                t = int(tp_dir.name)
                # Only add timepoint to t_set if it has at least one valid file
                # This prevents showing black images for empty/incomplete timepoints
                has_files = False
                for f in tp_dir.iterdir():
                    if f.suffix.lower() not in TIFF_EXTENSIONS:
                        continue
                    if m := FPATTERN.search(f.name):
                        region, fov = m.group("r"), int(m.group("f"))
                        z, channel = int(m.group("z")), m.group("c")
                        z_set.add(z)
                        c_set.add(channel)
                        file_index[(t, region, fov, z, channel)] = str(f)
                        has_files = True
                if has_files:
                    t_set.add(t)

            if not file_index:
                return None

            times = sorted(t_set)
            z_levels = sorted(z_set)
            channel_names = sorted(c_set)
            n_t, n_fov, n_z, n_c = (
                len(times),
                len(fovs),
                len(z_levels),
                len(channel_names),
            )

            sample = next(
                (
                    p
                    for p in file_index.values()
                    if Path(p).suffix.lower() in TIFF_EXTENSIONS
                ),
                None,
            )
            if sample is None:
                return None
            try:
                with tf.TiffFile(sample) as tif:
                    height, width = tif.pages[0].shape[-2:]
            except Exception as e:
                logger.debug("Failed to read sample TIFF: %s", e)
                return None

            luts = {
                i: wavelength_to_colormap(extract_wavelength(c))
                for i, c in enumerate(channel_names)
            }

            @lru_cache(maxsize=128)
            def load_plane(t, region, fov, z, channel):
                filepath = file_index.get((t, region, fov, z, channel))
                if not filepath:
                    return np.zeros((height, width), dtype=np.uint16)
                try:
                    ext = Path(filepath).suffix.lower()
                    if ext in TIFF_EXTENSIONS:
                        with tf.TiffFile(filepath) as tif:
                            return tif.pages[0].asarray()
                except Exception as e:
                    logger.debug("Failed to load plane %s: %s", filepath, e)
                return np.zeros((height, width), dtype=np.uint16)

            # Build on-demand loader via map_blocks over a dummy array
            chunks = (
                (1,) * n_t,
                (1,) * n_fov,
                (1,) * n_z,
                (1,) * n_c,
                (height,),
                (width,),
            )

            def _block_loader(block, block_info=None):
                loc = block_info[None]["chunk-location"]
                t_idx, f_idx, z_idx, c_idx = loc[0], loc[1], loc[2], loc[3]
                t = times[t_idx]
                region, fov = fovs[f_idx]["region"], fovs[f_idx]["fov"]
                z = z_levels[z_idx]
                channel = channel_names[c_idx]
                plane = load_plane(t, region, fov, z, channel)
                return plane.reshape(1, 1, 1, 1, height, width)

            dummy = da.zeros(
                (n_t, n_fov, n_z, n_c, height, width), chunks=chunks, dtype=np.uint16
            )
            stacked = da.map_blocks(
                _block_loader, dummy, dtype=np.uint16, chunks=chunks
            )

            # Read acquisition parameters for pixel size and dz
            pixel_size_um, dz_um = read_acquisition_parameters(base_path)

            # Fallback: try reading pixel size from TIFF metadata tags
            if pixel_size_um is None and sample is not None:
                pixel_size_um = read_tiff_pixel_size(sample)

            xarr = xr.DataArray(
                stacked,
                dims=["time", "fov", "z", "channel", "y", "x"],
                # Use actual values for time/z coords, numeric indices for fov/channel.
                # Channel names are stored in attrs and applied via _lut_controllers.
                coords={
                    "time": times,
                    "fov": list(range(n_fov)),
                    "z": z_levels,
                    "channel": list(range(n_c)),
                },
            )
            xarr.attrs["luts"] = luts
            xarr.attrs["channel_names"] = channel_names

            # Store physical pixel sizes (in micrometers)
            if pixel_size_um is not None:
                xarr.attrs["pixel_size_um"] = pixel_size_um
                xarr.attrs["pixel_size_x_um"] = pixel_size_um
                xarr.attrs["pixel_size_y_um"] = pixel_size_um
            if dz_um is not None:
                xarr.attrs["dz_um"] = dz_um
                xarr.attrs["pixel_size_z_um"] = dz_um

            return xarr
        except Exception as e:
            print(f"Single-TIFF load error: {e}")
            import traceback

            traceback.print_exc()
            return None

    def _set_ndv_data(self, data: xr.DataArray):
        """Update NDV viewer with lazy array."""
        global _current_voxel_scale

        if not NDV_AVAILABLE or not self.ndv_viewer:
            return

        # Log scale information and set voxel scale for 3D rendering
        pixel_size = data.attrs.get("pixel_size_um")
        dz = data.attrs.get("dz_um")
        if pixel_size is not None or dz is not None:
            scale_info = []
            if pixel_size is not None:
                scale_info.append(f"XY pixel size: {pixel_size:.4f} µm")
            if dz is not None:
                scale_info.append(f"Z step: {dz:.4f} µm")
            print(f"Scale metadata: {', '.join(scale_info)}")

            # Set voxel scale for 3D rendering (Z scaled relative to XY)
            if pixel_size is not None and dz is not None and pixel_size > 0:
                z_scale = dz / pixel_size
                _current_voxel_scale = (1.0, 1.0, z_scale)
                logger.info(f"Voxel aspect ratio (Z/XY): {z_scale:.2f}")
            else:
                _current_voxel_scale = None
        else:
            _current_voxel_scale = None

        luts = data.attrs.get("luts", {})
        channel_axis = data.dims.index("channel") if "channel" in data.dims else None

        # Recreate viewer with proper dimensions
        # Note: 3D button is always enabled - Downsampling3DXarrayWrapper handles
        # large volumes by automatically downsampling them for OpenGL rendering
        old_widget = self.ndv_viewer.widget()
        layout = self.layout()

        self.ndv_viewer = ndv.ArrayViewer(
            data,
            channel_axis=channel_axis,
            channel_mode="composite",
            luts=luts,
            visible_axes=(-2, -1),  # 2D display (y, x), sliders for rest
        )

        # Replace widget
        idx = layout.indexOf(old_widget)
        layout.removeWidget(old_widget)
        old_widget.deleteLater()
        layout.insertWidget(idx, self.ndv_viewer.widget(), 1)

        # Connect to nDimsRequested signal to re-hide sliders when 3D mode is toggled
        if hasattr(self.ndv_viewer, "_view") and hasattr(
            self.ndv_viewer._view, "nDimsRequested"
        ):
            self.ndv_viewer._view.nDimsRequested.connect(self._on_ndv_ndims_requested)

        # Update channel labels after viewer is ready.
        self._initiate_channel_label_update()

    def _initiate_channel_label_update(self):
        """Start the channel label update retry mechanism.

        Increments generation to cancel any pending retries from previous loads,
        then schedules retry attempts until NDV viewer is ready.
        """
        self._channel_label_generation += 1
        self._pending_channel_label_retries = CHANNEL_LABEL_UPDATE_MAX_RETRIES
        self._schedule_channel_label_update(self._channel_label_generation)

    def _schedule_channel_label_update(self, generation: int):
        """Retry updating channel labels until the NDV viewer is ready or we time out."""
        # Check if this callback is from a stale generation (viewer was replaced)
        if self._channel_label_generation != generation:
            return

        if not self.ndv_viewer or self._xarray_data is None:
            return

        remaining = self._pending_channel_label_retries
        if remaining <= 0:
            logger.warning(
                "Channel label update timed out - labels may show numeric indices"
            )
            return

        # Check if _lut_controllers is available (indicates viewer is ready).
        # Note: _lut_controllers is a private API that may change in future ndv versions;
        # at the time of writing there is no stable public API for this behavior in ndv.
        # If removed or renamed, this retry loop will timeout gracefully and channel
        # labels will not be updated, falling back to numeric indices in the UI.
        controllers = getattr(self.ndv_viewer, "_lut_controllers", None)
        if controllers:
            self._update_channel_labels()
            return

        # Not ready yet; schedule another check
        self._pending_channel_label_retries = remaining - 1
        QTimer.singleShot(
            CHANNEL_LABEL_UPDATE_RETRY_DELAY_MS,
            lambda: self._schedule_channel_label_update(generation),
        )

    def _update_channel_labels(self):
        """Manually update channel labels in the NDV viewer.

        This uses ndv's private _lut_controllers API to set display names.
        The approach is fragile and may break with future ndv updates.
        """
        if not self.ndv_viewer or self._xarray_data is None:
            return

        channel_names = self._xarray_data.attrs.get("channel_names", [])
        if not channel_names:
            return

        try:
            controllers = getattr(self.ndv_viewer, "_lut_controllers", None)
            if not controllers:
                return

            updated_names = []
            for i, name in enumerate(channel_names):
                if i in controllers:
                    controller = controllers[i]
                    controller.key = name
                    if hasattr(controller, "synchronize"):
                        # Propagate the updated key to the NDV UI so the channel
                        # label is displayed in the LUT controls.
                        controller.synchronize()
                    else:
                        logger.warning(
                            "LUT controller at index %d has no 'synchronize' method; "
                            "channel label '%s' may not appear in UI",
                            i,
                            name,
                        )
                    updated_names.append(name)
            logger.debug(
                "Updated %d channel labels: %s", len(updated_names), updated_names
            )
        except Exception as e:
            logger.debug("Failed to update channel labels: %s", e)


class LightweightMainWindow(QMainWindow):
    """Main window with dark theme."""

    viewer: LightweightViewer

    def __init__(self, dataset_path: str):
        super().__init__()
        self.setWindowTitle(f"NDViewer Lightweight - {Path(dataset_path).name}")
        self.setGeometry(100, 100, 720, 540)  # 4:3 aspect, smaller
        self._set_dark_theme()

        self.viewer = LightweightViewer(dataset_path)
        self.setCentralWidget(self.viewer)

    def _set_dark_theme(self):
        _apply_dark_theme(self)

    def closeEvent(self, event):
        """Ensure viewer cleanup when window closes."""
        self.viewer.close()
        super().closeEvent(event)


def main(dataset_path: str = None):
    """Launch lightweight viewer."""
    import sys

    app = QApplication(sys.argv)

    if dataset_path:
        # Direct launch with dataset
        window = LightweightMainWindow(dataset_path)
        window.show()
    else:
        # Show launcher window first
        launcher = LauncherWindow()
        launcher.show()

    sys.exit(app.exec_())


if __name__ == "__main__":
    import sys

    path = sys.argv[1] if len(sys.argv) > 1 else None
    main(path)
