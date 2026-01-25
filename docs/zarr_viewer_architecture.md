# Zarr Viewer Architecture

This document explains how ndviewer_light loads and displays zarr datasets.

## Overview

The viewer uses a three-layer stack to provide lazy, labeled access to zarr data:

```
┌─────────────────────────────────────────┐
│  xarray.DataArray                       │  ← Labeled dimensions (time, fov, z, channel, y, x)
│  - Named coordinates                    │
│  - Metadata (luts, channel_names)       │
├─────────────────────────────────────────┤
│  dask.array                             │  ← Lazy chunked computation
│  - Defines chunk boundaries             │
│  - Deferred execution graph             │
├─────────────────────────────────────────┤
│  tensorstore                            │  ← I/O backend
│  - Reads zarr v2 and v3 formats         │
│  - Async/parallel chunk fetching        │
│  - Supports array slicing interface     │
└─────────────────────────────────────────┘
```

| Layer | Role |
|-------|------|
| **Tensorstore** | Handles zarr format details (v2 vs v3, codecs, chunk layout). Provides NumPy-like slicing interface. |
| **Dask** | Adds lazy evaluation - nothing reads from disk until `.compute()` is called. Enables out-of-core processing for datasets larger than RAM. |
| **Xarray** | Provides semantic dimension names instead of axis indices. `arr.sel(channel=0, z=5)` is clearer than `arr[:, :, 5, 0, :, :]`. |

## How the Layers Connect

From `_load_zarr_v3_5d` in `core.py`:

```python
# 1. Tensorstore opens the zarr store
ts_arr = open_zarr_tensorstore(fov_path, array_path="0")

# 2. Dask wraps tensorstore with chunking strategy
chunks = (1, 1, 1, height, width)  # One chunk per plane
darr = da.from_array(ts_arr, chunks=chunks)

# 3. Xarray adds dimension labels
xarr = xr.DataArray(
    darr,
    dims=["time", "fov", "z", "channel", "y", "x"],
    coords={...}
)
```

Tensorstore implements the `__getitem__` protocol, so dask can treat it like a NumPy array:

```python
# When dask needs chunk [0, 0, 0, :, :], it calls:
ts_arr[0, 0, 0, :, :]  # Tensorstore fetches just that chunk from disk
```

## Lazy Loading

The viewer only loads planes that are currently displayed. When you move a slider:

```
User moves Z slider
        ↓
NDV requests arr.sel(z=new_z)
        ↓
Xarray translates to arr[:, :, new_z, :, :, :]
        ↓
Dask identifies which chunks are needed
        ↓
Dask calls tensorstore.__getitem__ for each chunk
        ↓
Tensorstore reads from zarr store on disk
        ↓
Data returned to NDV for display
```

This allows viewing datasets much larger than available RAM.

## Chunking Strategy

Data is chunked per-plane for optimal 2D viewing:

```python
# Per-plane chunks: optimal for 2D slice viewing
chunks = (1, 1, 1, height, width)

# This means:
# - Each (t, fov, z, channel) combination is one chunk
# - Moving sliders only loads the single plane needed
# - No wasted I/O reading adjacent planes
```

## Memory Considerations

- **Tensorstore**: Minimal memory - only caches what's requested
- **Dask**: Builds a task graph (lightweight) until `.compute()` is called
- **Xarray**: Thin wrapper, adds minimal overhead

The actual pixel data only enters memory when displayed.

## Why Tensorstore Instead of Zarr-Python?

1. **Zarr v3 support**: Tensorstore natively supports both zarr v2 and v3 formats with a unified API
2. **Performance**: Async I/O and parallel chunk fetching
3. **No zarr library dependency**: Avoids zarr-python's v2/v3 API differences

## Supported Zarr Structures

The viewer auto-detects and handles three zarr layouts:

| Structure | Description | Dimensions |
|-----------|-------------|------------|
| `per_fov` | One zarr per FOV (HCS plate or flat) | (T, C, Z, Y, X) per FOV |
| `6d_single` | Single zarr with FOV dimension | (FOV, T, C, Z, Y, X) |
| `6d_regions` | Multiple region zarrs | (FOV, T, C, Z, Y, X) per region |

## Push-Based API for Live Acquisition

During live acquisition, the viewer receives frame notifications instead of polling:

```python
# Acquisition system notifies viewer when a frame is written
viewer.notify_zarr_frame(t=0, fov_idx=0, z=0, channel="DAPI")
```

This triggers:
1. Cache invalidation for the updated plane
2. Slider range updates if new timepoints/FOVs appear
3. Display refresh if viewing the updated plane

## Metadata

Channel names, colors, and pixel sizes are read from `zarr.json` metadata following OME-NGFF conventions:

- `ome.multiscales` - axis definitions and transforms
- `ome.omero` - channel names and colors
- `_squid` - acquisition parameters (pixel size, z spacing)

## Related Files

- `ndviewer_light/core.py` - Main viewer implementation
- `ndviewer_light/zarr_v3.py` - Zarr v3 metadata parsing and tensorstore helpers
- `simulate_zarr_acquisition.py` - Test script for zarr viewing
