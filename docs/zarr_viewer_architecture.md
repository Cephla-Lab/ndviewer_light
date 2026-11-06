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

FOVs are found from the OME-Zarr 0.5 metadata in `ndviewer_light/ome_zarr.py`:
`plate.wells` → each well's `well.images` → each image's
`multiscales[0].datasets[0].path` (the full-resolution array), with axes, scale and
channels read from the same metadata. The old directory walk (`discover_zarr_v3_fovs`)
runs only when no OME-Zarr 0.5 layout is recognised (pre-0.5 Squid stores).

`_load_zarr_v3_5d` / `_load_zarr_v3_6d` in `core.py` then build the array per image:

```python
# 1. Tensorstore opens the image's array; dask wraps it with one chunk per plane
darr = self._open_zarr_image(image)  # open_zarr_tensorstore(image.group, image.array_path)

# 2. The image's axes are put in a fixed order (singleton axes added when missing)
darr = _arrange_axes(darr, image.dim_names, ["t", "c", "z", "y", "x"])

# 3. FOVs are stacked and xarray adds dimension labels, channel LUTs and scale
xarr = self._zarr_xarray(stacked, first_image)  # dims: time, fov, z, channel, y, x
```

A FOV that fails to open fails the whole load; the status label shows the error and
the path. There is no placeholder.

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

## Plane Cache

The viewer maintains an application-level LRU cache (`MemoryBoundedLRUCache`) for loaded planes. This cache is necessary because no other layer provides caching:

```
User slider → NDV → Dask → _load_zarr_plane() → Tensorstore → Disk
                                  ↑
                           CACHE IS HERE
```

- **NDV**: No caching (passes through to dask)
- **Dask**: No caching (lazy arrays recompute on each `.compute()`)
- **Tensorstore**: Minimal internal caching
- **OS page cache**: Helps but still has syscall overhead

### Performance Impact

Benchmarks with 1024x1024 uint16 planes:

| Interaction Pattern | Without Cache | With Cache | Speedup |
|---------------------|---------------|------------|---------|
| Z-stack back-and-forth | 86 ms | 13 ms | **6.5x** |
| Channel switching | 36 ms | 2 ms | **18.7x** |
| Time scrubbing | 21 ms | 10 ms | **2.2x** |

Channel switching benefits most because users typically cycle through the same channels repeatedly.

### Configuration

- Default size: 256 MB
- Holds ~128 planes at 1024x1024, or ~32 planes at 2048x2048
- LRU eviction when limit is reached

### Race Condition Handling

During live acquisition, a race condition can occur:

1. Viewer reads plane → gets zeros (data not yet flushed)
2. During read, `notify_zarr_frame()` is called
3. Read completes with stale zeros

To prevent caching stale data:
- Check `was_written_before_read` before caching
- Call `cache.invalidate()` in `notify_zarr_frame()`

## Why Tensorstore Instead of Zarr-Python?

1. **Zarr v3 support**: Tensorstore natively supports both zarr v2 and v3 formats with a unified API
2. **Performance**: Async I/O and parallel chunk fetching
3. **No zarr library dependency**: Avoids zarr-python's v2/v3 API differences

## Supported Zarr Structures

The viewer auto-detects these zarr layouts when browsing existing data:

| Structure | Description | Dimensions |
|-----------|-------------|------------|
| `hcs_plate` | HCS plate structure (row/col/field) | (T, C, Z, Y, X) per FOV |
| `per_fov` | One zarr per FOV (flat structure) | (T, C, Z, Y, X) per FOV |
| `6d` | Single zarr with FOV dimension | (FOV, T, C, Z, Y, X) |

For push-based live acquisition, use:
- `start_zarr_acquisition()` - per_fov structure
- `start_zarr_acquisition_6d()` - 6D structure (supports multiple regions)

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

- `ndviewer_light/core.py` - Main viewer implementation (includes zarr v3 support)
- `simulate_zarr_acquisition.py` - Test script for zarr viewing
