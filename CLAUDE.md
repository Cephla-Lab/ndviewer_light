# ndviewer_light Development Notes

## Project Overview
Lightweight microscopy image viewer built on ndv (n-dimensional viewer). Supports OME-TIFF and single-TIFF formats with lazy loading via Dask.

## Key Architecture

### Data Loading
- **OME-TIFF**: Uses Zarr backend via `tifffile.imread(..., aszarr=True)` then `da.from_zarr()` - no Dask graph building needed
- **Single-TIFF**: Uses `da.map_blocks()` with atomic single-plane chunks - flat graph, O(1) per slice
- **LRU cache** (`@lru_cache(maxsize=128)`) on plane loading bypasses Dask for repeated access

### Live Refresh (Acquisition Mode)
- Signature-based change detection: cheap filesystem checks (mtime, size) avoid re-reading unchanged data
- **Memory leak workaround**: ndv's data setter leaks GPU handles (see [ndv#209](https://github.com/pyapp-kit/ndv/issues/209))
  - Fix: Bypass setter via `wrapper._data = data` directly
  - Shape changes: emit `wrapper.dims_changed.emit()` to update sliders
  - Same shape: call `viewer._request_data()` to refresh display
  - See `_try_inplace_ndv_update()` in `core.py`

## Testing
```bash
# Run all tests
pytest -v

# Run memory leak comparison test
python tests/test_memory_leak_reproduction.py --compare
```

## CI Notes
- **Black version pinned to <26** in CI - version 26.x introduced formatting changes that break compatibility
- Tests require Qt system dependencies (installed via apt in CI)

## Code Style
- Use `black` for formatting (version <26)
- Prefer simple, minimal changes - avoid over-engineering
