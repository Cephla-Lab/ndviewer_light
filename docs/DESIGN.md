# NDViewer Light - Design and Implementation

A lightweight viewer built on [NDV](https://github.com/pyapp-kit/ndv) (`ndv>=0.5,<0.6`, vispy canvas, Qt through `qtpy`) for viewing 5D microscopy acquisitions with support for live acquisition streaming. The section [ndv Integration](#ndv-integration-ndv-05) lists which ndv APIs the viewer relies on and the few private points that remain.

## Architecture Overview

```
┌───────────────────────────────────────────────────────────────┐
│                      LightweightViewer                        │
├───────────────────────────────────────────────────────────────┤
│  ┌─────────────────────────────────────────────────────────┐  │
│  │                       UI Layer                          │  │
│  │  ┌───────────┐ ┌───────────┐ ┌───────────────────────┐  │  │
│  │  │ T Slider  │ │FOV Slider │ │      NDV Viewer       │  │  │
│  │  │  + Play   │ │  + Play   │ │ (vispy/OpenGL canvas) │  │  │
│  │  └─────┬─────┘ └─────┬─────┘ └───────────┬───────────┘  │  │
│  └────────┼─────────────┼───────────────────┼──────────────┘  │
│           │             │                   │                 │
│           ▼             ▼                   ▼                 │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │                  Coordination Layer                     │  │
│  │  _on_time_slider_changed() / _on_fov_slider_changed()   │  │
│  │  _load_current_fov() creates lazy dask array            │  │
│  │  _update_ndv_data() sends xarray to NDV                 │  │
│  └────────────────────────┬────────────────────────────────┘  │
│                           │                                   │
│           ┌───────────────┼───────────────┐                   │
│           ▼               ▼               ▼                   │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐              │
│  │ File Index  │ │ Plane Cache │ │Lazy Loading │              │
│  │(dict + lock)│ │(LRU + lock) │ │(dask.delayed│              │
│  │             │ │ 256MB limit │ │             │              │
│  │Key:(t,f,z,c)│ │Key:(t,f,z,c)│ │ Only loads  │              │
│  │Val: filepath│ │Val: ndarray │ │ on display  │              │
│  └─────────────┘ └─────────────┘ └─────────────┘              │
└───────────────────────────────────────────────────────────────┘
```

**Data Flow:**
1. Slider change → `_load_current_fov()` → creates dask array (no I/O)
2. NDV requests slice → dask triggers `_load_single_plane()`
3. `_load_single_plane()` checks cache → reads file if miss → caches result

## Two Operating Modes

### 1. File-Based Mode (Legacy)
- User opens a folder containing TIFF files
- Viewer scans folder structure and builds lazy xarray
- Periodic refresh timer polls for new files during acquisition
- Suitable for viewing completed acquisitions

### 2. Push-Based Mode (Live Acquisition)
- Acquisition controller calls `start_acquisition()` with configuration
- Each saved image is registered via `register_image()`
- Viewer updates in real-time without filesystem polling
- More efficient and responsive than file-based mode

## Mode Selection

The viewer determines its operating mode based on whether `start_acquisition()` has been called:

```python
def is_push_mode_active(self) -> bool:
    """Check if push-based mode is active (has FOV labels configured)."""
    return bool(self._fov_labels)
```

| Condition | Mode | Triggered by |
|-----------|------|--------------|
| `_fov_labels` is empty | File-based | User opens folder via `load_dataset()` |
| `_fov_labels` is set | Push-based | Acquisition controller calls `start_acquisition()` |

When `start_acquisition()` is called, it populates `_fov_labels`, switching the viewer to push mode. The lazy dask loading (`_load_current_fov()`) is only used in push mode.

## Push-Based API

### Initialization
```python
viewer.start_acquisition(
    channels=["BF", "DAPI", "GFP"],
    num_z=5,
    height=2048,
    width=2048,
    fov_labels=["A1:0", "A1:1", "A2:0"],  # well:fov format
)
```

### Image Registration (Thread-Safe)
```python
# Called from acquisition worker thread
viewer.register_image(
    t=0,           # timepoint
    fov_idx=0,     # FOV index
    z=2,           # z-level
    channel="GFP", # channel name
    filepath="/path/to/image.tiff"
)
```

### Navigation
```python
viewer.load_fov(fov=1, t=0)           # Load specific FOV
viewer.go_to_well_fov("A2", 0)        # Navigate by well ID
viewer.set_current_index("z", 2)      # Move one ndv slider programmatically
```

`set_current_index(dim, value)` validates `dim` against the public
`ArrayViewer.data_wrapper.dims` and `value` against `data_wrapper.sizes()`, then writes
`display_model.current_index[dim] = value` (ndv's own keybindings write the same
mapping; there is no setter method). Unknown dimensions and out-of-range values are
rejected: the method returns `False` and the model is left untouched. It never guards on
`dim in current_index`: on ndv 0.5 that mapping stays empty until a slider is moved
(index resolution became pure in ndv 0.5), so such a guard made the call a silent no-op.

### Cleanup
```python
viewer.end_acquisition()
```

## Lazy Loading with Dask

### Problem
Loading all z-planes and channels eagerly causes latency when navigating FOVs.
For 5 z-levels × 6 channels = 30 planes at ~12ms each = 360ms per FOV change.

### Solution
Create lazy dask arrays that only load planes when NDV actually displays them:

```python
def _load_current_fov(self):
    # Get current position from instance state
    t = self._current_time_idx
    fov = self._current_fov_idx
    h, w = self._image_height, self._image_width

    # Create lazy structure - no I/O here
    delayed_planes = []
    for z in self._z_levels:
        channel_planes = []
        for channel in self._channel_names:
            delayed_load = dask.delayed(self._load_single_plane)(t, fov, z, channel)
            da_plane = da.from_delayed(delayed_load, shape=(h, w), dtype=np.uint16)
            channel_planes.append(da_plane)
        delayed_planes.append(da.stack(channel_planes))
    data = da.stack(delayed_planes)  # Shape: (n_z, n_c, h, w)

    # NDV only loads displayed slice
    self._update_ndv_data(data)
```

### Result
- Only displayed z-plane is loaded from disk
- Changing z-level triggers load of just that plane
- 3D volume view still works (loads all planes when needed)

## Thread Safety

### Challenge
- `register_image()` called from acquisition worker thread
- `_load_single_plane()` called from dask worker threads
- Main thread handles GUI updates

### Solution: Two Locks

**1. File Index Lock** (`_file_index_lock`)
```python
# Write (worker thread)
with self._file_index_lock:
    self._file_index[(t, fov, z, channel)] = filepath

# Read (dask workers)
with self._file_index_lock:
    filepath = self._file_index.get(cache_key)
```

**2. Cache Lock** (internal to `MemoryBoundedLRUCache`)
```python
class MemoryBoundedLRUCache:
    def __init__(self, max_memory_bytes):
        self._lock = threading.Lock()

    def get(self, key):
        with self._lock:
            # LRU operations are atomic
```

## Dynamic Slider Ranges

### Per-Timepoint FOV Tracking
```python
self._max_fov_per_time: Dict[int, int] = {}  # timepoint -> max FOV index
```

### Behavior
1. FOV slider starts at max=0
2. As images are registered, slider range grows
3. When user changes timepoint, FOV slider adjusts to available FOVs

```python
def _on_time_slider_changed(self, value):
    # Adjust FOV slider for this timepoint
    available_fov_max = self._max_fov_per_time.get(value, 0)
    self._fov_slider.setMaximum(available_fov_max)

    # Clamp if needed
    if self._current_fov_idx > available_fov_max:
        self._fov_slider.setValue(available_fov_max)
```

## Caching Strategy

### Memory-Bounded LRU Cache
- 256MB default limit
- Evicts least-recently-used planes when full
- Thread-safe with internal lock

```python
class MemoryBoundedLRUCache:
    def put(self, key, value):
        with self._lock:
            # Evict LRU entries until we have room
            while self._current_memory + item_size > self._max_memory:
                oldest_key, oldest_value = self._cache.popitem(last=False)
                self._current_memory -= oldest_value.nbytes

            self._cache[key] = value
            self._current_memory += item_size
```

### Cache Key
```python
cache_key = (t, fov_idx, z, channel)  # Unique per plane
```

## Debouncing

### Problem
During acquisition, every saved image triggers `register_image()` → `_load_current_fov()`.
With 6 channels at ~50ms per image = 12 reloads/second, overwhelming the GUI.

### Solution
Debounce timer coalesces rapid updates:

```python
def _schedule_debounced_load(self):
    self._load_pending = True
    if not self._load_debounce_timer.isActive():
        self._load_debounce_timer.start(200)  # 200ms debounce

def _execute_debounced_load(self):
    if self._load_pending:
        self._load_pending = False
        self._load_current_fov()
```

## ndv Integration (ndv 0.5)

The viewer uses ndv's public API wherever one exists:

| Need | ndv 0.5 API | Where in `core.py` |
|------|-------------|--------------------|
| Display an array | `ndv.ArrayViewer(data, channel_axis=..., channel_mode="composite", luts=..., visible_axes=(-2, -1), scales=...)` | `_set_ndv_data()` |
| Channel names and colours | `luts={position: {"cmap": ..., "name": ...}}`; ndv labels each LUT row from `LUTModel.name` and falls back to the channel coordinate (`"0"`, `"1"`, ...) when it is empty | `build_luts()`, stored in `attrs["luts"]` by every array builder (push TIFF, the two zarr-push builders, OME-TIFF, single-TIFF, OME-Zarr) |
| Swap the displayed array without a rebuild | `ArrayViewer.data_wrapper` and the wrapper's documented `dims_changed` / `data_changed` signals | `_try_inplace_ndv_update()` |
| Navigate | `data_wrapper.dims` / `sizes()` for validation, `display_model.current_index[dim] = value` | `set_current_index()` |
| 3D voxel aspect | `scales={<z dim>: dz_um / pixel_size_um}` | `_set_ndv_data()` |
| Volumes larger than the GL 3D texture limit | ndv downsamples the texture itself (strided view) and compensates the scales, so a 2100-px volume keeps its 2100-unit extent | nothing: the former `Downsampling3DXarrayWrapper` is gone |

`_try_inplace_ndv_update` emits `dims_changed` only when the shape changed (ndv then
re-creates the sliders and re-resolves the view) and **always** `data_changed`
afterwards: on ndv 0.5 `dims_changed` alone re-fetches only when the resolved index
changed, so a shape change at an unchanged index would keep the old texture on screen.

### Private points that remain

Each one is pinned by a test so the next ndv rename fails loudly instead of silently:

- **`DataWrapper._data` write** in `_try_inplace_ndv_update`: ndv has no public setter for
  the wrapped array. The public `ArrayViewer.data = new` setter is not a substitute: it
  clears the canvas and re-adds every image on the next data response (probe on ndv 0.5,
  200 push frames: 85 ms per frame instead of 4 ms, camera reset each frame, 3353 vispy
  GL objects created instead of 0; neither path leaks handles). Pinned by
  `tests/test_inplace_update.py::test_data_wrapper_private_data_slot` and window smoke S1.
- **`display_model.current_index` written directly** (public model, no setter method on
  `ArrayViewer`). Pinned by `tests/test_set_current_index.py` and window smoke S3.
- **superqt `QLabeledSlider.setRange` patch** at module import (superqt-private, not ndv;
  untouched by the ndv 0.5 port).
- **Test-only:** `tests/_ndv_private.py` reads the uploaded texture
  (`_lut_controllers[key].handles[0].data()`), the dims sliders
  (`_view._qwidget.dims_sliders._sliders[dim]`) and the fetch state (`_is_idle()`), and
  `pytest.skip`s when a path is gone; `tests/conftest.py` pins
  `ndv.views._vispy._array_canvas.get_max_texture_sizes` for offscreen runs (below).
  `core.py` imports neither.

### Headless testing

ndv 0.5 queries `GL_MAX_TEXTURE_SIZE` on the first `add_image` / `add_volume`. Under
`QT_QPA_PLATFORM=offscreen` there is no GL context and the raw `glGetIntegerv` segfaults
(rc 139), so `tests/conftest.py` answers the query with fixed limits `(4096, 2048)` before
any viewer exists. `NDVL_REAL_GL=1` keeps the real query for runs with a window server
(`QT_QPA_PLATFORM=cocoa` on macOS, `xcb` under Xvfb in CI); only those runs execute the
`@pytest.mark.window` texture smokes. CI runs the suite both ways. See `tests/README.md`.

## 3D Volume Rendering

Z-step often differs from the XY pixel size. `_set_ndv_data` passes
`scales={"z" | "z_level": dz_um / pixel_size_um}` to `ndv.ArrayViewer` whenever both sizes
are in the array's attrs; ndv applies it as an `STTransform` on the volume and frames the
camera to match (no vispy patching). Volumes larger than the GL 3D texture limit are
downsampled by ndv with the scales compensated. Checked with a (3, 2100, 2100) volume,
dz 1.5 µm / pixel 0.5 µm, on vispy 0.17.0 and 0.14.3: texture (3, 1050, 1050),
`STTransform.scale` (2, 2, 3), camera x/y range (-21, 2121), z range (-0.09, 9.09).

## Signal Flow

```
Acquisition Worker Thread          Main Thread (Qt)
        │                               │
        │  register_image()             │
        ├──────────────────────────────►│
        │                               │ _image_registered signal
        │                               │         │
        │                               │         ▼
        │                               │ _on_image_registered()
        │                               │    - Update _max_fov_per_time
        │                               │    - Update slider ranges
        │                               │    - Schedule debounced load
        │                               │         │
        │                               │         ▼ (after 200ms)
        │                               │ _load_current_fov()
        │                               │    - Create dask array
        │                               │    - Update NDV
        │                               │         │
        │                               │         ▼
        │                               │ NDV requests slice
        │                               │         │
        │                     ┌─────────┴─────────┐
        │                     │ Dask Worker Pool  │
        │                     │  _load_single_plane()
        │                     │    - Check cache
        │                     │    - Load TIFF if needed
        │                     └───────────────────┘
```

## File Format Support

The viewer supports two file formats. **Note:** File formats and operating modes are independent - either format can be used with either mode.

### Single-TIFF Format
One file per (well, fov, z, channel) combination:
```
acquisition_folder/
├── 0/                          # timepoint
│   ├── A1_0_0_BF.tiff         # well_fov_z_channel.tiff
│   ├── A1_0_0_DAPI.tiff
│   ├── A1_0_1_BF.tiff
│   └── ...
└── 1/
    └── ...
```

### OME-TIFF Format
All z-planes and channels in one file per (well, fov):
```
acquisition_folder/
├── 0/
│   ├── A1_0.ome.tiff          # All z/channels in one file
│   └── ...
└── 1/
    └── ...
```

### Mode vs Format Matrix

|                  | Single-TIFF | OME-TIFF |
|------------------|-------------|----------|
| **Push Mode**    | ✅ Tested    | ⚠️ Untested |
| **File Mode**    | ✅ Tested    | ✅ Tested   |

**Testing Status (PR #23):** This PR primarily tests single-TIFF with push mode. Other combinations may work but have not been validated in this PR.

## Performance Characteristics

| Operation | Typical Time | Notes |
|-----------|-------------|-------|
| Dask array setup | <1ms | No I/O, just delayed objects |
| Single plane load (disk) | 10-15ms | Depends on disk speed |
| Single plane load (cached) | <0.1ms | Memory access only |
| NDV update (in-place) | ~4ms to the texture (max 46ms) | No viewer rebuild; `viewer.data = new` would take ~85ms and reset the camera |
| Slider animation | 100ms interval | 10 FPS playback |
| Debounce interval | 200ms | Max 5 loads/sec during acquisition |
