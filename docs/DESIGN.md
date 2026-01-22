# NDViewer Light - Design and Implementation

A lightweight viewer built on [NDV](https://github.com/pyapp-kit/ndv) for viewing 5D microscopy acquisitions with support for live acquisition streaming.

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
```

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

## 3D Volume Rendering

### Automatic Downsampling
For volumes exceeding OpenGL texture limits (typically 2048³):

```python
class Downsampling3DXarrayWrapper(XarrayWrapper):
    def isel(self, index):
        data = super().isel(index)

        if exceeds_texture_limit(data.shape):
            zoom_factors = compute_zoom_factors(data.shape, max_texture_size)
            return ndimage_zoom(data, zoom_factors, order=0)

        return data
```

### Anisotropic Voxels
Z-step often differs from XY pixel size. Vispy VolumeVisual is patched to scale vertices:

```python
# Patch applied at module load (simplified pseudocode)
def _patched_create_vertex_data(self):
    ...  # existing setup code
    scale = getattr(self, "_voxel_scale", None)
    if scale:
        z0, z1 = -0.5 * scale[2], (shape[0] - 0.5) * scale[2]
        ...  # create scaled vertices using z0, z1
```

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
| NDV update (in-place) | <1ms | No viewer rebuild |
| Slider animation | 100ms interval | 10 FPS playback |
| Debounce interval | 200ms | Max 5 loads/sec during acquisition |
