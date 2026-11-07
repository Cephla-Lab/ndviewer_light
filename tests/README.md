# Test Suite for ndviewer_light

This directory contains unit tests for the ndviewer_light application.

## Running Tests

Ensure pytest is installed in your environment:
```bash
pip install pytest
```

Run all tests:
```bash
pytest tests/
```

Run tests with verbose output:
```bash
pytest tests/ -v
```

Run specific test file:
```bash
pytest tests/test_channel_names.py -v
```

Run specific test class:
```bash
pytest tests/test_channel_names.py::TestOMETiffChannelNamesLogic -v
```

Run specific test:
```bash
pytest tests/test_channel_names.py::TestOMETiffChannelNamesLogic::test_empty_channel_names_fallback_to_default -v
```

## Test Files

### test_channel_names.py

Comprehensive unit tests for the channel name display feature. Tests cover four main areas:

#### 1. OME-TIFF Channel Name Logic (`TestOMETiffChannelNamesLogic`)

Tests the channel name extraction and preservation logic from OME-TIFF metadata (in `_load_ome_tiff` method):

- **Empty channel names fallback**: When no channel names are found in metadata, defaults to `Ch0`, `Ch1`, `Ch2`, etc.
- **Extension with fallbacks**: When fewer channel names than channels (e.g., 2 names for 5 channels), extends with `Ch2`, `Ch3`, `Ch4`
- **Truncation**: When more channel names than channels (e.g., 5 names for 2 channels), truncates to match
- **Exact match preservation**: When channel names count matches exactly, preserves all names
- **XML metadata parsing**: Tests parsing of OME-TIFF XML metadata with proper namespace handling
- **Error handling**: Tests fallback behavior when XML is invalid or channels have missing Name attributes

#### 2. Single-TIFF Channel Name Logic (`TestSingleTiffChannelNamesLogic`)

Tests channel name extraction from single-TIFF filenames (in `_load_single_tiff` method):

- **Alphabetical sorting**: Channel names extracted from filenames are sorted alphabetically
- **Set to list conversion**: Tests the conversion from set (during discovery) to sorted list
- **Storage in attrs**: Verifies channel names are properly stored in xarray attrs

#### 3. Channel Names Reaching ndv (`TestChannelNamesReachNdv`)

ndv 0.5 labels each LUT row from the public `LUTModel.name` field and falls back to the channel coordinate (`"0"`, `"1"`, ...) when it is empty. ndviewer_light keeps integer channel coordinates, so `build_luts(channel_names, channel_colors)` in `core.py` pairs every channel position with `{"cmap": ..., "name": ...}`; that dict is stored in `attrs["luts"]` and passed to `ndv.ArrayViewer(luts=...)` in `_set_ndv_data`. There is no retry loop and no private ndv access any more.

- **`build_luts` pairing**: Pure test of the helper (wavelength-derived colormap, hex colour override, fallback for missing/empty colours)
- **Push-mode names**: `start_acquisition(["BF", "GFP"], ...)` leaves `display_model.luts[i].name == "BF" / "GFP"` right after construction
- **OME-Zarr plate names**: `load_dataset` on a `write_plate(...)` store carries the omero channel labels into the model the same way
- **Checkbox labels**: polls the event loop (up to 3 s with `processEvents()`, never `sleep`) until the LUT rows ndv creates on its first data response show the names
- **No private access**: `core.py` contains no `_lut_controllers` reference

The viewer tests construct a real `LightweightViewer` under `QT_QPA_PLATFORM=offscreen`; `conftest.py` pins ndv's GL texture-size query so this works without a GL context.

#### 4. Integration Tests (`TestChannelNamesIntegration`)

Tests the end-to-end behavior of channel name handling:

- **Length consistency**: Verifies channel names list always matches channel dimension size
- **Default format**: Verifies default names follow `Ch0`, `Ch1`, `Ch2` format
- **Empty/None handling**: Tests that empty lists and None values become defaults
- **Custom name preservation**: Tests that custom names are preserved when counts match

## Test Philosophy

The name-processing tests follow a **logic-focused** approach, and the ndv-facing tests check the public model:

1. **Logic verification**: Tests directly verify the channel name processing logic by simulating the same code paths
2. **Public ndv API only**: Where a test needs ndv, it asserts on `display_model.luts[i].name` (public) and the widget text, never on ndv internals
3. **Headless**: Viewer tests run offscreen; anything that needs the event loop polls with `processEvents()` instead of sleeping
4. **Edge case coverage**: Comprehensive coverage of boundary conditions (empty, fewer, more, exact match)
5. **Clear documentation**: Each test has a descriptive name and docstring explaining what it verifies

This approach ensures:
- Fast test execution (a few seconds for the full suite)
- Clear verification of business logic and of what ndv is actually given
- Easy debugging when tests fail

## Test Organization

Tests are organized into classes by functional area:
- `TestOMETiffChannelNamesLogic`: OME-TIFF path logic
- `TestSingleTiffChannelNamesLogic`: Single-TIFF path logic
- `TestChannelNamesReachNdv`: `build_luts` and the names ndv holds in `LUTModel.name`
- `TestChannelNamesIntegration`: End-to-end integration tests

## Code References

Each test includes comments referencing the relevant methods in `ndviewer_light/core.py` that implement the tested logic, making it easy to trace tests back to the implementation using IDE navigation (e.g., "Go to Definition").
