# ndviewer_light

Minimal NDV-based viewer for viewing 5-D acquisitions.

## Installation

### Requirements

- Python 3.10+.
- [ndv](https://github.com/pyapp-kit/ndv) `>=0.5,<0.6` with the vispy canvas (installed by
  the package). ndv 0.5 depends on `ihist`, which needs **numpy >= 2**.
- A Qt binding through `qtpy`: the host application supplies PyQt5 or PyQt6 when the viewer
  is embedded; a standalone install brings its own with `pip3 install ".[pyqt6]"`.

### Windows (one line)

Needs [Git](https://git-scm.com/download/win) and [Python 3.10+ from python.org](https://www.python.org/downloads/windows/).
In PowerShell, from the folder you want the viewer in:

```powershell
git clone https://github.com/Cephla-Lab/ndviewer_light; cd ndviewer_light; py -3 -m venv .venv; .\.venv\Scripts\python -m pip install ".[pyqt6]"; .\.venv\Scripts\python create_shortcut.py
```

This installs the viewer into its own virtual environment (`ndviewer_light\.venv`, nothing
global is touched) and puts an **NDViewer Light** shortcut on your Desktop.

- It uses `py`, the launcher the python.org installer adds. On a fresh Windows, `python` is
  only a stub that opens the Microsoft Store.
- Update: `cd ndviewer_light; git pull; .\.venv\Scripts\python -m pip install ".[pyqt6]"`
- Run from a terminal: `.\.venv\Scripts\ndviewer-light [path\to\dataset]`

### Standalone (pip)

```bash
git clone https://github.com/Cephla-Lab/ndviewer_light.git
cd ndviewer_light
pip3 install ".[pyqt6]"
```

> On Ubuntu 24.04+, add `--break-system-packages` or use a virtual environment.

This installs an `ndviewer-light` command you can run from anywhere. The `[pyqt6]` extra brings
the Qt binding; without it nothing provides one and the viewer cannot open a window.

#### Create a desktop shortcut (optional)

```bash
python3 create_shortcut.py
```

- **macOS:** Creates an app in `~/Applications/NDViewer Light.app`. Drag it to your Dock if you like.
- **Windows:** Creates a shortcut on your Desktop.
- **Linux:** Creates a `.desktop` entry in `~/.local/share/applications/`.

The shortcut launches the `ndviewer-light` installed next to the Python that runs the script,
so running it with a virtual environment's Python (activated or not) targets that environment.

### Conda environment

```bash
conda env create -f environment.yml
conda activate ndviewer_light
```

### As a library

```bash
pip3 install /path/to/ndviewer_light
```

## Usage

### Command line

```bash
ndviewer-light                      # open with file dialog
ndviewer-light /path/to/dataset     # open a specific dataset
```

### As a library

```python
from ndviewer_light import LightweightViewer, LightweightMainWindow

# Embed viewer widget in your own application
viewer = LightweightViewer("/path/to/dataset")

# Or use the standalone window
window = LightweightMainWindow("/path/to/dataset")
window.show()
```

## Supported formats

- **OME-Zarr 0.5 (Zarr v3)**: a plate, well or image group, opened directly or inside a Squid experiment folder. Datasets, axes and pixel sizes are read from the store's metadata.
- **Squid's 6D layout** (time, FOV, z, channel, y, x).
- **OME-TIFF**.
- **Single TIFF**.

An array whose planned planes were not all written is labelled "(incomplete acquisition)".

## Development

```bash
pip3 install -e ".[pyqt6]" pytest
python -m pytest tests/                                             # offscreen, no GL context needed
QT_QPA_PLATFORM=cocoa NDVL_REAL_GL=1 python -m pytest tests/        # macOS: real GL, runs the window smokes
QT_QPA_PLATFORM=xcb NDVL_REAL_GL=1 xvfb-run python -m pytest tests/ # Linux / CI, same
black .                                                             # formatting (black<26)
```

On Windows (PowerShell), the real-GL run sets the variables with `$env:`:

```powershell
python -m pytest tests/                                                             # offscreen
$env:QT_QPA_PLATFORM = "windows"; $env:NDVL_REAL_GL = "1"; python -m pytest tests/  # real GL, runs the window smokes
```

The test modules default to `QT_QPA_PLATFORM=offscreen`, and `tests/conftest.py` pins ndv's
GL texture-size query there (ndv 0.5 segfaults on it without a GL context). The texture-level
smokes (`@pytest.mark.window`) run only with a real window server and `NDVL_REAL_GL=1`; CI runs
both configurations and fails the real-GL run if a smoke skipped. `tests/README.md` describes
the suite and `docs/DESIGN.md` the viewer
design, including which ndv 0.5 APIs it relies on.
