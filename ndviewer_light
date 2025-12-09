"""
Lightweight NDViewer - Fast, minimal viewer for multi-dimensional microscopy data.

Supports: OME-TIFF and single-TIFF acquisitions with lazy loading via dask.
"""

import re
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

# Core dependencies
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QLabel, QMainWindow, 
                             QPushButton, QFileDialog, QApplication)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPalette, QColor

# NDV viewer
try:
    import ndv
    NDV_AVAILABLE = True
except ImportError:
    NDV_AVAILABLE = False

# Lazy loading stack
try:
    import tifffile as tf
    import xarray as xr
    import dask.array as da
    from dask import delayed
    DASK_AVAILABLE = True
except ImportError:
    DASK_AVAILABLE = False

# Filename patterns (from common.py)
FPATTERN = re.compile(r"(?P<r>[^_]+)_(?P<f>\d+)_(?P<z>\d+)_(?P<c>.+)\.tiff?", re.IGNORECASE)
FPATTERN_OME = re.compile(r"(?P<r>[^_]+)_(?P<f>\d+)\.ome\.tiff?", re.IGNORECASE)


def extract_wavelength(channel_str: str) -> int:
    """Extract wavelength (nm) from channel string."""
    # Direct wavelength pattern
    if m := re.search(r'(\d{3,4})\s*nm', channel_str, re.IGNORECASE):
        return int(m.group(1))
    
    # Common fluorophores
    fluor_map = {
        'dapi': 405, 'hoechst': 405,
        'gfp': 488, 'fitc': 488, 'alexa488': 488,
        'tritc': 561, 'cy3': 561, 'mcherry': 561,
        'cy5': 640, 'alexa647': 640, 'cy7': 730
    }
    channel_lower = channel_str.lower()
    for fluor, wl in fluor_map.items():
        if fluor in channel_lower:
            return wl
    
    # Fallback
    if m := re.search(r'\d+', channel_str):
        return int(m.group(0))
    return 0


def detect_format(base_path: Path) -> str:
    """Detect OME-TIFF vs single-TIFF format."""
    ome_dir = base_path / "ome_tiff"
    if ome_dir.exists():
        if any('.ome' in f.name for f in ome_dir.glob('*.tif*')):
            return 'ome_tiff'
    
    first_tp = next((d for d in base_path.iterdir() if d.is_dir() and d.name.isdigit()), None)
    if first_tp:
        if any('.ome' in f.name for f in first_tp.glob('*.tif*')):
            return 'ome_tiff'
    return 'single_tiff'


def wavelength_to_colormap(wavelength: int, index: int = 0) -> str:
    """Map wavelength to NDV colormap."""
    if wavelength <= 420:
        return 'blue'
    elif 470 <= wavelength <= 510:
        return 'green'
    elif 540 <= wavelength <= 590:
        return 'yellow'
    elif 620 <= wavelength <= 660:
        return 'red'
    elif wavelength >= 700:
        return 'darkred'
    return ['blue', 'green', 'yellow', 'red', 'darkred'][index % 5]


class LauncherWindow(QMainWindow):
    """Separate launcher window with dropbox for dataset selection."""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("NDViewer Lightweight - Open Dataset")
        self.setGeometry(100, 100, 500, 300)
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
        layout.addWidget(self.status_label)
        
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
        
        self.viewer_window = LightweightMainWindow(path)
        self.viewer_window.show()
        self.close()
    
    def _set_dark_theme(self):
        from PyQt5.QtWidgets import QStyleFactory
        self.setStyle(QStyleFactory.create("Fusion"))
        
        p = self.palette()
        p.setColor(QPalette.Window, QColor(53, 53, 53))
        p.setColor(QPalette.WindowText, QColor(255, 255, 255))
        p.setColor(QPalette.Base, QColor(35, 35, 35))
        p.setColor(QPalette.AlternateBase, QColor(53, 53, 53))
        p.setColor(QPalette.Text, QColor(255, 255, 255))
        p.setColor(QPalette.Button, QColor(53, 53, 53))
        p.setColor(QPalette.ButtonText, QColor(255, 255, 255))
        p.setColor(QPalette.Highlight, QColor(42, 130, 218))
        p.setColor(QPalette.HighlightedText, QColor(35, 35, 35))
        self.setPalette(p)


class LightweightViewer(QWidget):
    """Minimal NDV-based viewer."""
    
    def __init__(self, dataset_path: str):
        super().__init__()
        self.dataset_path = dataset_path
        self.ndv_viewer = None
        self._setup_ui()
        self.load_dataset(dataset_path)
    
    def _setup_ui(self):
        layout = QVBoxLayout()
        
        # Status
        self.status_label = QLabel("Loading dataset...")
        self.status_label.setStyleSheet("color: #888; padding: 5px;")
        layout.addWidget(self.status_label)
        
        # NDV placeholder
        if NDV_AVAILABLE:
            dummy = np.zeros((1, 100, 100), dtype=np.uint16)
            self.ndv_viewer = ndv.ArrayViewer(dummy, channel_axis=0, 
                                              channel_mode="composite", 
                                              visible_axes=(-2, -1))
            layout.addWidget(self.ndv_viewer.widget(), 1)
        else:
            placeholder = QLabel("NDV not available.\npip install ndv[vispy,pyqt]")
            placeholder.setAlignment(Qt.AlignCenter)
            layout.addWidget(placeholder, 1)
        
        self.setLayout(layout)
    
    def load_dataset(self, path: str):
        """Load dataset and display in NDV."""
        self.dataset_path = path
        self.status_label.setText(f"Loading: {Path(path).name}...")
        QApplication.processEvents()
        
        try:
            data = self._create_lazy_array(Path(path))
            if data is not None:
                self._set_ndv_data(data)
                
                # Update status with dimensions
                dims_str = " × ".join(f"{d}={s}" for d, s in zip(data.dims, data.shape))
                self.status_label.setText(f"Loaded: {dims_str}")
            else:
                self.status_label.setText("Failed to load dataset")
        except Exception as e:
            self.status_label.setText(f"Error: {e}")
            import traceback
            traceback.print_exc()
    
    def _create_lazy_array(self, base_path: Path) -> Optional[xr.DataArray]:
        """Create lazy xarray from dataset - auto-detects format."""
        if not DASK_AVAILABLE:
            return None
        
        fmt = detect_format(base_path)
        fovs = self._discover_fovs(base_path, fmt)
        
        if not fovs:
            print("No FOVs found")
            return None
        
        print(f"Format: {fmt}, FOVs: {len(fovs)}")
        
        if fmt == 'ome_tiff':
            return self._load_ome_tiff(base_path, fovs)
        else:
            return self._load_single_tiff(base_path, fovs)
    
    def _discover_fovs(self, base_path: Path, fmt: str) -> List[Dict]:
        """Discover all FOVs (region, fov) pairs."""
        fov_set = set()
        
        if fmt == 'ome_tiff':
            ome_dir = base_path / "ome_tiff"
            if not ome_dir.exists():
                ome_dir = next((d for d in base_path.iterdir() 
                               if d.is_dir() and d.name.isdigit()), base_path)
            for f in ome_dir.glob("*.ome.tif*"):
                if m := FPATTERN_OME.search(f.name):
                    fov_set.add((m.group("r"), int(m.group("f"))))
        else:
            first_tp = next((d for d in base_path.iterdir() 
                            if d.is_dir() and d.name.isdigit()), None)
            if first_tp:
                for f in first_tp.glob("*.tiff"):
                    if m := FPATTERN.search(f.name):
                        fov_set.add((m.group("r"), int(m.group("f"))))
        
        return [{'region': r, 'fov': f} for r, f in sorted(fov_set)]
    
    def _load_ome_tiff(self, base_path: Path, fovs: List[Dict]) -> Optional[xr.DataArray]:
        """Load OME-TIFF with FOV-level chunking for minimal graph overhead."""
        try:
            # Build file index
            ome_dir = base_path / "ome_tiff"
            if not ome_dir.exists():
                ome_dir = next((d for d in base_path.iterdir() 
                               if d.is_dir() and d.name.isdigit()), base_path)
            
            file_index = {}
            for f in ome_dir.glob("*.ome.tif*"):
                if m := FPATTERN_OME.search(f.name):
                    file_index[(m.group("r"), int(m.group("f")))] = str(f)
            
            if not file_index:
                return None
            
            # Extract metadata from first file (ONE file open)
            first_file = next(iter(file_index.values()))
            with tf.TiffFile(first_file) as tif:
                series = tif.series[0]
                axes = series.axes
                shape = series.shape
                shape_dict = dict(zip(axes, shape))
                
                n_t = shape_dict.get('T', 1)
                n_c = shape_dict.get('C', 1)
                n_z = shape_dict.get('Z', 1)
                height = shape_dict.get('Y', shape[-2])
                width = shape_dict.get('X', shape[-1])
                
                # Extract channel names from OME-XML
                channel_names = []
                try:
                    if tif.ome_metadata:
                        import xml.etree.ElementTree as ET
                        root = ET.fromstring(tif.ome_metadata)
                        ns = {'ome': 'http://www.openmicroscopy.org/Schemas/OME/2016-06'}
                        for ch in root.findall('.//ome:Channel', ns):
                            name = ch.get('Name') or ch.get('ID', '')
                            if name:
                                channel_names.append(name)
                except:
                    pass
            
            # Build LUTs
            luts = {}
            for i in range(n_c):
                name = channel_names[i] if i < len(channel_names) else f'Ch{i}'
                luts[i] = wavelength_to_colormap(extract_wavelength(name), i)
            
            n_fov = len(fovs)
            
            # Memory protection for huge datasets
            if n_t * n_fov > 10000:
                print(f"WARNING: {n_t * n_fov} chunks - limiting to T=1")
                n_t = 1
            
            # FOV-level chunking: load entire (Z, C, Y, X) volume per chunk
            def load_fov_volume(fov_idx, t_idx):
                def _load():
                    try:
                        region, fov = fovs[fov_idx]['region'], fovs[fov_idx]['fov']
                        filepath = file_index.get((region, fov))
                        if not filepath:
                            return np.zeros((n_z, n_c, height, width), dtype=np.uint16)
                        
                        with tf.TiffFile(filepath) as tif:
                            series = tif.series[0]
                            volume = np.zeros((n_z, n_c, height, width), dtype=np.uint16)
                            for z in range(n_z):
                                for c in range(n_c):
                                    page_idx = t_idx * (n_z * n_c) + z * n_c + c
                                    volume[z, c] = series.pages[page_idx].asarray()
                        return volume
                    except Exception as e:
                        print(f"Error loading FOV {fov_idx}, T={t_idx}: {e}")
                        return np.zeros((n_z, n_c, height, width), dtype=np.uint16)
                return _load
            
            # Build minimal dask graph (T × FOV chunks only)
            time_arrays = []
            for t in range(n_t):
                fov_arrays = [
                    da.from_delayed(
                        delayed(load_fov_volume(f, t))(),
                        shape=(n_z, n_c, height, width),
                        dtype=np.uint16
                    ) for f in range(n_fov)
                ]
                time_arrays.append(da.stack(fov_arrays, axis=0))
            
            full_array = da.stack(time_arrays, axis=0)  # (T, FOV, Z, C, Y, X)
            
            xarr = xr.DataArray(
                full_array,
                dims=['time', 'fov', 'z_level', 'channel', 'y', 'x'],
                coords={
                    'time': list(range(n_t)),
                    'fov': list(range(n_fov)),
                    'z_level': list(range(n_z)),
                    'channel': list(range(n_c))
                }
            )
            xarr.attrs['luts'] = luts
            return xarr
            
        except Exception as e:
            print(f"OME-TIFF load error: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _load_single_tiff(self, base_path: Path, fovs: List[Dict]) -> Optional[xr.DataArray]:
        """Load single-TIFF with FOV-level chunking."""
        try:
            # Single scan to build complete file index
            file_index = {}  # (t, region, fov, z, channel) -> filepath
            t_set, z_set, c_set = set(), set(), set()
            
            for tp_dir in sorted(base_path.iterdir()):
                if not (tp_dir.is_dir() and tp_dir.name.isdigit()):
                    continue
                t = int(tp_dir.name)
                t_set.add(t)
                
                for f in tp_dir.glob("*.tiff"):
                    if m := FPATTERN.search(f.name):
                        region, fov = m.group("r"), int(m.group("f"))
                        z, channel = int(m.group("z")), m.group("c")
                        z_set.add(z)
                        c_set.add(channel)
                        file_index[(t, region, fov, z, channel)] = str(f)
            
            times = sorted(t_set)
            z_levels = sorted(z_set)
            channels = sorted(c_set)
            
            n_t, n_fov, n_z, n_c = len(times), len(fovs), len(z_levels), len(channels)
            
            # Get dimensions from sample file
            sample = next(iter(file_index.values()))
            with tf.TiffFile(sample) as tif:
                height, width = tif.pages[0].shape[-2:]
            
            # Build LUTs
            luts = {i: wavelength_to_colormap(extract_wavelength(c), i) 
                    for i, c in enumerate(channels)}
            
            # FOV-level chunking
            def load_fov_volume(fov_idx, t_idx):
                def _load():
                    try:
                        t = times[t_idx]
                        region, fov = fovs[fov_idx]['region'], fovs[fov_idx]['fov']
                        volume = np.zeros((n_z, n_c, height, width), dtype=np.uint16)
                        
                        for zi, z in enumerate(z_levels):
                            for ci, c in enumerate(channels):
                                filepath = file_index.get((t, region, fov, z, c))
                                if filepath:
                                    with tf.TiffFile(filepath) as tif:
                                        volume[zi, ci] = tif.pages[0].asarray()
                        return volume
                    except:
                        return np.zeros((n_z, n_c, height, width), dtype=np.uint16)
                return _load
            
            # Build dask graph
            time_arrays = []
            for t_idx in range(n_t):
                fov_arrays = [
                    da.from_delayed(
                        delayed(load_fov_volume(f, t_idx))(),
                        shape=(n_z, n_c, height, width),
                        dtype=np.uint16
                    ) for f in range(n_fov)
                ]
                time_arrays.append(da.stack(fov_arrays, axis=0))
            
            full_array = da.stack(time_arrays, axis=0)
            
            xarr = xr.DataArray(
                full_array,
                dims=['time', 'fov', 'z_level', 'channel', 'y', 'x'],
                coords={
                    'time': times,
                    'fov': list(range(n_fov)),
                    'z_level': z_levels,
                    'channel': channels
                }
            )
            xarr.attrs['luts'] = luts
            return xarr
            
        except Exception as e:
            print(f"Single-TIFF load error: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _set_ndv_data(self, data: xr.DataArray):
        """Update NDV viewer with lazy array."""
        if not NDV_AVAILABLE or not self.ndv_viewer:
            return
        
        luts = data.attrs.get('luts', {})
        channel_axis = data.dims.index('channel') if 'channel' in data.dims else None
        
        # Recreate viewer with proper dimensions
        old_widget = self.ndv_viewer.widget()
        layout = self.layout()
        
        self.ndv_viewer = ndv.ArrayViewer(
            data,
            channel_axis=channel_axis,
            channel_mode="composite",
            luts=luts,
            visible_axes=('y', 'x')  # 2D display, sliders for rest
        )
        
        # Replace widget
        idx = layout.indexOf(old_widget)
        layout.removeWidget(old_widget)
        old_widget.deleteLater()
        layout.insertWidget(idx, self.ndv_viewer.widget(), 1)


class LightweightMainWindow(QMainWindow):
    """Main window with dark theme."""
    
    def __init__(self, dataset_path: str):
        super().__init__()
        self.setWindowTitle(f"NDViewer Lightweight - {Path(dataset_path).name}")
        self.setGeometry(100, 100, 800, 700)
        self._set_dark_theme()
        
        self.viewer = LightweightViewer(dataset_path)
        self.setCentralWidget(self.viewer)
    
    def _set_dark_theme(self):
        from PyQt5.QtWidgets import QStyleFactory
        self.setStyle(QStyleFactory.create("Fusion"))
        
        p = self.palette()
        p.setColor(QPalette.Window, QColor(53, 53, 53))
        p.setColor(QPalette.WindowText, QColor(255, 255, 255))
        p.setColor(QPalette.Base, QColor(35, 35, 35))
        p.setColor(QPalette.AlternateBase, QColor(53, 53, 53))
        p.setColor(QPalette.Text, QColor(255, 255, 255))
        p.setColor(QPalette.Button, QColor(53, 53, 53))
        p.setColor(QPalette.ButtonText, QColor(255, 255, 255))
        p.setColor(QPalette.Highlight, QColor(42, 130, 218))
        p.setColor(QPalette.HighlightedText, QColor(35, 35, 35))
        self.setPalette(p)


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

