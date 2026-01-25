"""
Simulate a live acquisition using the push-based zarr API.

This script tests the zarr v3 push-based API:
- Runs the viewer in-process
- Creates a zarr store and writes frames to it
- Calls start_zarr_acquisition() to configure the viewer
- Calls notify_zarr_frame() for each written frame
- Does not rely on filesystem polling

This tests the push-based zarr API as would be used by Squid with zarr v3 support.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import zarr
from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import QApplication

from ndviewer_light import LightweightViewer


_FONT_5X7: dict[str, list[str]] = {
    "0": ["#####", "#...#", "#...#", "#...#", "#...#", "#...#", "#####"],
    "1": ["..#..", ".##..", "..#..", "..#..", "..#..", "..#..", ".###."],
    "2": ["#####", "....#", "....#", "#####", "#....", "#....", "#####"],
    "3": ["#####", "....#", "....#", "#####", "....#", "....#", "#####"],
    "4": ["#...#", "#...#", "#...#", "#####", "....#", "....#", "....#"],
    "5": ["#####", "#....", "#....", "#####", "....#", "....#", "#####"],
    "6": ["#####", "#....", "#....", "#####", "#...#", "#...#", "#####"],
    "7": ["#####", "....#", "...#.", "..#..", ".#...", ".#...", ".#..."],
    "8": ["#####", "#...#", "#...#", "#####", "#...#", "#...#", "#####"],
    "9": ["#####", "#...#", "#...#", "#####", "....#", "....#", "#####"],
    "F": ["#####", "#....", "#....", "#####", "#....", "#....", "#...."],
    "O": ["#####", "#...#", "#...#", "#...#", "#...#", "#...#", "#####"],
    "V": ["#...#", "#...#", "#...#", "#...#", "#...#", ".#.#.", "..#.."],
    "T": ["#####", "..#..", "..#..", "..#..", "..#..", "..#..", "..#.."],
    "C": ["#####", "#....", "#....", "#....", "#....", "#....", "#####"],
    "H": ["#...#", "#...#", "#...#", "#####", "#...#", "#...#", "#...#"],
    "Z": ["#####", "....#", "...#.", "..#..", ".#...", "#....", "#####"],
    "=": [".....", "#####", ".....", "#####", ".....", ".....", "....."],
    " ": [".....", ".....", ".....", ".....", ".....", ".....", "....."],
    "-": [".....", ".....", ".....", "#####", ".....", ".....", "....."],
    "_": [".....", ".....", ".....", ".....", ".....", ".....", "#####"],
    ":": [".....", "..#..", ".....", ".....", "..#..", ".....", "....."],
}


def _draw_text(
    img: np.ndarray, text: str, x: int, y: int, scale: int, value: int
) -> None:
    """Draw text into a uint16 image in-place using the bitmap font."""
    h, w = img.shape
    cursor_x = x
    cursor_y = y
    char_w = 5 * scale
    spacing = 1 * scale

    for ch in text:
        glyph = _FONT_5X7.get(ch.upper())
        if glyph is None:
            glyph = _FONT_5X7[" "]

        if cursor_x >= w or cursor_y >= h:
            break

        for gy in range(7):
            row = glyph[gy]
            for gx in range(5):
                if row[gx] != "#":
                    continue
                px0 = cursor_x + gx * scale
                py0 = cursor_y + gy * scale
                px1 = min(w, px0 + scale)
                py1 = min(h, py0 + scale)
                if px0 < 0 or py0 < 0 or px0 >= w or py0 >= h:
                    continue
                img[py0:py1, px0:px1] = np.uint16(value)

        cursor_x += char_w + spacing


class ZarrAcquisitionSimulator:
    """Simulates acquisition by writing to zarr and calling notify_zarr_frame()."""

    def __init__(
        self,
        viewer: LightweightViewer,
        zarr_path: Path,
        n_fov: int,
        n_z: int,
        n_t: int,
        channels: list[str],
        channel_colors: list[str],
        height: int,
        width: int,
        interval_ms: int,
        pixel_size_um: float = 0.325,
        z_step_um: float = 1.5,
    ):
        self.viewer = viewer
        self.zarr_path = zarr_path
        self.n_fov = n_fov
        self.n_z = n_z
        self.n_t = n_t
        self.channels = channels
        self.channel_colors = channel_colors
        self.height = height
        self.width = width
        self.interval_ms = interval_ms
        self.pixel_size_um = pixel_size_um
        self.z_step_um = z_step_um

        # Current position in acquisition
        self.current_t = 0
        self.current_fov = 0

        # Precompute base image pattern
        y = np.arange(height, dtype=np.uint16)[:, None]
        x = np.arange(width, dtype=np.uint16)[None, :]
        self.base = y + x

        # Generate FOV labels (well:fov format)
        self.fov_labels = []
        wells = ["A1", "A2", "A3", "B1", "B2", "B3", "C1", "C2", "C3"]
        fov_per_well = (n_fov + len(wells) - 1) // len(wells)
        for i in range(n_fov):
            well_idx = i // fov_per_well
            fov_in_well = i % fov_per_well
            well = wells[well_idx % len(wells)]
            self.fov_labels.append(f"{well}:{fov_in_well}")

        # Timer for periodic writes
        self.timer = QTimer()
        self.timer.timeout.connect(self._write_next_fov)

        # Zarr store handle
        self.zarr_store = None
        self.zarr_array = None

    def _create_zarr_store(self):
        """Create the zarr store with OME-NGFF metadata."""
        # Create zarr store
        self.zarr_store = zarr.open(str(self.zarr_path), mode="w")

        # Create data array with shape (T, C, Z, Y, X) for single-FOV
        # For multi-FOV, we use (T, FOV, C, Z, Y, X) but simplify here
        n_c = len(self.channels)
        shape = (self.n_t, n_c, self.n_z, self.height, self.width)
        chunks = (1, 1, 1, self.height, self.width)

        self.zarr_array = self.zarr_store.create_dataset(
            "0",  # Level 0 (highest resolution)
            shape=shape,
            chunks=chunks,
            dtype=np.uint16,
            overwrite=True,
        )

        # Write OME-NGFF metadata to .zattrs
        zattrs = {
            "multiscales": [
                {
                    "version": "0.4",
                    "axes": [
                        {"name": "t", "type": "time"},
                        {"name": "c", "type": "channel"},
                        {"name": "z", "type": "space", "unit": "micrometer"},
                        {"name": "y", "type": "space", "unit": "micrometer"},
                        {"name": "x", "type": "space", "unit": "micrometer"},
                    ],
                    "datasets": [
                        {
                            "path": "0",
                            "coordinateTransformations": [
                                {
                                    "type": "scale",
                                    "scale": [
                                        1,
                                        1,
                                        self.z_step_um,
                                        self.pixel_size_um,
                                        self.pixel_size_um,
                                    ],
                                }
                            ],
                        }
                    ],
                }
            ],
            "omero": {
                "channels": [
                    {"label": name, "color": color}
                    for name, color in zip(self.channels, self.channel_colors)
                ]
            },
            "_squid_metadata": {
                "pixel_size_um": self.pixel_size_um,
                "z_step_um": self.z_step_um,
                "acquisition_complete": False,
            },
        }

        # Write .zattrs file
        zattrs_path = self.zarr_path / ".zattrs"
        with open(zattrs_path, "w") as f:
            json.dump(zattrs, f, indent=2)

        print(f"Created zarr store at {self.zarr_path}")
        print(f"  Shape: {shape}")
        print(f"  Chunks: {chunks}")

    def start(self):
        """Start the simulated acquisition."""
        print(f"Starting zarr push-based acquisition simulation")
        print(f"  Output: {self.zarr_path}")
        print(
            f"  FOVs: {self.n_fov}, Z: {self.n_z}, T: {self.n_t}, Channels: {self.channels}"
        )

        # Create zarr store
        self._create_zarr_store()

        # Configure viewer via push-based zarr API
        self.viewer.start_zarr_acquisition(
            zarr_path=str(self.zarr_path),
            channels=self.channels,
            num_z=self.n_z,
            fov_labels=self.fov_labels,
            height=self.height,
            width=self.width,
        )

        # Start writing
        self.timer.start(self.interval_ms)

    def _write_next_fov(self):
        """Write all z-planes and channels for current FOV, then advance."""
        if self.current_t >= self.n_t:
            self._finish()
            return

        t = self.current_t
        fov = self.current_fov
        fov_label = self.fov_labels[fov]

        for z in range(self.n_z):
            for c, ch_name in enumerate(self.channels):
                # Create image with identifying pattern
                offset = np.uint16(t * 97 + fov * 11 + c * 301 + z * 50)
                img = (self.base + offset).astype(np.uint16, copy=True)

                # Overlay text label
                label = f"T={t:02d} F={fov} Z={z:02d} C={c}"
                _draw_text(img, label, x=20, y=20, scale=10, value=60000)

                # Write to zarr array: shape is (T, C, Z, Y, X)
                self.zarr_array[t, c, z, :, :] = img

                # Notify viewer (push-based zarr API)
                self.viewer.notify_zarr_frame(
                    t=t,
                    fov_idx=fov,
                    z=z,
                    channel=ch_name,
                )

        print(
            f"[t={t}] Wrote FOV {fov} ({fov_label}): {self.n_z} z x {len(self.channels)} ch"
        )

        # Advance to next FOV
        self.current_fov += 1
        if self.current_fov >= self.n_fov:
            self.current_fov = 0
            self.current_t += 1

    def _finish(self):
        """Called when acquisition is complete."""
        self.timer.stop()

        # Update metadata to mark acquisition as complete
        zattrs_path = self.zarr_path / ".zattrs"
        if zattrs_path.exists():
            with open(zattrs_path, "r") as f:
                zattrs = json.load(f)
            zattrs["_squid_metadata"]["acquisition_complete"] = True
            with open(zattrs_path, "w") as f:
                json.dump(zattrs, f, indent=2)

        self.viewer.end_zarr_acquisition()
        print("Acquisition complete. Browse the dataset in the viewer.")


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Simulate acquisition using push-based zarr API"
    )
    ap.add_argument(
        "dataset_root",
        nargs="?",
        default=None,
        help="Output zarr store path (default: ~/Downloads/ndv_zarr_test_<timestamp>.zarr).",
    )
    ap.add_argument(
        "--interval",
        type=float,
        default=0.1,
        help="Seconds between FOV writes (default: 0.1).",
    )
    ap.add_argument("--n-fov", type=int, default=1, help="Number of FOVs (default: 1)")
    ap.add_argument(
        "--n-ch", type=int, default=3, help="Number of channels (default: 3)"
    )
    ap.add_argument(
        "--n-t", type=int, default=5, help="Number of timepoints (default: 5)"
    )
    ap.add_argument(
        "--n-z", type=int, default=5, help="Number of z-levels (default: 5)"
    )
    ap.add_argument("--height", type=int, default=1000)
    ap.add_argument("--width", type=int, default=1000)
    ap.add_argument(
        "--channels",
        nargs="*",
        default=["DAPI", "GFP", "RFP"],
        help="Channel name strings (default: DAPI GFP RFP).",
    )
    ap.add_argument(
        "--colors",
        nargs="*",
        default=["0000FF", "00FF00", "FF0000"],
        help="Channel colors as hex RGB (default: 0000FF 00FF00 FF0000).",
    )
    ap.add_argument(
        "--pixel-size",
        type=float,
        default=0.325,
        help="Pixel size in micrometers (default: 0.325).",
    )
    ap.add_argument(
        "--z-step",
        type=float,
        default=1.5,
        help="Z step in micrometers (default: 1.5).",
    )
    args = ap.parse_args()

    if len(args.channels) != args.n_ch:
        print(
            f"Error: --channels length ({len(args.channels)}) must match --n-ch ({args.n_ch}).",
            file=sys.stderr,
        )
        return 2

    if len(args.colors) != args.n_ch:
        print(
            f"Error: --colors length ({len(args.colors)}) must match --n-ch ({args.n_ch}).",
            file=sys.stderr,
        )
        return 2

    if args.dataset_root is None:
        ts = time.strftime("%Y%m%d_%H%M%S")
        zarr_path = (Path.home() / "Downloads" / f"ndv_zarr_test_{ts}.zarr").resolve()
    else:
        zarr_path = Path(args.dataset_root).expanduser().resolve()

    # Create Qt application and viewer
    app = QApplication(sys.argv)
    viewer = LightweightViewer()
    viewer.setWindowTitle("NDViewer Light - Zarr Push-Based Simulation")
    viewer.resize(1200, 800)
    viewer.show()

    # Create and start simulator
    simulator = ZarrAcquisitionSimulator(
        viewer=viewer,
        zarr_path=zarr_path,
        n_fov=args.n_fov,
        n_z=args.n_z,
        n_t=args.n_t,
        channels=args.channels,
        channel_colors=args.colors,
        height=args.height,
        width=args.width,
        interval_ms=int(args.interval * 1000),
        pixel_size_um=args.pixel_size,
        z_step_um=args.z_step,
    )

    # Start acquisition after event loop starts
    QTimer.singleShot(100, simulator.start)

    return app.exec_()


if __name__ == "__main__":
    raise SystemExit(main())
