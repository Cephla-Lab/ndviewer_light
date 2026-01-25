"""
Simulate a live acquisition using the push-based zarr API.

This script tests the zarr v3 push-based API with different zarr structures:
- single: Simple 5D (T, C, Z, Y, X) - single region
- 6d: 6D with FOV dimension (T, FOV, C, Z, Y, X)
- per_fov: Separate zarr per FOV: zarr/region/fov_N.zarr
- hcs: HCS plate format: plate.zarr/row/col/field/acquisition.zarr

Usage:
    python simulate_zarr_acquisition.py --structure single
    python simulate_zarr_acquisition.py --structure 6d --n-fov 4
    python simulate_zarr_acquisition.py --structure per_fov --n-fov 4
    python simulate_zarr_acquisition.py --structure hcs --wells A1 A2 B1 B2 --fov-per-well 2
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Optional

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


def _write_zattrs(
    zarr_path: Path,
    channels: list[str],
    channel_colors: list[str],
    pixel_size_um: float,
    z_step_um: float,
    acquisition_complete: bool = False,
    axes: Optional[list[dict]] = None,
    scale: Optional[list[float]] = None,
) -> None:
    """Write OME-NGFF .zattrs metadata file."""
    if axes is None:
        axes = [
            {"name": "t", "type": "time"},
            {"name": "c", "type": "channel"},
            {"name": "z", "type": "space", "unit": "micrometer"},
            {"name": "y", "type": "space", "unit": "micrometer"},
            {"name": "x", "type": "space", "unit": "micrometer"},
        ]
    if scale is None:
        scale = [1, 1, z_step_um, pixel_size_um, pixel_size_um]

    zattrs = {
        "multiscales": [
            {
                "version": "0.4",
                "axes": axes,
                "datasets": [
                    {
                        "path": "0",
                        "coordinateTransformations": [
                            {"type": "scale", "scale": scale}
                        ],
                    }
                ],
            }
        ],
        "omero": {
            "channels": [
                {"label": name, "color": color}
                for name, color in zip(channels, channel_colors)
            ]
        },
        "_squid_metadata": {
            "pixel_size_um": pixel_size_um,
            "z_step_um": z_step_um,
            "acquisition_complete": acquisition_complete,
        },
    }

    zattrs_path = zarr_path / ".zattrs"
    with open(zattrs_path, "w") as f:
        json.dump(zattrs, f, indent=2)


class ZarrAcquisitionSimulator:
    """Simulates acquisition by writing to zarr and calling notify_zarr_frame()."""

    def __init__(
        self,
        viewer: LightweightViewer,
        output_path: Path,
        structure: str,
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
        wells: Optional[list[str]] = None,
        fov_per_well: int = 1,
    ):
        self.viewer = viewer
        self.output_path = output_path
        self.structure = structure
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
        self.wells = wells or ["A1"]
        self.fov_per_well = fov_per_well

        # Current position in acquisition
        self.current_t = 0
        self.current_fov = 0

        # Precompute base image pattern
        y = np.arange(height, dtype=np.uint16)[:, None]
        x = np.arange(width, dtype=np.uint16)[None, :]
        self.base = y + x

        # Generate FOV labels based on structure
        self.fov_labels = []
        self.fov_paths = []  # For per_fov and hcs structures
        self._setup_fov_structure()

        # Timer for periodic writes
        self.timer = QTimer()
        self.timer.timeout.connect(self._write_next_fov)

        # Zarr store handles
        self.zarr_stores = {}  # fov_idx -> zarr store
        self.zarr_arrays = {}  # fov_idx -> zarr array

    def _setup_fov_structure(self):
        """Set up FOV labels and paths based on structure type."""
        if self.structure == "hcs":
            # HCS plate: plate.zarr/row/col/field/acquisition.zarr
            for well in self.wells:
                row = well[0]  # e.g., "A"
                col = well[1:]  # e.g., "1"
                for field in range(self.fov_per_well):
                    self.fov_labels.append(f"{well}:{field}")
                    acq_path = (
                        self.output_path
                        / "plate.zarr"
                        / row
                        / col
                        / str(field)
                        / "acquisition.zarr"
                    )
                    self.fov_paths.append(acq_path)
            self.n_fov = len(self.fov_labels)

        elif self.structure == "per_fov":
            # Per-FOV: zarr/region_1/fov_N.zarr
            for i in range(self.n_fov):
                well_idx = i // max(1, self.n_fov // len(self.wells))
                well = self.wells[well_idx % len(self.wells)]
                fov_in_well = i % max(1, self.n_fov // len(self.wells))
                self.fov_labels.append(f"{well}:{fov_in_well}")
                fov_path = self.output_path / "zarr" / "region_1" / f"fov_{i}.zarr"
                self.fov_paths.append(fov_path)

        elif self.structure == "6d":
            # 6D single store with FOV dimension
            for i in range(self.n_fov):
                well_idx = i // max(1, self.n_fov // len(self.wells))
                well = self.wells[well_idx % len(self.wells)]
                fov_in_well = i % max(1, self.n_fov // len(self.wells))
                self.fov_labels.append(f"{well}:{fov_in_well}")
            # Single path for all FOVs
            self.fov_paths = [
                self.output_path / "zarr" / "region_1" / "acquisition.zarr"
            ]

        else:  # single
            # Simple 5D single store
            self.fov_labels = ["A1:0"]
            self.n_fov = 1
            self.fov_paths = [self.output_path]
            if not str(self.output_path).endswith(".zarr"):
                self.fov_paths = [self.output_path.with_suffix(".zarr")]

    def _create_zarr_stores(self):
        """Create zarr stores based on structure type."""
        n_c = len(self.channels)

        if self.structure == "single":
            # Single 5D store: (T, C, Z, Y, X)
            zarr_path = self.fov_paths[0]
            zarr_path.mkdir(parents=True, exist_ok=True)
            store = zarr.open(str(zarr_path), mode="w")
            shape = (self.n_t, n_c, self.n_z, self.height, self.width)
            chunks = (1, 1, 1, self.height, self.width)
            arr = store.create_dataset(
                "0", shape=shape, chunks=chunks, dtype=np.uint16, overwrite=True
            )
            self.zarr_stores[0] = store
            self.zarr_arrays[0] = arr
            _write_zattrs(
                zarr_path,
                self.channels,
                self.channel_colors,
                self.pixel_size_um,
                self.z_step_um,
            )
            print(f"Created single 5D zarr at {zarr_path}")
            print(f"  Shape: {shape}")

        elif self.structure == "6d":
            # Single 6D store: (T, FOV, C, Z, Y, X)
            zarr_path = self.fov_paths[0]
            zarr_path.mkdir(parents=True, exist_ok=True)
            store = zarr.open(str(zarr_path), mode="w")
            shape = (self.n_t, self.n_fov, n_c, self.n_z, self.height, self.width)
            chunks = (1, 1, 1, 1, self.height, self.width)
            arr = store.create_dataset(
                "0", shape=shape, chunks=chunks, dtype=np.uint16, overwrite=True
            )
            self.zarr_stores[0] = store
            self.zarr_arrays[0] = arr
            _write_zattrs(
                zarr_path,
                self.channels,
                self.channel_colors,
                self.pixel_size_um,
                self.z_step_um,
                axes=[
                    {"name": "t", "type": "time"},
                    {"name": "fov", "type": "position"},
                    {"name": "c", "type": "channel"},
                    {"name": "z", "type": "space", "unit": "micrometer"},
                    {"name": "y", "type": "space", "unit": "micrometer"},
                    {"name": "x", "type": "space", "unit": "micrometer"},
                ],
                scale=[1, 1, 1, self.z_step_um, self.pixel_size_um, self.pixel_size_um],
            )
            print(f"Created 6D zarr at {zarr_path}")
            print(f"  Shape: {shape}")

        elif self.structure in ("per_fov", "hcs"):
            # Separate store per FOV: (T, C, Z, Y, X) each
            for fov_idx, zarr_path in enumerate(self.fov_paths):
                zarr_path.mkdir(parents=True, exist_ok=True)
                store = zarr.open(str(zarr_path), mode="w")
                shape = (self.n_t, n_c, self.n_z, self.height, self.width)
                chunks = (1, 1, 1, self.height, self.width)
                arr = store.create_dataset(
                    "0", shape=shape, chunks=chunks, dtype=np.uint16, overwrite=True
                )
                self.zarr_stores[fov_idx] = store
                self.zarr_arrays[fov_idx] = arr
                _write_zattrs(
                    zarr_path,
                    self.channels,
                    self.channel_colors,
                    self.pixel_size_um,
                    self.z_step_um,
                )
            struct_name = "HCS plate" if self.structure == "hcs" else "per-FOV"
            print(f"Created {struct_name} zarr stores: {len(self.fov_paths)} FOVs")
            for i, p in enumerate(self.fov_paths[:3]):
                print(f"  [{i}] {p}")
            if len(self.fov_paths) > 3:
                print(f"  ... and {len(self.fov_paths) - 3} more")

    def start(self):
        """Start the simulated acquisition."""
        print(f"Starting zarr push-based acquisition simulation")
        print(f"  Structure: {self.structure}")
        print(f"  Output: {self.output_path}")
        print(
            f"  FOVs: {self.n_fov}, Z: {self.n_z}, T: {self.n_t}, Channels: {self.channels}"
        )

        # Create zarr stores
        self._create_zarr_stores()

        # Determine zarr path for viewer
        if self.structure == "single":
            viewer_zarr_path = str(self.fov_paths[0])
        elif self.structure == "6d":
            viewer_zarr_path = str(self.fov_paths[0])
        else:
            # For per_fov and hcs, use the first FOV path for now
            # The viewer will discover all FOVs from the structure
            viewer_zarr_path = str(self.fov_paths[0])

        # Configure viewer via push-based zarr API
        self.viewer.start_zarr_acquisition(
            zarr_path=viewer_zarr_path,
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

                # Write to appropriate zarr array based on structure
                if self.structure == "single":
                    # (T, C, Z, Y, X)
                    self.zarr_arrays[0][t, c, z, :, :] = img
                elif self.structure == "6d":
                    # (T, FOV, C, Z, Y, X)
                    self.zarr_arrays[0][t, fov, c, z, :, :] = img
                else:
                    # per_fov or hcs: each FOV has its own array (T, C, Z, Y, X)
                    self.zarr_arrays[fov][t, c, z, :, :] = img

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
        if self.structure in ("single", "6d"):
            zattrs_path = self.fov_paths[0] / ".zattrs"
            if zattrs_path.exists():
                with open(zattrs_path, "r") as f:
                    zattrs = json.load(f)
                zattrs["_squid_metadata"]["acquisition_complete"] = True
                with open(zattrs_path, "w") as f:
                    json.dump(zattrs, f, indent=2)
        else:
            # Update all FOV stores
            for zarr_path in self.fov_paths:
                zattrs_path = zarr_path / ".zattrs"
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
        description="Simulate acquisition using push-based zarr API",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single 5D zarr (simplest)
  python simulate_zarr_acquisition.py --structure single

  # 6D zarr with FOV dimension
  python simulate_zarr_acquisition.py --structure 6d --n-fov 4

  # Per-FOV zarr stores
  python simulate_zarr_acquisition.py --structure per_fov --n-fov 4

  # HCS plate format
  python simulate_zarr_acquisition.py --structure hcs --wells A1 A2 B1 B2 --fov-per-well 2
""",
    )
    ap.add_argument(
        "output_path",
        nargs="?",
        default=None,
        help="Output path (default: ~/Downloads/ndv_zarr_test_<timestamp>).",
    )
    ap.add_argument(
        "--structure",
        choices=["single", "6d", "per_fov", "hcs"],
        default="single",
        help="Zarr structure type (default: single).",
    )
    ap.add_argument(
        "--interval",
        type=float,
        default=0.1,
        help="Seconds between FOV writes (default: 0.1).",
    )
    ap.add_argument(
        "--n-fov",
        type=int,
        default=1,
        help="Number of FOVs (default: 1, ignored for hcs).",
    )
    ap.add_argument(
        "--n-ch", type=int, default=3, help="Number of channels (default: 3)."
    )
    ap.add_argument(
        "--n-t", type=int, default=5, help="Number of timepoints (default: 5)."
    )
    ap.add_argument(
        "--n-z", type=int, default=5, help="Number of z-levels (default: 5)."
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
    # HCS-specific options
    ap.add_argument(
        "--wells",
        nargs="*",
        default=["A1", "A2", "B1", "B2"],
        help="Well IDs for HCS structure (default: A1 A2 B1 B2).",
    )
    ap.add_argument(
        "--fov-per-well",
        type=int,
        default=2,
        help="FOVs per well for HCS structure (default: 2).",
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

    if args.output_path is None:
        ts = time.strftime("%Y%m%d_%H%M%S")
        output_path = (
            Path.home() / "Downloads" / f"ndv_zarr_test_{args.structure}_{ts}"
        ).resolve()
    else:
        output_path = Path(args.output_path).expanduser().resolve()

    # Create Qt application and viewer
    app = QApplication(sys.argv)
    viewer = LightweightViewer()
    viewer.setWindowTitle(f"NDViewer Light - Zarr Simulation ({args.structure})")
    viewer.resize(1200, 800)
    viewer.show()

    # Create and start simulator
    simulator = ZarrAcquisitionSimulator(
        viewer=viewer,
        output_path=output_path,
        structure=args.structure,
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
        wells=args.wells,
        fov_per_well=args.fov_per_well,
    )

    # Start acquisition after event loop starts
    QTimer.singleShot(100, simulator.start)

    return app.exec_()


if __name__ == "__main__":
    raise SystemExit(main())
