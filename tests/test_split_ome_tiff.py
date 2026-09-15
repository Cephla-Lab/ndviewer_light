"""Tests for OME-TIFF acquisitions split into one file per timepoint.

Squid can write OME-TIFF either as

* unsplit - ``<base>/ome_tiff/<region>_<fov>.ome.tiff`` with SizeT equal to the
  number of timepoints, or
* split - ``<base>/<t>/ome_tiff/<region>_<fov>.ome.tiff`` with SizeT = 1, so
  that finished timepoints can be offloaded while the run continues.

These tests build tiny 8x8 fixtures of both layouts and check that the viewer
browses a split run as one dataset with T = number of timepoint folders, while
unsplit runs behave exactly as before.
"""

import shutil
import sys
from unittest.mock import MagicMock

import numpy as np
import pytest
import tifffile

from ndviewer_light import LightweightViewer, detect_format, resolve_ome_tiff_dirs

SIZE = 8
CHANNEL_NAMES = ["BF LED matrix full", "Fluorescence 488 nm Ex"]
N_Z = 3
N_C = len(CHANNEL_NAMES)


def plane_value(t: int, fov: int, z: int, c: int) -> int:
    """A value unique to each (t, fov, z, c) plane."""
    return 1000 * (t + 1) + 100 * (fov + 1) + 10 * z + c


def make_stack(t: int, fov: int, n_t: int = 1) -> np.ndarray:
    """(T, Z, C, Y, X) stack whose planes carry `plane_value`."""
    data = np.zeros((n_t, N_Z, N_C, SIZE, SIZE), dtype=np.uint16)
    for ti in range(n_t):
        for z in range(N_Z):
            for c in range(N_C):
                data[ti, z, c] = plane_value(t + ti, fov, z, c)
    return data


def write_ome(path, data: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(
        path,
        data,
        ome=True,
        metadata={"axes": "TZCYX", "Channel": {"Name": CHANNEL_NAMES}},
    )


def make_split_dataset(base, n_t=3, fovs=(("A1", 0), ("A1", 1)), padding=0):
    """Write <base>/<t>/ome_tiff/<region>_<fov>.ome.tiff, one file per timepoint."""
    base.mkdir(parents=True, exist_ok=True)
    for t in range(n_t):
        tp_dir = base / f"{t:0{padding}d}"
        for region, fov in fovs:
            write_ome(
                tp_dir / "ome_tiff" / f"{region}_{fov}.ome.tiff",
                make_stack(t, fov),
            )
    return base


def make_unsplit_dataset(base, n_t=3, fovs=(("A1", 0), ("A1", 1))):
    """Write <base>/ome_tiff/<region>_<fov>.ome.tiff holding every timepoint."""
    base.mkdir(parents=True, exist_ok=True)
    for region, fov in fovs:
        write_ome(
            base / "ome_tiff" / f"{region}_{fov}.ome.tiff",
            make_stack(0, fov, n_t=n_t),
        )
    return base


def make_viewer(dataset_path=None):
    """Mock viewer carrying the real loading methods (no Qt construction)."""
    mock = MagicMock(spec=LightweightViewer)
    mock.dataset_path = str(dataset_path) if dataset_path is not None else ""
    for name in (
        "_discover_fovs",
        "_load_ome_tiff",
        "_load_ome_tiff_split",
        "_open_split_ome_stack",
        "_dataset_signature",
    ):
        setattr(mock, name, getattr(LightweightViewer, name).__get__(mock))
    return mock


def load(base):
    viewer = make_viewer()
    fovs = viewer._discover_fovs(base, "ome_tiff")
    return viewer._load_ome_tiff(base, fovs), fovs


class TestDetection:
    """detect_format() and resolve_ome_tiff_dirs() on both layouts."""

    @pytest.mark.parametrize("padding", [0, 5])
    def test_split_detected_as_ome_tiff(self, tmp_path, padding):
        base = make_split_dataset(tmp_path / "acq", padding=padding)
        assert detect_format(base) == "ome_tiff"

    @pytest.mark.parametrize("padding", [0, 5])
    def test_split_dirs_sorted_numerically(self, tmp_path, padding):
        base = make_split_dataset(tmp_path / "acq", n_t=11, padding=padding)
        entries, is_split = resolve_ome_tiff_dirs(base)
        assert is_split
        assert [t for t, _ in entries] == list(range(11))
        assert [d.parent.name for _, d in entries] == [
            f"{t:0{padding}d}" for t in range(11)
        ]

    def test_unsplit_detected_as_ome_tiff(self, tmp_path):
        base = make_unsplit_dataset(tmp_path / "acq")
        entries, is_split = resolve_ome_tiff_dirs(base)
        assert detect_format(base) == "ome_tiff"
        assert not is_split
        assert entries == [(0, base / "ome_tiff")]

    def test_timepoint_folder_without_ome_tiff_is_not_ome(self, tmp_path):
        """Individual-images runs keep their <t>/*.tiff layout untouched."""
        base = tmp_path / "acq"
        (base / "0").mkdir(parents=True)
        tifffile.imwrite(
            base / "0" / "A1_0_0_BF.tiff", np.zeros((SIZE, SIZE), dtype=np.uint16)
        )
        assert detect_format(base) == "single_tiff"


class TestSplitLoading:
    """Loading a split run as one dataset."""

    def test_fov_discovery_is_union_over_timepoints(self, tmp_path):
        base = make_split_dataset(tmp_path / "acq", n_t=2, fovs=[("A1", 0)])
        # A second FOV that only starts at the last timepoint.
        write_ome(base / "1" / "ome_tiff" / "A1_1.ome.tiff", make_stack(1, 1))

        viewer = make_viewer()
        assert viewer._discover_fovs(base, "ome_tiff") == [
            {"region": "A1", "fov": 0},
            {"region": "A1", "fov": 1},
        ]

    @pytest.mark.parametrize("padding", [0, 5])
    def test_t_is_number_of_timepoint_folders(self, tmp_path, padding):
        base = make_split_dataset(tmp_path / "acq", n_t=4, padding=padding)
        data, fovs = load(base)
        assert data is not None
        assert data.sizes["time"] == 4
        assert data.sizes["fov"] == len(fovs) == 2
        assert data.sizes["z"] == N_Z
        assert data.sizes["channel"] == N_C
        assert data.sizes["y"] == data.sizes["x"] == SIZE
        assert list(data.coords["time"].values) == [0, 1, 2, 3]

    def test_plane_values_per_t_fov_z_c(self, tmp_path):
        base = make_split_dataset(tmp_path / "acq", n_t=3)
        data, fovs = load(base)
        for t in range(3):
            for fov_idx, fov in enumerate(fovs):
                for z in range(N_Z):
                    for c in range(N_C):
                        plane = data[t, fov_idx, z, c].values
                        assert plane.shape == (SIZE, SIZE)
                        assert np.all(
                            plane == plane_value(t, fov["fov"], z, c)
                        ), f"wrong plane at t={t} fov={fov} z={z} c={c}"

    def test_channel_names_from_first_available_file(self, tmp_path):
        base = make_split_dataset(tmp_path / "acq", n_t=2)
        data, _ = load(base)
        assert data.attrs["channel_names"] == CHANNEL_NAMES

    def test_missing_timepoint_file_yields_blank_planes(self, tmp_path):
        """A FOV absent from one timepoint is blank there, not an error."""
        base = make_split_dataset(tmp_path / "acq", n_t=3)
        (base / "1" / "ome_tiff" / "A1_1.ome.tiff").unlink()

        data, fovs = load(base)
        assert data.sizes["time"] == 3
        assert [f["fov"] for f in fovs] == [0, 1]
        assert np.all(data[1, 1].values == 0)
        # Neighbouring timepoints and the other FOV are unaffected.
        assert np.all(data[1, 0, 0, 0].values == plane_value(1, 0, 0, 0))
        assert np.all(data[2, 1, 0, 0].values == plane_value(2, 1, 0, 0))

    def test_offloaded_timepoint_folder_drops_out(self, tmp_path):
        """A timepoint moved away mid-run shrinks T instead of failing."""
        base = make_split_dataset(tmp_path / "acq", n_t=3)
        shutil.rmtree(base / "0")

        data, _ = load(base)
        assert data.sizes["time"] == 2
        assert list(data.coords["time"].values) == [1, 2]
        assert np.all(data[0, 0, 0, 0].values == plane_value(1, 0, 0, 0))

    def test_file_vanishing_before_open_yields_blank_planes(
        self, tmp_path, monkeypatch
    ):
        """An offload tool can move a file away between scan and open."""
        base = make_split_dataset(tmp_path / "acq", n_t=2)
        core = sys.modules[LightweightViewer.__module__]

        vanished = str(base / "1" / "ome_tiff" / "A1_0.ome.tiff")
        real_tiff_file = core.tf.TiffFile

        def fake_tiff_file(path, *args, **kwargs):
            if str(path) == vanished:
                raise FileNotFoundError(vanished)
            return real_tiff_file(path, *args, **kwargs)

        monkeypatch.setattr(core.tf, "TiffFile", fake_tiff_file)

        data, _ = load(base)
        assert data.sizes["time"] == 2
        assert np.all(data[1, 0].values == 0)
        assert np.all(data[1, 1, 0, 0].values == plane_value(1, 1, 0, 0))

    def test_malformed_file_is_skipped(self, tmp_path):
        base = make_split_dataset(tmp_path / "acq", n_t=3)
        (base / "2" / "ome_tiff" / "A1_0.ome.tiff").write_bytes(b"not a tiff")

        data, _ = load(base)
        assert data.sizes["time"] == 3
        assert np.all(data[2, 0].values == 0)
        assert np.all(data[2, 1, 0, 0].values == plane_value(2, 1, 0, 0))

    def test_mismatched_geometry_is_skipped(self, tmp_path):
        """A file whose Z/C/Y/X disagree with the rest is blank, not fatal."""
        base = make_split_dataset(tmp_path / "acq", n_t=2)
        odd = np.zeros((1, N_Z, N_C, SIZE * 2, SIZE), dtype=np.uint16)
        write_ome(base / "1" / "ome_tiff" / "A1_0.ome.tiff", odd)

        data, _ = load(base)
        assert data.sizes["y"] == SIZE
        assert np.all(data[1, 0].values == 0)
        assert np.all(data[1, 1, 0, 0].values == plane_value(1, 1, 0, 0))


class TestLiveGrowth:
    """The refresh heuristic must re-scan folders, not read SizeT from a file."""

    def test_signature_changes_when_timepoint_folder_appears(self, tmp_path):
        base = make_split_dataset(tmp_path / "acq", n_t=2)
        viewer = make_viewer(base)
        before = viewer._dataset_signature()
        assert viewer._dataset_signature() == before

        make_split_dataset(base, n_t=3)
        assert viewer._dataset_signature() != before

    def test_signature_changes_when_fov_appears_in_latest_timepoint(self, tmp_path):
        base = make_split_dataset(tmp_path / "acq", n_t=2, fovs=[("A1", 0)])
        viewer = make_viewer(base)
        before = viewer._dataset_signature()

        write_ome(base / "1" / "ome_tiff" / "A1_1.ome.tiff", make_stack(1, 1))
        assert viewer._dataset_signature() != before

    def test_signature_changes_when_timepoint_is_offloaded(self, tmp_path):
        base = make_split_dataset(tmp_path / "acq", n_t=3)
        viewer = make_viewer(base)
        before = viewer._dataset_signature()

        shutil.rmtree(base / "0")
        assert viewer._dataset_signature() != before


class TestUnsplitUnchanged:
    """The single-file-per-FOV layout must behave exactly as before."""

    def test_unsplit_load(self, tmp_path):
        base = make_unsplit_dataset(tmp_path / "acq", n_t=3)
        data, fovs = load(base)
        assert data is not None
        assert data.sizes["time"] == 3
        assert data.sizes["fov"] == 2
        assert data.sizes["z"] == N_Z
        assert data.sizes["channel"] == N_C
        assert data.attrs["channel_names"] == CHANNEL_NAMES
        for t in range(3):
            for fov_idx, fov in enumerate(fovs):
                for z in range(N_Z):
                    for c in range(N_C):
                        assert np.all(
                            data[t, fov_idx, z, c].values
                            == plane_value(t, fov["fov"], z, c)
                        )

    def test_unsplit_fov_discovery(self, tmp_path):
        base = make_unsplit_dataset(tmp_path / "acq")
        viewer = make_viewer()
        assert viewer._discover_fovs(base, "ome_tiff") == [
            {"region": "A1", "fov": 0},
            {"region": "A1", "fov": 1},
        ]

    def test_unsplit_ignores_timepoint_folders(self, tmp_path):
        """Split discovery must not kick in when ome_tiff/ exists."""
        base = make_unsplit_dataset(tmp_path / "acq", n_t=2)
        (base / "0").mkdir()
        entries, is_split = resolve_ome_tiff_dirs(base)
        assert not is_split
        assert entries == [(0, base / "ome_tiff")]
