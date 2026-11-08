"""Helpers that write small OME-Zarr 0.5 stores for tests."""

import json
import re
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import tensorstore as ts

SQUID_AXES = [
    {"name": "t", "type": "time", "unit": "second"},
    {"name": "c", "type": "channel"},
    {"name": "z", "type": "space", "unit": "micrometer"},
    {"name": "y", "type": "space", "unit": "micrometer"},
    {"name": "x", "type": "space", "unit": "micrometer"},
]
DEFAULT_SCALE = {"z": 1.5, "y": 0.65, "x": 0.65}
DEFAULT_CHANNELS = ["BF LED matrix full", "Fluorescence 488 nm Ex"]
DEFAULT_COLORS = ["FFFFFF", "1FFF00"]

_POSITIONAL_NAMES = {
    4: ["c", "z", "y", "x"],
    5: ["t", "c", "z", "y", "x"],
    6: ["fov", "t", "c", "z", "y", "x"],
}


def plane_value(c: int, z: int) -> int:
    return 100 * c + z


def write_array(
    path: Path,
    shape: tuple,
    dimension_names: Optional[list] = None,
    dtype=np.uint16,
) -> None:
    """Write a zarr v3 array (one chunk) whose (y, x) planes hold plane_value(c, z)."""
    metadata = {
        "shape": list(shape),
        "chunk_grid": {
            "name": "regular",
            "configuration": {"chunk_shape": list(shape)},
        },
        "chunk_key_encoding": {"name": "default"},
        "data_type": np.dtype(dtype).name,
        "codecs": [{"name": "bytes", "configuration": {"endian": "little"}}],
    }
    if dimension_names is not None:
        metadata["dimension_names"] = list(dimension_names)
    store = ts.open(
        {
            "driver": "zarr3",
            "kvstore": {"driver": "file", "path": str(path)},
            "metadata": metadata,
            "create": True,
            "delete_existing": True,
        }
    ).result()

    names = dimension_names or _POSITIONAL_NAMES.get(len(shape))
    data = np.zeros(shape, dtype=dtype)
    if names is not None and "c" in names and "z" in names:
        c_axis, z_axis = names.index("c"), names.index("z")
    else:
        c_axis = z_axis = None
    for index in np.ndindex(*shape[:-2]):
        c = index[c_axis] if c_axis is not None else 0
        z = index[z_axis] if z_axis is not None else 0
        data[index] = plane_value(c, z)
    store[...].write(data).result()


def write_image_group(
    group: Path,
    shape: tuple = (1, 2, 3, 8, 8),
    axes: Optional[list] = None,
    *,
    dataset_path: str = "0",
    scale: Optional[Dict[str, float]] = None,
    dimension_names: Optional[list] = None,
    omero: bool = True,
    squid: Optional[dict] = None,
    version: str = "0.5",
    multiscale_transforms: Optional[list] = None,
    node_type: str = "group",
    dtype=np.uint16,
) -> None:
    """Write an image group; node_type="array" makes the array itself the group."""
    group = Path(group)
    axes = SQUID_AXES if axes is None else axes
    scale = DEFAULT_SCALE if scale is None else scale
    scale_vector = [
        scale.get(ax["name"], 1.0) if ax.get("type") == "space" else 1.0 for ax in axes
    ]
    multiscale = {
        "version": version,
        "axes": axes,
        "datasets": [
            {
                "path": dataset_path,
                "coordinateTransformations": [{"type": "scale", "scale": scale_vector}],
            }
        ],
    }
    if multiscale_transforms is not None:
        multiscale["coordinateTransformations"] = multiscale_transforms
    ome = {"version": version, "multiscales": [multiscale]}
    if omero:
        ome["omero"] = {
            "channels": [
                {"label": label, "color": color}
                for label, color in zip(DEFAULT_CHANNELS, DEFAULT_COLORS)
            ]
        }
    attributes = {"ome": ome}
    if squid is not None:
        attributes["_squid"] = squid

    if node_type == "array":
        write_array(group, shape, dimension_names, dtype=dtype)
        zarr_json = group / "zarr.json"
        node = json.loads(zarr_json.read_text())
        node["attributes"] = attributes
        zarr_json.write_text(json.dumps(node))
        return

    group.mkdir(parents=True, exist_ok=True)
    (group / "zarr.json").write_text(
        json.dumps({"zarr_format": 3, "node_type": "group", "attributes": attributes})
    )
    array_dir = group if dataset_path == "." else group / dataset_path
    write_array(array_dir, shape, dimension_names, dtype=dtype)


def _write_group(path: Path, attributes: dict) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "zarr.json").write_text(
        json.dumps({"zarr_format": 3, "node_type": "group", "attributes": attributes})
    )


def write_plate(
    root: Path,
    wells: Dict[str, List[int]],
    rows: list,
    cols: list,
    **image_kwargs,
) -> None:
    """Write a plate. wells maps a well id ("B3") to its list of field indices."""
    root = Path(root)
    row_names = [str(r) for r in rows]
    col_names = [str(c) for c in cols]
    well_paths = {}
    for well_id in wells:
        match = re.fullmatch(r"([A-Za-z]+)(\d+)", well_id)
        if match is None:
            raise ValueError(f"Bad well id {well_id!r}")
        well_paths[well_id] = (match.group(1), str(int(match.group(2))))
    well_entries = [
        {
            "path": f"{row}/{col}",
            "rowIndex": row_names.index(row),
            "columnIndex": col_names.index(col),
        }
        for row, col in well_paths.values()
    ]
    _write_group(
        root,
        {
            "ome": {
                "version": "0.5",
                "plate": {
                    "rows": [{"name": r} for r in row_names],
                    "columns": [{"name": c} for c in col_names],
                    "wells": well_entries,
                },
            }
        },
    )
    for row in sorted({row for row, _ in well_paths.values()}):
        _write_group(root / row, {})
    for well_id, fields in wells.items():
        row, col = well_paths[well_id]
        well_dir = root / row / col
        images = [{"path": str(f)} for f in fields]
        _write_group(well_dir, {"ome": {"version": "0.5", "well": {"images": images}}})
        for f in fields:
            write_image_group(well_dir / str(f), **image_kwargs)


def write_legacy_image(
    group: Path, shape: tuple = (1, 2, 3, 8, 8), axes=SQUID_AXES
) -> None:
    """Write a pre-0.5 Squid store: root-level multiscales, _squid_metadata, array "0".

    ``axes=None`` writes no axes at all (the oldest stores), so the reader must infer them from the rank.
    """
    group = Path(group)
    multiscale = {"datasets": [{"path": "0"}]}
    if axes is not None:
        multiscale["axes"] = axes
    _write_group(
        group,
        {"multiscales": [multiscale], "_squid_metadata": {"pixel_size_um": 0.65}},
    )
    write_array(group / "0", shape)
