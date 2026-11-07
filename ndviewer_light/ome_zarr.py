"""OME-Zarr 0.5 image metadata, read by the spec (no Qt, no dependency on core)."""

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger("ndviewer_light.ome_zarr")

_UM_PER_UNIT = {
    "angstrom": 1e-4,
    "picometer": 1e-6,
    "nanometer": 1e-3,
    "micrometer": 1.0,
    "millimeter": 1e3,
    "centimeter": 1e4,
    "decimeter": 1e5,
    "meter": 1e6,
    "kilometer": 1e9,
}
_FOV_NAMES = {"position", "p", "fov"}
_SPACE_BY_ORDER_FROM_END = ["x", "y", "z"]


class OmeZarrError(ValueError):
    """Raised when a node is not valid OME-Zarr 0.5 metadata."""


@dataclass
class OmeZarrImage:
    group: Path
    array_path: str
    axes: List[dict]
    dim_names: List[str]
    scale_um: Dict[str, float]
    channel_names: List[str] = field(default_factory=list)
    channel_colors: List[str] = field(default_factory=list)
    version: Optional[str] = None
    acquisition_complete: bool = True
    aborted: bool = False

    @property
    def array_dir(self) -> Path:
        return self.group if self.array_path == "." else self.group / self.array_path

    @property
    def pixel_size_um(self) -> Optional[float]:
        return self.scale_um.get("x")

    @property
    def dz_um(self) -> Optional[float]:
        return self.scale_um.get("z")


def read_node_attrs(path: Path) -> dict:
    """Return the ``attributes`` dict of ``path/zarr.json`` ({} when absent)."""
    zarr_json = Path(path) / "zarr.json"
    try:
        text = zarr_json.read_text()
    except FileNotFoundError:
        raise FileNotFoundError(f"No zarr.json in {path}") from None
    try:
        node = json.loads(text)
    except json.JSONDecodeError as exc:
        raise OmeZarrError(f"{zarr_json} is not valid JSON: {exc}") from exc
    if not isinstance(node, dict) or node.get("zarr_format") != 3:
        raise OmeZarrError(f"{zarr_json} is not a zarr v3 node")
    return node.get("attributes") or {}


def normalise_axis_names(axes: List[dict]) -> List[str]:
    """Map each axis to t, c, z, y, x or fov (array order)."""
    space_names = [ax.get("name", "") for ax in axes if ax.get("type") == "space"]
    if len(space_names) > 3:
        raise OmeZarrError(f"More than three space axes: {space_names}")
    by_name = all(name.lower() in ("x", "y", "z") for name in space_names)

    names = []
    space_seen = 0
    for ax in axes:
        ax_type = ax.get("type")
        raw = str(ax.get("name", ""))
        if ax_type == "time":
            names.append("t")
        elif ax_type == "channel":
            names.append("c")
        elif ax_type == "space":
            if by_name:
                names.append(raw.lower())
            else:
                from_end = len(space_names) - 1 - space_seen
                names.append(_SPACE_BY_ORDER_FROM_END[from_end])
            space_seen += 1
        else:
            lowered = raw.lower()
            names.append("fov" if lowered in _FOV_NAMES else lowered)
    return names


def _unit_factor(unit: Optional[str], axis_name: str) -> float:
    if not unit:
        logger.debug("Axis %s has no unit, assuming micrometer", axis_name)
        return 1.0
    try:
        return _UM_PER_UNIT[unit.lower()]
    except KeyError:
        raise OmeZarrError(f"Unsupported unit {unit!r} on axis {axis_name}") from None


def _scale_vector(transforms: Optional[List[dict]], n_axes: int) -> Optional[list]:
    for transform in transforms or []:
        if transform.get("type") == "scale":
            vector = transform.get("scale")
            if not isinstance(vector, list) or len(vector) != n_axes:
                raise OmeZarrError(f"Scale {vector} does not match {n_axes} axes")
            return vector
    return None


def compose_scale(
    axes: List[dict],
    dataset_transforms: List[dict],
    multiscale_transforms: Optional[List[dict]],
) -> Dict[str, float]:
    """Scale per space axis in micrometers (dataset scale x multiscale scale)."""
    names = normalise_axis_names(axes)
    dataset_scale = _scale_vector(dataset_transforms, len(axes))
    multiscale_scale = _scale_vector(multiscale_transforms, len(axes))
    if dataset_scale is None and multiscale_scale is None:
        return {}
    result = {}
    for i, ax in enumerate(axes):
        if ax.get("type") != "space":
            continue
        value = 1.0
        if dataset_scale is not None:
            value *= dataset_scale[i]
        if multiscale_scale is not None:
            value *= multiscale_scale[i]
        result[names[i]] = value * _unit_factor(ax.get("unit"), names[i])
    return result


def _color_text(color) -> str:
    return f"{color:06X}" if isinstance(color, int) else str(color or "")


def parse_image(group: Path, attrs: Optional[dict] = None) -> OmeZarrImage:
    """Parse the ``ome.multiscales`` image metadata of ``group``."""
    group = Path(group)
    if attrs is None:
        attrs = read_node_attrs(group)
    ome = attrs.get("ome") or {}
    multiscales = ome.get("multiscales")
    if not multiscales or not isinstance(multiscales, list):
        raise OmeZarrError(f"{group} has no ome.multiscales")
    multiscale = multiscales[0]
    axes = multiscale.get("axes") or []
    datasets = multiscale.get("datasets") or []
    if not axes or not datasets or "path" not in datasets[0]:
        raise OmeZarrError(f"{group} multiscales[0] lacks axes or datasets")

    version = ome.get("version")
    if version != "0.5":
        logger.warning(
            "%s declares OME-Zarr version %r, reading as 0.5", group, version
        )

    scale_um = compose_scale(
        axes,
        datasets[0].get("coordinateTransformations", []),
        multiscale.get("coordinateTransformations"),
    )

    channels = (ome.get("omero") or {}).get("channels") or []
    channel_names = [str(ch.get("label") or ch.get("name") or "") for ch in channels]
    channel_colors = [_color_text(ch.get("color")) for ch in channels]

    squid = attrs.get("_squid")
    acquisition_complete = True
    aborted = False
    if squid is not None:
        acquisition_complete = bool(squid.get("acquisition_complete", False))
        aborted = bool(squid.get("aborted", False))
        dim_names = normalise_axis_names(axes)
        for axis, key in (
            ("x", "pixel_size_um"),
            ("y", "pixel_size_um"),
            ("z", "z_step_um"),
        ):
            if axis not in dim_names or key not in squid:
                continue
            if axis not in scale_um:
                scale_um[axis] = squid[key]
            elif abs(scale_um[axis] - squid[key]) > 1e-9:
                logger.info(
                    "%s: _squid.%s=%s differs from spec scale %s on %s; using spec",
                    group,
                    key,
                    squid[key],
                    scale_um[axis],
                    axis,
                )

    return OmeZarrImage(
        group=group,
        array_path=datasets[0]["path"],
        axes=axes,
        dim_names=normalise_axis_names(axes),
        scale_um=scale_um,
        channel_names=channel_names,
        channel_colors=channel_colors,
        version=version,
        acquisition_complete=acquisition_complete,
        aborted=aborted,
    )


def array_dimension_names(image: OmeZarrImage) -> Optional[List[str]]:
    """The array's own ``dimension_names`` from its zarr.json, None when absent."""
    names = json.loads((image.array_dir / "zarr.json").read_text()).get(
        "dimension_names"
    )
    return list(names) if names else None


def _ome_attrs(path: Path) -> Optional[dict]:
    """The ``ome`` dict of ``path/zarr.json``; None when the node is unreadable."""
    try:
        return read_node_attrs(path).get("ome") or {}
    except (OSError, OmeZarrError):
        return None


def _ome_attrs_strict(path: Path) -> Optional[dict]:
    """The ``ome`` dict of an existing node, None when there is no node at all (no zarr.json).

    Unlike _ome_attrs, a node that exists but cannot be read raises OmeZarrError naming it:
    inside a recognised layout a broken region or plate is never silently dropped.
    """
    if not (Path(path) / "zarr.json").exists():
        return None
    return _read_ome(Path(path)).get("ome") or {}


def is_ome_zarr_node(path: Path) -> bool:
    """True when ``path/zarr.json`` carries ome.plate, ome.well or ome.multiscales."""
    ome = _ome_attrs(Path(path))
    return bool(ome) and any(key in ome for key in ("plate", "well", "multiscales"))


def _read_ome(path: Path) -> dict:
    """Attributes of an existing node, any failure reported with its path."""
    try:
        return read_node_attrs(path)
    except (OSError, OmeZarrError) as exc:
        raise OmeZarrError(f"{path}: {exc}") from exc


def _parse_existing_image(group: Path) -> OmeZarrImage:
    attrs = _read_ome(group)
    try:
        return parse_image(group, attrs)
    except OmeZarrError as exc:
        raise OmeZarrError(f"{group}: {exc}") from exc


def _entry(region: str, fov: int, image: OmeZarrImage) -> dict:
    return {"region": region, "fov": fov, "path": image.group, "image": image}


def _structure_of(images: List[OmeZarrImage]) -> str:
    return "6d" if any("fov" in image.dim_names for image in images) else "per_fov"


def _well_fields(well_dir: Path, region: str, skipped: List[str], root: Path):
    """Entries for the images listed by a well group; absent ones are skipped."""
    images = (_read_ome(well_dir).get("ome") or {}).get("well", {}).get("images", [])
    entries = []
    for j, listed in enumerate(images):
        name = str(listed.get("path", ""))
        field_dir = well_dir / name
        if not field_dir.is_dir():
            skipped.append(_relative(field_dir, root))
            continue
        fov = int(name) if name.isdigit() else j
        entries.append(_entry(region, fov, _parse_existing_image(field_dir)))
    return entries


def _relative(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return str(path)


def _discover_plate(plate_dir: Path, plate: dict):
    rows = [r.get("name", "") for r in plate.get("rows", [])]
    columns = [c.get("name", "") for c in plate.get("columns", [])]
    try:
        wells = sorted(
            plate.get("wells", []),
            key=lambda w: (w["rowIndex"], w["columnIndex"]),
        )
    except (KeyError, TypeError) as exc:
        raise OmeZarrError(
            f"{plate_dir}: plate well lacks rowIndex/columnIndex ({exc!r})"
        ) from exc
    skipped: List[str] = []
    entries = []
    for well in wells:
        try:
            well_path = well["path"]
            region = rows[well["rowIndex"]] + columns[well["columnIndex"]]
        except (KeyError, IndexError, TypeError) as exc:
            raise OmeZarrError(
                f"{plate_dir}: well {well.get('path')!r} has an invalid path, "
                f"rowIndex or columnIndex ({exc!r})"
            ) from exc
        well_dir = plate_dir / well_path
        if not well_dir.is_dir():
            skipped.append(_relative(well_dir, plate_dir))
            continue
        entries.extend(_well_fields(well_dir, region, skipped, plate_dir))
    _warn_skipped(plate_dir, skipped)
    return entries


def _warn_skipped(root: Path, skipped: List[str]) -> None:
    if skipped:
        logger.warning(
            "%s: %d listed entries not acquired (absent): %s",
            root,
            len(skipped),
            ", ".join(skipped),
        )


def _discover_experiment_folder(root: Path):
    """Plate, or zarr/<region>/{fov_<n>.ome.zarr, acquisition.zarr}; None if neither."""
    for name in ("plate.ome.zarr", "plate.zarr"):
        plate_dir = root / name
        plate = (_ome_attrs_strict(plate_dir) or {}).get("plate")
        if plate is not None:
            return _discover_plate(plate_dir, plate), "hcs_plate"

    zarr_dir = root / "zarr"
    if not zarr_dir.is_dir():
        return None
    entries = []
    images = []
    for region_dir in sorted(p for p in zarr_dir.iterdir() if p.is_dir()):
        fov_dirs = []
        for child in region_dir.iterdir():
            match = re.fullmatch(r"fov_(\d+)\.ome\.zarr", child.name)
            if match and child.is_dir():
                fov_dirs.append((int(match.group(1)), child))
        for fov, fov_dir in sorted(fov_dirs):
            image = _parse_existing_image(fov_dir)
            images.append(image)
            entries.append(_entry(region_dir.name, fov, image))
        acquisition = region_dir / "acquisition.zarr"
        # An existing store directory must carry metadata: missing or unreadable zarr.json is a broken
        # store (raised with its path), only an absent directory means "no 6D store here".
        acquisition_ome = (
            (_read_ome(acquisition).get("ome") or {}) if acquisition.is_dir() else {}
        )
        if "multiscales" in acquisition_ome:
            image = _parse_existing_image(acquisition)
            images.append(image)
            entries.append(_entry(region_dir.name, 0, image))
    if not entries:
        return None
    return entries, _structure_of(images)


def discover_ome_zarr(root: Path) -> Tuple[List[dict], str]:
    """Find the FOVs under ``root`` from OME-Zarr 0.5 metadata.

    Returns (entries, structure); entries are {"region", "fov", "path", "image"}.
    structure is "hcs_plate", "6d", "per_fov", or "unknown" when the layout is
    not recognised (the caller then tries the legacy directory walk).
    """
    root = Path(root)
    ome = _ome_attrs(root) or {}
    if "plate" in ome:
        return _discover_plate(root, ome["plate"]), "hcs_plate"
    if "well" in ome:
        skipped: List[str] = []
        region = root.parent.name + root.name
        entries = _well_fields(root, region, skipped, root)
        _warn_skipped(root, skipped)
        return entries, "hcs_plate"
    if "multiscales" in ome:
        image = _parse_existing_image(root)
        return [_entry("default", 0, image)], _structure_of([image])

    found = _discover_experiment_folder(root)
    return found if found is not None else ([], "unknown")


_LEGACY_DIMS_BY_RANK = {
    2: ["y", "x"],
    3: ["z", "y", "x"],
    4: ["c", "z", "y", "x"],
    5: ["t", "c", "z", "y", "x"],
    6: [
        "t",
        "fov",
        "c",
        "z",
        "y",
        "x",
    ],  # the order the pre-0.5 loaders assumed for axes-less 6D stores
}


def _legacy_dims_from_rank(path: Path) -> List[str]:
    """Dims of a pre-0.5 store without axes metadata, from the rank of its array "0" (as the old loaders did)."""
    try:
        with open(path / "0" / "zarr.json") as f:
            rank = len(json.load(f)["shape"])
    except (OSError, ValueError, KeyError, TypeError):
        rank = 5
    return list(_LEGACY_DIMS_BY_RANK.get(rank, _LEGACY_DIMS_BY_RANK[5]))


def image_from_legacy(path: Path, meta: dict) -> OmeZarrImage:
    """Describe a pre-0.5 store from the dict ``core.parse_zarr_v3_metadata`` returns."""
    axes = meta.get("axes") or []
    dim_names = (
        normalise_axis_names(axes) if axes else _legacy_dims_from_rank(Path(path))
    )
    scale_um: Dict[str, float] = {}
    if meta.get("pixel_size_um") is not None:
        scale_um["y"] = scale_um["x"] = meta["pixel_size_um"]
    if meta.get("dz_um") is not None:
        scale_um["z"] = meta["dz_um"]
    return OmeZarrImage(
        group=Path(path),
        array_path="0",
        axes=axes,
        dim_names=dim_names,
        scale_um=scale_um,
        channel_names=list(meta.get("channel_names") or []),
        channel_colors=list(meta.get("channel_colors") or []),
        version=None,
        acquisition_complete=meta.get("acquisition_complete", True),
        aborted=False,
    )
