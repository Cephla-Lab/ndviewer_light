"""OME-Zarr 0.5 image metadata, read by the spec (no Qt, no dependency on core)."""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

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
