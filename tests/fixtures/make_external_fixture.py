"""Generate tests/fixtures/external_05.ome.zarr with ngff-zarr, an independent OME-Zarr 0.5 writer.

Run from the clone root:
    uv run --python 3.12 --with "ngff-zarr==0.50.0" tests/fixtures/make_external_fixture.py
"""

import shutil
from pathlib import Path

import numpy as np
import ngff_zarr as nz

OUT = Path(__file__).parent / "external_05.ome.zarr"


def plane_value(c: int, z: int) -> int:
    return 100 * c + z


data = np.zeros((2, 3, 8, 8), dtype=np.uint16)
for c in range(2):
    for z in range(3):
        data[c, z] = plane_value(c, z)

image = nz.to_ngff_image(
    data,
    dims=["c", "z", "y", "x"],
    scale={"z": 2.0, "y": 0.5, "x": 0.5},
    axes_units={"z": "micrometer", "y": "micrometer", "x": "micrometer"},
)
multiscales = nz.to_multiscales(image, scale_factors=[])
if OUT.exists():
    shutil.rmtree(OUT)
nz.to_ngff_zarr(str(OUT), multiscales, version="0.5")
