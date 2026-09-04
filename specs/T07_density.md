# T07 — Density on three bases

DEPENDS ON: T06
CREATE: `src/solver/density.py`
DO NOT MODIFY: any other file. Never `tests/`.

## Purpose

Three different questions, three different denominators, never mixed. Every reported number
names its basis.

## Exact signature

```python
def container_density(sum_volume_mm3: float, width: float, depth: float,
                      fill_height: float) -> float:
    """sum(V) / (width * depth * fill_height). The shipping figure. Penalised by walls."""

def fill_height(placements: list) -> float:
    """Highest top surface across placements, in mm. Zero for an empty list."""

def hull_density(placements: list) -> float:
    """sum(V) / volume of the convex hull of the bodies. Removes the container.

    Approximate each body by its surface sample, at least 128 points per body,
    then take the hull of all samples with scipy.spatial.ConvexHull.
    """

def laguerre_density(placements: list):
    """Mean radical-cell fraction over INTERIOR cells only.

    Return None when no cell is interior. Do NOT return a number in that case.
    At small n no cell is interior and reporting one would be a fabrication.
    """

def report(placements, box) -> dict:
    """Return {"container": float, "hull": float, "laguerre": float | None,
               "fill_height_mm": float, "sum_volume_mm3": float}.

    Never sum, average, or otherwise combine the three bases.
    """
```

## Closed forms the gate checks

- **FCC identity.** Four spheres of radius `a / (2*sqrt(2))` in a cube of side `a` give
  container density `pi / sqrt(18) = 0.740480489...`. This is the face-centred cubic packing
  fraction and it is a theorem, not a measurement.
- **Sphere in a tight cube.** One sphere of radius `r` in a cube of side `2r` gives `pi / 6`.
- **Hull of one body.** The convex hull of a single ellipsoid is that ellipsoid, so
  `hull_density` for one placement approaches 1.0. The sampled hull is inscribed, so accept
  anything in `[0.95, 1.0]` and never above 1.0.

## Why radical and not plain Voronoi

For unequal particles the Voronoi bisector sits at the midpoint of two centres, which for a
large and a small neighbour falls INSIDE the large body. The cell volume is then wrong and a
per-particle fraction can exceed 1. The radical, or Laguerre, tessellation fixes the plane
by the power distance instead. If the radical tessellation is not implementable in budget,
`laguerre_density` returns `None` and the README says the basis is unavailable. That is an
acceptable outcome. Returning a plain Voronoi number in its place is not.

## Forbidden

- Never return a Laguerre number computed from cells that touch the boundary.
- Never average the three bases into one headline figure.
- `hull_density` must never exceed 1.0. If it does, the sampling or the hull is wrong.

## Gate, run this, do not edit it

```
python -m pytest tests/test_density.py -q
```

Expected: `8 passed`

## Done when

The gate passes and no file outside `src/solver/density.py` has changed.
