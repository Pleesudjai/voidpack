# T01 — Types and units

DEPENDS ON: nothing
CREATE: `src/types.py`
DO NOT MODIFY: any other file. Never `tests/`.

## Purpose

Every other task imports these. Units are the most common source of silent error in this
codebase, so they are fixed here once and stated in every docstring that crosses a boundary.

## Unit rule, absolute

Scan data arrives in **metres**. Everything inside the solver and everything reported is in
**millimetres**. Conversion happens once, at the capture boundary, and nowhere else.

## Exact contents

```python
from dataclasses import dataclass, field
import numpy as np

MM_PER_M = 1000.0

@dataclass
class Item:
    """One packable body, modelled as a triaxial ellipsoid. All lengths in mm."""
    label: str
    axes: tuple[float, float, float]      # semi-axes a, b, c in mm
    mass_g: float = 0.0
    fragile: bool = False
    deformable: bool = False
    compaction: float = 1.0               # multiplies each semi-axis, 1.0 = rigid
    keep_upright: bool = False

    def volume_mm3(self) -> float:
        """Ellipsoid volume, 4/3 pi a b c, in mm^3."""

    def shape_matrix(self, yaw: float = 0.0) -> np.ndarray:
        """(3,3) SPD matrix for the Perram-Wertheim test, rotated by yaw about z.

        Uses the EFFECTIVE semi-axes, that is axes scaled by compaction.
        """

@dataclass
class Box:
    """Container, inner dimensions in mm."""
    width: float
    depth: float
    height: float

    def volume_mm3(self) -> float: ...

@dataclass
class Placement:
    """One item placed. centre in mm from the box inner corner at the origin."""
    item: Item
    centre: np.ndarray                    # (3,) mm
    yaw: float = 0.0                      # radians about z
```

## Algorithm notes, do not deviate

- `shape_matrix` returns `R @ diag(1/ae**2, 1/be**2, 1/ce**2) @ R.T` where `ae, be, ce` are
  the effective semi-axes and `R` is the yaw rotation about z. This is the form
  Perram-Wertheim expects, so that a point `x` is inside when `(x-c) @ M @ (x-c) <= 1`.
- Effective semi-axis = semi-axis * compaction.
- `volume_mm3` on `Item` uses the RAW axes, not the compacted ones, because the physical
  item does not lose volume when it is squeezed.

## Forbidden

- No imports beyond `numpy` and the standard library.
- No I/O, no file reading, no printing.
- Do not add fields that are not listed.

## Gate, run this, do not edit it

```
python -m pytest tests/test_types.py -q
```

Expected: `6 passed`

## Done when

The gate passes and no file outside `src/types.py` has changed.
