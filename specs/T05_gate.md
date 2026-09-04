# T05 — Overlap gate

DEPENDS ON: T04
CREATE: `src/geometry/gate.py`
DO NOT MODIFY: any other file. Never `tests/`.

## Purpose

This function is why a density number from this codebase can be trusted. It refuses a
configuration rather than reporting a flattering number for it. It exists because the failure
already happened once, a run reported 0.50 density while bodies passed 5 to 8 mm through each
other.

## Exact signature

```python
TOLERANCE_MM = 1e-3

def pair_penetration_mm(pa, pb) -> float:
    """Linear interpenetration between two Placements, in mm. Zero when disjoint."""

def max_penetration_mm(placements: list) -> float:
    """Largest pairwise penetration across all pairs, in mm. Zero for an empty
    or single-item list."""

def gate(placements: list) -> None:
    """Raise OverlapRejected when max_penetration_mm exceeds TOLERANCE_MM.

    Return None when the configuration is acceptable.
    """

class OverlapRejected(Exception):
    """Carries .penetration_mm."""
```

## The penetration definition, exact

From T04, both bodies scale by `mu` to reach tangency, so along the line of centres the
sum of the two support radii is `|d| / mu`. Therefore

    penetration_mm = max(0, |d| / mu - |d|) = |d| * (1 - mu) / mu

For two spheres this reduces to `(r1 + r2) - |d|`, which is the correct physical overlap, and
the gate checks that identity.

## Forbidden

- Do not clamp, round, or soften a penetration value before comparing it to the tolerance.
- Do not add a "warn but continue" path. The gate raises or it returns.
- Do not catch `OverlapRejected` inside this module.

## Gate, run this, do not edit it

```
python -m pytest tests/test_gate.py -q
```

Expected: `6 passed`

## Done when

The gate passes and no file outside `src/geometry/gate.py` has changed.
