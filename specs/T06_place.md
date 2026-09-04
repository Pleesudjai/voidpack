# T06 — Drop and settle placement

DEPENDS ON: T05
CREATE: `src/solver/place.py`
DO NOT MODIFY: any other file. Never `tests/`.

## Purpose

**This is the packer.** The contact test answers whether two bodies overlap. It does not
answer where an item goes. This task produces the packing.

Budget 90 minutes. If it is not green by then, fall back to fixed-grid placement, take the
lower density, and record the fallback in the README.

## Exact signature

```python
def pack(items: list, box, ruleset: dict, seed: int = 0,
         grid: int = 12, yaws: int = 4) -> tuple[list, list]:
    """Place items into box under the compiled ruleset.

    Returns (placements, unplaced) where placements is a list of Placement and
    unplaced is a list of Item that did not fit.

    Deterministic for a given seed.
    """
```

## Algorithm, do not deviate

1. Sort items by `volume_mm3()` descending. Deformable items go LAST regardless of size,
   because they are void fillers.
2. Candidate orientations for an item: the up-axis is one of the three semi-axes, and the
   yaw is one of `yaws` values evenly spaced over `[0, pi)`. Build a permuted copy of the
   item whose third semi-axis is the chosen up-axis, then call `shape_matrix(yaw)`.
   If the item has `keep_upright` in the ruleset, the up-axis is fixed to the third axis.
3. Candidate `(x, y)` positions: a `grid` by `grid` lattice spanning the box footprint.
4. For each `(orientation, x, y)`, find the resting height by bisection on `z`.
   The predicate is "does the body at this z clear the floor and every placed body".
   Bracket `z` from the support radius in z, which is the lowest possible, up to the box
   height. Bisect until the bracket is under `1e-6` mm.
5. Reject a candidate that is not fully inside the box. Containment along axis `e` uses the
   support function, `extent = sqrt(e @ inv(M) @ e)`, so the body spans
   `centre_i +/- sqrt(inv(M)[i,i])` and both ends must lie inside.
6. Reject a candidate that would place the body above a `fragile` item whose footprint it
   overlaps in `x, y`.
7. Reject the item entirely if placing it would exceed a `max_weight` limit.
8. Accept the candidate with the LOWEST resting centre z. Ties break by lowest `(x, y)`
   lattice index, which is what makes the result deterministic.
9. An item with no acceptable candidate goes to `unplaced`.
10. Before returning, call `gate(placements)` from T05. It must not raise.

## Forbidden

- No `scipy.optimize`, no random restarts, no simulated annealing. Deterministic only.
- Do not relax the gate tolerance for deformable items. Deformation is already handled by the
  compaction factor shrinking the effective semi-axes in `shape_matrix`.
- Do not silently drop an item. It goes to `unplaced` and the caller decides.

## Gate, run this, do not edit it

```
python -m pytest tests/test_place.py -q
```

Expected: `7 passed`

## Done when

The gate passes and no file outside `src/solver/place.py` has changed.
