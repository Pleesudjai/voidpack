# T04 — Exact ellipsoid contact, Perram-Wertheim

DEPENDS ON: T01
CREATE: `src/geometry/contact.py`
DO NOT MODIFY: any other file. Never `tests/`.

## Purpose

The exact overlap criterion for two ellipsoids. This is the single function the whole
product claim rests on, so it is checked against closed forms and not against itself.

## Exact signature

```python
def contact_mu(A: np.ndarray, B: np.ndarray, d: np.ndarray) -> float:
    """Perram-Wertheim contact factor for two ellipsoids.

    A, B : (3,3) symmetric positive definite shape matrices, from
           Item.shape_matrix(). A point x is inside body A centred at c
           when (x-c) @ A @ (x-c) <= 1.
    d    : (3,) vector from the centre of A to the centre of B, in mm.

    Returns mu, the factor by which BOTH bodies must be scaled to reach
    tangency. mu < 1 means they overlap. mu > 1 means they are disjoint.
    mu == 1 means exact tangency.
    """
```

## Algorithm, do not deviate

1. `C(l) = (1 - l) * A + l * B`
2. `F(l) = l * (1 - l) * d @ A @ inv(C(l)) @ B @ d`
3. Maximise `F` over the OPEN interval `(0, 1)`.
4. Return `sqrt(max(F))`.

**Use golden-section search for step 3.** `F` is strictly concave on `(0,1)`, because
`F''(l) = -(1/2) w^T C(l)^-1 w` with `C` positive definite, so the maximiser is unique and a
derivative-free unimodal search converges. Do not attempt the analytic derivative. It exists
but it is not needed and it is where implementations go wrong.

Search parameters. Bracket `[1e-12, 1 - 1e-12]`. Iterate until the bracket is narrower than
`1e-12`, or for 200 iterations, whichever comes first.

5. Solve `C(l) @ y = B @ d` with `numpy.linalg.solve`. **Do not form `inv(C(l))`.**
   Then `F(l) = l * (1 - l) * (d @ A @ y)`.

## Closed forms this must reproduce, and the gate checks them

- **Two spheres**, radii `r1` and `r2`, centre separation `|d|`:
  `mu = |d| / (r1 + r2)` exactly.
- **Two identical ellipsoids** offset along a shared principal axis with semi-axis `a` in
  that direction: `mu = |d| / (2a)` exactly.
- **Symmetry**: `contact_mu(A, B, d) == contact_mu(B, A, -d)`.

## Forbidden

- No bounding-sphere shortcut, no early-out on centre distance. A proxy test silently admits
  interpenetration and inflates density, and the error is invisible in the density number.
- No `scipy`. Write the golden-section search.
- No `numpy.linalg.inv`. Use `solve`.
- Imports limited to `numpy`.

## Gate, run this, do not edit it

```
python -m pytest tests/test_contact.py -q
```

Expected: `7 passed`

## Done when

The gate passes and no file outside `src/geometry/contact.py` has changed.
