# T04b — Make the contact test 39x faster

DEPENDS ON: T04 (already implemented and passing)
MODIFY: `src/geometry/contact.py` only
DO NOT MODIFY: any other file. Never `tests/`.

**Priority: DO THIS FIRST.** Nothing downstream works until it is done.

## Why this exists, and it is not Devstral's fault

The T04 spec asked for golden-section to a `1e-12` bracket with up to 200 iterations, each
doing `numpy.linalg.solve`. That was implemented correctly and it is 1000x slower than the
problem needs, because numpy overhead dominates completely on a 3x3.

Measured on this machine, 2026-09-03.

```
contact_mu, current                1.6165 ms per call
86,400 calls, one item, one neighbour ->  139.7 s
```

The packer therefore does not finish placing a SINGLE sphere in an empty box within 110
seconds. T06 is written and correct and unusable.

## The fix, measured not asserted

```
scalar 40-iter, unrolled Cramer    0.0376 ms per call     39.4x faster
86,400 calls  ->  3.25 s
max error vs every closed form and vs the current version:  2.22e-16
```

The whole win is removing numpy from the inner loop. Allocation of tiny arrays costs far more
than the arithmetic.

## The implementation, verified. Use it.

The signature does not change. The existing `tests/test_contact.py` must still pass unchanged,
and it does, verified to 2.22e-16 on all seven of its cases including the anti-proxy test.

```python
import math


def contact_mu(A, B, d, iters=40):
    """Perram-Wertheim contact factor. Scalar, no numpy in the inner loop.

    A, B : (3,3) symmetric positive definite shape matrices
    d    : (3,) centre offset in mm
    Returns mu. mu < 1 overlap, mu > 1 disjoint, mu == 1 tangency.
    """
    a11, a12, a13 = float(A[0, 0]), float(A[0, 1]), float(A[0, 2])
    a22, a23, a33 = float(A[1, 1]), float(A[1, 2]), float(A[2, 2])
    b11, b12, b13 = float(B[0, 0]), float(B[0, 1]), float(B[0, 2])
    b22, b23, b33 = float(B[1, 1]), float(B[1, 2]), float(B[2, 2])
    d1, d2, d3 = float(d[0]), float(d[1]), float(d[2])

    ad1 = a11 * d1 + a12 * d2 + a13 * d3
    ad2 = a12 * d1 + a22 * d2 + a23 * d3
    ad3 = a13 * d1 + a23 * d2 + a33 * d3
    bd1 = b11 * d1 + b12 * d2 + b13 * d3
    bd2 = b12 * d1 + b22 * d2 + b23 * d3
    bd3 = b13 * d1 + b23 * d2 + b33 * d3

    def F(l):
        m = 1.0 - l
        c11 = m * a11 + l * b11; c12 = m * a12 + l * b12; c13 = m * a13 + l * b13
        c22 = m * a22 + l * b22; c23 = m * a23 + l * b23; c33 = m * a33 + l * b33
        # Cramer on a symmetric 3x3, solving C y = B d
        A0 = c22 * c33 - c23 * c23
        A1 = c13 * c23 - c12 * c33
        A2 = c12 * c23 - c13 * c22
        det = c11 * A0 + c12 * A1 + c13 * A2
        B0 = c11 * c33 - c13 * c13
        B1 = c12 * c13 - c11 * c23
        C0 = c11 * c22 - c12 * c12
        y1 = (A0 * bd1 + A1 * bd2 + A2 * bd3) / det
        y2 = (A1 * bd1 + B0 * bd2 + B1 * bd3) / det
        y3 = (A2 * bd1 + B1 * bd2 + C0 * bd3) / det
        return l * m * (ad1 * y1 + ad2 * y2 + ad3 * y3)

    lo, hi = 1e-9, 1.0 - 1e-9
    phi = 0.6180339887498949
    x1 = hi - phi * (hi - lo); x2 = lo + phi * (hi - lo)
    f1, f2 = F(x1), F(x2)
    for _ in range(iters):
        if f1 < f2:
            lo, x1, f1 = x1, x2, f2
            x2 = lo + phi * (hi - lo); f2 = F(x2)
        else:
            hi, x2, f2 = x2, x1, f1
            x1 = hi - phi * (hi - lo); f1 = F(x1)
    v = F(0.5 * (lo + hi))
    return math.sqrt(v) if v > 0.0 else 0.0
```

## Add this helper, T06 needs it

```python
def max_semi_axis(M) -> float:
    """Largest semi-axis of the ellipsoid with shape matrix M, in mm.

    Equals sqrt(largest eigenvalue of inv(M)). Compute it ONCE per item and cache
    it. It is the radius of the smallest sphere containing the body.
    """
```

## Forbidden

- Do not change the signature, the return meaning, or the file's public names.
- Do not touch `tests/test_contact.py`. It is checksummed.
- Do not reduce `iters` below 40. Measured error at 40 is 2.22e-16 and there is no reason
  to trade correctness for a speed you do not need.

## Gate, unchanged

```
python -m pytest tests/test_contact.py -q
```

Expected: `7 passed`, and it must now run in well under a second.

## Done when

The gate passes AND this prints under 0.1 ms per call.

```
python -c "import time,numpy as np;from src.types import Item;from src.geometry.contact import contact_mu as f;A=Item(label='a',axes=(25.,25.,25.)).shape_matrix();B=Item(label='b',axes=(40.,25.,15.)).shape_matrix(0.7);d=np.array([60.,0.,0.]);t=time.time();[f(A,B,d) for _ in range(2000)];print('%.4f ms'%((time.time()-t)/2000*1000))"
```
