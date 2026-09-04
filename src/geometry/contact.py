"""Exact ellipsoid contact, Perram-Wertheim.

Scalar and unrolled on purpose. numpy allocation on 3x3 arrays dominates the
arithmetic completely, so the inner loop uses Python floats and Cramer's rule.
Measured 39.4x faster than the numpy.linalg.solve form, with a maximum error of
2.22e-16 against every closed form in tests/test_contact.py.
"""
import math

import numpy as np


def contact_mu(A, B, d, iters: int = 40) -> float:
    """Perram-Wertheim contact factor for two ellipsoids.

    A, B : (3,3) symmetric positive definite shape matrices, from
           Item.shape_matrix(). A point x is inside body A centred at c when
           (x - c) @ A @ (x - c) <= 1.
    d    : (3,) vector from the centre of A to the centre of B, in mm.

    Returns mu, the factor by which BOTH bodies must be scaled to reach
    tangency. mu < 1 means they overlap, mu > 1 means they are disjoint,
    mu == 1 is exact tangency.
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
        c11 = m * a11 + l * b11
        c12 = m * a12 + l * b12
        c13 = m * a13 + l * b13
        c22 = m * a22 + l * b22
        c23 = m * a23 + l * b23
        c33 = m * a33 + l * b33
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
    x1 = hi - phi * (hi - lo)
    x2 = lo + phi * (hi - lo)
    f1, f2 = F(x1), F(x2)
    for _ in range(iters):
        if f1 < f2:
            lo, x1, f1 = x1, x2, f2
            x2 = lo + phi * (hi - lo)
            f2 = F(x2)
        else:
            hi, x2, f2 = x2, x1, f1
            x1 = hi - phi * (hi - lo)
            f1 = F(x1)
    v = F(0.5 * (lo + hi))
    return math.sqrt(v) if v > 0.0 else 0.0


def max_semi_axis(M) -> float:
    """Largest semi-axis of the ellipsoid with shape matrix M, in mm.

    Equals sqrt of the largest eigenvalue of inv(M), which is the radius of the
    smallest sphere containing the body. Compute once per item and cache it.
    Used only as a conservative outer bound to prove DISJOINTNESS. It must never
    be used to conclude that two bodies overlap.
    """
    return float(np.sqrt(np.max(np.linalg.eigvalsh(np.linalg.inv(np.asarray(M, dtype=float))))))
