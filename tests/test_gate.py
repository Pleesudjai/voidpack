"""Gate for T05. Written before the implementation. Do not edit."""
import numpy as np
import pytest

from src.types import Item, Placement
from src.geometry.gate import (TOLERANCE_MM, OverlapRejected, gate,
                               max_penetration_mm, pair_penetration_mm)


def place(r, x):
    return Placement(item=Item(label="s", axes=(r, r, r)),
                     centre=np.array([float(x), 0.0, 0.0]), yaw=0.0)


def test_penetration_matches_the_sphere_closed_form():
    """For spheres, penetration = (r1 + r2) - |d|."""
    a, b = place(10.0, 0.0), place(10.0, 15.0)
    assert pair_penetration_mm(a, b) == pytest.approx(5.0, rel=1e-6)


def test_disjoint_bodies_have_zero_penetration():
    a, b = place(10.0, 0.0), place(10.0, 30.0)
    assert pair_penetration_mm(a, b) == pytest.approx(0.0, abs=1e-9)


def test_tangent_bodies_have_zero_penetration():
    a, b = place(10.0, 0.0), place(10.0, 20.0)
    assert pair_penetration_mm(a, b) == pytest.approx(0.0, abs=1e-6)


def test_max_penetration_is_zero_for_trivial_lists():
    assert max_penetration_mm([]) == 0.0
    assert max_penetration_mm([place(10.0, 0.0)]) == 0.0


def test_gate_accepts_a_clean_configuration():
    gate([place(10.0, 0.0), place(10.0, 25.0), place(10.0, 50.0)])


def test_gate_rejects_and_reports_the_depth():
    """The 5 to 8 mm interpenetration that once reported as 0.50 density must raise."""
    with pytest.raises(OverlapRejected) as exc:
        gate([place(10.0, 0.0), place(10.0, 14.0)])
    assert exc.value.penetration_mm == pytest.approx(6.0, rel=1e-6)
    assert TOLERANCE_MM == 1e-3
