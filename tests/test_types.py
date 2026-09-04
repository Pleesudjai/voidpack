"""Gate for T01. Written before the implementation. Do not edit."""
import numpy as np
import pytest

from src.types import Item, Box, Placement, MM_PER_M


def test_mm_per_m_is_exact():
    assert MM_PER_M == 1000.0


def test_sphere_volume_matches_closed_form():
    """Closed form, not a previous run. A sphere of radius 10 mm."""
    it = Item(label="s", axes=(10.0, 10.0, 10.0))
    expected = 4.0 / 3.0 * np.pi * 10.0 ** 3
    assert it.volume_mm3() == pytest.approx(expected, rel=1e-12)


def test_triaxial_volume_matches_closed_form():
    it = Item(label="e", axes=(30.0, 20.0, 15.0))
    expected = 4.0 / 3.0 * np.pi * 30.0 * 20.0 * 15.0
    assert it.volume_mm3() == pytest.approx(expected, rel=1e-12)


def test_volume_ignores_compaction():
    """A squeezed item does not lose physical volume."""
    rigid = Item(label="a", axes=(10.0, 10.0, 10.0))
    soft = Item(label="b", axes=(10.0, 10.0, 10.0), deformable=True, compaction=0.8)
    assert soft.volume_mm3() == pytest.approx(rigid.volume_mm3(), rel=1e-12)


def test_shape_matrix_puts_surface_points_at_unity():
    """x^T M x = 1 exactly on the ellipsoid surface, for every semi-axis tip."""
    a, b, c = 30.0, 20.0, 15.0
    M = Item(label="e", axes=(a, b, c)).shape_matrix()
    for v in (np.array([a, 0, 0]), np.array([0, b, 0]), np.array([0, 0, c])):
        assert float(v @ M @ v) == pytest.approx(1.0, rel=1e-12)


def test_shape_matrix_yaw_rotates_the_body():
    """After 90 degrees of yaw, the a-axis tip lies along +y."""
    a, b, c = 30.0, 20.0, 15.0
    M = Item(label="e", axes=(a, b, c)).shape_matrix(yaw=np.pi / 2)
    v = np.array([0.0, a, 0.0])
    assert float(v @ M @ v) == pytest.approx(1.0, rel=1e-9)
