"""Gate for T04. Theorem checks only. Written before the implementation. Do not edit."""
import numpy as np
import pytest

from src.types import Item
from src.geometry.contact import contact_mu


def sphere(r):
    return Item(label="s", axes=(r, r, r)).shape_matrix()


def test_two_spheres_match_the_closed_form():
    """mu = |d| / (r1 + r2) exactly. This is the whole test of correctness."""
    r1, r2 = 30.0, 20.0
    A, B = sphere(r1), sphere(r2)
    for dist in (10.0, 49.0, 50.0, 51.0, 120.0):
        d = np.array([dist, 0.0, 0.0])
        assert contact_mu(A, B, d) == pytest.approx(dist / (r1 + r2), rel=1e-9)


def test_tangent_spheres_give_exactly_one():
    r1, r2 = 30.0, 20.0
    d = np.array([50.0, 0.0, 0.0])
    assert contact_mu(sphere(r1), sphere(r2), d) == pytest.approx(1.0, rel=1e-9)


def test_overlap_is_below_one_and_disjoint_is_above():
    A = B = sphere(10.0)
    assert contact_mu(A, B, np.array([15.0, 0, 0])) < 1.0
    assert contact_mu(A, B, np.array([25.0, 0, 0])) > 1.0


def test_closed_form_holds_off_axis():
    """Direction must not matter for spheres."""
    r1, r2 = 12.0, 7.0
    d = np.array([3.0, -4.0, 12.0])          # |d| = 13
    assert np.linalg.norm(d) == pytest.approx(13.0)
    assert contact_mu(sphere(r1), sphere(r2), d) == pytest.approx(13.0 / 19.0, rel=1e-9)


def test_identical_ellipsoids_offset_along_a_principal_axis():
    """Two identical bodies offset along a shared axis: mu = |d| / (2a)."""
    a, b, c = 40.0, 25.0, 15.0
    M = Item(label="e", axes=(a, b, c)).shape_matrix()
    for dist in (50.0, 80.0, 110.0):
        d = np.array([dist, 0.0, 0.0])
        assert contact_mu(M, M, d) == pytest.approx(dist / (2 * a), rel=1e-9)
    d = np.array([0.0, 40.0, 0.0])
    assert contact_mu(M, M, d) == pytest.approx(40.0 / (2 * b), rel=1e-9)


def test_symmetry_under_swapping_the_bodies():
    A = Item(label="a", axes=(30.0, 20.0, 15.0)).shape_matrix()
    B = Item(label="b", axes=(18.0, 18.0, 40.0)).shape_matrix(yaw=0.7)
    d = np.array([44.0, -12.0, 9.0])
    assert contact_mu(A, B, d) == pytest.approx(contact_mu(B, A, -d), rel=1e-9)


def test_no_bounding_sphere_shortcut_is_used():
    """A prolate body separated along its SHORT axis must overlap even though the
    centre distance exceeds the short semi-axis sum. A bounding-sphere proxy passes
    this case as disjoint and is therefore detectable here."""
    long_body = Item(label="p", axes=(60.0, 10.0, 10.0)).shape_matrix()
    d = np.array([0.0, 15.0, 0.0])           # 15 < 10 + 10, so genuinely overlapping
    assert contact_mu(long_body, long_body, d) == pytest.approx(15.0 / 20.0, rel=1e-9)
