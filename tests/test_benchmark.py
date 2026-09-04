"""Gate for T13. Theorem checks. Written before the implementation. Do not edit."""
import math

import pytest

from src.types import Item, Box
from src.solver.benchmark import (GENSANE_N15_CUBE, aa_target_cpft, actual_cpft,
                                  benchmark, best_q, grading_deviation, wall_ratio)


def sphere(label, r):
    return Item(label=label, axes=(r, r, r))


def test_gensane_constant_matches_the_closed_form():
    """Best known packing of 15 equal spheres in a cube, 2500 pi / 17576."""
    assert GENSANE_N15_CUBE == pytest.approx(2500.0 * math.pi / 17576.0, rel=1e-12)


def test_aa_curve_is_zero_at_dmin_and_one_at_dmax():
    assert aa_target_cpft(10.0, 10.0, 80.0, q=0.37) == pytest.approx(0.0, abs=1e-12)
    assert aa_target_cpft(80.0, 10.0, 80.0, q=0.37) == pytest.approx(1.0, abs=1e-12)


def test_aa_curve_with_q_equal_one_is_linear():
    """q = 1 collapses the curve to (D - dmin) / (dmax - dmin)."""
    for d in (20.0, 45.0, 70.0):
        assert aa_target_cpft(d, 10.0, 80.0, q=1.0) == pytest.approx(
            (d - 10.0) / 70.0, rel=1e-12)


def test_aa_curve_is_monotonic_increasing():
    prev = -1.0
    for d in range(10, 81, 5):
        v = aa_target_cpft(float(d), 10.0, 80.0, q=0.37)
        assert v > prev
        prev = v


def test_aa_curve_clamps_outside_the_range():
    assert aa_target_cpft(5.0, 10.0, 80.0) == pytest.approx(0.0, abs=1e-12)
    assert aa_target_cpft(200.0, 10.0, 80.0) == pytest.approx(1.0, abs=1e-12)


def test_actual_cpft_is_volume_weighted_not_count_weighted():
    """One big sphere and one small one. By count the small is half. By volume it is not.

    r = 10 and r = 20, so volumes are in ratio 1 to 8 and the small body is 1/9 of
    the total volume.
    """
    items = [sphere("small", 10.0), sphere("big", 20.0)]
    at_small = actual_cpft(items, 20.0)          # equivalent diameter of the small = 20 mm
    assert at_small == pytest.approx(1.0 / 9.0, rel=1e-9)
    assert actual_cpft(items, 40.0) == pytest.approx(1.0, rel=1e-12)


def test_a_set_built_on_the_aa_curve_has_low_deviation():
    """Build a real gradation on the q = 0.37 curve, then check the deviation is small.

    The construction matters. A&A CPFT is a VOLUME cumulative, so the COUNT in each
    size bin is chosen to make that bin carry its target volume fraction. Taking one
    item per CPFT quantile instead is wrong and gives a deviation near 0.27.
    Verified independently: 10 log-spaced bins, 96 items, deviation 0.0303.
    """
    import numpy as np

    d_min, d_max, q, bins, scale = 20.0, 80.0, 0.37, 10, 2e6
    edges = np.logspace(math.log10(d_min), math.log10(d_max), bins + 1)
    items = []
    for k in range(bins):
        d_rep = math.sqrt(edges[k] * edges[k + 1])
        frac = (aa_target_cpft(edges[k + 1], d_min, d_max, q)
                - aa_target_cpft(edges[k], d_min, d_max, q))
        v_rep = (4.0 / 3.0) * math.pi * (d_rep / 2.0) ** 3
        count = max(1, int(round(frac * scale / v_rep)))
        for j in range(count):
            items.append(sphere("b%d_%d" % (k, j), d_rep / 2.0))

    assert len(items) > 50
    assert grading_deviation(items, q=q) < 0.06


def test_wall_ratio_is_smallest_box_dimension_over_largest_item_diameter():
    box = Box(300.0, 200.0, 250.0)
    items = [sphere("a", 20.0), sphere("b", 25.0)]      # largest diameter = 50 mm
    assert wall_ratio(box, items) == pytest.approx(200.0 / 50.0, rel=1e-9)


def test_benchmark_refuses_a_ceiling_when_n_is_not_fifteen():
    """There is no published value to interpolate, so None is the only honest answer."""
    box = Box(300.0, 300.0, 300.0)
    items = [sphere("s%d" % i, 20.0) for i in range(6)]
    r = benchmark(items, box, achieved_container_density=0.36)
    assert r["n_items"] == 6
    assert r["equal_sphere_ceiling"] is None
    assert r["fraction_of_ceiling"] is None
    assert 0.20 <= r["best_q"] <= 0.50
