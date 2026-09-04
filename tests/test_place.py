"""Gate for T06. Written before the implementation. Do not edit."""
import numpy as np
import pytest

from src.types import Item, Box
from src.schema import empty_ruleset, rule_fragile, rule_max_weight
from src.geometry.gate import max_penetration_mm, TOLERANCE_MM
from src.solver.place import pack


def sphere_item(label, r, mass_g=0.0, **kw):
    return Item(label=label, axes=(r, r, r), mass_g=mass_g, **kw)


def extent_z(p):
    M = p.item.shape_matrix(p.yaw)
    return float(np.sqrt(np.linalg.inv(M)[2, 2]))


def test_one_sphere_rests_on_the_floor():
    """Closed form. A sphere of radius r at rest has centre_z = r exactly."""
    r = 20.0
    placements, unplaced = pack([sphere_item("a", r)], Box(200.0, 200.0, 200.0),
                                empty_ruleset())
    assert unplaced == []
    assert placements[0].centre[2] == pytest.approx(r, abs=1e-4)


def test_two_spheres_in_a_column_stack_to_three_r():
    """Box footprint admits one column only, so the second sphere rests at z = 3r."""
    r = 20.0
    box = Box(2 * r, 2 * r, 400.0)
    placements, unplaced = pack([sphere_item("a", r), sphere_item("b", r)], box,
                                empty_ruleset())
    assert unplaced == []
    zs = sorted(p.centre[2] for p in placements)
    assert zs[0] == pytest.approx(r, abs=1e-3)
    assert zs[1] == pytest.approx(3 * r, abs=1e-3)


def test_every_body_is_inside_the_box():
    box = Box(300.0, 300.0, 300.0)
    items = [sphere_item("s%d" % i, 25.0) for i in range(6)]
    placements, _ = pack(items, box, empty_ruleset())
    for p in placements:
        M = p.item.shape_matrix(p.yaw)
        inv = np.linalg.inv(M)
        for i, span in enumerate((box.width, box.depth, box.height)):
            e = float(np.sqrt(inv[i, i]))
            assert p.centre[i] - e >= -1e-6
            assert p.centre[i] + e <= span + 1e-6


def test_the_result_passes_the_overlap_gate():
    box = Box(300.0, 300.0, 300.0)
    items = [sphere_item("s%d" % i, 25.0) for i in range(8)]
    placements, _ = pack(items, box, empty_ruleset())
    assert max_penetration_mm(placements) <= TOLERANCE_MM


def test_packing_is_deterministic_for_a_seed():
    box = Box(300.0, 300.0, 300.0)
    items = [sphere_item("s%d" % i, 22.0) for i in range(5)]
    a, _ = pack(items, box, empty_ruleset(), seed=7)
    b, _ = pack(items, box, empty_ruleset(), seed=7)
    assert [p.centre.tolist() for p in a] == [p.centre.tolist() for p in b]


def test_max_weight_rule_limits_what_is_placed():
    box = Box(400.0, 400.0, 400.0)
    items = [sphere_item("s%d" % i, 20.0, mass_g=100.0) for i in range(3)]
    rs = empty_ruleset()
    rs["rules"] = [rule_max_weight(250.0, "keep it under 250 g")]
    placements, unplaced = pack(items, box, rs)
    assert sum(p.item.mass_g for p in placements) <= 250.0
    assert len(unplaced) >= 1


def test_nothing_is_placed_above_a_fragile_item():
    box = Box(2 * 20.0, 2 * 20.0, 400.0)          # single column, forces stacking
    rs = empty_ruleset()
    rs["rules"] = [rule_fragile("egg", "the egg is fragile")]
    items = [sphere_item("egg", 20.0), sphere_item("rock", 20.0)]
    placements, unplaced = pack(items, box, rs)
    egg = [p for p in placements if p.item.label == "egg"]
    if egg:
        top = egg[0].centre[2]
        for p in placements:
            if p.item.label != "egg":
                assert p.centre[2] <= top + 1e-6
