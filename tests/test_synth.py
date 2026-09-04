"""Gate for T14. Written before the implementation. Do not edit."""
import math

import numpy as np
import pytest

from src.types import MM_PER_M
from src.capture.synth import SynthScene, synth_scene


def one_sphere(r_mm=30.0, noise_mm=0.0, bias_frac=0.0, **kw):
    return synth_scene([((r_mm, r_mm, r_mm), (0.0, 0.0))],
                       noise_mm=noise_mm, bias_frac=bias_frac, **kw)


def test_output_is_in_metres_not_millimetres():
    """A 30 mm sphere spans 0.06 m. A generator that forgot the conversion spans 60."""
    s = one_sphere(30.0)
    span = float(np.ptp(s.points[:, 0]))
    assert span < 1.0, "points look like mm, not m"
    assert MM_PER_M == 1000.0


def test_body_rests_tangent_to_the_plane():
    """Ground truth centre z equals the vertical semi-axis exactly."""
    s = one_sphere(30.0)
    t = s.truth[0]
    assert t["centre_mm"][2] == pytest.approx(t["axes_mm"][2], rel=1e-12)


def test_ground_truth_volume_matches_the_closed_form():
    a, b, c = 40.0, 25.0, 15.0
    s = synth_scene([((a, b, c), (0.0, 0.0))], noise_mm=0.0, bias_frac=0.0)
    expected = 4.0 / 3.0 * math.pi * a * b * c
    assert s.truth[0]["volume_mm3"] == pytest.approx(expected, rel=1e-12)


def test_no_points_exist_below_the_view_horizon():
    """Single-view occlusion. A closed surface would make T09 trivial and is forbidden.

    With zero noise and zero bias, every body point must sit in the upper part of the
    body, well above the tangent contact at z = 0.
    """
    r = 30.0
    s = one_sphere(r)
    body = s.points[s.points[:, 2] > 1e-6] * MM_PER_M       # drop the table, back to mm
    assert len(body) > 50
    assert body[:, 2].min() > 0.2 * r, "lower cap was emitted, occlusion not applied"


def test_zero_noise_points_lie_exactly_on_the_surface():
    """With no noise and no bias, every body point satisfies the ellipsoid equation."""
    a, b, c = 40.0, 25.0, 15.0
    s = synth_scene([((a, b, c), (0.0, 0.0))], noise_mm=0.0, bias_frac=0.0)
    pts = s.points * MM_PER_M
    body = pts[pts[:, 2] > 1e-6]
    cx, cy, cz = s.truth[0]["centre_mm"]
    q = (((body[:, 0] - cx) / a) ** 2 + ((body[:, 1] - cy) / b) ** 2
         + ((body[:, 2] - cz) / c) ** 2)
    assert np.allclose(q, 1.0, atol=1e-6)


def test_inward_bias_shrinks_the_apparent_body():
    """-3% bias must pull points inside the true surface, systematically, not on average."""
    r = 30.0
    s = synth_scene([((r, r, r), (0.0, 0.0))], noise_mm=0.0, bias_frac=-0.03)
    pts = s.points * MM_PER_M
    body = pts[pts[:, 2] > 1e-6]
    cx, cy, cz = s.truth[0]["centre_mm"]
    radii = np.linalg.norm(body - np.array([cx, cy, cz]), axis=1)
    assert np.allclose(radii, r * 0.97, rtol=1e-6)


def test_noise_sigma_is_what_was_requested():
    """Residual scatter about the true surface must equal the requested sigma.

    Body points are selected by DISTANCE FROM THE KNOWN CENTRE, not by z > 0.
    At 8 mm of table noise a large share of table points clear z = 0, and
    including them inflated the measured residual from 8 mm to 28.7 mm. The
    generator was right and the earlier selection was wrong.
    """
    r, sigma = 30.0, 8.0
    s = synth_scene([((r, r, r), (0.0, 0.0))], noise_mm=sigma, bias_frac=0.0,
                    n_points=20000, seed=3)
    pts = s.points * MM_PER_M
    cx, cy, cz = s.truth[0]["centre_mm"]
    dist = np.linalg.norm(pts - np.array([cx, cy, cz]), axis=1)
    body = pts[dist < 2.0 * r]
    assert len(body) > 500
    resid = np.linalg.norm(body - np.array([cx, cy, cz]), axis=1) - r
    assert resid.std() == pytest.approx(sigma, rel=0.25)


def test_two_bodies_closer_than_the_dbscan_eps_can_be_generated():
    """The merge failure must be TESTABLE, not discovered on real data at 8 PM.

    DBSCAN runs at eps 12 mm. Bodies 5 mm apart must be producible so T08 can assert
    the merge, and bodies 40 mm apart so T08 can assert the clean split.
    """
    r = 20.0
    near = synth_scene([((r, r, r), (0.0, 0.0)), ((r, r, r), (2 * r + 5.0, 0.0))],
                       noise_mm=0.0, bias_frac=0.0)
    far = synth_scene([((r, r, r), (0.0, 0.0)), ((r, r, r), (2 * r + 40.0, 0.0))],
                      noise_mm=0.0, bias_frac=0.0)
    assert len(near.truth) == 2 and len(far.truth) == 2
    gap_near = near.truth[1]["centre_mm"][0] - near.truth[0]["centre_mm"][0] - 2 * r
    gap_far = far.truth[1]["centre_mm"][0] - far.truth[0]["centre_mm"][0] - 2 * r
    assert gap_near == pytest.approx(5.0, abs=1e-9)
    assert gap_far == pytest.approx(40.0, abs=1e-9)
