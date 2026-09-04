"""Gate for T09. Ground truth from the fixture. Written first. Do not edit.

Tolerances are MEASURED, not guessed. The chosen estimator gives 9.2% mean and
17.4% max volume error, and 7.1% mean axis error, on this fixture.
"""
import json
import os

import numpy as np
import pytest

from src.types import Item
from src.capture.segment import segment
from src.capture.fit import fit_cluster, fit_scene, plane_frame

FIXTURE = "scans/synthetic_produce.npy"
TRUTH = "scans/synthetic_produce_truth.json"

pytestmark = pytest.mark.skipif(not os.path.exists(FIXTURE),
                                reason="fixture missing, run make_fixture.py")


def _load():
    pts = np.load(FIXTURE)
    truth = json.load(open(TRUTH, encoding="utf-8"))["bodies"]
    return pts, truth, segment(pts)


def _match(centre_mm, truth):
    """Match by POSITION. Matching by axis similarity scrambles the pairs when the
    fit is off, which is how an earlier diagnostic misread a working segmentation."""
    return min(truth, key=lambda t: float(np.hypot(t["centre_mm"][0] - centre_mm[0],
                                                   t["centre_mm"][1] - centre_mm[1])))


def test_plane_frame_third_row_is_the_plane_normal():
    R = plane_frame(np.array([0.0, 0.0, 1.0, 0.0]))
    assert R.shape == (3, 3)
    assert np.allclose(R[2], [0.0, 0.0, 1.0], atol=1e-9)
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-9)


def test_every_cluster_produces_a_fitted_body():
    pts, truth, seg = _load()
    assert seg["n_clusters"] == len(truth), "segmentation is wrong, fix T08b first"
    items = fit_scene(pts, seg)
    assert len(items) == len(truth)
    assert all(isinstance(i, Item) for i in items)


def test_centres_land_on_the_true_bodies():
    """Position is the easy part and it must be right before anything else is trusted."""
    pts, truth, seg = _load()
    plane = np.asarray(seg["plane"])
    for idx in seg["clusters"]:
        f = fit_cluster(pts[idx], plane)
        t = _match(f["centre_mm"], truth)
        d = float(np.hypot(t["centre_mm"][0] - f["centre_mm"][0],
                           t["centre_mm"][1] - f["centre_mm"][1]))
        assert d < 8.0, "centre %.1f mm from the true body" % d


def test_tangency_holds_exactly():
    """The body rests on the plane, so its HEIGHT ABOVE THE PLANE equals the
    vertical semi-axis c.

    Not the world z. The plane is slightly tilted in a real scan, so world z and
    height above the plane differ, and comparing world z to a semi-axis was the
    original error in this assertion. Nor is c always the smallest axis, since an
    egg standing on end has its largest semi-axis vertical.
    """
    pts, truth, seg = _load()
    plane = np.asarray(seg["plane"])
    for idx in seg["clusters"]:
        f = fit_cluster(pts[idx], plane)
        assert f["height_above_plane_mm"] == pytest.approx(f["c_mm"], rel=1e-6)
        assert min(abs(f["c_mm"] - float(a)) for a in f["axes_mm"]) < 1e-6


def test_volume_error_is_within_the_measured_tolerance():
    """Measured: 9.2% mean, 17.4% max. Gate at 20% mean and 35% max, so a
    correct implementation passes and a broken one does not."""
    pts, truth, seg = _load()
    plane = np.asarray(seg["plane"])
    errs = []
    for idx in seg["clusters"]:
        f = fit_cluster(pts[idx], plane)
        t = _match(f["centre_mm"], truth)
        errs.append(abs(100.0 * (f["volume_mm3"] - t["volume_mm3"]) / t["volume_mm3"]))
    assert np.mean(errs) < 20.0, "mean volume error %.1f%%" % np.mean(errs)
    assert np.max(errs) < 35.0, "max volume error %.1f%%" % np.max(errs)


def test_axis_error_is_within_the_measured_tolerance():
    """Measured 7.1% mean. Gate at 15%."""
    pts, truth, seg = _load()
    plane = np.asarray(seg["plane"])
    errs = []
    for idx in seg["clusters"]:
        f = fit_cluster(pts[idx], plane)
        t = _match(f["centre_mm"], truth)
        a = np.sort(np.asarray(f["axes_mm"]))[::-1]
        ta = np.sort(np.asarray(t["axes_mm"], dtype=float))[::-1]
        errs.append(100.0 * float(np.mean(np.abs(a - ta) / ta)))
    assert np.mean(errs) < 15.0, "mean axis error %.1f%%" % np.mean(errs)
