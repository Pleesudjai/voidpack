"""Gate for T08. Ground truth comes from T14. Written first. Do not edit."""
import math

import numpy as np
import pytest

from src.types import MM_PER_M
from src.capture.synth import synth_scene
from src.capture.segment import (cluster_objects, dbscan, fit_plane_ransac,
                                 points_above_plane, segment)


def scene(bodies, **kw):
    kw.setdefault("noise_mm", 2.0)
    kw.setdefault("bias_frac", 0.0)
    return synth_scene(bodies, **kw)


def test_ransac_recovers_the_table_normal():
    """Ground truth plane is z = 0, so the unit normal must be +/- z."""
    s = scene([((30.0, 30.0, 30.0), (0.0, 0.0))])
    plane, inliers = fit_plane_ransac(s.points)
    assert abs(abs(float(plane[2])) - 1.0) < 1e-2
    assert abs(float(plane[3])) < 0.01          # offset near zero, in metres
    assert inliers.sum() > 100


def test_normal_is_oriented_so_objects_are_above():
    """The table scatter splits evenly, so only non-inlier points may vote."""
    s = scene([((30.0, 30.0, 30.0), (0.0, 0.0))])
    plane, inliers = fit_plane_ransac(s.points)
    signed = s.points @ plane[:3] + plane[3]
    assert signed[~inliers].mean() > 0, "normal points into the table"


def test_height_band_keeps_the_body_and_drops_the_table():
    s = scene([((30.0, 30.0, 30.0), (0.0, 0.0))])
    plane, _ = fit_plane_ransac(s.points)
    mask = points_above_plane(s.points, plane)
    kept = s.points[mask] * MM_PER_M
    assert mask.sum() > 50
    assert kept[:, 2].min() > 5.0


def test_dbscan_labels_two_well_separated_blobs():
    rng = np.random.default_rng(0)
    a = rng.normal(0.0, 0.002, size=(200, 3))
    b = rng.normal(0.0, 0.002, size=(200, 3)) + np.array([0.5, 0, 0])
    labels = dbscan(np.vstack([a, b]), eps_m=0.012, min_points=12)
    assert len(set(labels[labels >= 0])) == 2


def test_two_bodies_forty_mm_apart_split_into_two_clusters():
    r = 20.0
    s = scene([((r, r, r), (0.0, 0.0)), ((r, r, r), (2 * r + 40.0, 0.0))])
    out = segment(s.points, min_cluster_points=50)
    assert out["n_clusters"] == 2


def test_two_bodies_five_mm_apart_MERGE_into_one_cluster():
    """This is the documented failure, and it is why the capture protocol demands
    20 mm of clear space. DBSCAN eps is 12 mm, so a 5 mm gap bridges."""
    r = 20.0
    s = scene([((r, r, r), (0.0, 0.0)), ((r, r, r), (2 * r + 5.0, 0.0))])
    out = segment(s.points, min_cluster_points=50)
    assert out["n_clusters"] == 1


def test_a_dropped_small_cluster_raises_a_warning_rather_than_vanishing():
    """A silently dropped object corrupts every downstream number with no diagnostic."""
    r = 20.0
    s = scene([((r, r, r), (0.0, 0.0)), ((r, r, r), (2 * r + 40.0, 0.0))],
              n_points=1200)
    out = segment(s.points, min_cluster_points=100000)
    assert out["n_clusters"] == 0
    assert out["warnings"], "clusters were discarded with no warning"


def test_cluster_count_matches_ground_truth_for_three_bodies():
    r = 22.0
    bodies = [((r, r, r), (0.0, 0.0)),
              ((r, r, r), (2 * r + 40.0, 0.0)),
              ((r, r, r), (0.0, 2 * r + 40.0))]
    s = scene(bodies, n_points=9000)
    out = segment(s.points, min_cluster_points=50)
    assert out["n_clusters"] == len(s.truth) == 3
