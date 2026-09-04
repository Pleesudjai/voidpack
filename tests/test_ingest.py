"""Gate for T17. Builds its own depth frames, needs no real scan. Do not edit."""
import math

import numpy as np
import pytest

from src.capture.ingest import backproject, ingest, read_camera_matrix, voxel_downsample


FX, FY, CX, CY = 600.0, 600.0, 320.0, 240.0
K = np.array([[FX, 0.0, CX], [0.0, FY, CY], [0.0, 0.0, 1.0]])


def flat_depth(w=640, h=480, d_mm=500):
    """A fronto-parallel wall at a constant depth."""
    return np.full((h, w), d_mm, dtype=np.uint16)


def test_backprojection_of_a_flat_wall_lands_at_the_right_depth():
    """Closed form. A constant-depth frame maps to a plane at z = d, in metres."""
    pts = backproject(flat_depth(d_mm=500), K, stride=8)
    assert len(pts) > 100
    assert np.allclose(pts[:, 2], 0.500, atol=1e-9)


def test_pixel_geometry_matches_the_pinhole_closed_form():
    """A single pixel offset from the principal point maps to x = (u-cx) d / fx."""
    d = np.zeros((480, 640), dtype=np.uint16)
    d[240, 420] = 500                      # v = cy, u = cx + 100
    pts = backproject(d, K, stride=1)
    assert len(pts) == 1
    assert pts[0, 0] == pytest.approx((420 - CX) * 0.500 / FX, rel=1e-9)
    assert pts[0, 1] == pytest.approx((240 - CY) * 0.500 / FY, abs=1e-12)
    assert pts[0, 2] == pytest.approx(0.500, rel=1e-12)


def test_zero_depth_is_a_no_return_and_is_dropped():
    """A zero treated as z = 0 puts a phantom point at the camera. This is the
    single most common ingest bug and it corrupts the plane fit."""
    d = np.zeros((480, 640), dtype=np.uint16)
    d[100:110, 100:110] = 500
    pts = backproject(d, K, stride=1)
    assert len(pts) == 100
    assert (pts[:, 2] > 0.1).all()


def test_depths_outside_the_working_range_are_dropped():
    d = np.zeros((480, 640), dtype=np.uint16)
    d[10, 10] = 50            # too near
    d[20, 20] = 5000          # too far, the room behind the table
    d[30, 30] = 500           # good
    pts = backproject(d, K, stride=1, min_mm=150.0, max_mm=1500.0)
    assert len(pts) == 1
    assert pts[0, 2] == pytest.approx(0.500, rel=1e-12)


def test_output_is_metres_not_millimetres():
    pts = backproject(flat_depth(d_mm=500), K, stride=16)
    assert pts[:, 2].max() < 1.0, "looks like millimetres"


def test_voxel_downsample_cannot_exceed_the_cell_count():
    """Hard geometric bound, not an empirical guess. The output can never contain
    more points than there are cells spanning the bounding box.

    60000 points uniform in a 0.20 m cube at 0.01 m voxels: at most 20^3 = 8000
    cells, so at least a 7.5x reduction. Verified independently, got 7993.

    A 0.005 m voxel over the same cloud gives 64000 cells and only a 1.5x
    reduction, which is why the voxel size and the extent must be reasoned about
    together rather than assumed.
    """
    rng = np.random.default_rng(0)
    pts = rng.uniform(0.0, 0.20, size=(60000, 3))
    voxel_m = 0.01
    out = voxel_downsample(pts, voxel_m=voxel_m)
    max_cells = math.ceil(0.20 / voxel_m) ** 3
    assert len(out) <= max_cells
    assert len(out) < len(pts) / 5
    assert np.allclose(out.min(axis=0), pts.min(axis=0), atol=0.02)
    assert np.allclose(out.max(axis=0), pts.max(axis=0), atol=0.02)


def test_voxel_downsample_of_one_voxel_returns_one_point():
    pts = np.array([[0.0, 0.0, 0.0], [0.001, 0.001, 0.001], [0.0005, 0.0, 0.0]])
    assert len(voxel_downsample(pts, voxel_m=0.01)) == 1


def test_camera_matrix_is_parsed_into_intrinsics(tmp_path):
    p = tmp_path / "camera_matrix.csv"
    p.write_text("600.0,0.0,320.0\n0.0,600.0,240.0\n0.0,0.0,1.0\n", encoding="utf-8")
    out = read_camera_matrix(str(p))
    assert out.shape == (3, 3)
    assert out[0, 0] == pytest.approx(600.0)
    assert out[0, 2] == pytest.approx(320.0)
