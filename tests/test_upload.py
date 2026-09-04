"""Gate for T18. Written before the implementation. Do not edit."""
import io as _io

import numpy as np
import pytest

from src.air.upload import ALLOWED, MAX_BYTES, handle_upload, parse_cloud


def npy_bytes(arr):
    buf = _io.BytesIO()
    np.save(buf, arr, allow_pickle=False)
    return buf.getvalue()


def test_npy_round_trip():
    pts = np.random.default_rng(0).uniform(0, 0.2, size=(500, 3))
    out = parse_cloud("scan.npy", npy_bytes(pts))
    assert out.shape == (500, 3)
    assert np.allclose(out, pts)


def test_csv_with_and_without_a_header():
    body = b"x,y,z\n0.1,0.2,0.3\n0.4,0.5,0.6\n"
    assert parse_cloud("s.csv", body).shape == (2, 3)
    assert parse_cloud("s.csv", b"0.1,0.2,0.3\n0.4,0.5,0.6\n").shape == (2, 3)


def test_ascii_ply_is_parsed():
    ply = (b"ply\nformat ascii 1.0\nelement vertex 2\n"
           b"property float x\nproperty float y\nproperty float z\n"
           b"end_header\n0.1 0.2 0.3\n0.4 0.5 0.6\n")
    assert parse_cloud("s.ply", ply).shape == (2, 3)


def test_unknown_extension_is_rejected_by_name():
    out = handle_upload("scan.exe", b"whatever")
    assert "error" in out
    assert ".exe" in out["error"] or "extension" in out["error"].lower()
    assert set(ALLOWED) == {".npy", ".ply", ".csv"}


def test_oversized_upload_is_rejected():
    out = handle_upload("big.npy", b"0" * (MAX_BYTES + 1))
    assert "error" in out
    assert MAX_BYTES == 50 * 1024 * 1024


def test_path_traversal_in_the_name_is_rejected():
    for bad in ("../../etc/passwd.npy", r"..\..\win.npy", "/abs/path.npy", "C:\\x.npy"):
        out = handle_upload(bad, npy_bytes(np.zeros((3, 3))))
        assert "error" in out, "accepted a traversal name: %s" % bad


def test_a_transposed_array_is_rejected_with_that_message():
    """A silent transpose corrupts every downstream number. Say what is wrong."""
    out = handle_upload("t.npy", npy_bytes(np.zeros((3, 500))))
    assert "error" in out
    assert "transpose" in out["error"].lower() or "(N, 3)" in out["error"]


def test_millimetre_input_is_detected_and_converted_not_guessed_silently():
    """A 300 mm box in a file labelled metres has a diagonal of 300, not 0.3."""
    pts = np.random.default_rng(0).uniform(0, 300.0, size=(200, 3))
    out = handle_upload("mm.npy", npy_bytes(pts))
    assert out.get("units_converted") is True
    assert out["bounds_mm"][1][0] == pytest.approx(pts[:, 0].max(), rel=0.05)


def test_large_cloud_is_downsampled_for_the_viewport():
    pts = np.random.default_rng(0).uniform(0, 0.2, size=(400000, 3))
    out = handle_upload("big.npy", npy_bytes(pts), target_points=150000)
    assert out["n_points"] == 400000
    assert out["downsampled_to"] <= 150000
