"""T17. Stray Scanner scan folder or .zip to a metric point cloud.

UNITS. The depth PNGs written by Stray Scanner are uint16 MILLIMETRES. Every
array returned by this module is in METRES, which is what `src.capture.segment`
and `src.capture.fit` expect. The millimetre boundary is crossed exactly once,
inside `backproject`, and never again.

INTRINSICS. `camera_matrix.csv` holds K for the RGB stream, 1920x1440 on an
iPhone Pro. The depth frames are 256x192, smaller by 7.5 in both axes. K must be
rescaled by the resolution ratio before it is used on a depth frame or every
back-projected point is wrong by that factor. `backproject` takes K as given and
does not guess, so the rescale lives in `ingest` where both resolutions are known.

No open3d, see CLAUDE.md. No rgb.mp4 decode, colour is not used by the solver.
"""
from __future__ import annotations

import io
import os
import zipfile
from typing import List, Optional, Sequence

import numpy as np
from PIL import Image

MM_PER_M = 1000.0

# ARKit camera axes are +x right, +y up, -z forward. The pinhole back-projection
# below produces the vision convention, +x right, +y down, +z forward. This flips
# between them. Used only when fusing frames through the odometry poses.
_CV_TO_ARKIT = np.diag([1.0, -1.0, -1.0])


class _Source:
    """Reads a scan that is either an unpacked folder or a .zip, same calls."""

    def __init__(self, path: str):
        self.path = path
        self.zf: Optional[zipfile.ZipFile] = None
        self.prefix = ""
        if os.path.isdir(path):
            return
        if not zipfile.is_zipfile(path):
            raise FileNotFoundError(f"not a scan folder or zip: {path}")
        self.zf = zipfile.ZipFile(path)
        names = self.zf.namelist()
        roots = {n.split("/", 1)[0] for n in names if "/" in n}
        if len(roots) == 1 and "camera_matrix.csv" not in names:
            self.prefix = roots.pop() + "/"

    def read(self, rel: str) -> bytes:
        if self.zf is None:
            with open(os.path.join(self.path, rel), "rb") as fh:
                return fh.read()
        return self.zf.read(self.prefix + rel)

    def listdir(self, rel: str) -> List[str]:
        if self.zf is None:
            return sorted(os.listdir(os.path.join(self.path, rel)))
        head = self.prefix + rel.rstrip("/") + "/"
        return sorted(
            n[len(head):] for n in self.zf.namelist()
            if n.startswith(head) and len(n) > len(head)
        )

    def close(self) -> None:
        if self.zf is not None:
            self.zf.close()


def read_camera_matrix(path) -> np.ndarray:
    """Parse camera_matrix.csv into a (3,3) intrinsics matrix K.

    The file is three comma separated rows. K is at the RGB resolution, see the
    module docstring.
    """
    with open(path, "r", encoding="utf-8") as fh:
        return _parse_camera_matrix(fh.read())


def _parse_camera_matrix(text: str) -> np.ndarray:
    rows = [r for r in text.strip().splitlines() if r.strip()]
    K = np.array([[float(v) for v in r.split(",")] for r in rows], dtype=float)
    if K.shape != (3, 3):
        raise ValueError(f"camera_matrix must be 3x3, got {K.shape}")
    return K


def scale_intrinsics(K: np.ndarray, from_wh: Sequence[int], to_wh: Sequence[int]) -> np.ndarray:
    """Rescale K from one image resolution to another. Returns a new (3,3)."""
    sx = float(to_wh[0]) / float(from_wh[0])
    sy = float(to_wh[1]) / float(from_wh[1])
    out = np.asarray(K, dtype=float).copy()
    out[0, 0] *= sx
    out[0, 2] *= sx
    out[1, 1] *= sy
    out[1, 2] *= sy
    return out


def backproject(depth_mm, K, stride=2, min_mm=150.0, max_mm=1500.0,
                confidence=None, min_confidence=0) -> np.ndarray:
    """One uint16 depth frame to (N,3) camera-frame points in METRES.

    For pixel (u, v) at depth d,
        x = (u - cx) * d / fx,  y = (v - cy) * d / fy,  z = d.

    A depth of 0 is a NO RETURN and is dropped. Treating it as z = 0 puts a
    phantom point at the camera and corrupts the plane fit. Depths outside
    [min_mm, max_mm] are dropped, which removes the far wall and the near-limit
    sensor artefacts.

    `confidence` is the matching Stray Scanner confidence frame, values 0, 1, 2.
    Pixels below `min_confidence` are dropped. The default 0 keeps every pixel.
    """
    depth = np.asarray(depth_mm)
    K = np.asarray(K, dtype=float)
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]

    sub = depth[::stride, ::stride].astype(np.float64)
    h, w = sub.shape
    vv, uu = np.meshgrid(
        np.arange(h, dtype=np.float64) * stride,
        np.arange(w, dtype=np.float64) * stride,
        indexing="ij",
    )

    keep = (sub > 0.0) & (sub >= float(min_mm)) & (sub <= float(max_mm))
    if confidence is not None and min_confidence > 0:
        keep &= np.asarray(confidence)[::stride, ::stride] >= min_confidence
    if not keep.any():
        return np.empty((0, 3), dtype=np.float64)

    d = sub[keep] / MM_PER_M
    u = uu[keep]
    v = vv[keep]
    return np.stack([(u - cx) * d / fx, (v - cy) * d / fy, d], axis=1)


def voxel_downsample(points, voxel_m=0.002) -> np.ndarray:
    """Grid-average to at most one point per voxel. 2 mm default.

    A viewport with two million points is slow and the demo looks bad, so this is
    a presentation requirement as much as a compute one. Input and output are
    both in metres. Absolute scale is preserved, nothing is recentred.
    """
    pts = np.asarray(points, dtype=np.float64)
    if len(pts) == 0:
        return pts.reshape(0, 3)
    idx = np.floor((pts - pts.min(axis=0)) / float(voxel_m)).astype(np.int64)
    _, inverse = np.unique(idx, axis=0, return_inverse=True)
    inverse = np.asarray(inverse).ravel()
    n = int(inverse.max()) + 1
    counts = np.bincount(inverse, minlength=n).astype(np.float64)
    return np.stack(
        [np.bincount(inverse, weights=pts[:, k], minlength=n) / counts for k in range(3)],
        axis=1,
    )


def _quat_to_matrix(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    """Unit quaternion to a (3,3) rotation. Stray Scanner order is qx qy qz qw."""
    q = np.array([qx, qy, qz, qw], dtype=float)
    q /= np.linalg.norm(q)
    x, y, z, w = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def read_odometry(src) -> dict:
    """frame index to (R, t), camera to world. Empty dict when there is no file."""
    if not isinstance(src, _Source):
        src = _Source(str(src))
    try:
        text = src.read("odometry.csv").decode("utf-8")
    except (KeyError, FileNotFoundError):
        return {}
    poses = {}
    for line in text.strip().splitlines()[1:]:
        f = line.split(",")
        if len(f) < 9:
            continue
        R = _quat_to_matrix(float(f[5]), float(f[6]), float(f[7]), float(f[8]))
        t = np.array([float(f[2]), float(f[3]), float(f[4])])
        poses[int(f[1])] = (R, t)
    return poses


def _png(data: bytes) -> np.ndarray:
    return np.array(Image.open(io.BytesIO(data)))


def ingest(folder, out_path=None, frames=1, stride=2, voxel_m=0.002,
           min_mm=150.0, max_mm=1500.0, min_confidence=0,
           frame_indices=None) -> np.ndarray:
    """Read a Stray Scanner folder or .zip and return (N,3) points in METRES.

    frames=1 uses the single middle depth frame, the deliberate default. T09 fits
    an upper cap under a plane-tangency constraint precisely because one view
    sees only the upper surface, so single-frame ingest matches the mathematics
    the rest of the pipeline already assumes.

    frames>1 fuses evenly spaced frames through the odometry poses, giving a
    world-frame cloud. Pose drift over a long orbit is a systematic error, so
    check the plane fit residual before trusting a fused cloud.
    """
    src = _Source(str(folder))
    try:
        K_rgb = _parse_camera_matrix(src.read("camera_matrix.csv").decode("utf-8"))
        names = [n for n in src.listdir("depth") if n.endswith(".png")]
        if not names:
            raise FileNotFoundError(f"no depth frames in {folder}")

        if frame_indices is not None:
            picks = [names[i] for i in frame_indices]
        elif frames <= 1:
            picks = [names[len(names) // 2]]
        else:
            take = min(int(frames), len(names))
            picks = [names[i] for i in np.linspace(0, len(names) - 1, take).astype(int)]

        poses = read_odometry(src) if len(picks) > 1 else {}
        chunks = []
        for name in picks:
            depth = _png(src.read(f"depth/{name}"))
            conf = None
            if min_confidence > 0:
                try:
                    conf = _png(src.read(f"confidence/{name}"))
                except (KeyError, FileNotFoundError):
                    conf = None
            # K is at the RGB resolution, the depth frame is not. Rescale first.
            K = scale_intrinsics(K_rgb, (1920, 1440), (depth.shape[1], depth.shape[0]))
            pts = backproject(depth, K, stride=stride, min_mm=min_mm, max_mm=max_mm,
                              confidence=conf, min_confidence=min_confidence)
            if len(picks) > 1:
                idx = int(os.path.splitext(name)[0])
                if idx not in poses:
                    continue
                R, t = poses[idx]
                pts = (R @ (_CV_TO_ARKIT @ pts.T)).T + t
            chunks.append(pts)

        points = np.vstack(chunks) if chunks else np.empty((0, 3))
        points = voxel_downsample(points, voxel_m=voxel_m)
        if out_path:
            os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
            np.save(out_path, points)
        return points
    finally:
        src.close()


def _main(argv=None) -> int:
    import argparse

    p = argparse.ArgumentParser(description="Stray Scanner folder or zip to .npy, metres.")
    p.add_argument("folder")
    p.add_argument("-o", "--out", default=None)
    p.add_argument("--frames", type=int, default=1)
    p.add_argument("--stride", type=int, default=2)
    p.add_argument("--voxel", type=float, default=0.002)
    p.add_argument("--min-confidence", type=int, default=0)
    a = p.parse_args(argv)
    pts = ingest(a.folder, out_path=a.out, frames=a.frames, stride=a.stride,
                 voxel_m=a.voxel, min_confidence=a.min_confidence)
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    print(f"{len(pts)} points, metres")
    print(f"  x {lo[0]:+.3f} to {hi[0]:+.3f}   y {lo[1]:+.3f} to {hi[1]:+.3f}   "
          f"z {lo[2]:+.3f} to {hi[2]:+.3f}")
    if a.out:
        print(f"  wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
