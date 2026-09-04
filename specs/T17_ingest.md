# T17 — Stray Scanner ingest, phone folder to point cloud

DEPENDS ON: T01
CREATE: `src/capture/ingest.py`
DO NOT MODIFY: any other file. Never `tests/`.

## The gap this fills

Every other spec starts from a `points` array and **nothing produced one**. This is the
missing first stage.

## The flow, three stages, and only the middle one is code

```
1. phone -> laptop      AirDrop, cable, or iCloud. Outside the app.
                        The phone cannot reach 127.0.0.1, so there is no
                        direct phone upload and there does not need to be.

2. folder -> .npy       THIS TASK. A CLI that runs once per scan.
                        python -m src.capture.ingest scans/raw/orange_set \
                               -o scans/orange_set.npy

3. .npy -> browser      T18. One small file, drag onto the viewport.
```

Stage 2 turns hundreds of megabytes into a few megabytes, once, on the laptop. Stage 3 then
moves a small single file, which a browser handles well.

## Exact signature

```python
def read_camera_matrix(path) -> np.ndarray:
    """Parse camera_matrix.csv into a (3,3) intrinsics matrix K."""

def backproject(depth_mm, K, stride=2, min_mm=150.0, max_mm=1500.0) -> np.ndarray:
    """One uint16 depth frame to (N,3) camera-frame points in METRES.

    For pixel (u, v) with depth d:
        x = (u - cx) * d / fx
        y = (v - cy) * d / fy
        z = d

    A depth of 0 means no return and MUST be dropped, never treated as z = 0.
    Depths outside [min_mm, max_mm] are dropped, which removes the far wall and
    sensor artefacts at the near limit.
    """

def voxel_downsample(points, voxel_m=0.002) -> np.ndarray:
    """Grid-average to at most one point per voxel. 2 mm default.

    A viewport with two million points is slow and the demo looks bad. This is a
    presentation requirement as much as a compute one.
    """

def ingest(folder, out_path=None, frames=1, stride=2, voxel_m=0.002) -> np.ndarray:
    """Read a Stray Scanner folder and return (N,3) points in METRES.

    frames=1 uses the single middle depth frame. See the note below, this is the
    DEFAULT and it is deliberate.
    """
```

## Single frame is the default, and that is not a shortcut

Fusing many frames needs the odometry poses, and pose drift over a 90 second orbit introduces
a systematic error that is hard to detect and easy to blame on the fit.

**A single frame is what the rest of the pipeline already expects.** T09 fits an upper cap
under a plane-tangency constraint precisely because one view sees only the upper surface. The
mirroring in T09 rebuilds the underside. So single-frame ingest matches the mathematics that
is already specified, and multi-frame fusion would give better coverage of a surface the fit
does not use.

Multi-frame is the stretch goal, `frames > 1` with poses from `odometry.csv`. Attempt it only
after the whole pipeline is green.

## Forbidden

- No `open3d`, it is not installable on Python 3.13.
- Do not treat depth 0 as a valid measurement at the origin. It is a no-return.
- Do not normalise, recentre or rescale. Absolute scale is the entire point of this project.
- Do not decode `rgb.mp4`. Colour is not used by the solver and a video decode dependency is
  a failure waiting to happen.

## Gate, run this, do not edit it

```
python -m pytest tests/test_ingest.py -q
```

Expected: `8 passed`

The gate builds its own synthetic depth frames, so it needs no real scan and runs before the
capture trip returns.

## Budget

45 minutes. Fallback if it slips, hand-convert one scan with a throwaway script, commit the
resulting `.npy` as the frozen fixture, and note in the README that ingest is manual.
