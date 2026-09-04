# T14 — Synthetic capture generator

DEPENDS ON: T01
CREATE: `src/capture/synth.py`
DO NOT MODIFY: any other file. Never `tests/`.

## Purpose

Produce a point cloud that looks like a single-orbit iPhone LiDAR scan of ellipsoids resting on
a table, **with exact ground truth**, so that T08 segmentation and T09 fit can be gated before
the real capture exists.

Read `docs/huggingface_lidar_datasets.md` first. The short version, no public dataset carries
the true semi-axes of an orange, so no public dataset can gate a fit. Synthetic data can,
because the axes are chosen rather than measured.

Budget 45 minutes. It unblocks three tasks.

## Exact signature

```python
@dataclass
class SynthScene:
    points: np.ndarray        # (N,3) in METRES, matching what a real capture delivers
    plane: np.ndarray         # (4,) unit normal a,b,c and offset d, so n.x + d = 0
    truth: list               # list of dicts, one per body, with the GROUND TRUTH

def synth_scene(bodies, noise_mm=10.0, bias_frac=-0.03, n_points=4000,
                seed=0, view_elev_deg=37.5) -> SynthScene:
    """Render a single-orbit capture of ellipsoids resting on the z = 0 plane.

    bodies : list of (axes_mm, centre_xy_mm) pairs. The body rests tangent to the
             plane, so its centre z equals its vertical semi-axis exactly.
    noise_mm  : per-point Gaussian sigma along the view ray. Default 10 mm is the
                centimetre-class figure from the sensor.
    bias_frac : systematic INWARD surface offset as a fraction of the local radius.
                Default -0.03 is the TSDF bias recorded in the prior notes. This is a
                bias, not noise, and it does not average away.
    n_points  : total points across the whole scene, table included.
    view_elev_deg : orbit elevation above the table. Points below this horizon on each
                body are NOT emitted, which is the single-view occlusion.

    Deterministic for a given seed. UNITS ARE METRES OUT, matching a real capture,
    while `bodies` is given in mm. The conversion happens here and nowhere else.
    """
```

Each `truth` entry is
`{"axes_mm": (a, b, c), "centre_mm": (x, y, z), "volume_mm3": float, "label": str}`.

## Algorithm, do not deviate

1. Emit table points on `z = 0` across the footprint bounding all bodies plus a 100 mm margin.
   Table points get the same `noise_mm`, because RANSAC has to cope with real table scatter.
2. For each body, sample surface points uniformly in the parameter domain, then **reject any
   point whose outward normal is below the view horizon**, `n_z < sin(view_elev_deg)`. This is
   the occlusion and it is the reason the fit needs plane tangency.
3. Apply the inward bias, scale each surviving surface point toward the body centre by
   `(1 + bias_frac)`.
4. Add Gaussian noise of `noise_mm` along the view direction, which is the vector from the
   point to a virtual sensor placed at the orbit elevation.
5. Convert mm to metres exactly once, dividing by `MM_PER_M` from T01.

## Forbidden

- **Do not emit a closed surface.** A body whose lower hemisphere is present removes the
  hardest part of the problem and turns T09 into a trivially solvable fit.
- Do not skip the bias. A generator with noise but no bias tests the wrong failure.
- No `open3d`. Numpy and the standard library only. This module must import with no heavy
  dependency so the gates run fast.
- Do not clip a body to the plane. It rests tangent, it does not intersect.

## Gate, run this, do not edit it

```
python -m pytest tests/test_synth.py -q
```

Expected: `8 passed`

## What this unblocks

| Task | Before T14 | After T14 |
|---|---|---|
| T08 segmentation | waits for the real fixture | gated now, with a KNOWN cluster count |
| T09 ellipsoid fit | waits for the real fixture | gated now, against KNOWN semi-axes |
| T11 error propagation | asserts an axis error from a datasheet | **sweeps `noise_mm` and MEASURES it** |

T11 is the important one. Sweeping the noise parameter converts the weakest claim in the pitch,
that the verdict is exact when the inputs carry a centimetre of error, into a measured curve.
