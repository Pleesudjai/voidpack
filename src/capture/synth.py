"""Synthetic single-orbit capture with exact ground truth.

Exists because no public dataset carries the true semi-axes of an orange, so no
public dataset can gate a fit. Here the axes are chosen, so they are known.

Deliberately adversarial. A generator that emits clean closed surfaces passes and
then real data fails at 8 PM, so this one injects the three failure modes that
actually occur: single-view occlusion, centimetre-class noise, and the systematic
inward TSDF bias.
"""
import math
from dataclasses import dataclass, field

import numpy as np

from src.types import MM_PER_M


@dataclass
class SynthScene:
    points: np.ndarray                 # (N,3) in METRES, as a real capture delivers
    plane: np.ndarray                  # (4,) unit normal and offset, n.x + d = 0
    truth: list = field(default_factory=list)


def synth_scene(bodies, noise_mm=10.0, bias_frac=-0.03, n_points=4000,
                seed=0, view_elev_deg=37.5) -> SynthScene:
    """Render a single-orbit capture of ellipsoids resting on the z = 0 plane.

    bodies : list of (axes_mm, (centre_x_mm, centre_y_mm)). Each body rests
             tangent to the plane, so its centre z equals its vertical semi-axis.
    noise_mm  : per-point Gaussian sigma. 10 mm is the centimetre-class figure.
    bias_frac : systematic INWARD surface offset as a fraction of the local
                radius. -0.03 is the TSDF bias. A bias, not noise, and it does
                not average away.
    n_points  : total across the scene. THE TABLE TAKES HALF. RANSAC is a
                majority vote and a table in the minority loses the plane fit.
    view_elev_deg : orbit elevation. A surface point is visible from some azimuth
                during a full orbit when its polar angle is under 90 + elevation,
                which is n_z > -sin(elevation). The minus sign matters: the
                positive form keeps only a 9 mm polar cap on a 44 mm sphere, and
                nine thousand points in a 9 mm slab look like a plane to RANSAC.

    Deterministic for a given seed. Output is METRES, input is millimetres, and
    the conversion happens here and nowhere else.
    """
    rng = np.random.default_rng(seed)
    elev = math.radians(view_elev_deg)
    keep_nz = -math.sin(elev)

    bodies = list(bodies)
    if not bodies:
        raise ValueError("synth_scene needs at least one body")

    xs = [c[0] for _, c in bodies]
    ys = [c[1] for _, c in bodies]

    n_table = max(1, n_points // 2)
    tx = rng.uniform(min(xs) - 100.0, max(xs) + 100.0, n_table)
    ty = rng.uniform(min(ys) - 100.0, max(ys) + 100.0, n_table)
    tz = rng.normal(0.0, noise_mm, n_table)
    chunks = [np.c_[tx, ty, tz]]

    per = max(1, (n_points - n_table) // len(bodies))
    truth = []
    for i, (axes, (cx, cy)) in enumerate(bodies):
        a, b, c = (float(v) for v in axes)
        u = rng.normal(size=(per * 12, 3))
        u /= np.linalg.norm(u, axis=1, keepdims=True)
        u = u[u[:, 2] > keep_nz][:per]              # single-view occlusion
        p = u * np.array([a, b, c]) * (1.0 + bias_frac)   # inward bias, then
        if noise_mm > 0.0:
            p = p + rng.normal(0.0, noise_mm, p.shape)    # noise, in that order
        p = p + np.array([cx, cy, c])               # tangent: centre z equals c
        chunks.append(p)
        truth.append({"label": "body_%d" % i,
                      "axes_mm": (a, b, c),
                      "centre_mm": (cx, cy, c),
                      "volume_mm3": 4.0 / 3.0 * math.pi * a * b * c})

    pts_mm = np.vstack(chunks)
    return SynthScene(points=pts_mm / MM_PER_M,
                      plane=np.array([0.0, 0.0, 1.0, 0.0]),
                      truth=truth)
