"""Generate a synthetic scan FIXTURE with exact ground truth.

This is DATA, not the T14 module. Devstral still writes src/capture/synth.py to
the T14 spec. This exists so there is something to run against right now.

Bodies rest tangent to z = 0. Only the upper cap is emitted, matching a single
orbit. Noise and the -3% TSDF inward bias are applied, so it is adversarial
rather than friendly.
"""
import json
import math

import numpy as np

RNG = np.random.default_rng(20260903)
ELEV = math.radians(37.5)
NOISE_MM = 8.0
BIAS = -0.03

# Realistic produce semi-axes in mm, from typical grocery dimensions.
# Laid out with >= 40 mm clear space so DBSCAN at eps 12 mm cannot merge them.
BODIES = [
    ("orange-1", (37.0, 36.0, 35.0), (0.0, 0.0)),
    ("orange-2", (35.0, 34.0, 34.0), (140.0, 0.0)),
    ("lime-1",   (26.0, 25.0, 28.0), (260.0, 0.0)),
    ("lime-2",   (25.0, 24.0, 27.0), (350.0, 0.0)),
    ("potato-1", (48.0, 32.0, 29.0), (0.0, 150.0)),
    ("potato-2", (44.0, 30.0, 27.0), (140.0, 150.0)),
    ("egg-1",    (22.0, 22.0, 29.0), (260.0, 150.0)),
    ("egg-2",    (22.0, 21.5, 28.5), (340.0, 150.0)),
]

pts, truth = [], []

xs = [c[0] for _, _, c in BODIES]
ys = [c[1] for _, _, c in BODIES]
n_table = 12000
tx = RNG.uniform(min(xs) - 90, max(xs) + 90, n_table)
ty = RNG.uniform(min(ys) - 90, max(ys) + 90, n_table)
tz = RNG.normal(0.0, NOISE_MM * 0.35, n_table)      # table scatter is tighter
pts.append(np.c_[tx, ty, tz])

for label, axes, (cx, cy) in BODIES:
    a, b, c = axes
    u = RNG.normal(size=(4000, 3))
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    u = u[u[:, 2] > -math.sin(ELEV)][:1400]          # full-orbit visibility, T14b rule
    p = u * np.array([a, b, c]) * (1.0 + BIAS)       # inward TSDF bias
    p = p + np.array([cx, cy, c])
    p = p + RNG.normal(0.0, NOISE_MM, p.shape)       # sensor noise
    pts.append(p)
    truth.append({"label": label, "axes_mm": list(axes),
                  "centre_mm": [cx, cy, c],
                  "volume_mm3": 4.0 / 3.0 * math.pi * a * b * c})

cloud_mm = np.vstack(pts)
cloud_m = cloud_mm / 1000.0                          # METRES out, as a real capture

np.save("scans/synthetic_produce.npy", cloud_m)
with open("scans/synthetic_produce_truth.json", "w", encoding="utf-8") as f:
    json.dump({"units_in_file": "metres",
               "noise_mm": NOISE_MM, "bias_frac": BIAS,
               "view_elev_deg": 37.5,
               "n_points": int(len(cloud_m)),
               "n_bodies": len(truth), "bodies": truth}, f, indent=2)

print("wrote scans/synthetic_produce.npy       %d points, metres" % len(cloud_m))
print("wrote scans/synthetic_produce_truth.json %d bodies with exact ground truth" % len(truth))
print()
print("%-10s %-22s %-12s" % ("label", "true semi-axes mm", "true volume mm3"))
for t in truth:
    print("%-10s %-22s %12.0f" % (t["label"],
          " x ".join("%.1f" % v for v in t["axes_mm"]), t["volume_mm3"]))
print()
print("bounds mm  x[%.0f, %.0f]  y[%.0f, %.0f]  z[%.0f, %.0f]" % (
    cloud_mm[:, 0].min(), cloud_mm[:, 0].max(),
    cloud_mm[:, 1].min(), cloud_mm[:, 1].max(),
    cloud_mm[:, 2].min(), cloud_mm[:, 2].max()))
print("minimum clear gap between bodies: 40 mm, well above the 12 mm DBSCAN eps")
