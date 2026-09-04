"""Compose the four extracted LiDAR groceries into one coloured table scene.

WHY A COMPOSED SCENE. Each capture holds ONE item, held in the air on a stick, so
there is no scene to segment and no table to sit on. The four items are laid out
on a synthetic table plane and written as a single cloud in the format
`src.capture.segment` already reads. The in-plane geometry and every surface
colour are measured. The table, the spacing and the third axis are constructed,
and the README says so.

THE THIRD AXIS IS NOT OBSERVABLE FROM ONE VIEW. What the sensor returns is a cap,
the surface facing the lens, and its rim is the item silhouette. The depth of that
cap is bounded by how far the near band can open before it swallows the hand
behind the item, so it under-measures. Left alone the grapefruit fitted as a 47 mm
tall disc against 94 x 82 mm across. Each cap is therefore scaled about its own
rim plane until the body half-height equals half the measured MINOR in-plane
extent, that is until the cross-section about the long axis is circular. That is
an ASSUMPTION, approved by the author on 2026-09-03, and it is right for these
four items, an egg, a grapefruit, an avocado and a potato, which are all bodies of
near revolution about their long axis.

COLOUR. The depth stream and the RGB stream come from the same camera, 256x192
and 1920x1440, an exact 7.5 in both axes, so a depth pixel maps into the video
frame by that scale alone. Colour is carried for DISPLAY only. The solver never
reads it, per the architecture rule that every number comes from geometry.

UNITS. Metres in the file, matching `scans/synthetic_produce.npy`.

    python tools/compose_scene.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.capture.ingest import voxel_downsample  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

GAP_MM = 55.0             # at least the 20 mm the capture rules ask between items
TABLE_MARGIN_MM = 70.0
# 2.7 mm. The resampled item surfaces carry 14826 points, and the capture rules
# require the table to be the majority of the cloud or RANSAC fits an item as
# the plane. Measured, a 3.0 mm pitch gives 14784 table points and loses that.
TABLE_PITCH_MM = 2.7
TABLE_NOISE_MM = 1.5      # so RANSAC sees a band and not a perfect plane
TABLE_RGB = (198, 196, 192)
# 3.0 mm, chosen by sweep this session and NOT by eye. `segment` sets its DBSCAN
# eps to 3.0 times the median spacing of the whole cloud, so a fine item cloud
# drives eps down until each body fragments. Measured, an item voxel of 2.0 mm
# gave eps 4.6 mm and 8 clusters for 4 bodies, 2.5 mm gave 5.4 mm and 7 clusters,
# 3.0 mm gives 6.4 mm and exactly 4. Holding eps by hand would not help, because
# `src.air.bridge` calls `segment(pts)` with its defaults.
ITEM_VOXEL_MM = 3.0
RIM_PCT = 4.0             # rim plane percentile, see close_body
# Surface resampling. The measured cloud has complete directional coverage, the
# worst angular gap is 21 degrees on the egg, but far too few points to read as
# a solid, 264 on the egg over the whole shell. These control the resample.
SURFACE_SPACING_MM = 1.6  # target point spacing on the finished surface
BLEND_DEG = 12.0          # angular scale over which the fill fades to the ellipsoid
# Nominal dimensions, major x minor x height in mm, for items the capture cannot
# measure because the hand occludes them. Applied as an anisotropic scale about
# the body centre, so the measured surface relief and the sampled colour are
# kept and only the overall size changes.
#
# The egg is the only entry and the reason is specific. It is the one item held
# directly in the fingers rather than on a stick. Checked frame by frame against
# the video this session, the thumb covers its lower third in every frame, so the
# largest clean silhouette the sensor ever sees is 47.1 x 41.6 mm of a real egg
# that is about 57 x 44 mm. The frames that look longer, 000853 to 000856 at 61
# to 76 mm, are the egg plus a detached fingertip blob inside the same connected
# component. So the egg reconstructed as measured is near spherical, aspect 1.13,
# and reads as a ball. Authorised by the author on 2026-09-03. Recorded in the
# truth file as `scaled_to_nominal`, and it is not a measurement.
NOMINAL_MM = {"egg": (57.0, 44.0, 44.0)}
SEED = 7


def upright_cap(points_m: np.ndarray) -> np.ndarray:
    """Camera-frame cap to a table-frame cap, metres in and metres out.

    The camera looks along +z, so the surface nearest the lens has the SMALLEST
    z. Negating z turns the cap to face upward, which is how the packing pipeline
    views a table.
    """
    p = np.asarray(points_m, dtype=float).copy()
    p[:, 2] = -p[:, 2]
    p -= [p[:, 0].mean(), p[:, 1].mean(), p[:, 2].min()]
    return p


def inplane_extent_mm(cap_m: np.ndarray) -> tuple:
    """Major and minor extent of the footprint, mm, by PCA in the xy plane."""
    xy = cap_m[:, :2] - cap_m[:, :2].mean(axis=0)
    proj = xy @ np.linalg.svd(xy, full_matrices=False)[2].T
    ext = np.sort((proj.max(axis=0) - proj.min(axis=0)) * 1000.0)[::-1]
    return float(ext[0]), float(ext[1])


def close_body(cap_m: np.ndarray, colours=None):
    """Trim the rim, scale to a circular cross-section, mirror. Metres.

    Returns (points, colours). The rim plane is a low PERCENTILE of height and
    never the minimum. Reflecting about the single lowest point joins the two
    halves through that one point, which is not a DBSCAN core point, so the egg
    and the potato lost their whole underside that way while the denser
    grapefruit and avocado kept theirs. The percentile also cuts the
    flying-pixel skirt, the mixed depth returns in the one pixel ring where a
    time-of-flight sample straddles the item edge and the floor behind it.
    """
    cap = np.asarray(cap_m, dtype=float)
    rim = float(np.percentile(cap[:, 2], RIM_PCT))
    keep = cap[:, 2] >= rim
    cap = cap[keep]
    if colours is not None:
        colours = np.asarray(colours)[keep]

    # Scale the measured cap depth up to a circular cross-section, see the note
    # at the top of this file. Nothing in x or y is touched.
    _, minor_mm = inplane_extent_mm(cap)
    have = float(cap[:, 2].max() - rim)
    want = 0.5 * minor_mm / 1000.0
    if have > 1e-6:
        cap[:, 2] = rim + (cap[:, 2] - rim) * (want / have)

    under = cap.copy()
    under[:, 2] = 2.0 * rim - under[:, 2]
    lower = under[:, 2] < rim - 1e-9          # the rim ring is shared, keep it once
    closed = np.vstack([cap, under[lower]])
    closed[:, 2] -= closed[:, 2].min()        # the body now rests on z = 0
    if colours is None:
        return closed, None
    return closed, np.vstack([colours, colours[lower]])


def fibonacci_directions(n: int) -> np.ndarray:
    """n roughly equal-area unit vectors on the sphere, (n,3)."""
    i = np.arange(n, dtype=float) + 0.5
    phi = np.arccos(1.0 - 2.0 * i / n)
    theta = np.pi * (1.0 + 5.0 ** 0.5) * i
    return np.stack([np.sin(phi) * np.cos(theta),
                     np.sin(phi) * np.sin(theta),
                     np.cos(phi)], axis=1)


def densify_surface(points_m, colours, spacing_mm=None, blend_deg=None):
    """Resample a body to a dense closed surface. Metres in, metres out.

    WHY. Directional coverage is already complete, the worst angular gap
    measured over the four bodies is 21 degrees on the egg and there are no
    voids. What looks like an open surface is DENSITY. At a 3 mm voxel the egg
    carries 264 points over its whole shell, so the eye sees straight through it.

    HOW. The body is treated as star-shaped about its centroid, which holds for
    convex produce. Each measured point gives a radius in its own direction. The
    surface is then resampled on an equal-area direction set, taking an inverse
    angle weighted mean of the nearest measured radii, and falling back to the
    fitted ellipsoid only where the nearest measured direction is far. The two
    are blended by a Gaussian in the angular gap, so there is no visible seam
    between measured surface and filled surface.

    Returns (points, colours, synthetic_fraction). `synthetic_fraction` is the
    share of directions where the ellipsoid carried more than half the weight,
    which is the honest count of how much of the surface is invented.
    """
    spacing_mm = SURFACE_SPACING_MM if spacing_mm is None else spacing_mm
    blend_deg = BLEND_DEG if blend_deg is None else blend_deg
    pts = np.asarray(points_m, dtype=float)
    centre = pts.mean(axis=0)
    d = pts - centre
    vt = np.linalg.svd(d, full_matrices=False)[2]      # principal axes, rows
    semi = np.ptp(d @ vt.T, axis=0) / 2.0              # ellipsoid semi-axes, m
    semi = np.maximum(semi, 1e-5)

    r = np.linalg.norm(d, axis=1)
    good = r > 1e-9
    r, u = r[good], d[good] / r[good][:, None]
    src = np.nonzero(good)[0]

    # Enough directions that the mean spacing on the surface hits the target.
    n = int(4.0 * np.pi * float(np.mean(r)) ** 2 / (spacing_mm / 1000.0) ** 2)
    dirs = fibonacci_directions(max(600, min(n, 40000)))

    k = int(min(6, len(u)))
    chord, idx = cKDTree(u).query(dirs, k=k)
    chord = chord.reshape(len(dirs), k)
    idx = idx.reshape(len(dirs), k)
    ang = 2.0 * np.arcsin(np.clip(chord / 2.0, 0.0, 1.0))

    w = 1.0 / (ang + 1e-6)
    r_measured = (w * r[idx]).sum(axis=1) / w.sum(axis=1)
    comp = dirs @ vt.T
    r_ellipsoid = 1.0 / np.sqrt(((comp / semi) ** 2).sum(axis=1))

    blend = np.exp(-(ang[:, 0] / np.radians(blend_deg)) ** 2)
    r_new = blend * r_measured + (1.0 - blend) * r_ellipsoid
    out = centre + r_new[:, None] * dirs
    col = None if colours is None else np.asarray(colours)[src][idx[:, 0]]
    return out, col, float(np.mean(blend < 0.5))


def nominal_body(closed_m, colours, nominal_mm, spacing_mm=None):
    """A clean ellipsoid at the given size, wearing the measured colour. Metres.

    Used ONLY where the capture cannot support a reconstruction at all. The egg
    is the sole case. It is gripped directly in the fingers, the thumb covers its
    lower third in every frame, so the cap that survives segmentation is a partial
    patch rather than a hemisphere. Mirroring a partial patch about its rim gives
    a wedge with flat facets, which is what made the reconstructed egg look wrong,
    and resampling cannot round a truncation that is in the input.

    Geometry here is fully synthetic and the truth file says so. Only the colour
    is measured, carried across by matching direction from the body centre, so
    the shell keeps the real shading of the shell it came from.
    """
    spacing_mm = SURFACE_SPACING_MM if spacing_mm is None else spacing_mm
    semi = np.asarray(nominal_mm, dtype=float) / 2000.0        # mm across to m semi
    n = int(4.0 * np.pi * float(np.mean(semi)) ** 2 / (spacing_mm / 1000.0) ** 2)
    dirs = fibonacci_directions(max(600, min(n, 40000)))
    r = 1.0 / np.sqrt(((dirs / semi) ** 2).sum(axis=1))
    pts = r[:, None] * dirs
    pts[:, 2] -= pts[:, 2].min()                               # rest it on z = 0

    col = None
    if colours is not None:
        src = np.asarray(closed_m, dtype=float)
        d = src - src.mean(axis=0)
        rr = np.linalg.norm(d, axis=1)
        keep = rr > 1e-9
        _, idx = cKDTree(d[keep] / rr[keep][:, None]).query(dirs, k=1)
        col = np.asarray(colours)[keep][idx]
    return pts, col


def table_points(x0, x1, y0, y1, rng) -> np.ndarray:
    """A jittered sample of the table plane at z = 0, metres."""
    pitch = TABLE_PITCH_MM / 1000.0
    gx, gy = np.meshgrid(np.arange(x0, x1, pitch), np.arange(y0, y1, pitch), indexing="ij")
    n = gx.size
    jitter = rng.normal(0.0, pitch / 4.0, size=(n, 2))
    z = rng.normal(0.0, TABLE_NOISE_MM / 1000.0, size=n)
    return np.stack([gx.ravel() + jitter[:, 0], gy.ravel() + jitter[:, 1], z], axis=1)


def _downsample_with_colour(points, colours, voxel_m):
    """Voxel grid-average the points and carry the nearest source colour."""
    out = voxel_downsample(points, voxel_m=voxel_m)
    if colours is None:
        return out, None
    _, idx = cKDTree(points).query(out, k=1)
    return out, np.asarray(colours)[idx]


def compose(obj_dir: str, out_npy: str, out_json: str, out_rgb: str) -> dict:
    rng = np.random.default_rng(SEED)
    with open(os.path.join(obj_dir, "manifest.json"), "r", encoding="utf-8") as fh:
        manifest = json.load(fh)

    clouds, tints, bodies = [], [], []
    x_cursor = 0.0
    for item in manifest["items"]:
        raw = np.load(os.path.join(ROOT, item["npy"]))
        rgb = np.load(os.path.join(ROOT, item["rgb_npy"])) if item.get("rgb_npy") else None
        # Mirror the RAW cap and downsample the closed body, never the reverse.
        # Downsampling first leaves the rim as sparse as the measurement made it,
        # and the halves then meet through isolated points.
        closed, closed_rgb = close_body(upright_cap(raw), rgb)
        # Resample to a dense closed surface. See densify_surface, the measured
        # cloud covers every direction but is far too sparse to read as a solid.
        # Scale BEFORE the resample, never after. Stretching a finished surface
        # spreads its points along the stretched axis and reopens the gaps the
        # resample just closed, which is visible on the egg at a 1.46 stretch.
        nominal = NOMINAL_MM.get(item["label"])
        if nominal is not None:
            pts, col = nominal_body(closed, closed_rgb, nominal)
            synth_frac = 1.0
        else:
            pts, col, synth_frac = densify_surface(closed, closed_rgb)

        half_w = float(pts[:, 0].max() - pts[:, 0].min()) / 2.0
        pts[:, 0] += x_cursor + half_w
        clouds.append(pts)
        tints.append(col if col is not None
                     else np.full((len(pts), 3), 160, dtype=np.uint8))
        lo, hi = pts.min(axis=0), pts.max(axis=0)
        major, minor = inplane_extent_mm(pts)
        bodies.append({
            "label": item["label"],
            "scan": item["scan"],
            "frame": item["frame"],
            "n_points": int(len(pts)),
            "coloured": bool(col is not None),
            "synthetic_surface_fraction": round(synth_frac, 4),
            "scaled_to_nominal": bool(NOMINAL_MM.get(item["label"]) is not None),
            "inplane_mm": [round(major, 2), round(minor, 2)],
            "inplane_is": ("nominal ellipsoid, geometry synthetic and colour measured, "
                           "the thumb occludes this item in every frame"
                           if NOMINAL_MM.get(item["label"]) else "measured"),
            "height_mm": round(float(hi[2] * 1000), 2),
            "height_is": "constructed, circular cross-section about the long axis",
            # `centre_mm` is the key src/capture/naming.py matches on, so the
            # fitted body-N labels are renamed to these real names in the app.
            "centre_mm": [round(float((lo[0] + hi[0]) / 2 * 1000), 2),
                          round(float((lo[1] + hi[1]) / 2 * 1000), 2),
                          round(float((lo[2] + hi[2]) / 2 * 1000), 2)],
            "centre_xy_mm": [round(float((lo[0] + hi[0]) / 2 * 1000), 2),
                             round(float((lo[1] + hi[1]) / 2 * 1000), 2)],
        })
        x_cursor += 2.0 * half_w + GAP_MM / 1000.0

    items = np.vstack(clouds)
    item_rgb = np.vstack(tints)
    m = TABLE_MARGIN_MM / 1000.0
    table = table_points(items[:, 0].min() - m, items[:, 0].max() + m,
                         items[:, 1].min() - m, items[:, 1].max() + m, rng)
    # The capture rules require the table to hold at least half the cloud, or
    # RANSAC finds an item instead of the plane.
    if len(table) < len(items):
        raise RuntimeError(f"table {len(table)} points under items {len(items)}")

    scene = np.vstack([table, items])
    scene_rgb = np.vstack([np.full((len(table), 3), TABLE_RGB, dtype=np.uint8), item_rgb])
    order = rng.permutation(len(scene))       # shuffle BOTH arrays the same way
    scene, scene_rgb = scene[order], scene_rgb[order]
    np.save(out_npy, scene)
    np.save(out_rgb, scene_rgb)

    truth = {
        "units_in_file": "metres",
        "source": "Stray Scanner iPhone LiDAR, one frame per item, background removed",
        "measured": "the two in-plane axes of every item, and every surface colour",
        "constructed": ("the table plane, the spacing between items, and the third axis, "
                        "which one view cannot observe, see tools/compose_scene.py"),
        "colour_file": os.path.basename(out_rgb),
        "n_points": int(len(scene)),
        "n_item_points": int(len(items)),
        "n_table_points": int(len(table)),
        "n_bodies": len(bodies),
        "bodies": bodies,
    }
    with open(out_json, "w", encoding="utf-8") as fh:
        json.dump(truth, fh, indent=1)
    return truth


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--objects", default=os.path.join(ROOT, "scans", "objects"))
    ap.add_argument("--out", default=os.path.join(ROOT, "scans", "real_produce.npy"))
    a = ap.parse_args(argv)
    out_json = a.out.replace(".npy", "_truth.json")
    out_rgb = a.out.replace(".npy", "_rgb.npy")
    truth = compose(a.objects, a.out, out_json, out_rgb)
    print(f"{truth['n_points']} points, {truth['n_item_points']} item + "
          f"{truth['n_table_points']} table, {truth['n_bodies']} bodies")
    for b in truth["bodies"]:
        e = b["inplane_mm"]
        tag = "NOMINAL " if b["scaled_to_nominal"] else "measured"
        print(f"  {b['label']:11s} {b['n_points']:5d} pts  {tag} {e[0]:6.1f} x {e[1]:5.1f} mm"
              f"  height {b['height_mm']:5.1f} mm  {b['synthetic_surface_fraction']*100:4.1f}% filled")
    print(f"wrote {a.out}\nwrote {out_rgb}\nwrote {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
