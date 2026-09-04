"""Isolate the held grocery from a Stray Scanner capture, background removed.

WHAT THE CAPTURES ACTUALLY ARE. Measured from `odometry.csv` in this session, the
phone is nearly stationary in all four scans, camera span 20 to 80 mm and viewing
direction spread within 8 degrees for the egg, avocado and potato, 31 degrees for
the grapefruit. The operator held the phone still and rotated the item on a stick.
World-frame odometry fusion is therefore wrong, and camera-frame fusion over a
rotating item would return the rotational envelope, which over-states the volume
of anything that is not a sphere. So ONE frame per item is used, which is also
the T17 default and matches the cap-and-mirror mathematics in T09.

HOW THE BACKGROUND GOES. The item is held in mid air, 100 to 300 mm from the
lens, and the floor is 1000 to 1500 mm behind it. A near depth band separates
them outright, so no RANSAC plane is needed and none is used. What survives the
band is the item, the hand and the forearm. The stick is thin and returns low
confidence, so it breaks the connection and the item falls out as its own
connected component. The component is chosen on shape and size, never on colour.

UNITS. Depth PNGs are uint16 millimetres, every array written is METRES.

    python tools/extract_objects.py --render
"""
from __future__ import annotations

import argparse
import io
import json
import os
import sys
import zipfile
from typing import Optional

import numpy as np
from PIL import Image
from scipy import ndimage

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.capture.ingest import _parse_camera_matrix, backproject, scale_intrinsics  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Scan id to label. Identified from one decoded RGB frame per capture, 2026-09-03.
SCANS = {
    "0eac8a9a8b": "egg",
    "6b060981ba": "grapefruit",
    "93390ea0af": "avocado",
    "f9a19bb445": "potato",
}

NEAR_MIN_MM = 80.0        # below the closest observed item pixel, 95 mm on the egg
NEAR_MAX_MM = 450.0       # far below the floor at 1000 mm, so the room never enters
# Confidence 1, not 2. The LiDAR returns confidence 1 in a one to two pixel ring
# around every silhouette, so demanding 2 erodes the rim and under-measures the
# item. Measured on the grapefruit, 64 mm across at confidence 2 against 96 mm at
# confidence 1, an under-measurement of a third.
MIN_CONFIDENCE = 1
# The item is held out toward the lens and the hand is behind it, so a band that
# starts at the nearest surface and is only as deep as a grocery separates them
# even where the fingers touch the item and the two merge into one component.
TIGHT_DEPTH_MM = 62.0     # deeper than the deepest measured cap, 41 mm on the grapefruit
NEAREST_PCT = 2.0         # percentile used for the nearest surface, robust to a stray pixel
WIDE_DEPTH_FACTOR = 1.15  # second-pass band as a multiple of the item width
MAX_BAND_MM = 130.0       # never open the band past the largest plausible grocery
MIN_EXTENT_MM = 25.0      # smaller than any grocery here
MAX_EXTENT_MM = 120.0     # above the largest item, the grapefruit at 104 mm, and
                          # below the merged egg-and-palm blob at 130 to 146 mm
# Silhouette fill, component pixels over the area of the ellipse fitted to those
# pixels. A rounded item fills its own ellipse, measured 0.88 to 1.13 across all
# four captures. A hand gripping the egg does not, measured 0.66 to 0.81, because
# the fingers leave the ellipse largely empty. This is what separates the egg
# from the fist holding it, where depth alone cannot.
MIN_FILL = 0.88
# A convex cap slightly OVER-fills its fitted ellipse because the ellipse is
# sized on extents while the silhouette bulges inside them, so 1.0 is a normal
# reading for a clean item and a useful strict gate, not an impossible one.
STRICT_FILL = 1.00
# Aspect is the longest PCA extent over the SECOND longest, never over the third.
# One view of a rounded item is a cap, so its third extent is the cap depth and is
# small by construction, measured 9 to 14 mm against a 43 to 66 mm width. Testing
# the longest over the thinnest rejects every real item at about 4.6.
MAX_ASPECT = 2.2          # in-plane elongation, excludes the hand and the forearm
MAX_FLATNESS = 12.0       # longest over thinnest, excludes a flat sheet of floor
MIN_PIXELS = 250          # a component smaller than this is stick or noise
MIN_PIXELS_FOR_PICK = 800 # a view thinner than this is a glancing edge, not the item


def _png(zf: zipfile.ZipFile, name: str) -> np.ndarray:
    return np.array(Image.open(io.BytesIO(zf.read(name))))


def rgb_frame(zf: zipfile.ZipFile, root: str, index: int):
    """Decode one frame of rgb.mp4 as (H,W,3) uint8, or None if it cannot be read.

    Colour is used for DISPLAY only. The solver never sees it, per the
    architecture rule that every number comes from geometry.
    """
    try:
        import av
    except ImportError:
        return None
    try:
        container = av.open(io.BytesIO(zf.read(f"{root}/rgb.mp4")))
        stream = container.streams.video[0]
        for i, frame in enumerate(container.decode(stream)):
            if i == index:
                out = frame.to_ndarray(format="rgb24")
                container.close()
                return out
        container.close()
    except Exception:
        return None
    return None


def sample_colours(rgb, mask: np.ndarray) -> Optional[np.ndarray]:
    """Colour for each kept depth pixel, (N,3) uint8, in backproject order.

    The depth stream and the RGB stream come from the same camera, 256x192 and
    1920x1440, an exact 7.5 in both axes, so a depth pixel maps to RGB by that
    scale alone and no extrinsic calibration is involved. `mask` must be the same
    boolean array `backproject` keeps, and `np.nonzero` walks it row-major, which
    is the order `backproject` emits, so the two line up index for index.
    """
    if rgb is None:
        return None
    ys, xs = np.nonzero(mask)
    sy = rgb.shape[0] / float(mask.shape[0])
    sx = rgb.shape[1] / float(mask.shape[1])
    ry = np.clip((ys * sy).astype(int), 0, rgb.shape[0] - 1)
    rx = np.clip((xs * sx).astype(int), 0, rgb.shape[1] - 1)
    return rgb[ry, rx].astype(np.uint8)


def _components(mask: np.ndarray):
    """Connected components of the near-band mask, 8 connected, largest first."""
    lab, n = ndimage.label(mask, structure=np.ones((3, 3), dtype=int))
    out = [(int((lab == i).sum()), i) for i in range(1, n + 1)]
    out.sort(reverse=True)
    return lab, out


def _fill_ratio(sel: np.ndarray) -> float:
    """Component pixels over the area of the ellipse fitted to those pixels."""
    ys, xs = np.nonzero(sel)
    P = np.stack([xs, ys], axis=1).astype(float)
    P -= P.mean(axis=0)
    proj = P @ np.linalg.svd(P, full_matrices=False)[2].T
    a = (proj[:, 0].max() - proj[:, 0].min()) / 2.0
    b = (proj[:, 1].max() - proj[:, 1].min()) / 2.0
    return float(sel.sum() / (np.pi * a * b + 1e-9))


def trim_grazing(pts_m: np.ndarray, k: int = 12, min_cos: float = 0.30) -> np.ndarray:
    """Drop the flying-pixel skirt at the silhouette. Points in and out, METRES.

    A time-of-flight pixel that straddles the item edge averages the item at
    about 150 mm with the floor at about 1200 mm and returns something in
    between, so the silhouette carries a skirt of points stretched away from the
    lens. Left in, the skirt tripled the measured cap depth, 31 mm on the egg
    against the 21 mm its own width implies.

    The skirt is detected by geometry rather than by depth. A local plane is
    fitted to each point's k nearest neighbours and the point is dropped when its
    normal is more than `arccos(min_cos)` from the view direction, which is where
    a real surface is turning away and stops being measurable anyway. T09 mirrors
    the cap to rebuild the underside, so the trimmed sliver is reconstructed.
    """
    from scipy.spatial import cKDTree

    if len(pts_m) < k + 1:
        return pts_m
    tree = cKDTree(pts_m)
    _, idx = tree.query(pts_m, k=k)
    nbr = pts_m[idx]                                  # (N,k,3)
    centred = nbr - nbr.mean(axis=1, keepdims=True)
    # Smallest right-singular vector of each neighbourhood is its plane normal.
    normals = np.linalg.svd(centred, full_matrices=False)[2][:, 2, :]
    keep = np.abs(normals[:, 2]) >= min_cos          # view direction is +z
    return pts_m[keep] if keep.sum() >= MIN_PIXELS else pts_m


def _score(pts_m: np.ndarray, npix: int, touches_border: bool, fill: float = 1.0):
    """Shape score for one component. Returns (ok, aspect, extent_mm, reason)."""
    if npix < MIN_PIXELS:
        return False, 0.0, 0.0, f"only {npix} px"
    if fill < MIN_FILL:
        return False, 0.0, 0.0, f"fill {fill:.2f}, a hand not an item"
    ext = _pca_extent_mm(pts_m)
    longest, second, thinnest = float(ext[0]), float(max(ext[1], 1e-6)), float(max(ext[2], 1e-6))
    aspect = longest / second
    if longest < MIN_EXTENT_MM:
        return False, aspect, longest, f"{longest:.0f} mm too small"
    if longest > MAX_EXTENT_MM:
        return False, aspect, longest, f"{longest:.0f} mm too large, hand or arm"
    if aspect > MAX_ASPECT:
        return False, aspect, longest, f"aspect {aspect:.1f} too elongated"
    if longest / thinnest > MAX_FLATNESS:
        return False, aspect, longest, f"flatness {longest / thinnest:.1f}, a sheet not an item"
    if touches_border and longest > 120.0:
        return False, aspect, longest, "runs off the frame"
    return True, aspect, longest, "ok"


def _pca_extent_mm(pts_m: np.ndarray) -> np.ndarray:
    """PCA extents in mm, descending. Robust where an axis-aligned box is not,
    because a tilted forearm fills a large box while being narrow in its own frame."""
    centred = pts_m - pts_m.mean(axis=0)
    _, _, vt = np.linalg.svd(centred, full_matrices=False)
    proj = centred @ vt.T
    return np.sort((proj.max(axis=0) - proj.min(axis=0)) * 1000.0)[::-1]


def extract_one(zip_path: str, every: int = 5, verbose: bool = True) -> dict:
    """Best single frame of one capture to the item cloud. Points in METRES.

    Returns a dict with `points` (N,3 metres, camera frame), the frame name, the
    measured PCA extents in mm and the rejected components, so a wrong pick is
    visible rather than silent.
    """
    zf = zipfile.ZipFile(zip_path)
    root = sorted({n.split("/", 1)[0] for n in zf.namelist() if "/" in n})[0]
    K_rgb = _parse_camera_matrix(zf.read(f"{root}/camera_matrix.csv").decode("utf-8"))
    depths = sorted(n for n in zf.namelist() if "/depth/" in n and n.endswith(".png"))

    best: Optional[dict] = None
    strict: Optional[dict] = None
    loose: Optional[dict] = None
    for name in depths[::every]:
        depth = _png(zf, name)
        conf = _png(zf, name.replace("/depth/", "/confidence/"))
        near = (depth >= NEAR_MIN_MM) & (depth <= NEAR_MAX_MM) & (conf >= MIN_CONFIDENCE)
        if near.sum() < MIN_PIXELS:
            continue
        # Anchor a shallow band on the nearest surface, which is the held item.
        d0 = float(np.percentile(depth[near], NEAREST_PCT))
        tight = near & (depth <= d0 + TIGHT_DEPTH_MM)
        if tight.sum() < MIN_PIXELS:
            continue
        K = scale_intrinsics(K_rgb, (1920, 1440), (depth.shape[1], depth.shape[0]))
        lab, comps = _components(tight)
        # The item owns the nearest pixel, so take the component holding it and
        # never simply the largest, which is the forearm in several frames.
        masked = np.where(tight, depth, np.uint16(65535))
        i = int(lab.flat[int(np.argmin(masked))])
        if i == 0:
            continue
        npix = int((lab == i).sum())
        sel = lab == i
        border = bool(sel[0, :].any() or sel[-1, :].any()
                      or sel[:, 0].any() or sel[:, -1].any())
        pts = backproject(np.where(sel, depth, 0).astype(np.uint16), K, stride=1,
                          min_mm=NEAR_MIN_MM, max_mm=NEAR_MAX_MM)
        fill = _fill_ratio(sel)
        ok, aspect, longest, why = _score(pts, npix, border, fill)
        if not ok or npix < MIN_PIXELS_FOR_PICK:
            continue

        # Second pass, widen the band to the item's own width. A convex body is
        # never deeper than it is wide, so a fixed 62 mm band truncates anything
        # larger. It cut the grapefruit, 94 mm across, down to a 48 mm tall disc.
        # The wider band is only accepted when it still passes the shape gate, so
        # a frame where the hand sits just behind the item falls back to pass one.
        want = min(WIDE_DEPTH_FACTOR * longest, MAX_BAND_MM)
        if want > TIGHT_DEPTH_MM:
            wide = near & (depth <= d0 + want)
            lab2, _ = _components(wide)
            masked2 = np.where(wide, depth, np.uint16(65535))
            j = int(lab2.flat[int(np.argmin(masked2))])
            if j != 0:
                sel2 = lab2 == j
                border2 = bool(sel2[0, :].any() or sel2[-1, :].any()
                               or sel2[:, 0].any() or sel2[:, -1].any())
                pts2 = backproject(np.where(sel2, depth, 0).astype(np.uint16), K,
                                   stride=1, min_mm=NEAR_MIN_MM, max_mm=NEAR_MAX_MM)
                fill2 = _fill_ratio(sel2)
                ok2, aspect2, longest2, why2 = _score(pts2, int(sel2.sum()), border2, fill2)
                if ok2:
                    sel, pts, fill = sel2, pts2, fill2
                    aspect, longest, why = aspect2, longest2, why2
                    npix = int(sel2.sum())
        # Choose on fill, NOT on size. The largest accepted component is the hand
        # in several egg frames, which clears the gate at 0.88 while the clean egg
        # frames sit at 1.02 to 1.04. Fill is what the downstream model cares
        # about anyway, since the pipeline fits an ellipsoid to this cap.
        # `mask` is exactly what backproject keeps, so the colour sampled from it
        # lines up with `points` index for index.
        cand = {"points": pts, "frame": name.split("/")[-1], "n_pixels": npix,
                "aspect": aspect, "longest_mm": longest, "fill": fill, "reason": why,
                "mask": sel & (depth >= NEAR_MIN_MM) & (depth <= NEAR_MAX_MM)}
        # Prefer the LARGEST view that is cleanly convex, and fall back to the
        # single cleanest only when nothing reaches the strict fill. Taking the
        # largest over the loose gate returns a hand, measured 139 x 60 mm for
        # the egg at fill 0.96. Taking the cleanest alone gives a small partial
        # view, 89 x 77 mm for the grapefruit against 95 x 87 mm here.
        if fill >= STRICT_FILL:
            if strict is None or npix > strict["n_pixels"]:
                strict = cand
        if loose is None or fill > loose["fill"]:
            loose = cand

    best = strict if strict is not None else loose
    if best is not None:
        idx = int(os.path.splitext(best["frame"])[0])
        colours = sample_colours(rgb_frame(zf, root, idx), best["mask"])
        if colours is not None and len(colours) == len(best["points"]):
            best["colours"] = colours
        else:
            best["colours"] = None
            if verbose:
                print("  no colour, rgb.mp4 frame could not be read or did not align")
        best.pop("mask", None)
    zf.close()
    if best is None:
        raise RuntimeError(f"no component in {zip_path} passed the shape gate")

    best["pca_extent_mm"] = _pca_extent_mm(best["points"]).tolist()
    best["n_points"] = int(len(best["points"]))
    if verbose:
        e = best["pca_extent_mm"]
        print(f"  frame {best['frame']}, {best['n_points']} points, "
              f"PCA extent {e[0]:.1f} x {e[1]:.1f} x {e[2]:.1f} mm, "
              f"aspect {best['aspect']:.2f}, fill {best['fill']:.2f}")
    return best


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--lidar", default=os.path.join(ROOT, "Lidar"))
    ap.add_argument("--out", default=os.path.join(ROOT, "scans", "objects"))
    ap.add_argument("--every", type=int, default=5)
    ap.add_argument("--render", action="store_true", help="write a PNG of every pick")
    a = ap.parse_args(argv)

    os.makedirs(a.out, exist_ok=True)
    manifest = {"units_in_file": "metres", "frame": "camera, +z away from the lens",
                "source": "Stray Scanner iPhone LiDAR, one frame per capture",
                "items": []}
    picks = {}
    for scan, label in SCANS.items():
        zp = os.path.join(a.lidar, f"{scan}.zip")
        if not os.path.exists(zp):
            print(f"{label:11s} MISSING {zp}")
            continue
        print(f"{label:11s} {scan}")
        got = extract_one(zp, every=a.every)
        out_npy = os.path.join(a.out, f"{label}.npy")
        np.save(out_npy, got["points"])
        out_rgb = None
        if got.get("colours") is not None:
            out_rgb = os.path.join(a.out, f"{label}_rgb.npy")
            np.save(out_rgb, got["colours"])
        picks[label] = got
        manifest["items"].append({
            "label": label, "scan": scan, "frame": got["frame"],
            "n_points": got["n_points"], "pca_extent_mm": got["pca_extent_mm"],
            "aspect": got["aspect"], "npy": os.path.relpath(out_npy, ROOT).replace("\\", "/"),
            "rgb_npy": (os.path.relpath(out_rgb, ROOT).replace("\\", "/")
                        if out_rgb else None),
        })

    with open(os.path.join(a.out, "manifest.json"), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=1)
    print(f"wrote {len(picks)} clouds and manifest.json to {a.out}")

    if a.render and picks:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(len(picks), 3, figsize=(11, 3.4 * len(picks)))
        axes = np.atleast_2d(axes)
        for r, (label, got) in enumerate(picks.items()):
            p = (got["points"] - got["points"].mean(axis=0)) * 1000.0
            for k, (i, j, nm) in enumerate([(0, 1, "XY"), (0, 2, "XZ"), (1, 2, "YZ")]):
                ax = axes[r, k]
                ax.scatter(p[:, i], p[:, j], s=1.2, c=p[:, 2], cmap="magma")
                ax.set_aspect("equal")
                ax.set_title(f"{label} {nm}, mm", fontsize=9)
                ax.grid(alpha=0.25, lw=0.4)
        plt.tight_layout()
        png = os.path.join(a.out, "picks.png")
        plt.savefig(png, dpi=90)
        print(f"wrote {png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
