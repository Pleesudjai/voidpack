"""Attach real produce names to fitted bodies by matching a capture sidecar.

The fitter is deliberately name-blind. `fit_scene` labels clusters `body-1`
onward in cluster order, because a scan cannot know that a point cloud is an
egg. When a capture ships a `*_truth.json` sidecar carrying the real labels and
the table positions they were placed at, this module matches each fitted
cluster to one sidecar body and renames the Item.

Units. Point clouds arrive in metres, sidecar centres are recorded in
millimetres, and every distance computed or reported here is in millimetres.

The match is refused rather than guessed. It must be a bijection, every
assignment must sit within MAX_MATCH_MM of its sidecar centre, and the runner
up must be at least MIN_MARGIN_MM further away. If any of that fails the
original `body-N` labels stand and the caller is told which test failed. A
wrong name is worse than no name, because a rule written against it would
compile cleanly and then silently constrain nothing.

This module never sets `fragile`, `keep_upright` or `mass_g`. Those are
constraints, they come from the user through the AIR rule compiler, and a
sidecar that reads "egg" is not the user saying an egg is fragile.
"""
import json
import os
from typing import Any, Dict, List, Optional

import numpy as np
from scipy.optimize import linear_sum_assignment

# A cluster centroid further than this from its assigned sidecar centre is not
# that body. Observed worst case on scans/synthetic_produce.npy is 1.2 mm.
MAX_MATCH_MM = 25.0

# The runner up must be at least this much further away than the winner, so a
# near tie is refused instead of coin-flipped. Observed worst case margin on
# the same capture is 79.4 mm.
MIN_MARGIN_MM = 20.0


def sidecar_path(scan_path: str) -> str:
    """Return the `*_truth.json` path that pairs with a scan path."""
    base, _ = os.path.splitext(scan_path)
    return base + "_truth.json"


def load_sidecar(scan_path: str) -> Optional[Dict[str, Any]]:
    """Load the ground-truth sidecar for a scan, or None when absent.

    Returns None on a missing or unreadable file rather than raising, because
    a capture without a sidecar is the normal case for a real scan and must
    still pack.
    """
    path = sidecar_path(scan_path)
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except (ValueError, OSError):
        return None
    if not isinstance(data, dict) or not isinstance(data.get("bodies"), list):
        return None
    return data


def cluster_centroids_mm(points_m: np.ndarray, segmentation: Dict[str, Any]) -> np.ndarray:
    """Return an (n_clusters, 3) array of cluster centroids in millimetres.

    Input points are in metres. Only the horizontal pair is used for matching,
    because a partial-view cloud biases the vertical centroid downward while
    the horizontal centroid stays on the body.
    """
    out = []
    for cluster in segmentation["clusters"]:
        out.append(points_m[cluster].mean(axis=0) * 1000.0)
    return np.asarray(out, dtype=float)


def match_clusters(points_m: np.ndarray, segmentation: Dict[str, Any],
                   sidecar: Dict[str, Any]) -> Dict[str, Any]:
    """Match fitted clusters to sidecar bodies by horizontal centre, in mm.

    Returns a dict with `ok`, `reason`, `mapping` and `rows`. `mapping` is
    keyed by the `body-N` label the fitter minted, valued by the real name.
    It is empty whenever `ok` is False.
    """
    names = [b.get("label") for b in sidecar["bodies"]]
    truth_xy = np.asarray([b["centre_mm"][:2] for b in sidecar["bodies"]], dtype=float)
    fit_xyz = cluster_centroids_mm(points_m, segmentation)
    fit_xy = fit_xyz[:, :2]

    n_fit, n_truth = len(fit_xy), len(truth_xy)
    if n_fit != n_truth:
        return {"ok": False, "mapping": {}, "rows": [],
                "reason": "count mismatch, %d fitted bodies against %d sidecar bodies"
                          % (n_fit, n_truth)}
    if n_fit == 0:
        return {"ok": False, "mapping": {}, "rows": [], "reason": "no clusters"}

    # Full pairwise distance, then the optimal bijection. Greedy nearest can
    # assign one sidecar body twice and strand another, an assignment solve
    # cannot.
    dist = np.linalg.norm(fit_xy[:, None, :] - truth_xy[None, :, :], axis=2)
    rows_idx, cols_idx = linear_sum_assignment(dist)

    mapping, rows = {}, []
    for i, j in zip(rows_idx, cols_idx):
        d_assigned = float(dist[i, j])
        others = np.delete(dist[i], j)
        d_runner_up = float(others.min())
        label = "body-%d" % (i + 1)
        rows.append({"label": label, "name": names[j],
                     "centre_xy_mm": [float(fit_xy[i, 0]), float(fit_xy[i, 1])],
                     "d_mm": d_assigned, "runner_up_mm": d_runner_up,
                     "margin_mm": d_runner_up - d_assigned})
        if d_assigned > MAX_MATCH_MM:
            return {"ok": False, "mapping": {}, "rows": rows,
                    "reason": "%s sits %.1f mm from %s, over the %.1f mm limit"
                              % (label, d_assigned, names[j], MAX_MATCH_MM)}
        if d_runner_up - d_assigned < MIN_MARGIN_MM:
            return {"ok": False, "mapping": {}, "rows": rows,
                    "reason": "%s is a near tie, %.1f mm to %s against %.1f mm to the runner up"
                              % (label, d_assigned, names[j], d_runner_up)}
        mapping[label] = names[j]

    if len(set(mapping.values())) != len(mapping):
        return {"ok": False, "mapping": {}, "rows": rows,
                "reason": "assignment is not a bijection"}
    return {"ok": True, "mapping": mapping, "rows": rows, "reason": "matched"}


def apply_names(items: List[Any], mapping: Dict[str, str]) -> int:
    """Rename Items in place from a `body-N` to real-name mapping.

    Returns the number renamed. Items absent from the mapping keep the label
    the fitter gave them.
    """
    renamed = 0
    for item in items:
        new = mapping.get(item.label)
        if new:
            item.label = new
            renamed += 1
    return renamed


def name_items(items: List[Any], points_m: np.ndarray, segmentation: Dict[str, Any],
               scan_path: str) -> Dict[str, Any]:
    """Rename fitted Items from the scan sidecar when one is present and clean.

    Returns the match report with a `renamed` count added. On any refusal the
    Items are untouched and keep their `body-N` labels.
    """
    sidecar = load_sidecar(scan_path)
    if sidecar is None:
        return {"ok": False, "mapping": {}, "rows": [], "renamed": 0,
                "reason": "no sidecar at %s" % sidecar_path(scan_path)}
    report = match_clusters(points_m, segmentation, sidecar)
    report["renamed"] = apply_names(items, report["mapping"]) if report["ok"] else 0
    return report
