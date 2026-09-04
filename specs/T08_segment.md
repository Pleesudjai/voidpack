# T08 — Segmentation, table plane and clustering

DEPENDS ON: T14 (supplies test scenes with ground truth)
CREATE: `src/capture/segment.py`
DO NOT MODIFY: any other file. Never `tests/`.

## Dependency reality, read this first

**`open3d` is NOT installed and cannot be installed.** No wheel exists for Python 3.13,
verified 2026-09-03. `scikit-learn` is not installed either. Both RANSAC and DBSCAN are
written here, on `numpy` and `scipy.spatial.cKDTree`. This is about 50 lines and it removes a
heavy dependency from a repository the judges will read.

## Exact signature

```python
def fit_plane_ransac(points, threshold_m=0.008, iterations=2000, seed=0):
    """RANSAC the dominant plane. Points in METRES.

    Returns (plane, inlier_mask) where plane is (4,) as [nx, ny, nz, d] with the
    normal unit length, so a point x on the plane satisfies n.x + d = 0.

    The normal is oriented so that the MAJORITY of non-inlier points have positive
    signed distance. That is the "above the table" direction. Deterministic for a seed.
    """

def points_above_plane(points, plane, min_h_m=0.010, max_h_m=0.200):
    """Boolean mask of points whose signed distance to the plane lies in the band."""

def dbscan(points, eps_m=0.012, min_points=12):
    """DBSCAN on scipy.spatial.cKDTree. Returns an int label array, -1 for noise.

    Standard definition. A core point has at least min_points neighbours within eps,
    itself included. Clusters are the connected components of core points, and a
    non-core point within eps of a core point joins that cluster as a border point.
    """

def cluster_objects(points, eps_m=0.012, min_points=12, min_cluster_points=150):
    """Return a list of index arrays, one per cluster, largest first.
    Clusters smaller than min_cluster_points are DISCARDED."""

def segment(points, **kw) -> dict:
    """Full pipeline. Returns
    {"plane": (4,), "above_mask": bool array, "clusters": list of index arrays,
     "n_clusters": int, "warnings": list[str]}

    A warning MUST be emitted when any cluster was discarded for being under
    min_cluster_points, naming how many points it had. A silently dropped object is
    the failure this project cannot afford, because every downstream volume and
    density is then computed on the wrong item set.
    """
```

## The normal orientation subtlety, do not skip it

The scatter of the table itself splits roughly evenly about the fitted plane, so voting on all
points decides nothing. **Orient the normal using only points OUTSIDE the inlier band.** Count
how many lie above versus below, and flip the normal if more lie below.

## Known failure modes the gate asserts

- **Merging.** Two bodies whose surfaces are closer than `eps_m` join into one cluster. The
  gate asserts this happens at a 5 mm gap and does not happen at a 40 mm gap. It is a
  property of DBSCAN, not a bug, and the capture protocol exists because of it.
- **Silent dropping.** A body returning fewer than `min_cluster_points` disappears. The gate
  asserts a warning is raised rather than the object vanishing quietly.

## Forbidden

- No `open3d`, no `sklearn`. They are not available.
- Do not renormalise, recentre, or rescale the input. Absolute scale is the whole point.
- Do not silently discard a cluster without a warning.

## Gate, run this, do not edit it

```
python -m pytest tests/test_segment.py -q
```

Expected: `8 passed`

## Done when

The gate passes and no file outside `src/capture/segment.py` has changed.
