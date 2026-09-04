# T09 — Fit an ellipsoid to each cluster

DEPENDS ON: T08b (adaptive eps). It will not work without it.
CREATE: `src/capture/fit.py`
DO NOT MODIFY: any other file. Never `tests/`.

**This is the missing link.** Nothing else turns a cluster of points into a packable `Item`.
Without it there is no path from a scan to a packing.

## The method, measured and chosen, not invented

Full table-constrained Sampson-distance ellipsoid fitting is research-grade and is out of
scope tonight. This is the robust estimator that was measured to work.

1. **Height from tangency.** The body rests on the plane, so its full height is `2c`.
   `c = percentile(height_above_plane, 99.0) / 2`.
   The 99th percentile rather than the max, because the max is one noisy point.
2. **Footprint for `a` and `b`.** Project the cluster onto the table plane. Take a 2D PCA of
   the footprint to get the two horizontal principal directions. Then
   `a, b = percentile(abs(projection), 85.0)` along each.
   **85, not 100.** The extreme projection is a noise sample. This single choice is the
   difference between a 7% axis error and a 300% one.
3. **Centre.** `x, y` from the footprint centroid, and `z = c` exactly, by tangency.
4. Sort the semi-axes descending and build the `Item`.

## Measured accuracy on `scans/synthetic_produce.npy`, 8 bodies with exact ground truth

```
footprint pct   height pct    mean|vol err|   max|vol|   mean axis err
     85            99.0            9.2%        17.4%         7.1%     <- USE THIS
     90            97.0           11.8%        27.9%         4.5%
     80            99.5           16.9%        26.5%        12.8%
     97            99.5           67.5%       116.4%        18.4%
```

The mean **signed** volume error at the chosen setting is `-4.9%`. The fixture carries a
deliberate `-3%` inward TSDF bias, worth about `-9%` on volume, so the estimator is recovering
the true surface rather than drifting.

## Exact signature

```python
FOOTPRINT_PCT = 85.0
HEIGHT_PCT = 99.0

def plane_frame(plane) -> np.ndarray:
    """(3,3) rotation whose third row is the unit plane normal."""

def fit_cluster(points_m, plane, footprint_pct=FOOTPRINT_PCT,
                height_pct=HEIGHT_PCT) -> dict:
    """One cluster to one fitted body. Points in METRES, output in MILLIMETRES.

    Returns {"axes_mm": (a,b,c) sorted descending, "centre_mm": (x,y,z),
             "yaw": float, "volume_mm3": float, "n_points": int,
             "height_mm": float}
    """

def fit_scene(points_m, segmentation) -> list:
    """Every cluster to an Item. Returns a list of src.types.Item.

    The label is "body-1", "body-2", ... in descending volume order. The mass is
    left at 0.0 because a scan cannot weigh anything, and inventing a mass would
    put a number in front of the user that no measurement produced.
    """
```

## Report the error, do not hide it

`fit_cluster` also returns `"n_points"`. A cluster with fewer than 200 points gives a poor
footprint percentile and the caller must be able to see it.

## Forbidden

- Do not use the maximum projection as a semi-axis. Measured at 300% error.
- Do not invent a mass. A scan measures geometry, not weight.
- Do not normalise or recentre the cloud. Absolute scale is the point.
- Do not call the solver, the AIR model, or anything under `src/air/`.

## Gate, run this, do not edit it

```
python -m pytest tests/test_fit.py -q
```

Expected: `6 passed`

## Done when

The gate passes and this prints a mean volume error under 20%.

```
python -c "
import json,numpy as np
from src.capture.segment import segment
from src.capture.fit import fit_scene
p=np.load('scans/synthetic_produce.npy')
T={b['label']:b for b in json.load(open('scans/synthetic_produce_truth.json'))['bodies']}
its=fit_scene(p, segment(p))
print('fitted', len(its), 'of', len(T))"
```
