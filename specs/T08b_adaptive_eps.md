# T08b — Make the DBSCAN eps adapt to point density

DEPENDS ON: T08 (implemented)
MODIFY: `src/capture/segment.py` only
DO NOT MODIFY: any other file. Never `tests/`.

**Ten minutes. Do this before T09, which cannot work without it.**

## The measured problem

A fixed eps is knife-edge. On `scans/synthetic_produce.npy`, 8 bodies of ground truth.

```
eps = 0.008 m ->  9 clusters
eps = 0.010 m ->  8 clusters   correct
eps = 0.012 m ->  7 clusters   <- the current default
eps = 0.025 m ->  4 clusters   <- the current cluster_objects default
```

One millimetre either side of correct changes the answer. A demo resting on a hand-tuned
constant will break on a real scan, because point spacing changes with standoff distance.

## The fix

Scale eps to the actual point density.

```python
def median_spacing(points) -> float:
    """Median nearest-neighbour distance, in metres. scipy.spatial.cKDTree, k=2."""

def cluster_objects(points, eps_m=None, k_eps=3.0, min_points=12,
                    min_cluster_points=150):
    """When eps_m is None, use eps = k_eps * median_spacing(points)."""
```

Measured on the same cloud, median spacing above the plane is **3.15 mm**.

```
k = 2.5  ->  9 clusters
k = 3.0  ->  8 clusters   correct
k = 3.5  ->  8 clusters   correct
k = 4.0  ->  6 clusters
```

**k = 3.0 to 3.5 is a band, not a knife edge.** Default `k_eps = 3.0`.

`segment()` must accept and forward both `eps_m` and `k_eps`, and must report the eps it
actually used in its return dict as `"eps_m_used"`, so an operator can see what happened.

## Forbidden

- Do not keep a fixed default of 0.012 or 0.025 anywhere. `cluster_objects` currently
  defaults to 0.025 while `dbscan` defaults to 0.012, and that inconsistency is part of why
  this went unnoticed.
- Do not tune `k_eps` per scan to hit an expected count. It is a density-scaled constant and
  it stays at 3.0 unless a measurement says otherwise.

## Gate

```
python -m pytest tests/test_segment.py -q
```

Plus this must print 8.

```
python -c "
import numpy as np
from src.capture.segment import segment
print(segment(np.load('scans/synthetic_produce.npy'))['n_clusters'])"
```
