# T14b — Fix the synthetic capture geometry

DEPENDS ON: T14 (implemented, 222 lines)
MODIFY: `src/capture/synth.py` only
DO NOT MODIFY: any other file. Never `tests/`.

## Two of these three bugs are in the T14 spec, not in the implementation

Measured 2026-09-03 14:45. A three-body scene from the current generator segments into
**12 clusters instead of 3**, and RANSAC fits the wrong plane entirely.

```
point split:   table 2250   body 9000   ratio 0.25
RANSAC plane:  n=[-0.004 -0.001 -1.0]  d=0.0405   inliers=8952
expected:      n=[0,0,1]  d=0
```

The same segmentation code gets **8 of 8 clusters** on `scans/synthetic_produce.npy`, so
`segment.py` is sound and the generator is what is wrong.

## Bug 1, the occlusion rule is too aggressive. MY SPEC WAS WRONG.

T14 says to reject a point whose outward normal satisfies `n_z < sin(view_elev_deg)`. At
37.5 degrees that keeps only the cap within 52.5 degrees of the top, which on a 22 mm sphere
is a band **9 mm thick**. Nine thousand points inside a 9 mm slab are very nearly coplanar,
so RANSAC prefers them over the real table.

**Measured effect, 2026-09-03.** On a 22 mm sphere.

```
old rule:  cap spans 35.4 to 44.0 mm =  8.6 mm thick   20% of the surface kept
new rule:  cap spans  8.6 to 44.0 mm = 35.4 mm thick   80% of the surface kept
```

And the segmentation outcome across table fractions, three bodies, truth = 3.

```
table_frac=0.20   old -> 10 clusters FAIL     new -> 3 PASS
table_frac=0.35   old ->  1 cluster  FAIL     new -> 3 PASS
table_frac=0.50   old ->  3 clusters PASS     new -> 3 PASS
```

**The corrected rule is robust to the point split. The old rule works only at exactly 50/50
and fails on both sides of it**, fragmenting into 10 clusters when the table is sparse and
collapsing to 1 when it is at 0.35. Fix both, and the occlusion is the one that buys the
robustness.

**The correct rule for a full orbit.** A surface point is visible from at least one azimuth
during a complete orbit at elevation `phi` when its polar angle satisfies
`theta < 90 deg + phi`. In terms of the normal that is

```python
keep = n_z > -math.sin(math.radians(view_elev_deg))
```

Note the **minus sign**. At 37.5 degrees this keeps everything down to 127.5 degrees, a cap
about 35 mm deep on the same sphere, reaching below the equator.

This also reconciles the generator with the fit. `measure_cluster` mirrors the visible cap
about the object **mid-height**, which only makes sense if the visible region reaches roughly
to the equator. Under the old rule it never did.

## Bug 2, the table-to-body point split was never specified. MY SPEC WAS WRONG.

RANSAC is a majority vote. If body points outnumber table points the plane fit can lose.

**Required.** The table receives **at least 50%** of `n_points`. Emit the table first, at
`n_table = n_points // 2`, then divide the remainder evenly across the bodies.

For reference, `scans/synthetic_produce.npy` has a table-to-body ratio of 1.13 and segments
correctly at 8 of 8.

## Bug 3, the surface, bias and noise tests fail. This one is the implementation.

Three gate tests fail and each checks a property the generator must have exactly.

| Failing test | What it requires |
|---|---|
| `test_zero_noise_points_lie_exactly_on_the_surface` | with `noise_mm=0, bias_frac=0`, every body point satisfies `((x-cx)/a)^2 + ((y-cy)/b)^2 + ((z-cz)/c)^2 == 1` to 1e-6 |
| `test_inward_bias_shrinks_the_apparent_body` | with `bias_frac=-0.03` and no noise, every radius is exactly `0.97 * r`, systematically, not on average |
| `test_noise_sigma_is_what_was_requested` | the residual standard deviation about the true surface equals `noise_mm` within 25% |

Order of operations matters and must be exactly this.

```
1. sample the surface           point lies exactly on the ellipsoid
2. apply the inward bias        scale the offset from the centre by (1 + bias_frac)
3. add Gaussian noise           sigma = noise_mm
4. translate to the centre
5. convert mm to metres, ONCE, dividing by MM_PER_M
```

Applying noise before the bias, or scaling the whole point rather than the offset from the
centre, breaks tests 1 and 2.

## Verification the executor must run before declaring this done

```
python -m pytest tests/test_synth.py tests/test_segment.py -q
```

Expected: `16 passed`. Both files, together. `test_segment.py` builds its scenes from this
generator, so a broken generator fails nine tests in a file that is not its own.

Then this must print 3.

```
python -c "
from src.capture.synth import synth_scene
from src.capture.segment import segment
s=synth_scene([((22.,22.,22.),(0.,0.)),((22.,22.,22.),(84.,0.)),((22.,22.,22.),(0.,84.))],
              noise_mm=2.0,bias_frac=0.0,n_points=9000)
print(segment(s.points, min_cluster_points=50)['n_clusters'])"
```

## Forbidden

- Do not emit a closed surface. The occlusion is corrected, not removed.
- Do not tune `eps` or `min_cluster_points` in `segment.py` to make this pass. The
  segmentation is correct and is proven so on the real fixture. Fix the generator.
- Do not normalise or rescale the cloud.
