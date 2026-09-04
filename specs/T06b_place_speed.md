# T06b — Make the packer finish

DEPENDS ON: T04b (do that one first, this is useless without it)
MODIFY: `src/solver/place.py` only
DO NOT MODIFY: any other file. Never `tests/`.

## The measured problem

The current search evaluates, per item,

```
3 up-axes x 12 x-positions x 12 y-positions x 4 yaws = 1,728 candidate poses
x 50 bisection steps on z                            = 86,400 validity checks
```

At the current contact cost that is 140 seconds for one item against one neighbour. After
T04b it is 3.25 seconds, which is still too slow for twelve items and far too slow for the
drag interaction in T16, which needs a verdict in well under 50 ms.

Three changes fix it. All three are exact. **None of them approximates the contact test.**

## Change 1, the conservative outer-bound early-out

Before running the exact test on a pair, check

```python
if |d|**2 > (max_semi_a + max_semi_b)**2:
    # provably disjoint, skip the exact test
```

`max_semi_axis` comes from T04b and is cached per item, per orientation.

**This is NOT the forbidden bounding-sphere proxy, and the distinction matters.**

| | Forbidden proxy | This early-out |
|---|---|---|
| What it claims | "these are disjoint" from a sphere test | "these are disjoint" from a sphere test |
| When it is used | as the ANSWER, in all cases | only when the spheres are already separated |
| Can it hide an overlap | **YES**, that is why it is banned | **NO**, separated bounding spheres means separated bodies, always |

The banned version admits interpenetration because it also claims *overlap* from the proxy.
This one only ever proves *disjointness*, which is a theorem. Every pair that is not proven
disjoint still gets the exact Perram-Wertheim test. `test_no_bounding_sphere_shortcut_is_used`
in `tests/test_contact.py` still guards `contact_mu` itself and must keep passing.

Measured cost of the check: 0.00126 ms, against 0.0376 ms for the exact test after T04b. It
is 30x cheaper and it skips the large majority of pairs.

## Change 2, bisect to the tolerance that matters, not to machine precision

The current loop runs 50 steps for `1e-15` precision on a quantity whose gate tolerance is
`1e-3` mm. Stop at `1e-4` mm, which is 10x finer than the gate and about **25 steps**. Halves
the work.

## Change 3, shrink the grid ONLY. Do NOT reduce yaws.

**CORRECTION, measured 2026-09-03 17:25.** An earlier draft of this spec proposed
`yaws=2`. That is wrong and it breaks the gate.

```
grid=12 yaws=4   69.2 s   placed 3/3   container 0.246   penetration 0.00 mm   OK
grid=10 yaws=4   48.8 s   placed 3/3   container 0.246   penetration 0.00 mm   OK
grid= 8 yaws=4   34.4 s   placed 3/3   container 0.246   penetration 0.00 mm   OK
grid= 6 yaws=2    -       GATE REJECTED, 1.056 mm of interpenetration
```

**`grid=8` is free.** It returns the identical density to `grid=12` in half the time.
**`yaws` stays at 4.** Two yaw angles leave the search unable to find a non-overlapping
pose and the gate correctly refuses the result.

New default is `grid=8, yaws=4`, which is `3 x 8 x 8 x 4 = 768` poses against 1,728, a
2.25x reduction with no loss of quality. Keep both as parameters.

For a `keep_upright` item the up-axis is fixed, cutting its poses by a further 3x.

## Projected result

```
384 poses x 25 steps            =  9,600 checks per item
of which most take the early-out at 0.00126 ms
budget: well under 1 s per item, roughly 5 s for a twelve-item order
```

## Add a fast path for the drag interaction, T16 needs it

```python
def check_move(placements, label, new_centre_mm) -> dict:
    """Penetration for ONE moved item against all others. No search, no bisection.

    Returns {"penetration_mm": float, "ok": bool, "against": str | None}
    where `against` names the deepest offender.

    Must return in under 5 ms for twenty items. This is what the browser calls on
    every pointermove, so it is the latency budget for the whole demo.
    """
```

## Forbidden

- Do not weaken, skip, or make optional the final `gate(placements)` call. The early-out is a
  search optimisation and the gate is the guarantee. They are not the same thing.
- Do not use the outer bound to conclude that two bodies DO overlap. It can only prove they
  do not.
- Do not cache a contact result across a change of position or orientation.

## Gate, unchanged

```
python -m pytest tests/test_place.py -q
```

Expected: `7 passed`, **and the whole file must run in under 60 seconds.** If it does not
finish, the task is not done, regardless of what the individual assertions say.

## Done when

The gate passes inside 60 s and this prints under 1 second.

```
python -c "import time;from src.types import Item,Box;from src.schema import empty_ruleset;from src.solver.place import pack;its=[Item(label='s%d'%i,axes=(25.,25.,25.)) for i in range(6)];t=time.time();p,u=pack(its,Box(300.,300.,300.),empty_ruleset());print('%.2f s, placed %d'%(time.time()-t,len(p)))"
```
