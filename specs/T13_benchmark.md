# T13 — Concrete packing benchmark, Andreasen and Andersen

DEPENDS ON: T01, T07
CREATE: `src/solver/benchmark.py`
DO NOT MODIFY: any other file. Never `tests/`.

## Purpose, and why this task exists

The solver answers whether a specific arrangement fits. It has no way to say whether the
density it achieved is good. Judging an n = 15 container result against the Kepler bound
`pi/sqrt(18) = 0.7405` is a category error, because that bound is for INFINITE packings of
EQUAL spheres.

This task supplies the missing reference, and it is the part of the product that genuinely
comes from concrete mix design rather than from liquid-crystal physics. The exact contact test
in T04 is Perram-Wertheim, which is not a concrete method. **Without T13 the claim that this
work carries a concrete particle packing method into packaging is not true.** With it, it is.

## The two published methods, with what the sources actually say

**Modified Andreasen and Andersen**, Funk and Dinger. A target grading curve for maximum
packing density with a distribution modulus `q`. Funk and Dinger added a finite lower size
limit, which the Fuller curve lacks. Sources give `q <= 0.37` for maximum packing on an
infinite distribution, typically `0.30` to `0.37` for densely packed materials. Andreasen
originally bounded `q` to `0.33` to `0.50`.

**Compressible Packing Model**, de Larrard 1999. A virtual packing density, the maximum
attainable for a given mixture, reduced to a real density by a compaction index `K`
representing compaction energy and stacking type, with two geometric interactions, the wall
effect and the loosening effect.

**SCOPE DECISION.** Implement A&A only. **Do NOT implement CPM.** Its equations are not
verified against a primary source in this repository, the standing rule forbids reporting a
number from an unverified equation, and the clock does not allow the source check. CPM enters
the deck as the named next step, and the wall-effect RATIO is reported as a diagnostic rather
than as a correction.

## Exact signature

```python
GENSANE_N15_CUBE = 2500.0 * math.pi / 17576.0   # = 0.446858308715, best known for
                                                # 15 EQUAL spheres in a cube (Gensane 2004).
                                                # Compute it, do not hardcode the decimal.

def aa_target_cpft(d_mm, d_min, d_max, q=0.37) -> float:
    """Modified Andreasen and Andersen cumulative percent finer than, as a fraction 0..1.

        CPFT(D) = (D**q - d_min**q) / (d_max**q - d_min**q)

    Returns 0.0 at d_min and 1.0 at d_max exactly. Clamped outside [d_min, d_max].
    UNVERIFIED-EQUATION: the functional form above has not been cross-checked against
    the Funk and Dinger primary source in this repository. Any number this produces
    that reaches a deliverable must carry that tag until the check is done.
    """

def actual_cpft(items, d_mm) -> float:
    """Volume fraction of the item set whose equivalent diameter is at or below d_mm.

    Equivalent diameter is 2 * (3V / 4pi)**(1/3), the diameter of the sphere of the
    same volume. Weighted by volume, not by count.
    """

def grading_deviation(items, q=0.37, samples=64) -> float:
    """Root mean square difference between actual_cpft and aa_target_cpft, sampled
    uniformly in log10(d) across the item size range. Zero means the item set already
    sits on the ideal grading curve."""

def best_q(items, q_lo=0.20, q_hi=0.50, steps=61) -> float:
    """The q in [q_lo, q_hi] minimising grading_deviation. Grid search, deterministic."""

def wall_ratio(box, items) -> float:
    """Smallest box inner dimension divided by the largest item equivalent diameter.

    A diagnostic, not a correction. In a concrete pour this is large and the wall
    effect is a footnote. In a shipping box it is small and the wall effect is a
    first-order term, which is the reason this quantity is reported at all.
    """

def benchmark(items, box, achieved_container_density) -> dict:
    """Return
    {"q_used": float, "best_q": float, "grading_deviation": float,
     "wall_ratio": float, "n_items": int,
     "equal_sphere_ceiling": float | None,
     "fraction_of_ceiling": float | None,
     "notes": list[str]}

    equal_sphere_ceiling is GENSANE_N15_CUBE when n == 15, otherwise None.
    Do NOT interpolate or extrapolate a ceiling for other n. There is no published
    value to interpolate and inventing one would be a fabrication.
    """
```

## What the report must say, and must not say

**Must say.** Which q was used. How far the real item set sits from the ideal grading. The
wall ratio. When n is 15, the achieved density as a fraction of the equal-sphere ceiling.

**Must not say.** That the packing "achieved 0.36 against a theoretical 0.74". That comparison
is the category error this task exists to prevent, and it is the exact mistake the prior notes
recorded and corrected.

## Forbidden

- No CPM implementation. No compaction index, no virtual packing density, no wall or loosening
  correction term. Diagnostics only.
- No invented ceiling for `n != 15`.
- Imports limited to `numpy` and the standard library.

## Gate, run this, do not edit it

```
python -m pytest tests/test_benchmark.py -q
```

Expected: `9 passed`

## Done when

The gate passes and no file outside `src/solver/benchmark.py` has changed.

## References, cite these in REFERENCES.md

- Funk and Dinger, modified Andreasen and Andersen distribution, distribution modulus q.
- de Larrard, Compressible Packing Model, 1999. Cited as the framework for the wall and
  loosening effects, not implemented here.
- Gensane, Electron. J. Combin. 11 (2004) #R33, best known packing of 15 equal spheres in a
  cube, `2500*pi/17576`.
