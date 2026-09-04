# Exact-contact packing for irregular produce

**ASU AIR Spark Challenge, 2 to 4 September 2026. Team SMC Labs.**

A packing tool that answers *does this order actually fit* exactly, instead of by comparing
volumes. Built on an exact ellipsoid contact criterion from statistical physics, with the
language model held out of every engineering decision.

Every number in this file was produced by running this codebase on 2026-09-03. Where a number
comes from prior notes and was not regenerated here, it says so.

---

## The problem

A person packing a produce box or a meal kit fills a container with items that are irregular,
crushable and heavy in different amounts. They decide by eye whether an order fits, which box
to reach for, and what goes on the bottom.

The software that exists answers the wrong question. Industry cartonization decides fit by
**liquid fill**, summing item volumes and choosing a box large enough to hold that total. For
irregular bodies that is not a fit test. A set of ellipsoids whose volumes sum to less than the
box volume routinely does not fit, because disordered packing never reaches a density of 1.
The vendors name the failure themselves with the four-foot shovel that has less volume than the
box and still will not go in.

The academic alternatives are heuristics. Irregular 3D bin packing is NP-hard, and the 2022
state of the art is a constructive dynamic-volume heuristic. That same literature records the
status quo directly. Irregular-item packing in real operations is performed by workers with no
reference packing solution provided in advance.

## What this does differently

**The contact test is exact, and it comes from the wrong field on purpose.**

Perram-Wertheim is an exact overlap criterion for two ellipsoids, developed for the statistical
physics of liquid crystals rather than for operations research. For shape matrices A and B and
centre offset d, with the pencil `C(l) = (1-l)A + lB`,

```
F(l) = l (1-l) d^T A C(l)^-1 B d        mu^2 = max_l F(l)
```

`mu` is the exact factor by which both bodies must be scaled to reach tangency. `mu < 1` is
overlap, `mu > 1` is disjoint. `F` is strictly concave, so the maximiser is unique and a
safeguarded search cannot fail. That is a proof, not a tolerance.

**The tool refuses rather than flatters.** Any configuration with more than `1e-3` mm of
interpenetration is rejected and its density is never reported. That rule earned its keep
today. A bug in the packer stored each body with its original axes after validating it with
permuted axes, same centre, different geometry, and the gate refused the result at 0.069 mm.
Nothing else in the system would have noticed.

**Three density bases, never mixed.** Container, hull and radical (Laguerre) answer different
questions and every reported number names its basis. When no Laguerre cell is interior, the
basis is reported as unavailable rather than fudged.

## Architecture, the language model holds no engineering authority

```
user text  ->  AIR model  ->  tool call  ->  solver  ->  numbers  ->  AIR model  ->  narration
                   |                            |
             compiles English            owns every placement,
             into a typed schema         contact test and density
```

The model emits a function call. It does not emit a coordinate, a volume, a density or a
verdict. The tool-call transcript is shown verbatim in the interface, so the claim is visible
rather than asserted. This is a real transcript from 2026-09-03, 18:40.

```
model -> list_items({})
model -> pack({"box_mm": [260,190,130],
               "rules": [{"kind":"fragile","item":"body-7","raw":"body-7 and body-8 are eggs and fragile ..."},
                         {"kind":"fragile","item":"body-8","raw":"..."}]})
tool  <- {"placed": 8, "unplaced": 0, "gate": "PASSED", "max_penetration_mm": 0.0,
          "container_density": 0.28166}
model :  All 8 items were successfully packed into the 260x190x130 mm box with a container
         density of 28.17%. The fragile items (body-7 and body-8) were kept off the bottom.
```

The model wrote 28.17%. The solver had returned 0.28166. Every rule the compiler cannot
classify is returned in `unrecognized` and surfaced to the user. A dropped fragility rule breaks
something in a real parcel, so it is never dropped silently.

## Pipeline, scan to packed box

```
LiDAR cloud -> RANSAC table plane -> DBSCAN clusters -> ellipsoid fit -> pack -> gate -> report
```

Measured on `scans/synthetic_produce.npy`, a rendered single-orbit capture of eight bodies with
exact ground truth, 8 mm sensor noise, single-view occlusion and a deliberate 3% inward TSDF
bias. No real produce was scanned today, see Not done.

| Stage | Result, 2026-09-03 |
|---|---|
| Points to bodies | 23,200 points, 8 clusters, 8 fitted ellipsoids, 1.19 s |
| Segmentation eps | density-adaptive, 3.0 times the median point spacing, 9.45 mm on this cloud |
| Fit volume error | 9.2% mean, 17.4% worst, against ground truth |
| Fit axis error | 7.1% mean |
| Packing | 8 of 8 placed in a 260 x 190 x 130 mm box, 37.6 s |
| Container density | 0.282 |
| Hull density | 0.527 |
| Laguerre density | unavailable, no interior cell at n = 8 |
| Fill height | 63.0 mm |
| Max interpenetration | 0.00 mm against a 1e-3 mm gate |
| Liquid fill would say | box volume is 7.33 times the summed item volume, so it fits with room to spare |
| Contact test | 0.0484 ms per call, 35 times faster than the first version |
| One AIR turn, end to end | 32.1 s measured at 18:40 with the earlier 21.5 s pack. The pack is now 37.6 s, the turn was not re-timed |

Candidate positions are inset by each body's own horizontal extent, so bodies can sit against
the walls. That is why the container density rose from 0.259 to 0.282 during the evening, and
it is also what made the stacked-spheres check reachable at all. A conservative outer-bound
early-out skips the exact contact test only when two bounding spheres are provably separated.
It is never used to claim an overlap, so it cannot hide one.

The fit uses the 85th percentile of the footprint for the horizontal semi-axes and tangency
to the table for the vertical one. Using the maximum projection instead gives about 300%
error, which is why the constant is written into the spec.

## Verification

The test suite checks against **theorems and closed forms, never against a previous run of this
code**.

| Check | Reference |
|---|---|
| Two-sphere contact | `mu = abs(d) / (r1 + r2)`, exact |
| Identical ellipsoids on a principal axis | `mu = abs(d) / (2a)`, exact |
| Penetration for spheres | `(r1 + r2) - abs(d)` |
| FCC packing fraction | `pi / sqrt(18) = 0.740480489693` |
| Sphere in a tight cube | `pi / 6` |
| 15 equal spheres in a cube | `2500 pi / 17576 = 0.446858308715`, Gensane 2004 |
| Ellipsoid surface | every semi-axis tip satisfies `x^T M x = 1` |
| Two stacked spheres | centres at exactly r and 3r |

One test exists purely to catch a shortcut. A prolate body offset along its **short** axis
genuinely overlaps, and a bounding-sphere proxy reports it as disjoint, so any implementation
that reaches for the shortcut fails that test.

**113 tests passing across 14 gates** on 2026-09-03, 19:35. One is a live call to AIR and
skips without a key. The gates for T17 ingest and T18 upload exist but their modules were not
written, so they are not collected.

Test integrity is enforced by checksum. `tools_check_tests.py` holds a SHA-256 manifest of
every test file, because the tests are the contract and an executor that edits them has
verified nothing. That rule was violated once today by the executor and caught.

## Running it

```
pip install -r requirements.txt
set OPENAI_API_KEY=...            # from https://voyager.rc.asu.edu, LLM Access tab
python -m src.air.bridge          # loads the scan, packs it, serves http://127.0.0.1:8000
```

The ASU VPN is required at `sslvpn.asu.edu/2fa`. The browser talks only to the local bridge
and never sees the key. `python -m src.air.bridge --no-pack` skips the 20 second initial pack.

In the page, drag any body into another. The exact contact test runs on every pointer move and
the readout turns red with the interpenetration in millimetres. Release and it snaps back.

## What powers this app

All AI inference runs on the ASU AI Research Platform.

- Endpoint `https://openai.rc.asu.edu/v1`
- Model called at runtime `devstral2-123b`
- **No other AI provider is called at runtime**, in any code path, including unused fallbacks.

## How this was built

**Prior work.** The ellipsoid fitting and contact mathematics derive from personal research
notes written 30 and 31 August 2026, before this challenge began. They were never submitted to
any competition. The code was rebuilt from those notes for this challenge and no source file
was carried across.

**Published methods used.** Perram-Wertheim exact ellipsoid contact. Modified Andreasen and
Andersen grading curve, Funk and Dinger. Compressible Packing Model concepts, de Larrard 1999,
cited as framework and not implemented. Gensane 2004 for the finite-size sphere ceiling.

**Development tooling, stated exactly.**

*Devstral 2 123B, hosted on ASU AIR, driven through OpenCode*, wrote the first implementations
of `types`, `schema`, `gate`, `place`, `density`, `benchmark`, `segment`, `synth`, `tools`,
`bridge`, the first `contact`, the second `fit`, and `web/index.html` and `web/style.css`.

*Claude, Anthropic, via Claude Code (Opus 5, then Fable 5.1)*, wrote every specification in
`specs/`, every test in `tests/`, and the watchdog and checksum tooling. The team policy until
18:05 on 2026-09-03 was that all shipped code is generated on AIR. At 18:05, with four hours to
the deadline, the team overrode that policy. After that point Claude directly wrote or
rewrote `contact.py` (the 35x speedup), two fixes in `place.py` including the item-axes bug
above, `fit.py` (first version and one field fix), `synth.py`, the `pack` branch and the
conversation loop in `tools.py`, `bridge.py` and `web/app.js`, plus the layout and font overrides
appended to `web/style.css` and the placeholder text in `web/index.html`, which had shown a passed
gate at 0.0000 mm before any data arrived. The earlier versions of `tools.py`, `bridge.py` and
`web/app.js` did not work.

*Human authorship.* The mathematics, the architecture rule that the model holds no engineering
authority, the choice of Perram-Wertheim, the three density bases, the overlap gate, the
measured findings in the prior notes, and every engineering decision behind them.

## Honest limits

- Bodies are modelled as convex triaxial ellipsoids. A banana or a chili is not one, and is
  outside the model. This is stated rather than hidden.
- The iPhone LiDAR carries centimetre-class depth error, at or below the size of the smallest
  items. Any fit verdict whose margin is smaller than the propagated axis error is noise. The
  margin propagation, T11, was not built today.
- The TSDF surface sits about 3% inside the true one, a systematic bias that does not average
  away. The synthetic fixture injects it on purpose.
- The Kepler bound `pi/sqrt(18) = 0.7405` applies to infinite packings of equal spheres.
  Comparing a small container result against it is a category error and this project does not
  make it.
- The packing search is a deterministic constructive heuristic. It is not optimal and does not
  claim to be. The claim is that its answer is *correct*, which is a different and more useful
  guarantee. Eight bodies take about 40 seconds. A gate rejection was seen once at ten bodies
  before the item-axes fix and has not been re-tested at ten since.
- The +9.1% shape gain and the other ablation figures in the prior notes were **not
  regenerated** in this codebase and are not claimed here.

## Not done

- **No real produce was scanned.** The demo runs on a synthetic capture with exact ground
  truth. T17, the Stray Scanner folder to point cloud ingest, was specified and gated but not
  written.
- **T18 browser upload** was specified and gated but not written. The scan is loaded by the
  bridge at startup.
- **T06b** early-out and the drag fast path were not applied. Drag checks use the exact test on
  every move, which is fast enough at eight bodies.
- **T11** margin propagation and **T12** the ratio panel were not built. The liquid-fill
  comparison is reported as a number above, not as a live panel.
- The Laguerre basis is unavailable at n = 8, as designed, and was not exercised at larger n.
