# Task Ledger

**Executor** Devstral 2 123B on ASU AIR, driven through OpenCode with tools.
**Verified 2026-09-03** Tool calling works. All four candidate models returned a correct
structured `tool_calls` entry on a live probe. The gates below are therefore self-driving,
the model runs pytest itself and iterates until green.

## Dependency reality, verified 2026-09-03

`open3d` has NO wheel for Python 3.13 and cannot be installed. `scikit-learn` is not
installed. Available and verified: `numpy 2.2.3`, `scipy 1.16.1`, `matplotlib 3.10.5`,
`pytest 9.0.3`. RANSAC and DBSCAN are implemented in T08. See `requirements.txt`.

## The rules that make this work

1. **One task, one fresh OpenCode session.** Never chain tasks in one loop. The gateway does
   not expose context length and agent endurance is unproven, so short loops sidestep both
   unknowns.
2. **The executor never writes a test.** Tests are the contract and they were written before
   any implementation existed. A model that writes both its code and its test has verified
   nothing, which is the same defect as validating against a previous run.
3. **The executor never edits `tests/`.** If a gate seems wrong, it stops and reports.
4. **One task touches one file.** Anything outside `CREATE:` is off limits.
5. **A gate is a command with an expected output.** Never a judgement.

## How to run one task

```
# fresh session, project root
opencode
> read specs/T04_contact.md and implement it exactly. Run the gate until it passes.
> Do not modify any file outside the CREATE line. Do not edit tests/.
```

## Dependency graph

```
T01 types ──┬── T02 schema ── T03 validator ──┬── T10 AIR compiler
            │                                  │
            ├── T04 contact ── T05 gate ───────┼── T06 placement ── T07 density
            │                                  │
            └── T08 segment ── T09 fit ────────┘        │
                   (needs scan)                          └── T11 error propagation
                                                              └── T12 panel
```

## The tasks

| ID | Title | Creates | Depends | Needs scan | Gate |
|---|---|---|---|---|---|
| **T00** | **Assets and compliance gate** | *(no code, BINDING)* | — | no | manual checklist |
| T01 | Types and units | `src/types.py` | T00 | no | `pytest tests/test_types.py` |
| **T17** | **Stray Scanner ingest, folder to .npy** | `src/capture/ingest.py` | T01 | no | `pytest tests/test_ingest.py` |
| **T18** | **Browser upload of a cloud** | `src/air/upload.py` | T15, T17 | no | `pytest tests/test_upload.py` |
| T02 | Constraint schema | `src/schema.py` | T01 | no | `pytest tests/test_schema.py` |
| T03 | Schema validator | `src/validate_rules.py` | T02 | no | `pytest tests/test_validate_rules.py` |
| T04 | Exact contact, Perram-Wertheim | `src/geometry/contact.py` | T01 | no | `pytest tests/test_contact.py` |
| T05 | Overlap gate | `src/geometry/gate.py` | T04 | no | `pytest tests/test_gate.py` |
| T06 | Drop and settle placement | `src/solver/place.py` | T05 | no | `pytest tests/test_place.py` |
| T07 | Density on three bases | `src/solver/density.py` | T06 | no | `pytest tests/test_density.py` |
| T08 | Segmentation | `src/capture/segment.py` | T14 | **no, T14 supplies it** | `pytest tests/test_segment.py` |
| **T09** | **Ellipsoid fit, THE MISSING LINK** | `src/capture/fit.py` | T08b | no | `pytest tests/test_fit.py` |
| **T08b** | **Density-adaptive DBSCAN eps** | `src/capture/segment.py` | T08 | no | `pytest tests/test_segment.py` |
| **T10** | **AIR tool layer, model calls the solver** | `src/air/tools.py` | T02, T06, T07 | no | `pytest tests/test_air_tools.py` |
| **T15** | **Local bridge, browser never sees the key** | `src/air/bridge.py` | T10 | no | `pytest tests/test_bridge.py` |
| T11 | Error propagation on verdict | `src/solver/margin.py` | T07 | no | `pytest tests/test_margin.py` |
| T12 | Comparison panel, ratio not binary | `src/report/panel.py` | T11 | no | `pytest tests/test_panel.py` |
| **T16** | **Web viewport, 3D and drag** | `web/*` | T15 | no | `pytest tests/test_viewport.py` |
| **T13** | **A&A grading benchmark** | `src/solver/benchmark.py` | T01, T07 | no | `pytest tests/test_benchmark.py` |
| **T14** | **Synthetic capture generator** | `src/capture/synth.py` | T01 | **no** | `pytest tests/test_synth.py` |

**T00 is read first by everyone, it is the compliance gate and it takes five minutes.**

**T01 to T07, T10 and T13 need no scan data.** They can start the moment T02 is frozen and run
while the capture trip is still out.

## Parallel assignment, three people

| Track | Tasks | Blocked by |
|---|---|---|
| A, geometry and solver | T04, T05, T06, T07, T11, T13 | T01 only |
| B, AIR and product | T02, T03, T10, T15, T12 | T01 only |
| C, capture and delivery | T14, then T08, T09, then the scan trip | nothing, T14 unblocks it |

T01 is written by whoever starts first and takes fifteen minutes. Everything else forks.

## Status

Set to DONE only when the gate passes with the stated expected output.

| ID | Status | Owner | Notes |
|---|---|---|---|
| T01 | TODO | | |
| T02 | TODO | | blocker for two tracks |
| T03 | TODO | | |
| T04 | TODO | | |
| T05 | TODO | | |
| T06 | TODO | | the packer, missing from the first design draft |
| T07 | TODO | | |
| T08 | TODO | | needs the frozen fixture |
| T09 | TODO | | needs the frozen fixture |
| T10 | TODO | | tool-calling architecture, gate runs offline, live test auto-skips |
| T15 | TODO | | stdlib http.server only, 30 min budget |
| T17 | TODO | | THE MISSING FIRST STAGE, nothing else produced a point cloud |
| T18 | TODO | | raw body upload, no multipart parsing |
| T16 | TODO | | engineering console, NOT a chat product. Drag-to-overlap is the one must-ship interaction |
| T11 | TODO | | |
| T12 | TODO | | |
| T13 | TODO | | the only genuinely concrete-derived method in the build |
| T14 | TODO | | unblocks T08, T09 and T11 without the real scan |

## Abort gates

If a task is not green by its stated budget, cut it and record the cut in the README as
future work. Do not let one task eat the freeze.

| ID | Budget | If it fails |
|---|---|---|
| T06 | 90 min | Fall back to fixed-grid placement, lower density, still honest |
| T09 | 60 min | Fall back to the mirrored-hull volume, drop the ellipsoid axes |
| T10 | 45 min | Hard-code two compiled rules, keep the live call as one demo path |
| T11 | 30 min | Report the verdict without a margin and say so on the slide |
| T13 | 45 min | Report the wall ratio alone and drop the grading deviation |
| T14 | 45 min | Hand-build one fixture cloud in a .npy and gate against that |
| T16 | 120 min | Drop drag interaction, keep orbit plus a static render. Never drop the transcript strip |
