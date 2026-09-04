# T16 — Web viewport, 3D point cloud, box and interactive placement

DEPENDS ON: T15 bridge
CREATE: `web/index.html`, `web/app.js`, `web/style.css`, `web/vendor/three.min.js`
DO NOT MODIFY: any other file. Never `tests/`.

## What it shows

Three things at once, because the argument only lands when they are visible together.

1. The **raw point cloud** from the scan, as points.
2. The **box**, as a wireframe.
3. The **fitted ellipsoids** in their packed positions, solid.

The user orbits, pans, zooms, and **drags a placed item**. On drag, the exact contact test
runs and the readout updates live.

## The killer interaction, build this before anything decorative

Drag an item into another one. The moment interpenetration exceeds the gate tolerance, the
item turns red and the readout says

```
OVERLAP REJECTED    4.213 mm    tolerance 0.001 mm
```

Release it and it snaps back to its last valid position.

That is the whole thesis, interactive. Every other packing tool answers "does it fit" with a
volume estimate. Yours answers it exactly, and the user can feel it by dragging. **If only one
interaction ships, ship this one.**

## Layout, fixed

```
+--------------------------------------------------------------+
|  TOOLBAR   box selector | reset view | re-pack | [ASU AIR: ok]|
+---------------------------+----------------------------------+
|                           |  MEASUREMENTS                    |
|                           |  container   0.361               |
|      3D VIEWPORT          |  hull        0.548               |
|      point cloud          |  laguerre    unavailable (n<int) |
|      box wireframe        |  fill height 128.4 mm            |
|      placed ellipsoids    |  gate        PASSED 0.0002 mm    |
|                           |  ---------------------------     |
|      ~65% width           |  BENCHMARK                       |
|                           |  best q      0.34                |
|                           |  wall ratio  4.0                 |
|                           |  ceiling     n/a (n=12)          |
+---------------------------+----------------------------------+
|  RULES  [ the eggs are fragile, keep under 20 lb    ] [SEND]  |
|  TRANSCRIPT                                                   |
|  model -> pack({"box_mm":[300,200,150],"rules":[...]})        |
|  tool  <- {"placed":11,"unplaced":1,"container_density":0.361}|
+--------------------------------------------------------------+
```

The transcript strip is not decoration. It is the evidence that the model called a tool and
did not compute an answer. Render it **verbatim**, monospace, no prettifying.

## Visual contract, binding

This is an engineering console. It should look like it belongs beside CAD software or a
warehouse management system, not beside a chat product.

**Palette, use exactly these.**

| Token | Hex | Use |
|---|---|---|
| `--bg` | `#f4f5f6` | page |
| `--panel` | `#ffffff` | panels |
| `--line` | `#c9ccd1` | borders, 1px |
| `--ink` | `#1c1f23` | primary text |
| `--muted` | `#5b6068` | labels, secondary |
| `--accent` | `#0b6b53` | one accent only, selected state and PASS |
| `--warn` | `#b3261e` | overlap, REJECTED, unplaced |
| `--viewport` | `#e8eaec` | 3D background |

**Type.** UI in `system-ui, "Segoe UI", Arial, sans-serif` at 13px.
**Every number** in `ui-monospace, "Cascadia Mono", Consolas, monospace` with
`font-variant-numeric: tabular-nums`, right-aligned in its column, fixed decimal places.

**Geometry.** Border radius 2px maximum. 1px solid borders. No shadows except a single
`0 1px 0 var(--line)` under the toolbar. Dense spacing, 6px to 10px padding, not 24px.

## FORBIDDEN, this list is the point of the section

- No gradients anywhere. Not in the background, not on a button, not in the 3D scene.
- No purple, violet, indigo, or teal-to-purple. The accent is the single green above.
- No emoji, anywhere, including in the transcript or a status badge.
- No rounded chat bubbles, no avatar circles, no typing indicator, no "thinking" animation.
- No glassmorphism, blur, translucency, or neon glow.
- No hero heading, no marketing copy, no tagline in the interface.
- No animated transitions longer than 120ms. No easing curves with bounce.
- No icon font and no icon library. Text labels. `RE-PACK`, not a sparkle icon.
- No dark mode toggle. One theme, light, done well.
- No React, Vue, Svelte, Tailwind, or any build step. Plain HTML, CSS and JS.

If a reviewer could mistake a screenshot for an AI chat product, the contract is broken.

## Technical constraints for the executor

- **`web/vendor/three.min.js` is VENDORED, not loaded from a CDN.** A CDN that is slow at
  9 PM during a live pitch is an avoidable failure. Download it once and commit it.
- Point cloud is `THREE.Points` with `PointsMaterial`, size 1.5, no vertex colours needed.
- The box is `THREE.LineSegments` from `EdgesGeometry(BoxGeometry(w, h, d))`.
- An ellipsoid is `SphereGeometry(1, 24, 16)` scaled to the semi-axes. Do not build a custom
  geometry.
- Orbit control: write about 60 lines of pointer handling. **Do not import OrbitControls from
  a separate CDN file.** One vendored file only.
- Drag: raycast on pointerdown, move in the plane parallel to the ground at the item centre
  height, POST to `/api/check` on each move at most every 50 ms, colour by the response.
- All geometry arrives from the bridge in **millimetres**. Convert once at load.

## Routes this needs from T15

| Route | Body | Returns |
|---|---|---|
| `/api/scene` | GET | `{"points_mm": [...], "box_mm": [...], "placements": [...]}` |
| `/api/check` | POST `{"label": str, "centre_mm": [x,y,z]}` | `{"penetration_mm": float, "ok": bool}` |
| `/api/turn` | POST `{"text": str}` | the T10 `run_turn` dict, transcript included |

## Gate, run this, do not edit it

```
python -m pytest tests/test_viewport.py -q
```

Expected: `9 passed`

The gate is a static contract check, not a visual test. It asserts no key is present, no
external host is contacted, the required element ids exist, and the forbidden style tokens are
absent. **Visual quality is checked by a human against the contract above**, and that review
is on the checklist in T00.

## Done when

The gate passes, and a human has confirmed against the visual contract that the page does not
look like a chat product.
