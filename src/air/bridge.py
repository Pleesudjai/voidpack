"""Local bridge. The browser talks to 127.0.0.1:8000, this process holds the key.

Standard library http.server only. Binds to loopback, never 0.0.0.0. Fails at
startup if OPENAI_API_KEY is absent rather than 500-ing during a live pitch.
The key never appears in a response body, a header, or a log line.

Routes
    GET  /                      web/index.html
    GET  /app.js /style.css /vendor/three.min.js
    GET  /api/scene             points, box, placements, report, benchmark
    POST /api/turn   {"text"}   one AIR turn, transcript returned verbatim
    POST /api/optimize {}             smallest catalog box that takes every item
    POST /api/counts {"counts"}       set how many of each item type, then re-pack
    POST /api/pack   {"box_index"?}   re-run the solver with the last compiled rules
    POST /api/check  {"label","centre_mm"}   penetration for one moved item

Run:  python -m src.air.bridge [--scan scans/synthetic_produce.npy] [--port 8000]
"""
import argparse
import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import urlparse

import numpy as np

from src.geometry.gate import OverlapRejected, TOLERANCE_MM, max_penetration_mm, pair_penetration_mm
from src.schema import empty_ruleset
from src.solver.density import report as density_report
from src.solver.place import pack
from src.types import Box, Item, Placement
from src.air.tools import run_turn

WEB_DIR = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "web"))

# Only these paths are ever served, and none is built from user input, so there
# is no traversal surface.
STATIC = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/index.html": ("index.html", "text/html; charset=utf-8"),
    "/app.js": ("app.js", "application/javascript; charset=utf-8"),
    "/style.css": ("style.css", "text/css; charset=utf-8"),
    "/vendor/three.min.js": (os.path.join("vendor", "three.min.js"),
                             "application/javascript; charset=utf-8"),
}

# Shipping catalog, inner dimensions in mm, ordered smallest volume first so
# the index order and the size letters agree. M is the box every earlier result
# in the README was measured in, 260 x 190 x 130.
# A typo in a spinner must not ask the solver for a thousand bodies mid-pitch.
MAX_COPIES = 12

# Container density the free-size search opens at. Measured packs on these
# captures land near 0.32, so 0.28 gives a start box that holds everything
# without wasting the first round on a container nothing needs.
START_DENSITY = 0.28

# Points kept per library thumbnail. The whole scene is about 500 kB of JSON
# already, so the four thumbnails are thinned rather than sent whole.
THUMB_POINTS = 420

DEFAULT_CATALOG = [
    Box(220.0, 160.0, 120.0, "S"),
    Box(260.0, 190.0, 130.0, "M"),
    Box(300.0, 220.0, 180.0, "L"),
    Box(360.0, 260.0, 210.0, "XL"),
]


def _placement_json(p: Placement) -> dict:
    """The item's axes as STORED, so axes[2] is the vertical semi-axis once placed."""
    return {"label": p.item.label,
            "axes_mm": [float(a) for a in p.item.axes],
            "centre_mm": [float(v) for v in p.centre],
            "yaw": float(p.yaw),
            "mass_g": float(p.item.mass_g),
            "fragile": bool(p.item.fragile),
            "volume_mm3": float(p.item.volume_mm3())}


def _placement_from_json(d: dict) -> Placement:
    it = Item(label=d["label"], axes=tuple(float(a) for a in d["axes_mm"]),
              mass_g=float(d.get("mass_g", 0.0)), fragile=bool(d.get("fragile", False)))
    return Placement(item=it, centre=np.asarray(d["centre_mm"], dtype=float),
                     yaw=float(d.get("yaw", 0.0)))


class State:
    """Everything the browser can see. Numbers come from the solver or not at all."""

    def __init__(self, items, box_catalog, points_m=None, point_rgb=None,
                 mass_source=None, n_table_points=0, anchors=None):
        # `items` is the LIBRARY, one fitted body per produce type. The packed
        # scene is the library expanded by `self.counts`, so three eggs are
        # three separate bodies with their own centres and yaws.
        from src.capture.library import default_counts, expand
        self.library = list(items)
        self.counts = default_counts(self.library)
        self.mass_source = mass_source
        self.anchors = dict(anchors or {})
        # The counts the OPT box was sized for. Changing the item set makes
        # that box stale, and a stale OPT box silently strands items.
        self.opt_counts = None
        # Points arrive ordered table first, bodies after, so the page can
        # draw the two halves at different sizes without a per-point flag.
        self.n_table_points = int(n_table_points)
        self.items = expand(self.library, self.counts)
        self.box_catalog = list(box_catalog) or list(DEFAULT_CATALOG)
        # Default to M, the 260 x 190 x 130 box every published result in the
        # README was measured in. Index 0 is S once the catalog is sorted by
        # volume, so an index literal would have moved the baseline silently.
        self.box_index = next((i for i, b in enumerate(self.box_catalog)
                               if getattr(b, "name", "").upper() == "M"), 0)
        self.points_mm = (np.asarray(points_m, dtype=float) * 1000.0
                          if points_m is not None and len(points_m) else np.zeros((0, 3)))
        # Per-point colour sampled from the capture video, DISPLAY only. The
        # solver never reads it, per the architecture rule that every number
        # comes from geometry. Empty when the scan carries no colour sidecar.
        self.point_rgb = (np.asarray(point_rgb, dtype=np.uint8)
                          if point_rgb is not None and len(point_rgb) == len(self.points_mm)
                          else np.zeros((0, 3), dtype=np.uint8))
        self.placements = []
        self.unplaced = []
        self.report = None
        self.gate = "not run"
        self.max_penetration_mm = None
        self.ruleset = empty_ruleset()
        self.benchmark = None
        self.last_error = None
        self.lock = threading.Lock()

    @property
    def box(self) -> Box:
        return self.box_catalog[self.box_index]

    @property
    def opt_stale(self) -> bool:
        """True when the selected box is an OPT box sized for a different set."""
        if getattr(self.box_catalog[self.box_index], "name", "") != "OPT":
            return False
        return self.opt_counts is not None and self.opt_counts != self.counts

    @property
    def masses_known(self) -> bool:
        """True only when every library body carries a positive mass."""
        return bool(self.library) and all((getattr(i, "mass_g", 0.0) or 0.0) > 0
                                          for i in self.library)

    def occupied_mm(self, placements):
        """Bounding box the placements actually fill, in mm, yaw included.

        The horizontal half-extent of a yawed ellipsoid is
        hypot(a cos y, b sin y) along x and hypot(a sin y, b cos y) along y,
        which is exact, not a bounding sphere. Compaction is included because
        the solver tests contact on the effective axes.
        """
        import math
        w = d = h = 0.0
        for q in placements:
            a, b, c = (float(v) * float(q.item.compaction) for v in q.item.axes)
            cy, sy = math.cos(q.yaw), math.sin(q.yaw)
            ex = math.hypot(a * cy, b * sy)
            ey = math.hypot(a * sy, b * cy)
            w = max(w, float(q.centre[0]) + ex)
            d = max(d, float(q.centre[1]) + ey)
            h = max(h, float(q.centre[2]) + c)
        return w, d, h

    def optimize_box(self, rounds=3, clearance_mm=1.5, search_grid=5) -> dict:
        """Smallest FREE-SIZE box found by shrink and re-pack. Millimetres.

        Not restricted to the catalog. Each round packs for real and then
        shrinks the container to what the placements occupy, plus a clearance
        so a body tangent to a wall is not pushed through it. A round that
        fails to place everything is discarded and the last good box stands.

        Greedy descent, reported as such. It is not a proof of optimality.
        """
        # Starting from the largest catalog box wastes the most expensive round
        # on a container nothing needs. Start instead from a cube whose volume
        # is the item volume at START_DENSITY, floored by the largest single
        # body so the biggest item always fits, and fall back to the largest
        # catalog box if that first pack cannot place everything.
        import math
        sum_v = sum(i.volume_mm3() for i in self.items) or 1.0
        side = (sum_v / START_DENSITY) ** (1.0 / 3.0)
        floor = 2.0 * max((max(i.axes) * i.compaction) for i in self.items) * 1.05
        side = float(max(side, floor))
        start_box = Box(side, side, side, "OPT")
        fallback = max(self.box_catalog, key=lambda b: b.volume_mm3())
        box = Box(start_box.width, start_box.depth, start_box.height, "OPT")
        best, rounds_log = None, []
        for k in range(int(rounds)):
            try:
                pl, un = pack(self.items, box, self.ruleset, grid=search_grid)
                pen = float(max_penetration_mm(pl))
            except OverlapRejected as e:
                rounds_log.append({"box_mm": [box.width, box.depth, box.height],
                                   "rejected_mm": float(e.penetration_mm)})
                break
            except Exception as e:
                rounds_log.append({"error": str(e)[:120]})
                break
            fits = (len(un) == 0 and pen <= TOLERANCE_MM)
            rounds_log.append({"box_mm": [round(float(box.width), 1), round(float(box.depth), 1),
                                          round(float(box.height), 1)],
                               "volume_L": round(float(box.volume_mm3()) / 1e6, 3),
                               "placed": len(pl), "unplaced": len(un),
                               "max_penetration_mm": pen, "fits": fits})
            if not fits:
                if best is None and k == 0 and fallback.volume_mm3() > box.volume_mm3():
                    box = Box(fallback.width, fallback.depth, fallback.height, "OPT")
                    start_box = fallback
                    continue
                break
            best = box
            w, d, h = self.occupied_mm(pl)
            nxt = Box(w + clearance_mm, d + clearance_mm, h + clearance_mm, "OPT")
            # Stop when a round buys less than 1% of the volume.
            if nxt.volume_mm3() >= box.volume_mm3() * 0.99:
                break
            box = nxt
        if best is None:
            self.do_pack()
            return {"found": False, "rounds": rounds_log,
                    "reason": "no box in the search placed every item"}
        # The optimised box replaces any previous OPT entry, never a catalog one.
        self.box_catalog = [b for b in self.box_catalog if getattr(b, "name", "") != "OPT"]
        self.box_catalog.append(best)
        self.box_index = len(self.box_catalog) - 1
        # The search ran coarse. The answer only counts once the full-resolution
        # packer reproduces it, so grow the box a little and retry rather than
        # report a box the shipped settings cannot actually fill.
        grew = 0
        for _ in range(3):
            self.do_pack()
            if self.gate == "PASSED" and not self.unplaced:
                break
            grew += 1
            best = Box(best.width * 1.04, best.depth * 1.04, best.height * 1.04, "OPT")
            self.box_catalog[self.box_index] = best
        smallest_fitting = None
        for b in sorted(self.box_catalog, key=lambda z: z.volume_mm3()):
            if getattr(b, "name", "") == "OPT":
                continue
            if b.width >= best.width and b.depth >= best.depth and b.height >= best.height:
                smallest_fitting = b
                break
        self.opt_counts = dict(self.counts)
        return {"found": True, "rounds": rounds_log, "grown_after_search": grew,
                "box_mm": [round(float(best.width), 1), round(float(best.depth), 1),
                           round(float(best.height), 1)],
                "volume_L": round(float(best.volume_mm3()) / 1e6, 3),
                "start_mm": [round(float(start_box.width), 1), round(float(start_box.depth), 1),
                             round(float(start_box.height), 1)],
                "start_volume_L": round(float(start_box.volume_mm3()) / 1e6, 3),
                "saved_frac": round(1.0 - float(best.volume_mm3()) / float(start_box.volume_mm3()), 4),
                "beats_catalog": (getattr(smallest_fitting, "name", None)
                                  if smallest_fitting is not None else None)}

    def set_counts(self, counts) -> dict:
        """Replace the per-type counts and rebuild the scene. Counts are clamped.

        A count is capped at MAX_COPIES so a typo cannot ask for a thousand
        bodies and hang the solver during a pitch.
        """
        from src.capture.library import expand
        if not isinstance(counts, dict):
            return {"error": "counts must be an object of label to integer"}
        clean = {}
        for item in self.library:
            raw = counts.get(item.label, self.counts.get(item.label, 1))
            try:
                n = int(raw)
            except (TypeError, ValueError):
                return {"error": "count for %r is not an integer" % item.label}
            if n < 0:
                return {"error": "count for %r is negative" % item.label}
            clean[item.label] = min(n, MAX_COPIES)
        if sum(clean.values()) == 0:
            return {"error": "at least one item is needed"}
        with self.lock:
            self.counts = clean
            self.items = expand(self.library, clean)
        return {"counts": clean, "n_items": len(self.items)}

    def do_pack(self, grid=8, yaws=4):
        """Run the solver. A refusal by the gate is reported, never hidden."""
        with self.lock:
            try:
                pl, un = pack(self.items, self.box, self.ruleset, grid=grid, yaws=yaws)
                self.placements, self.unplaced = pl, un
                self.report = density_report(pl, self.box)
                self.max_penetration_mm = float(max_penetration_mm(pl))
                self.gate = "PASSED"
                self.last_error = None
            except OverlapRejected as e:
                self.gate = "REJECTED"
                self.max_penetration_mm = float(e.penetration_mm)
                self.last_error = "overlap gate refused the packing: %.4f mm" % e.penetration_mm
            except Exception as e:  # never let a solver error take the server down
                self.gate = "ERROR"
                self.last_error = str(e)[:200]
            self._benchmark()

    def _benchmark(self):
        try:
            from src.solver.benchmark import benchmark
            ach = self.report["container"] if self.report else 0.0
            self.benchmark = benchmark(self.items, self.box, ach)
        except Exception:
            self.benchmark = None

    def adopt(self, placements_json, rules=None):
        """Take placements the model's pack tool produced, so the view matches the transcript."""
        with self.lock:
            self.placements = [_placement_from_json(d) for d in placements_json]
            if rules is not None:
                self.ruleset = {"rules": list(rules), "unrecognized": []}
            try:
                self.report = density_report(self.placements, self.box)
                self.max_penetration_mm = float(max_penetration_mm(self.placements))
                self.gate = "PASSED" if self.max_penetration_mm <= TOLERANCE_MM else "REJECTED"
            except Exception as e:
                self.last_error = str(e)[:200]
            self._benchmark()

    def scene(self) -> dict:
        with self.lock:
            pts = self.points_mm
            rgb = self.point_rgb
            if len(pts) > 60000:
                step = int(np.ceil(len(pts) / 60000.0))
                pts = pts[::step]
                rgb = rgb[::step] if len(rgb) else rgb
            b = self.box
            return {
                "points_mm": np.round(pts, 1).tolist(),
                "points_rgb": rgb.tolist() if len(rgb) == len(pts) else None,
                "box_mm": [b.width, b.depth, b.height],
                "box_index": self.box_index,
                "box_catalog": [[c.width, c.depth, c.height] for c in self.box_catalog],
                "box_names": [c.name for c in self.box_catalog],
                "library": [{"label": i.label, "axes_mm": [float(a) for a in i.axes],
                             "mass_g": float(i.mass_g)} for i in self.library],
                "counts": dict(self.counts),
                "max_copies": MAX_COPIES,
                "opt_stale": self.opt_stale,
                "n_table_points": self.n_table_points,
                "scan_anchors": self.anchors,
                "masses_known": self.masses_known,
                "mass_source": self.mass_source,
                "total_mass_g": float(sum((i.mass_g or 0.0) for i in self.items)),
                "packed_mass_g": float(sum((p.item.mass_g or 0.0) for p in self.placements)),
                "items": [{"label": i.label, "axes_mm": [float(a) for a in i.axes],
                           "mass_g": float(i.mass_g), "fragile": bool(i.fragile)} for i in self.items],
                "placements": [_placement_json(p) for p in self.placements],
                "unplaced": [i.label for i in self.unplaced],
                "report": self.report,
                "gate": self.gate,
                "max_penetration_mm": self.max_penetration_mm,
                "tolerance_mm": TOLERANCE_MM,
                "rules": self.ruleset.get("rules", []),
                "benchmark": self.benchmark,
                "error": self.last_error,
                "air_ok": bool(os.environ.get("OPENAI_API_KEY")),
            }

    def check_move(self, label, centre_mm) -> dict:
        """Exact penetration for ONE moved item against all others. No search."""
        with self.lock:
            me = next((p for p in self.placements if p.item.label == label), None)
            if me is None:
                return {"error": "no placed item named %r" % label}
            moved = Placement(item=me.item, centre=np.asarray(centre_mm, dtype=float), yaw=me.yaw)
            worst, against = 0.0, None
            for other in self.placements:
                if other is me:
                    continue
                pen = float(pair_penetration_mm(moved, other))
                if pen > worst:
                    worst, against = pen, other.item.label
            # containment via the support function, so a body cannot leave the box quietly
            M = me.item.shape_matrix(me.yaw)
            ext = np.sqrt(np.diag(np.linalg.inv(M)))
            b = self.box
            outside = bool(np.any(moved.centre - ext < -1e-6) or
                           moved.centre[0] + ext[0] > b.width + 1e-6 or
                           moved.centre[1] + ext[1] > b.depth + 1e-6 or
                           moved.centre[2] + ext[2] > b.height + 1e-6)
            ok = (worst <= TOLERANCE_MM) and not outside
            return {"penetration_mm": worst, "ok": ok, "against": against,
                    "outside_box": outside, "tolerance_mm": TOLERANCE_MM}


class BridgeHandler(BaseHTTPRequestHandler):
    state: State = None  # set per server by make_server

    def _send(self, code, body, ctype="application/json"):
        data = body if isinstance(body, (bytes, bytearray)) else json.dumps(body).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def _body(self):
        n = int(self.headers.get("Content-Length", 0) or 0)
        raw = self.rfile.read(n) if n else b""
        return json.loads(raw.decode("utf-8")) if raw else {}

    def do_GET(self):
        path = urlparse(self.path).path
        if path in STATIC:
            rel, ctype = STATIC[path]
            full = os.path.join(WEB_DIR, rel)
            if not os.path.exists(full):
                return self._send(404, {"error": "missing web asset %s" % rel})
            with open(full, "rb") as f:
                return self._send(200, f.read(), ctype)
        if path == "/api/scene":
            return self._send(200, self.state.scene())
        return self._send(404, {"error": "Not found"})

    def do_POST(self):
        path = urlparse(self.path).path
        try:
            data = self._body()
        except Exception:
            return self._send(400, {"error": "Invalid JSON"})

        if path == "/api/turn":
            text = str(data.get("text", "")).strip()
            if not text:
                return self._send(400, {"error": "empty text"})
            try:
                out = run_turn(text, self.state.items, self.state.box_catalog)
            except Exception as e:
                return self._send(500, {"error": "turn failed: %s" % str(e)[:160]})
            last = out.get("last_result") or {}
            if isinstance(last, dict) and last.get("placements"):
                rules = None
                for e in reversed(out.get("transcript", [])):
                    if e.get("role") == "model" and e.get("name") == "pack":
                        rules = (e.get("arguments") or {}).get("rules")
                        break
                self.state.adopt(last["placements"], rules)
            out["scene"] = self.state.scene()
            return self._send(200, out)

        if path == "/api/optimize":
            out = self.state.optimize_box()
            scene = self.state.scene()
            scene["optimize"] = out
            return self._send(200, scene)

        if path == "/api/counts":
            out = self.state.set_counts(data.get("counts"))
            if "error" in out:
                return self._send(400, out)
            self.state.do_pack()
            return self._send(200, self.state.scene())

        if path == "/api/pack":
            idx = data.get("box_index")
            if isinstance(idx, int) and 0 <= idx < len(self.state.box_catalog):
                self.state.box_index = idx
            self.state.do_pack()
            return self._send(200, self.state.scene())

        if path == "/api/check":
            if "label" not in data or "centre_mm" not in data:
                return self._send(400, {"error": "need label and centre_mm"})
            return self._send(200, self.state.check_move(data["label"], data["centre_mm"]))

        return self._send(404, {"error": "Not found"})

    def log_message(self, fmt, *args):
        # Suppress default access logging. Nothing here may ever print a key.
        return


def make_server(items, box_catalog, port=8000, points_m=None, point_rgb=None,
                mass_source=None, n_table_points=0, anchors=None):
    """Create the server without starting it. Raises if the key is absent."""
    if "OPENAI_API_KEY" not in os.environ:
        raise RuntimeError("OPENAI_API_KEY environment variable must be set to start the bridge server")
    state = State(items, box_catalog, points_m, point_rgb=point_rgb,
                  mass_source=mass_source, n_table_points=n_table_points,
                  anchors=anchors)
    handler = type("BridgeHandlerWithState", (BridgeHandler,), {"state": state})
    server = HTTPServer(("127.0.0.1", port), handler)
    server.state = state
    return server


def serve(items, box_catalog, port=8000, points_m=None, initial_pack=True, point_rgb=None,
          mass_source=None, n_table_points=0, anchors=None):
    server = make_server(items, box_catalog, port, points_m, point_rgb=point_rgb,
                         mass_source=mass_source, n_table_points=n_table_points,
                         anchors=anchors)
    if initial_pack:
        print("packing the initial scene, about 20 s for eight bodies ...", flush=True)
        server.state.do_pack()
        print("initial pack: gate %s, %d placed, %d unplaced" % (
            server.state.gate, len(server.state.placements), len(server.state.unplaced)), flush=True)
    print("Bridge server running on http://127.0.0.1:%d" % port, flush=True)
    print("Press Ctrl+C to stop", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped")
    finally:
        server.server_close()


def _load_scan(path, footprint_pct=None):
    """scan -> segment -> fit -> name. Returns (items, points_m, name_report).

    Points are metres in the file. Naming is a rename only, it never sets a
    constraint, and it leaves the `body-N` labels alone unless the sidecar
    match is a clean bijection. See src/capture/naming.py.
    """
    from src.capture.fit import fit_scene, footprint_pct_for
    from src.capture.naming import load_sidecar, name_items
    from src.capture.segment import segment
    pts = np.load(path)
    seg = segment(pts)
    pct, why = footprint_pct_for(load_sidecar(path))
    if footprint_pct is not None:
        pct, why = float(footprint_pct), "set on the command line"
    items = fit_scene(pts, seg, footprint_pct=pct)
    report = name_items(items, pts, seg, path)
    report["footprint_pct"] = pct
    report["footprint_why"] = why
    report["anchors"] = _anchors(pts, seg, report.get("mapping") or {})
    mask = np.zeros(len(pts), dtype=bool)
    for cluster in seg["clusters"]:
        mask[np.asarray(cluster)] = True
    report["item_mask"] = mask
    return items, pts, report


def _anchors(points_m, seg, mapping):
    """Where a floating name label goes for each scanned body. Millimetres.

    `xy_mm` is the cluster centroid and `top_mm` its highest point, so the page
    can lift the label clear of the body. Keyed by the real label when naming
    succeeded, otherwise by `body-N`. Display only, no solver input.
    """
    out = {}
    for idx, cluster in enumerate(seg["clusters"]):
        label = mapping.get("body-%d" % (idx + 1), "body-%d" % (idx + 1))
        pts_mm = points_m[np.asarray(cluster)] * 1000.0
        centre = pts_mm.mean(axis=0)
        out[label] = {"xy_mm": [float(centre[0]), float(centre[1])],
                      "top_mm": float(pts_mm[:, 2].max())}
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description="Local bridge between the browser and ASU AIR.")
    ap.add_argument("--scan", default=os.path.join("scans", "synthetic_produce.npy"))
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--no-pack", action="store_true", help="skip the initial pack")
    ap.add_argument("--footprint-pct", type=float, default=None,
                    help="override the footprint percentile chosen from the scan sidecar")
    a = ap.parse_args(argv)
    if not os.path.exists(a.scan):
        print("scan not found: %s" % a.scan, file=sys.stderr)
        return 2
    items, pts, naming = _load_scan(a.scan, a.footprint_pct)
    print("scan %s: %d points -> %d fitted bodies" % (a.scan, len(pts), len(items)), flush=True)
    print("footprint percentile p%g, %s" % (naming.get("footprint_pct", 0.0),
                                            naming.get("footprint_why", "")), flush=True)
    from src.capture.library import (apply_flags, apply_masses, load_flags,
                                     load_masses, mass_path)
    masses, mass_source = load_masses(a.scan)
    n_mass = apply_masses(items, masses)
    flags = load_flags(a.scan)
    n_flag = apply_flags(items, flags)
    if n_flag:
        print("declared properties: fragile %s, keep_upright %s"
              % (sorted(flags['fragile']) or '-', sorted(flags['keep_upright']) or '-'),
              flush=True)
    if n_mass == len(items) and n_mass:
        print("masses %d of %d from %s, source: %s"
              % (n_mass, len(items), mass_path(a.scan), mass_source or "unstated"), flush=True)
    else:
        print("NO MASSES, %d of %d weighed. A max_weight rule will compile and "
              "constrain nothing. Write %s to fix."
              % (n_mass, len(items), mass_path(a.scan)), flush=True)
    if naming.get("renamed"):
        worst = min(r["margin_mm"] for r in naming["rows"])
        print("named %d bodies from the sidecar, worst match margin %.1f mm: %s"
              % (naming["renamed"], worst, ", ".join(i.label for i in items)), flush=True)
    else:
        print("bodies keep their fitted labels, %s" % naming.get("reason", "no sidecar"), flush=True)
    # Optional colour sidecar, one uint8 RGB triple per point, display only.
    rgb = None
    rgb_path = a.scan[:-4] + "_rgb.npy" if a.scan.endswith(".npy") else None
    if rgb_path and os.path.exists(rgb_path):
        candidate = np.load(rgb_path)
        if len(candidate) == len(pts):
            rgb = candidate
            print("colour sidecar %s, %d points" % (rgb_path, len(rgb)), flush=True)
        else:
            print("colour sidecar ignored, %d colours against %d points"
                  % (len(candidate), len(pts)), flush=True)
    # Order the cloud table first, bodies last, and tell the page where the
    # split falls. figures/scene_common.py draws bodies at PT_SIZE 3.4 in
    # measured colour and the table at 0.5 in pale grey, and the page now
    # matches that. Reordering is DISPLAY only, no solver input is derived
    # from point order.
    item_mask = naming.get("item_mask")
    n_table = 0
    if item_mask is not None and len(item_mask) == len(pts):
        order = np.concatenate([np.flatnonzero(~item_mask), np.flatnonzero(item_mask)])
        pts = pts[order]
        n_table = int((~item_mask).sum())
        if rgb is not None and len(rgb) == len(order):
            rgb = rgb[order]
        print("cloud ordered, %d table then %d body points" % (n_table, len(pts) - n_table),
              flush=True)
    serve(items, list(DEFAULT_CATALOG), port=a.port, points_m=pts,
          initial_pack=not a.no_pack, point_rgb=rgb, mass_source=mass_source,
          n_table_points=n_table, anchors=naming.get("anchors"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
