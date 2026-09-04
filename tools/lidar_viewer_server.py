from __future__ import annotations

import csv
import json
import mimetypes
import shutil
import subprocess
from io import BytesIO
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np
from scipy import ndimage
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[1]
SOURCE_CAPTURE = ROOT / "Lidar" / "75b9d04d0a"
TRIMS = ROOT / "Lidar" / "trims"
FPS = 60.0


def capture_options() -> list[dict]:
    options = [{"id": "original", "label": "original - 75b9d04d0a", "path": str(SOURCE_CAPTURE)}]
    if TRIMS.exists():
        for folder in sorted(TRIMS.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
            if folder.is_dir() and (folder / "depth").exists():
                options.append({"id": f"trim:{folder.name}", "label": folder.name, "path": str(folder)})
    return options


def selected_capture(value: str | None = None) -> Path:
    if value and value.startswith("trim:"):
        name = value.split(":", 1)[1]
        candidate = (TRIMS / name).resolve()
        trims_root = TRIMS.resolve()
        if trims_root in candidate.parents and candidate.exists():
            return candidate
    newest = next((Path(opt["path"]) for opt in capture_options() if opt["id"].startswith("trim:")), None)
    return newest or SOURCE_CAPTURE


def metadata(capture: Path) -> dict:
    depth_files = sorted((capture / "depth").glob("*.png"))
    first = Image.open(depth_files[0])
    options = capture_options()
    current_id = "original" if capture == SOURCE_CAPTURE else f"trim:{capture.name}"
    return {
        "capture": str(capture),
        "capture_id": current_id,
        "options": options,
        "frames": len(depth_files),
        "fps": FPS,
        "duration_s": round(len(depth_files) / FPS, 3),
        "depth_size": first.size,
        "has_confidence": (capture / "confidence").exists(),
        "has_rgb": (capture / "rgb.mp4").exists(),
        "has_odometry": (capture / "odometry.csv").exists(),
        "has_camera_matrix": (capture / "camera_matrix.csv").exists(),
        "rgb_url": "/" + capture.relative_to(ROOT).as_posix() + "/rgb.mp4",
    }


def depth_preview(capture: Path, frame: int) -> bytes:
    path = capture / "depth" / f"{frame:06d}.png"
    arr = np.asarray(Image.open(path), dtype=np.float32)
    valid = arr[arr > 0]
    if valid.size:
        lo, hi = np.percentile(valid, [2, 98])
    else:
        lo, hi = 0.0, 1.0
    scaled = np.clip((arr - lo) / max(hi - lo, 1.0), 0, 1)
    gray = (scaled * 255).astype(np.uint8)
    img = Image.fromarray(gray, mode="L").resize((512, 384), Image.Resampling.NEAREST)
    out = BytesIO()
    img.save(out, format="PNG")
    return out.getvalue()


def confidence_preview(capture: Path, frame: int) -> bytes:
    path = capture / "confidence" / f"{frame:06d}.png"
    arr = np.asarray(Image.open(path), dtype=np.uint8)
    gray = (arr * 127).clip(0, 255).astype(np.uint8)
    img = Image.fromarray(gray, mode="L").resize((512, 384), Image.Resampling.NEAREST)
    out = BytesIO()
    img.save(out, format="PNG")
    return out.getvalue()


def depth_frame(capture: Path, frame: int, size: tuple[int, int]) -> np.ndarray:
    path = capture / "depth" / f"{frame:06d}.png"
    if not path.exists():
        return np.zeros((size[1], size[0]), dtype=np.float32)
    img = Image.open(path).resize(size, Image.Resampling.NEAREST)
    return np.asarray(img, dtype=np.float32)


def rgb_frame(capture: Path, frame: int) -> Image.Image:
    second = max(frame, 0) / FPS
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-ss",
        f"{second:.3f}",
        "-i",
        str(capture / "rgb.mp4"),
        "-frames:v",
        "1",
        "-f",
        "image2pipe",
        "-vcodec",
        "png",
        "-",
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if not result.stdout:
        return Image.new("RGB", (512, 384), "black")
    return Image.open(BytesIO(result.stdout)).convert("RGB")


def hsv_channels(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rgb = arr.astype(np.float32) / 255.0
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    maxc = np.max(rgb, axis=2)
    minc = np.min(rgb, axis=2)
    delta = maxc - minc
    hue = np.zeros_like(maxc)
    red = delta > 0
    hue[red & (maxc == r)] = ((g[red & (maxc == r)] - b[red & (maxc == r)]) / delta[red & (maxc == r)]) % 6
    hue[red & (maxc == g)] = ((b[red & (maxc == g)] - r[red & (maxc == g)]) / delta[red & (maxc == g)]) + 2
    hue[red & (maxc == b)] = ((r[red & (maxc == b)] - g[red & (maxc == b)]) / delta[red & (maxc == b)]) + 4
    hue *= 60.0
    saturation = np.where(maxc == 0, 0, delta / maxc)
    value = maxc
    return hue, saturation, value


def largest_component(mask: np.ndarray) -> np.ndarray:
    labels, count = ndimage.label(mask)
    if not count:
        return mask
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    return labels == sizes.argmax()


def convex_hull_mask(mask: np.ndarray) -> np.ndarray:
    boundary = mask ^ ndimage.binary_erosion(mask, structure=np.ones((3, 3)))
    ys, xs = np.nonzero(boundary)
    if len(xs) < 3:
        return mask
    points = sorted(set(zip(xs.tolist(), ys.tolist())))

    def cross(o: tuple[int, int], a: tuple[int, int], b: tuple[int, int]) -> int:
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower: list[tuple[int, int]] = []
    for p in points:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper: list[tuple[int, int]] = []
    for p in reversed(points):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    hull = lower[:-1] + upper[:-1]
    out = Image.new("L", (mask.shape[1], mask.shape[0]), 0)
    ImageDraw.Draw(out).polygon(hull, fill=1)
    return np.asarray(out, dtype=bool)


def orange_object_mask(capture: Path, frame: int, arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    hue, saturation, value = hsv_channels(arr)
    r = arr[:, :, 0].astype(np.int16)
    g = arr[:, :, 1].astype(np.int16)
    b = arr[:, :, 2].astype(np.int16)
    orange_color = (r > 95) & (g > 45) & ((r - g) > 18) & ((g - b) > 34) & ((r - b) > 75)
    seed = (hue >= 16) & (hue <= 52) & (saturation > 0.32) & (value > 0.28) & orange_color
    seed = ndimage.binary_opening(seed, structure=np.ones((3, 3)))
    seed = ndimage.binary_closing(seed, structure=np.ones((9, 9)))
    seed = largest_component(seed)
    if not seed.any():
        return seed, seed

    seed = ndimage.binary_fill_holes(seed)
    depth = depth_frame(capture, frame, (arr.shape[1], arr.shape[0]))
    valid = depth > 0
    seed_depth = depth[seed & valid]

    hull = convex_hull_mask(seed)
    hull = ndimage.binary_dilation(hull, structure=np.ones((5, 5)))
    if seed_depth.size >= 20:
        median_depth = float(np.median(seed_depth))
        mad = float(np.median(np.abs(seed_depth - median_depth)))
        tolerance = max(70.0, min(220.0, 4.0 * 1.4826 * mad + 45.0))
        depth_neighborhood = valid & (np.abs(depth - median_depth) <= tolerance)
        depth_grown = ndimage.binary_propagation(seed, mask=(hull & depth_neighborhood))
        object_mask = hull | depth_grown
    else:
        object_mask = hull

    object_mask = ndimage.binary_closing(object_mask, structure=np.ones((9, 9)))
    object_mask = ndimage.binary_fill_holes(object_mask)
    object_mask = largest_component(object_mask)
    return object_mask, seed


def skin_mask(arr: np.ndarray) -> np.ndarray:
    hue, saturation, value = hsv_channels(arr)
    r = arr[:, :, 0].astype(np.int16)
    g = arr[:, :, 1].astype(np.int16)
    b = arr[:, :, 2].astype(np.int16)
    return (
        (hue >= 0)
        & (hue <= 30)
        & (saturation > 0.10)
        & (saturation < 0.55)
        & (value > 0.35)
        & (r > 95)
        & (g > 55)
        & (b > 35)
        & (r > g)
        & (g > b)
    )


def label_mask(arr: np.ndarray) -> np.ndarray:
    hue, saturation, value = hsv_channels(arr)
    r = arr[:, :, 0].astype(np.int16)
    g = arr[:, :, 1].astype(np.int16)
    b = arr[:, :, 2].astype(np.int16)
    neutral = (np.abs(r - g) < 32) & (np.abs(g - b) < 32) & (np.abs(r - b) < 32)
    paper = (saturation < 0.34) & (value > 0.45) & neutral
    cool_gray = (b >= g - 12) & (g >= r - 12) & (saturation < 0.38) & (value > 0.34)
    return paper | cool_gray


def fill_from_neighbors(arr: np.ndarray, donor: np.ndarray, remove_mask: np.ndarray) -> np.ndarray:
    if not donor.any() or not remove_mask.any():
        return arr
    _, indices = ndimage.distance_transform_edt(~donor, return_indices=True)
    filled = arr.copy()
    ys, xs = np.nonzero(remove_mask)
    filled[ys, xs] = arr[indices[0, ys, xs], indices[1, ys, xs]]
    return filled


def smooth_repaired_pixels(arr: np.ndarray, donor: np.ndarray, repair_mask: np.ndarray) -> np.ndarray:
    if not repair_mask.any():
        return arr
    filled = arr.astype(np.float32)
    for _ in range(30):
        for channel in range(3):
            blurred = ndimage.uniform_filter(filled[:, :, channel], size=5)
            filled[:, :, channel][repair_mask] = blurred[repair_mask]
        filled[donor] = arr[donor]
    return np.clip(filled, 0, 255).astype(np.uint8)


def orange_preview(capture: Path, frame: int) -> bytes:
    img = rgb_frame(capture, frame).resize((512, 384), Image.Resampling.BILINEAR)
    arr = np.asarray(img, dtype=np.uint8)
    mask, seed = orange_object_mask(capture, frame, arr)
    hue, saturation, _ = hsv_channels(arr)
    r = arr[:, :, 0].astype(np.int16)
    g = arr[:, :, 1].astype(np.int16)
    b = arr[:, :, 2].astype(np.int16)
    donor = seed & (hue > 26) & (saturation > 0.50) & ((g - b) > 60) & ((r - b) > 105)
    donor = largest_component(donor)
    if donor.sum() < 500:
        donor = seed
    fill_area = mask & ~donor
    arr = fill_from_neighbors(arr, donor, fill_area)
    arr = smooth_repaired_pixels(arr, donor, fill_area)
    cut = np.zeros_like(arr)
    cut[mask] = arr[mask]
    out_img = Image.fromarray(cut, mode="RGB")
    out = BytesIO()
    out_img.save(out, format="PNG")
    return out.getvalue()


def trim_capture(start: int, end: int, label: str) -> dict:
    meta = metadata(SOURCE_CAPTURE)
    start = max(0, min(start, meta["frames"] - 1))
    end = max(start, min(end, meta["frames"] - 1))
    safe_label = "".join(c for c in label if c.isalnum() or c in ("-", "_")).strip() or "orange"
    out = TRIMS / f"{safe_label}_{start:06d}_{end:06d}"
    suffix = 2
    while out.exists():
        out = TRIMS / f"{safe_label}_{start:06d}_{end:06d}_v{suffix}"
        suffix += 1
    (out / "depth").mkdir(parents=True)
    (out / "confidence").mkdir(parents=True)

    for i in range(start, end + 1):
        name = f"{i:06d}.png"
        new = f"{i - start:06d}.png"
        shutil.copy2(SOURCE_CAPTURE / "depth" / name, out / "depth" / new)
        shutil.copy2(SOURCE_CAPTURE / "confidence" / name, out / "confidence" / new)

    shutil.copy2(SOURCE_CAPTURE / "camera_matrix.csv", out / "camera_matrix.csv")
    copy_odometry(start, end, out / "odometry.csv")

    rgb_out = out / "rgb.mp4"
    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-ss",
        f"{start / FPS:.3f}",
        "-to",
        f"{(end + 1) / FPS:.3f}",
        "-i",
        str(SOURCE_CAPTURE / "rgb.mp4"),
        "-c",
        "copy",
        str(rgb_out),
    ]
    subprocess.run(cmd, check=False)
    return {"output": str(out), "frames": end - start + 1, "start": start, "end": end}


def copy_odometry(start: int, end: int, out_path: Path) -> None:
    source = SOURCE_CAPTURE / "odometry.csv"
    with source.open(newline="") as src, out_path.open("w", newline="") as dst:
        reader = csv.DictReader(src, skipinitialspace=True)
        fieldnames = [name.strip() for name in (reader.fieldnames or [])]
        writer = csv.DictWriter(dst, fieldnames=fieldnames)
        writer.writeheader()
        for row in reader:
            clean = {key.strip(): value for key, value in row.items() if key is not None}
            frame = int(clean["frame"])
            if start <= frame <= end:
                clean["frame"] = f"{frame - start:06d}"
                writer.writerow(clean)


class Handler(SimpleHTTPRequestHandler):
    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        qs = parse_qs(parsed.query)
        cap = selected_capture(qs.get("capture", [None])[0])

        if parsed.path == "/":
            self.path = "/tools/lidar_viewer.html"
            return super().do_GET()
        if parsed.path == "/api/meta":
            return self.json(metadata(cap))
        if parsed.path == "/api/depth":
            return self.png(depth_preview(cap, int(qs.get("frame", ["0"])[0])))
        if parsed.path == "/api/confidence":
            return self.png(confidence_preview(cap, int(qs.get("frame", ["0"])[0])))
        if parsed.path == "/api/orange":
            return self.png(orange_preview(cap, int(qs.get("frame", ["0"])[0])))
        if parsed.path == "/api/trim":
            result = trim_capture(
                int(qs.get("start", ["0"])[0]),
                int(qs.get("end", ["0"])[0]),
                qs.get("label", ["orange"])[0],
            )
            return self.json(result)

        return super().do_GET()

    def json(self, payload: dict) -> None:
        data = json.dumps(payload, indent=2).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def png(self, data: bytes) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "image/png")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def guess_type(self, path: str) -> str:
        if path.endswith(".mp4"):
            return "video/mp4"
        return mimetypes.guess_type(path)[0] or "application/octet-stream"


if __name__ == "__main__":
    server = ThreadingHTTPServer(("127.0.0.1", 8765), Handler)
    print("LiDAR viewer: http://127.0.0.1:8765")
    server.serve_forever()
