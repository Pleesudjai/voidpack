# T18 — Browser upload of a point cloud

DEPENDS ON: T15 bridge, T17 ingest
CREATE: `src/air/upload.py`
DO NOT MODIFY: any other file. Never `tests/`.

## What the user does

Drag a `.npy`, `.ply` or `.csv` onto the viewport, or press `LOAD SCAN`. The cloud replaces
the current scene and the packing re-runs.

The file came from T17, so it is a few megabytes, not hundreds.

## Route

```
POST /api/upload?name=orange_set.npy
     body = raw file bytes, Content-Length set
     -> {"n_points": 48213, "downsampled_to": 12044,
         "bounds_mm": [[x0,y0,z0],[x1,y1,z1]], "units": "mm"}
```

**Raw body, not multipart.** Parsing multipart with the standard library is fiddly and it is a
classic place for an implementation to fail. The browser reads the file with `FileReader` and
POSTs the bytes with the name in the query string. Two lines of JavaScript, no parser.

## Exact signature

```python
MAX_BYTES = 50 * 1024 * 1024
ALLOWED = (".npy", ".ply", ".csv")

def parse_cloud(name: str, data: bytes) -> np.ndarray:
    """Bytes to (N,3) points in METRES. Dispatch on the extension.

    .npy  numpy array, must be (N,3) float
    .ply  ASCII header, x y z as the first three properties, ascii or
          binary_little_endian body
    .csv  one point per line, x,y,z, optional header row skipped
    """

def handle_upload(name: str, data: bytes, target_points=150000) -> dict:
    """Validate, parse, downsample if needed, return the summary above."""
```

## Validation, all of it required

1. **Extension allowlist.** Anything not in `ALLOWED` is rejected with a named error.
2. **Size cap 50 MB.** Reject before reading into memory where possible.
3. **Sanitise the name.** Reject any name containing `..`, `/`, `\` or a drive letter. The
   name is used only for the extension and for display, never to open a path.
4. **Shape check.** The parsed array must be `(N, 3)` and float. A `(3, N)` array is a
   transposed file and is rejected with that exact message, because it is the most common
   mistake and a silent transpose would corrupt everything downstream.
5. **Downsample above `target_points`.** A viewport with two million points is slow and the
   demo looks bad.
6. **Unit sniff.** If the bounding box diagonal exceeds 20, the file is almost certainly in
   millimetres rather than metres. Convert and say so in the response. Do not guess silently.

## Forbidden

- No multipart parsing.
- No `eval`, no `pickle.loads`, and `np.load` must be called with `allow_pickle=False`.
  A pickle in a `.npy` is arbitrary code execution.
- Never write the uploaded bytes to a path derived from the user-supplied name.

## Gate, run this, do not edit it

```
python -m pytest tests/test_upload.py -q
```

Expected: `9 passed`
