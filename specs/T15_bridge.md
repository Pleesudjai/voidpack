# T15 — Local bridge, the browser never sees the key

DEPENDS ON: T10
CREATE: `src/air/bridge.py`
DO NOT MODIFY: any other file. Never `tests/`.

## Purpose

The browser must never hold the API key. If the page calls the AIR endpoint directly, the key
sits in the bundle, anyone opens devtools and reads it, and the bundle is in a repository that
is a submitted deliverable.

```
browser  ->  http://127.0.0.1:8000  ->  https://openai.rc.asu.edu/v1
             (holds OPENAI_API_KEY)
```

## Exact signature

Standard library `http.server` only. No FastAPI, no flask, no uvicorn. This is 60 lines and a
dependency you do not add is a dependency that cannot fail at 8 PM.

```python
def make_server(items, box_catalog, port=8000): ...
def serve(items, box_catalog, port=8000): ...
```

Two routes.

| Route | Method | Body | Returns |
|---|---|---|---|
| `/api/turn` | POST | `{"text": "..."}` | the `run_turn` dict from T10, transcript included |
| `/` | GET | — | the static page |

## Rules

1. **Bind to `127.0.0.1`, never `0.0.0.0`.** The demo is local and a bound-to-all server on
   conference wifi is a different kind of mistake.
2. **The key never appears in a response body, a header, or a log line.** Not even truncated.
3. If `OPENAI_API_KEY` is missing, fail at startup with a clear message. Do not start a server
   that will 500 on the first request during a live pitch.
4. Return the transcript verbatim. It is the demo artefact.

## Forbidden

- No third-party web framework.
- No CORS wildcard. Same origin only.
- No key in any client-side file, including a template.

## Gate

```
python -m pytest tests/test_bridge.py -q
```

Write this gate only after T10 is green. Budget 30 minutes. If it is not working, fall back to
running the demo from a terminal and screen-sharing that, which costs presentation polish and
costs nothing in correctness.
