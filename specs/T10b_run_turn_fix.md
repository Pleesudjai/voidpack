# T10b — Make `run_turn` reliable and testable

DEPENDS ON: T10 (implemented, `call_tool` is green at 8 of 8)
MODIFY: `src/air/tools.py` only
DO NOT MODIFY: any other file. Never `tests/`.

**Priority: second, after T04b.** The whole demo is this function.

## The measured problem

Two live runs through the bridge, same code, different failures.

```
run 1:  model -> list_items({})
        tool  <- {"error": "Tool execution failed: 'NoneType' object is not iterable"}
        final_text: none

run 2:  transcript: []          <- completely empty
        final_text: none
```

`call_tool` is correct and passes all 8 of its tests. `make_server` passes `self.items`
correctly. The failure is inside the `run_turn` loop, and it is **non-deterministic**, which
is worse than a consistent failure because it will work in rehearsal and fail on stage.

## Why my spec let this through, and what changes

`tests/test_air_tools.py` tests `call_tool` thoroughly and never tests the loop offline. The
only loop test is the live one, which skips without a key and is non-deterministic with one.
**A loop that only has a live test has no gate.**

The fix is a seam. `run_turn` gains an injectable transport so the loop can be driven by
canned model responses, deterministically, offline.

## Required changes

### 1. Add the injectable seam

```python
def _post_chat(payload: dict, timeout: float = 60.0) -> dict:
    """POST to the AIR chat endpoint. The ONLY place that touches the network."""

def run_turn(user_text, items, box_catalog, model="devstral2-123b",
             max_steps=4, _transport=None) -> dict:
    """_transport, when given, replaces _post_chat. Test seam only.

    It is called as _transport(payload) and returns a parsed response dict.
    Production code NEVER passes it. It exists so the loop has a gate.
    """
```

### 2. Never return an empty transcript silently

Every turn appends at least one entry. If the response has no `choices`, no `message`, or
neither `content` nor `tool_calls`, append

```python
{"role": "system", "name": None, "arguments": None,
 "content": "model returned no usable message: <the raw response, truncated to 400 chars>"}
```

An empty transcript is never an acceptable return value. It tells the operator nothing and it
is what happened in run 2.

### 3. Handle every shape the model actually sends

A 123B model varies its output. All of these must work.

| Shape | Handling |
|---|---|
| `tool_calls` present, `content` null | dispatch the calls, this is the normal path |
| `content` present, no `tool_calls` | record it, set `final_text`, stop |
| both present | dispatch the calls AND record the content |
| `arguments` is `""`, `null`, or absent | treat as `{}`, do not crash |
| `arguments` is a JSON string | `json.loads` it, and on failure record a rejection rather than raising |
| `arguments` is already a dict | use it directly |
| more than one tool call in a message | dispatch each in order, append a tool entry for each |

### 4. Feed the tool result back and continue

After dispatching, append the tool result to `messages` as a `role: "tool"` message with the
matching `tool_call_id`, then loop. That is how the model gets to narrate the result. Without
it there is never a `final_text`, which is exactly what both runs showed.

### 5. Never raise out of `run_turn`

Any exception is caught, recorded in the transcript as a `system` entry, and the function
returns normally. A live pitch cannot survive a traceback.

## Add these tests to `tests/test_air_tools.py`

I will add them and re-freeze the manifest. They are listed here so you can see what the gate
will demand.

```python
def test_run_turn_dispatches_an_injected_tool_call()
def test_run_turn_handles_null_arguments()
def test_run_turn_handles_arguments_as_a_json_string()
def test_run_turn_never_returns_an_empty_transcript()
def test_run_turn_records_final_text_when_the_model_answers_in_prose()
def test_run_turn_never_raises_on_a_malformed_response()
def test_run_turn_feeds_the_tool_result_back_so_the_model_can_narrate()
```

## Forbidden

- Do not change `call_tool`. It is green at 8 of 8 and it is not the problem.
- Do not let `_transport` be reachable from the bridge or from any user input.
- Do not swallow an error without putting it in the transcript. Silent is worse than wrong.

## Gate

```
python -m pytest tests/test_air_tools.py -q
```

Expected: `15 passed` (8 existing plus 7 new), with the live test skipping when no key.

## Done when

The gate passes AND this prints a transcript containing a model tool call, a tool result, and
a non-empty final text.

```
python -c "
from src.types import Item,Box
from src.air.tools import run_turn
i=[Item(label='egg',axes=(22.,22.,29.),mass_g=60.),Item(label='orange',axes=(37.,36.,35.),mass_g=180.)]
o=run_turn('pack these into the 300 by 200 by 150 box, eggs are fragile',i,[Box(300.,200.,150.)])
[print(e) for e in o['transcript']]
print('FINAL:',o['final_text'])"
```
