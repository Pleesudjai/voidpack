# T10 — AIR tool layer, the model calls the solver

DEPENDS ON: T02 schema, T06 pack, T07 density
CREATE: `src/air/tools.py`
DO NOT MODIFY: any other file. Never `tests/`.

## The architecture, and why it is this way

**The tools lead. The model only communicates.**

The model is given function tools and it calls them. It never computes a coordinate, a
volume, a density, or a verdict. It translates English into a tool call, and it narrates
what came back.

This is not a style preference. It produces the single best artefact in the demo, the
**tool-call transcript**, which answers the question every jury asks.

```
user  : pack this order, the eggs are fragile and keep it under 20 lb
model → pack({"box_mm": [300, 200, 150],
              "rules": [{"kind": "fragile", "item": "egg", "raw": "the eggs are fragile"},
                        {"kind": "max_weight", "limit_g": 9071.85, "raw": "under 20 lb"}]})
tool  ← {"placed": 11, "unplaced": 1, "container_density": 0.361,
         "gate": "passed", "max_penetration_mm": 0.0002}
model : Eleven of twelve items fit. The egg carton sits on top because the fragile rule
        forbids load above it. One potato did not fit within the weight cap.
```

The model emitted a function call. **It did not emit a single number.** Put that log on the
screen beside the 3D view.

Verified 2026-09-03, tool calling works on AIR. All four candidate models returned a correct
structured `tool_calls` entry on a live probe, `devstral2-123b` at 0.6 s.

## Exact signature

```python
BASE_URL = "https://openai.rc.asu.edu/v1"

def tool_specs() -> list:
    """The OpenAI-format tool schemas handed to the model. Exactly TWO tools.

    list_items()  -> the current item library, so the model can reference labels
    pack(box_mm, rules) -> run the solver and return the result

    Two is the whole set. A small closed tool set is called correctly far more often
    than a large one, and every extra tool is a new way for a 123B model to go wrong.
    """

def call_tool(name: str, arguments: dict, items, box_catalog) -> dict:
    """Dispatch one tool call. NEVER raises on bad input from the model.

    On an invalid rule, return {"error": ..., "rejected": [...]} so the model can
    read it and tell the user. An unparseable rule is REPORTED, never dropped.
    """

def run_turn(user_text, items, box_catalog, model="devstral2-123b",
             max_steps=4) -> dict:
    """One conversational turn against AIR.

    Returns {"transcript": [...], "final_text": str, "last_result": dict | None}

    `transcript` is the demo artefact. Every entry is
    {"role": "model"|"tool", "name": str|None, "arguments": dict|None,
     "content": str|dict}
    and it is rendered verbatim on screen. Do not summarise it.
    """
```

## The rules that keep the model out of the engineering

1. **Validate every tool call against the T02 schema before dispatch.** A field the schema
   does not define is rejected with a named error, not silently dropped.
2. **The tool return is the only source of numbers.** `run_turn` must never let a number
   appear in `final_text` that is absent from `last_result`. If the model invents one, that
   is a defect in the prompt and the prompt gets fixed.
3. **A rejected rule is surfaced.** `{"rejected": [...]}` goes back to the model and the
   model must tell the user. A dropped fragility rule breaks something in a real parcel.
4. **Cap `max_steps`.** A 123B model in a loop will call `pack` repeatedly. Four steps is
   enough for compile, pack, and narrate.

## Key handling, non-negotiable

Read from `os.environ["OPENAI_API_KEY"]`. Never a literal, never a default, never a config
file inside the repository. **The browser must never receive it.** The browser talks to the
local bridge in T15, and the bridge talks to AIR.

## Forbidden

- No provider other than AIR. No fallback to any other endpoint, not even one that never
  fires, because a reviewer reads the code and not the execution.
- Do not let the model compute. If a tool result is missing, say so, do not fill the gap.
- No streaming for the first version. It complicates the transcript for no demo value.

## Gate, run this, do not edit it

```
python -m pytest tests/test_air_tools.py -q
```

Expected: `8 passed`

The gate does NOT require the network. The live-call test is skipped automatically when
`OPENAI_API_KEY` is absent, so the suite is green offline and exercises the real gateway when
the key is present.

## Done when

The gate passes and no file outside `src/air/tools.py` has changed.
