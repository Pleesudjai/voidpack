# T02 — Constraint schema

DEPENDS ON: T01
CREATE: `src/schema.py`
DO NOT MODIFY: any other file. Never `tests/`.

## Purpose

This is the contract between the AIR constraint compiler and the solver. It is deliberately
small. Three rule kinds only. A model asked to emit into a small closed schema succeeds far
more often than one asked to emit into an open one.

## The schema, exact

A compiled rule set is a JSON object.

```json
{
  "rules": [
    {"kind": "fragile",      "item": "egg",     "raw": "the eggs are fragile"},
    {"kind": "keep_upright", "item": "carton",  "raw": "milk carton must stay upright"},
    {"kind": "max_weight",   "limit_g": 9070.0, "raw": "keep it under 20 lb"}
  ],
  "unrecognized": ["do not ship on a Sunday"]
}
```

### Rule kinds, closed set of three

| kind | required fields | meaning for the solver |
|---|---|---|
| `fragile` | `item` | nothing may be placed in the column above this item |
| `keep_upright` | `item` | yaw is the only permitted rotation, no tumbling |
| `max_weight` | `limit_g` | total placed mass may not exceed this, in grams |

`raw` is required on every rule and carries the original English fragment, so the explainer
can quote the user rather than paraphrase them.

`unrecognized` is required and may be empty. **A rule the compiler cannot classify goes here.
It is never dropped silently.** A dropped fragility rule breaks something in a real parcel.

## Exact contents

```python
RULE_KINDS = ("fragile", "keep_upright", "max_weight")

SCHEMA_VERSION = "1.0"

def empty_ruleset() -> dict:
    """Return {"rules": [], "unrecognized": []}."""

def rule_fragile(item: str, raw: str) -> dict: ...
def rule_keep_upright(item: str, raw: str) -> dict: ...
def rule_max_weight(limit_g: float, raw: str) -> dict: ...

def schema_prompt() -> str:
    """Return the exact system prompt text handed to the AIR model in T10.

    Must describe the three kinds, require `raw` on every rule, and require that
    anything unclassifiable goes to `unrecognized` rather than being dropped.
    Must instruct the model to return JSON and nothing else.
    """
```

## Unit notes

- `limit_g` is grams. If English says pounds, the COMPILER converts, not the solver.
  1 lb = 453.59237 g. 20 lb = 9071.85 g.

## Forbidden

- Do not add a fourth rule kind. Not `segregate`, not `heavy_low`, not `orientation`.
  Three kinds is the scope decision and it is deliberate.
- No validation logic here. That is T03.
- No network calls. That is T10.
- Standard library only.

## Gate, run this, do not edit it

```
python -m pytest tests/test_schema.py -q
```

Expected: `7 passed`

## Done when

The gate passes and no file outside `src/schema.py` has changed.
