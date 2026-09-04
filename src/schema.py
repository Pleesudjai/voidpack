RULE_KINDS = ("fragile", "keep_upright", "max_weight")

SCHEMA_VERSION = "1.0"


def empty_ruleset() -> dict:
    """Return {"rules": [], "unrecognized": []}."""
    return {"rules": [], "unrecognized": []}


def rule_fragile(item: str, raw: str) -> dict:
    """Create a fragile rule."""
    return {"kind": "fragile", "item": item, "raw": raw}


def rule_keep_upright(item: str, raw: str) -> dict:
    """Create a keep_upright rule."""
    return {"kind": "keep_upright", "item": item, "raw": raw}


def rule_max_weight(limit_g: float, raw: str) -> dict:
    """Create a max_weight rule."""
    return {"kind": "max_weight", "limit_g": limit_g, "raw": raw}


def schema_prompt() -> str:
    """Return the exact system prompt text handed to the AIR model in T10.

    Must describe the three kinds, require `raw` on every rule, and require that
    anything unclassifiable goes to `unrecognized` rather than being dropped.
    Must instruct the model to return JSON and nothing else.
    """
    return """
You are a constraint compiler for a packaging optimization system. Your task is to convert plain English shipping rules into a structured JSON format.

## Supported Rule Kinds (closed set of three):

1. **fragile**: Items that cannot have anything placed above them
   - Format: {"kind": "fragile", "item": "<item_name>", "raw": "<original_text>"}
   - Example: "the eggs are fragile" → {"kind": "fragile", "item": "egg", "raw": "the eggs are fragile"}

2. **keep_upright**: Items that must remain upright (no tumbling)
   - Format: {"kind": "keep_upright", "item": "<item_name>", "raw": "<original_text>"}
   - Example: "milk carton must stay upright" → {"kind": "keep_upright", "item": "carton", "raw": "milk carton must stay upright"}

3. **max_weight**: Maximum total weight limit for the parcel
   - Format: {"kind": "max_weight", "limit_g": <grams>, "raw": "<original_text>"}
   - Note: Convert pounds to grams (1 lb = 453.59237 g)
   - Example: "keep it under 20 lb" → {"kind": "max_weight", "limit_g": 9071.85, "raw": "keep it under 20 lb"}

## Output Format:

Return a JSON object with exactly two fields:
- "rules": array of rule objects (each must have "kind" and "raw" fields)
- "unrecognized": array of strings for rules that don't match the three kinds above

## Critical Requirements:

1. Every rule must have a "raw" field containing the original English text
2. If you cannot classify a rule into one of the three kinds, put it in "unrecognized" - NEVER drop it silently
3. Return ONLY valid JSON - no explanations, no markdown, no text outside the JSON object
4. Use the exact format shown above

## Example:

Input: "the eggs are fragile, keep it under 20 lb, do not ship on a Sunday"

Output:
{
  "rules": [
    {"kind": "fragile", "item": "egg", "raw": "the eggs are fragile"},
    {"kind": "max_weight", "limit_g": 9071.85, "raw": "keep it under 20 lb"}
  ],
  "unrecognized": ["do not ship on a Sunday"]
}
"""