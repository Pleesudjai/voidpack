"""Gate for T10. Written before the implementation. Do not edit.

The offline tests are the contract. The single live test skips without a key so the
suite stays green off the VPN.
"""
import json
import os

import pytest

from src.types import Item, Box
from src.air.tools import BASE_URL, call_tool, tool_specs


ITEMS = [Item(label="egg", axes=(22.0, 22.0, 30.0), mass_g=60.0),
         Item(label="potato", axes=(35.0, 28.0, 25.0), mass_g=170.0),
         Item(label="lime", axes=(24.0, 24.0, 26.0), mass_g=70.0)]
CATALOG = [Box(300.0, 200.0, 150.0)]


def test_endpoint_is_air_and_nothing_else():
    assert BASE_URL == "https://openai.rc.asu.edu/v1"


def test_exactly_two_tools_are_exposed():
    """A small closed tool set is called correctly far more often than a large one."""
    specs = tool_specs()
    names = {s["function"]["name"] for s in specs}
    assert names == {"list_items", "pack"}


def test_tool_specs_are_valid_openai_shape():
    for s in tool_specs():
        assert s["type"] == "function"
        f = s["function"]
        assert isinstance(f["name"], str) and f["description"]
        assert f["parameters"]["type"] == "object"
        json.dumps(s)          # must be serialisable as-is


def test_list_items_returns_the_labels_the_model_can_reference():
    out = call_tool("list_items", {}, ITEMS, CATALOG)
    labels = {i["label"] for i in out["items"]}
    assert labels == {"egg", "potato", "lime"}


def test_an_unknown_rule_kind_is_REJECTED_not_dropped():
    """A silently dropped rule is the failure this contract exists to prevent."""
    out = call_tool("pack", {"box_mm": [300.0, 200.0, 150.0],
                             "rules": [{"kind": "ship_on_tuesday", "raw": "ship Tuesday"}]},
                    ITEMS, CATALOG)
    assert out.get("rejected"), "unknown rule kind vanished without a report"
    assert "ship_on_tuesday" in json.dumps(out["rejected"])


def test_a_rule_missing_a_required_field_is_rejected_with_a_named_error():
    out = call_tool("pack", {"box_mm": [300.0, 200.0, 150.0],
                             "rules": [{"kind": "fragile", "raw": "it is fragile"}]},
                    ITEMS, CATALOG)
    assert out.get("rejected")
    assert "item" in json.dumps(out["rejected"]).lower()


def test_bad_model_input_returns_an_error_and_never_raises():
    """The model will send malformed arguments. The tool layer must survive it."""
    for args in ({}, {"box_mm": "big"}, {"box_mm": [1.0], "rules": "none"},
                 {"box_mm": [300.0, 200.0, 150.0], "rules": [42]}):
        out = call_tool("pack", args, ITEMS, CATALOG)
        assert isinstance(out, dict)
        assert "error" in out or "rejected" in out
    assert isinstance(call_tool("no_such_tool", {}, ITEMS, CATALOG), dict)


@pytest.mark.skipif(not os.environ.get("OPENAI_API_KEY"),
                    reason="no OPENAI_API_KEY, live AIR call skipped")
def test_live_air_model_emits_a_structured_tool_call():
    """Integration. Requires the key and the ASU VPN.

    Verified by direct probe on 2026-09-03, devstral2-123b returns a correct
    structured tool_calls entry in about 0.6 s.
    """
    from src.air.tools import run_turn
    out = run_turn("pack this order into the 300 by 200 by 150 box, the eggs are fragile",
                   ITEMS, CATALOG, max_steps=2)
    calls = [t for t in out["transcript"] if t["role"] == "model" and t.get("name")]
    assert calls, "model returned prose instead of a tool call"
    assert calls[0]["name"] in {"list_items", "pack"}


# ---------------------------------------------------------------------------
# T10b, the run_turn loop. Added 2026-09-03 after two live runs failed in two
# different ways. A loop whose only test is a live call has no gate.
# ---------------------------------------------------------------------------

def _resp(content=None, tool_calls=None):
    """One canned chat-completions response."""
    return {"choices": [{"message": {"role": "assistant",
                                     "content": content,
                                     "tool_calls": tool_calls}}]}


def _tc(name, arguments, cid="call_1"):
    return [{"id": cid, "type": "function",
             "function": {"name": name, "arguments": arguments}}]


def _transport(*responses):
    """Injectable transport that yields canned responses in order."""
    seq = list(responses)

    def _t(payload):
        return seq.pop(0) if seq else _resp(content="done")
    return _t


def _run(*responses, text="pack it"):
    from src.air.tools import run_turn
    return run_turn(text, ITEMS, CATALOG,
                    _transport=_transport(*responses))


def test_run_turn_dispatches_an_injected_tool_call():
    out = _run(_resp(tool_calls=_tc("list_items", "{}")),
               _resp(content="Three items are available."))
    kinds = [(e["role"], e.get("name")) for e in out["transcript"]]
    assert ("model", "list_items") in kinds, kinds
    assert any(r == "tool" for r, _ in kinds), kinds


def test_run_turn_handles_null_arguments():
    """A 123B model sends null, empty string or an absent field. None may crash."""
    for args in (None, "", "{}"):
        out = _run(_resp(tool_calls=_tc("list_items", args)),
                   _resp(content="ok"))
        assert out["transcript"], "empty transcript for arguments=%r" % (args,)
        assert not any("Traceback" in str(e.get("content", ""))
                       for e in out["transcript"])


def test_run_turn_handles_arguments_as_a_json_string():
    out = _run(_resp(tool_calls=_tc(
                   "pack", '{"box_mm": [300.0, 200.0, 150.0], "rules": []}')),
               _resp(content="Packed."))
    tools = [e for e in out["transcript"] if e["role"] == "tool"]
    assert tools, "the tool was never dispatched"


def test_run_turn_never_returns_an_empty_transcript():
    """An empty transcript tells the operator nothing. This happened live."""
    for bad in ({}, {"choices": []}, {"choices": [{"message": {}}]}):
        out = _run(bad)
        assert out["transcript"], "empty transcript for response %r" % (bad,)


def test_run_turn_records_final_text_when_the_model_answers_in_prose():
    out = _run(_resp(content="I need the box dimensions first."))
    assert out["final_text"]
    assert "box dimensions" in out["final_text"]


def test_run_turn_never_raises_on_a_malformed_response():
    """A live pitch cannot survive a traceback."""
    for bad in (None, "not a dict", {"unexpected": True},
                {"choices": [{"message": {"tool_calls": [{"bogus": 1}]}}]}):
        out = _run(bad)
        assert isinstance(out, dict)
        assert "transcript" in out


def test_run_turn_feeds_the_tool_result_back_so_the_model_can_narrate():
    """Without feeding the result back there is never a final_text, which is
    exactly what both live runs showed."""
    out = _run(_resp(tool_calls=_tc("list_items", "{}")),
               _resp(content="You have an egg, a potato and a lime."))
    assert out["final_text"], "the model never got to narrate"
    assert "egg" in out["final_text"]


# ---------------------------------------------------------------------------
# Added 2026-09-03 18:30. The gate never exercised a VALID pack call, so a pack
# tool that had never once worked passed 8 of 8. That was my defect.
# ---------------------------------------------------------------------------

def test_a_valid_pack_call_returns_placements_and_a_solver_report():
    """The happy path. Every number in the result must come from the solver."""
    out = call_tool("pack", {"box_mm": [300.0, 200.0, 150.0], "rules": []}, ITEMS, CATALOG)
    assert "error" not in out, out.get("error")
    assert isinstance(out.get("placed"), int) and out["placed"] >= 1
    assert isinstance(out.get("placements"), list) and len(out["placements"]) == out["placed"]
    p = out["placements"][0]
    for k in ("label", "axes_mm", "centre_mm", "yaw"):
        assert k in p, "placement lacks %s" % k
    assert out["report"]["container"] > 0.0
    assert out["gate"] in ("PASSED", "REJECTED")
    assert out["max_penetration_mm"] <= 1e-3 or out["gate"] == "REJECTED"


def test_a_valid_pack_call_honours_a_fragile_rule():
    """The compiled rule reaches the solver. Nothing may be placed above the fragile item."""
    out = call_tool("pack", {"box_mm": [60.0, 60.0, 400.0],
                             "rules": [{"kind": "fragile", "item": "egg", "raw": "egg is fragile"}]},
                    ITEMS, CATALOG)
    assert "error" not in out, out.get("error")
    egg = [p for p in out["placements"] if p["label"] == "egg"]
    if egg:
        top = egg[0]["centre_mm"][2]
        for p in out["placements"]:
            if p["label"] != "egg":
                assert p["centre_mm"][2] <= top + 1e-6, "an item sits above the fragile egg"
