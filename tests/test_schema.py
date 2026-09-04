"""Gate for T02. Written before the implementation. Do not edit."""
import json

from src.schema import (RULE_KINDS, SCHEMA_VERSION, empty_ruleset,
                        rule_fragile, rule_keep_upright, rule_max_weight,
                        schema_prompt)


def test_rule_kinds_is_the_closed_set_of_three():
    assert set(RULE_KINDS) == {"fragile", "keep_upright", "max_weight"}
    assert len(RULE_KINDS) == 3


def test_empty_ruleset_has_both_required_lists():
    r = empty_ruleset()
    assert r == {"rules": [], "unrecognized": []}


def test_fragile_rule_shape():
    r = rule_fragile("egg", "the eggs are fragile")
    assert r["kind"] == "fragile"
    assert r["item"] == "egg"
    assert r["raw"] == "the eggs are fragile"


def test_keep_upright_rule_shape():
    r = rule_keep_upright("carton", "milk carton must stay upright")
    assert r["kind"] == "keep_upright"
    assert r["item"] == "carton"
    assert r["raw"]


def test_max_weight_rule_carries_grams():
    r = rule_max_weight(9071.85, "keep it under 20 lb")
    assert r["kind"] == "max_weight"
    assert abs(r["limit_g"] - 9071.85) < 1e-6
    assert "item" not in r


def test_every_rule_is_json_serialisable():
    rs = empty_ruleset()
    rs["rules"] = [rule_fragile("egg", "a"), rule_keep_upright("box", "b"),
                   rule_max_weight(100.0, "c")]
    rs["unrecognized"] = ["do not ship on a Sunday"]
    assert json.loads(json.dumps(rs)) == rs


def test_schema_prompt_names_all_three_kinds_and_the_unrecognized_rule():
    p = schema_prompt().lower()
    for k in ("fragile", "keep_upright", "max_weight", "unrecognized", "raw"):
        assert k in p, "schema_prompt must mention %s" % k
    assert SCHEMA_VERSION
