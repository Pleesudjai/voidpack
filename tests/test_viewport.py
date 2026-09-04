"""Gate for T16. A static contract check, not a visual test. Do not edit.

A UI cannot be unit tested for taste. It CAN be tested for the two things that
actually go wrong, a leaked key and a page that ignores the visual contract.
"""
import os
import re

import pytest

WEB = "web"
INDEX = os.path.join(WEB, "index.html")
APP = os.path.join(WEB, "app.js")
CSS = os.path.join(WEB, "style.css")


def read(p):
    assert os.path.exists(p), "missing required file: %s" % p
    return open(p, encoding="utf-8").read()


def client_text():
    return "\n".join(read(p) for p in (INDEX, APP, CSS))


def test_the_required_files_exist():
    for p in (INDEX, APP, CSS, os.path.join(WEB, "vendor", "three.min.js")):
        assert os.path.exists(p), "missing %s" % p


def test_no_api_key_anywhere_in_the_client():
    """The browser must never hold a key. This is the disqualifying failure."""
    t = client_text()
    assert "sk-" not in t
    assert "OPENAI_API_KEY" not in t
    assert "openai.rc.asu.edu" not in t, "client must talk to the bridge, not to AIR"


def test_client_only_calls_same_origin_routes():
    """Every fetch must be a relative /api path. No external host, no CDN call."""
    for m in re.findall(r"fetch\(\s*[`'\"]([^`'\"]+)", read(APP)):
        assert m.startswith("/api/"), "non-local fetch target: %s" % m


def test_three_js_is_vendored_not_cdn():
    """A CDN that is slow at 9 PM during a live pitch is an avoidable failure."""
    html = read(INDEX)
    for src in re.findall(r'<script[^>]+src=["\']([^"\']+)', html):
        assert not src.startswith("http"), "external script tag: %s" % src
    assert "vendor/three.min.js" in html


def test_no_build_step_and_no_framework():
    t = client_text().lower()
    for banned in ("react", "vue.js", "svelte", "tailwind", "webpack", "vite"):
        assert banned not in t, "framework or build tool referenced: %s" % banned
    assert not os.path.exists("package.json"), "no build step is permitted"


def test_the_required_element_ids_are_present():
    html = read(INDEX)
    for eid in ("viewport", "measurements", "transcript", "rules-input", "gate-status"):
        assert 'id="%s"' % eid in html or "id='%s'" % eid in html, "missing #%s" % eid


def test_the_forbidden_visual_tokens_are_absent():
    """Enforces the part of the contract a model most reliably violates."""
    css = read(CSS).lower()
    for banned in ("linear-gradient", "radial-gradient", "backdrop-filter",
                   "box-shadow: 0 0 ", "border-radius: 9999", "border-radius: 50%"):
        assert banned not in css, "forbidden style: %s" % banned
    for purple in ("#6b46c1", "#7c3aed", "#8b5cf6", "#a855f7", "rebeccapurple", "indigo"):
        assert purple not in css, "forbidden colour: %s" % purple


def test_no_emoji_in_the_interface():
    t = client_text()
    assert not re.search(r"[\U0001F300-\U0001FAFF\u2600-\u27BF]", t), "emoji in the UI"


def test_numbers_are_monospace_and_tabular():
    """Every reported number is a measurement. It is set in monospace, right aligned,
    with tabular figures, so columns line up and digits do not shift."""
    css = read(CSS).lower()
    assert "tabular-nums" in css
    assert "monospace" in css
