"""An item library, one fitted body per produce type, instantiated N times.

A scan holds one specimen of each type. Packing three eggs means three separate
bodies in the solver, each with its own centre and yaw, so the library is
expanded into copies before `pack` ever sees it.

MASS. A scan cannot weigh anything. `fit_scene` sets `mass_g = 0.0` and says so.
Mass is therefore read from a sidecar the author fills in, `*_mass.json`, keyed
by the library label and recorded in grams. Nothing here estimates a mass from a
density, because a density this module invented would be a fabricated number in
a deliverable. When the sidecar is absent every mass stays 0.0 and
`masses_known` is False, so the caller can refuse to advertise a weight rule.

UNITS. Grams for mass. Every length that passes through is already millimetres.
"""
import copy
import json
import os
from typing import Any, Dict, List, Optional, Tuple


def mass_path(scan_path: str) -> str:
    """Return the `*_mass.json` path that pairs with a scan path."""
    base, _ = os.path.splitext(scan_path)
    return base + "_mass.json"


def load_masses(scan_path: str) -> Tuple[Dict[str, float], Optional[str]]:
    """Load author-supplied masses in grams. Returns (by_label, provenance).

    The file is a JSON object with a `masses` map of label to grams, and a
    `source` string saying how the numbers were obtained. A missing file gives
    an empty map and None, which is not an error.
    """
    path = mass_path(scan_path)
    if not os.path.exists(path):
        return {}, None
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except (ValueError, OSError):
        return {}, None
    raw = data.get("masses")
    if not isinstance(raw, dict):
        return {}, None
    out = {}
    for label, grams in raw.items():
        if isinstance(grams, (int, float)) and grams > 0:
            out[str(label)] = float(grams)
    return out, data.get("source")


def load_flags(scan_path: str) -> dict:
    """Author-declared per-item properties from the `*_mass.json` sidecar.

    Returns {"fragile": set(), "keep_upright": set()}. These are DECLARED by
    the author, never inferred from a label. A body called "egg" is not fragile
    because of its name, it is fragile because the sidecar says so, and a scan
    that ships no sidecar declares nothing.
    """
    out = {"fragile": set(), "keep_upright": set()}
    path = mass_path(scan_path)
    if not os.path.exists(path):
        return out
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except (ValueError, OSError):
        return out
    for key in ("fragile", "keep_upright"):
        raw = data.get(key)
        if isinstance(raw, list):
            out[key] = {str(v) for v in raw}
    return out


def apply_flags(library, flags: dict) -> int:
    """Set `fragile` and `keep_upright` on library items. Returns count set."""
    n = 0
    for item in library:
        if item.label in flags.get("fragile", ()):
            item.fragile = True
            n += 1
        if item.label in flags.get("keep_upright", ()):
            item.keep_upright = True
            n += 1
    return n


def apply_masses(library: List[Any], by_label: Dict[str, float]) -> int:
    """Set `mass_g` on library items from the sidecar. Returns the count set."""
    n = 0
    for item in library:
        grams = by_label.get(item.label)
        if grams:
            item.mass_g = float(grams)
            n += 1
    return n


def expand(library: List[Any], counts: Dict[str, int]) -> List[Any]:
    """Return one Item per requested copy, labelled `<type>-1`, `<type>-2`, ...

    A count of 1 keeps the bare type label, so a single-specimen scene reads
    `egg` and not `egg-1`. A count of 0 drops the type. A type absent from
    `counts` defaults to one copy, so an empty dict reproduces the library.
    """
    out = []
    for item in library:
        n = counts.get(item.label, 1)
        try:
            n = max(0, int(n))
        except (TypeError, ValueError):
            n = 1
        for k in range(n):
            clone = copy.deepcopy(item)
            if n > 1:
                clone.label = "%s-%d" % (item.label, k + 1)
            out.append(clone)
    return out


def total_mass_g(items: List[Any]) -> float:
    """Sum of `mass_g` over items, in grams. Zero when nothing was weighed."""
    return float(sum(getattr(i, "mass_g", 0.0) or 0.0 for i in items))


def default_counts(library: List[Any]) -> Dict[str, int]:
    """One of each type."""
    return {item.label: 1 for item in library}
