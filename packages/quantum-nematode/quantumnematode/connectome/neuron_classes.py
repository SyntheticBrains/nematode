"""Anatomical classes of the hermaphrodite's 302 neurons, from their names.

A class groups the cells that share a name stem: a left/right pair, a four- or six-fold radial set,
or a numbered ventral-cord class. The rules follow *C. elegans* nomenclature:

- the numbered ventral-cord classes (AS, DA, DB, DD, VA, VB, VC, VD) drop their digits;
- any other name drops a side suffix (``DL``, ``DR``, ``VL``, ``VR``, ``L``, ``R``, ``D``, ``V``)
  when the names sharing that stem carry one of nomenclature's suffix patterns: a left/right pair,
  the four radial positions, those six together, dorsal/ventral/left/right, or dorsal with a ventral
  pair. Where two stems qualify, the larger pattern wins: OLQDL is OLQ, not OLQD. A stem whose names
  form no pattern is not a class, so AVL stays its own class beside AVDL and AVDR, and RIVL pairs
  with RIVR rather than joining RID.

Over the 302 neurons this gives the 118 classes White et al. (1986) describe.
"""

from __future__ import annotations

import re
from collections import defaultdict
from typing import TYPE_CHECKING

from quantumnematode.connectome.neurons import NEURON_CLASSIFICATION

if TYPE_CHECKING:
    from collections.abc import Iterable

VENTRAL_CORD_CLASSES = ("AS", "DA", "DB", "DD", "VA", "VB", "VC", "VD")
# Longest first, so DL is tried before L.
_SIDE_SUFFIXES = ("DL", "DR", "VL", "VR", "L", "R", "D", "V")
_NUMBERED = re.compile(r"^(AS|DA|DB|DD|VA|VB|VC|VD)(\d+)$")


_PATTERNS = (
    frozenset({"L", "R"}),
    frozenset({"DL", "DR", "VL", "VR"}),
    frozenset({"DL", "DR", "VL", "VR", "L", "R"}),
    frozenset({"D", "V", "L", "R"}),
    frozenset({"D", "VL", "VR"}),
)


def _splits(name: str) -> list[tuple[str, str]]:
    return [(name[: -len(s)], s) for s in _SIDE_SUFFIXES if name.endswith(s) and len(name) > len(s)]


def neuron_classes(names: Iterable[str] | None = None) -> dict[str, str]:
    """Return each neuron's anatomical class, for ``names`` or all 302 neurons."""
    pool = sorted(names if names is not None else NEURON_CLASSIFICATION)
    out: dict[str, str] = {}
    for name in pool:
        if match := _NUMBERED.match(name):
            out[name] = match.group(1)
    suffixes: dict[str, set[str]] = defaultdict(set)
    for name in pool:
        if name not in out:
            for stem, suffix in _splits(name):
                suffixes[stem].add(suffix)
    valid = {stem: len(found) for stem, found in suffixes.items() if frozenset(found) in _PATTERNS}
    for name in pool:
        if name in out:
            continue
        candidates = [stem for stem, _ in _splits(name) if stem in valid]
        out[name] = max(candidates, key=lambda stem: valid[stem]) if candidates else name
    return out


def class_members(classes: dict[str, str]) -> dict[str, list[str]]:
    """Return each class's members, in name order."""
    members: dict[str, list[str]] = defaultdict(list)
    for name in sorted(classes):
        members[classes[name]].append(name)
    return dict(members)
