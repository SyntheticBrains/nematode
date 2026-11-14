#!/usr/bin/env python
"""Export the signed connectome and the neuromodulator layer as one versioned JSON file for Wormturn.

Wormturn takes its connectome data from this repository rather than curating its own. Beyond what
the Wormlight export carries, it needs each chemical connection's sign with the step that gave it,
and the neuromodulator layer. This script writes them under the schema ``wormturn.connectome/1``:

- ``neurons``: as in the Wormlight export: name, class, release identities (primary first), and
  ``ruleSign``, the sign ``TRANSMITTER_SIGN`` gives the primary identity, or null;
- ``chemical``: directed neuron-to-neuron synapses, as EM serial-section counts, each with the sign
  ``per_connection_signs`` gives it, ``signSource`` (``physiology``, ``expression``, ``rule`` or
  ``none``, which has sign 0), and ``citation``, the key of a physiology sign or null;
- ``gap``: as in the Wormlight export, each pair once with ``a < b``;
- ``signCitations``: the full citation for each key;
- ``ligands``, ``couples``, ``links`` and ``drives``: the neuromodulator layer as
  ``data/neuromodulators/layer.json`` holds it, each ligand with its name added;
- ``receptorCouplingRule`` and ``receptorCouplings``: the G-protein coupling rule and each
  receptor's entry as ``data/neuromodulators/receptor_couplings.json`` holds them, each with its
  name added;
- ``provenance``: the commit exported from, whether tracked files had uncommitted changes, and the
  path, SHA256 and role of every file the values come from.

The layer's values are carried unchanged, and every list keeps its file's order. There are no
muscles: the Wormlight export carries them.

The wiring is the Emmons 2024 release of the Cook et al. 2019 matrices (CC BY 4.0), and the signs
are computed on that wiring. The output is deterministic, with one record per line, so the same
commit gives the same bytes and a re-export diffs line by line. The script refuses a tree whose
tracked files have uncommitted changes, because the commit it records would not describe what it
exported; ``--allow-dirty`` exports anyway and records that the tree was dirty.

Usage::

    uv run python scripts/export_wormturn.py --out connectome.v1.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from quantumnematode.connectome.loader import (
    EMMONS_2024_HERMAPHRODITE_PATH,
    load_emmons_2024_hermaphrodite,
)
from quantumnematode.connectome.neuromodulator_layer import LAYER_PATH, RECEPTOR_COUPLINGS_PATH
from quantumnematode.connectome.neurotransmitters import ATLAS_PATH
from quantumnematode.connectome.signs import (
    FENYVES_S1_PATH,
    FENYVES_S5_PATH,
    PHYSIOLOGY_CITATIONS,
    PHYSIOLOGY_OVERRIDES_PATH,
    per_connection_signs,
)

# The Wormlight exporter sits beside this script and supplies the git check, the input records, the
# neuron and gap-junction records, and the renderer.
_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import export_wormlight as ew  # noqa: E402

SCHEMA = "wormturn.connectome/1"
REPO_ROOT = ew.REPO_ROOT
SIGN_TABLE_PATH = (
    REPO_ROOT / "packages" / "quantum-nematode" / "quantumnematode" / "connectome" / "signs.py"
)
LFS_POINTER_PREFIX = b"version https://git-lfs.github.com/spec/v1"

# The top-level keys after the schema and the provenance, in the order they are written: each is a
# list written one record per line.
_RECORD_LISTS = (
    "neurons",
    "chemical",
    "gap",
    "signCitations",
    "ligands",
    "couples",
    "links",
    "drives",
    "receptorCouplingRule",
    "receptorCouplings",
)


def checked_input(path: Path, role: str) -> dict[str, str]:
    """Return an input's provenance entry, refusing a Git LFS pointer in place of the file.

    A pointer's digest would otherwise be recorded as the file's.
    """
    with path.open("rb") as handle:
        head = handle.read(len(LFS_POINTER_PREFIX))
    if head == LFS_POINTER_PREFIX:
        msg = f"{path} is a Git LFS pointer, not the file; run `git lfs pull` first"
        raise ValueError(msg)
    return ew.input_entry(path, role)


def _inputs() -> list[dict[str, str]]:
    return [
        checked_input(
            EMMONS_2024_HERMAPHRODITE_PATH,
            "chemical synapses and gap junctions: Cook et al. 2019 as released in Emmons 2024, "
            "PLoS Biology 22:e3002939, S1 File (CC BY 4.0)",
        ),
        checked_input(
            ew.NEURON_TABLE_PATH,
            "neuron classes, and release identities as committed values generated from the atlas "
            "below",
        ),
        checked_input(
            ATLAS_PATH,
            "release identities: Wang et al. 2024, eLife 13:RP95402, Supplementary File 2",
        ),
        checked_input(
            PHYSIOLOGY_OVERRIDES_PATH,
            "physiology signs: cited per-connection measurements",
        ),
        checked_input(
            FENYVES_S1_PATH,
            "expression signs: Fenyves et al. 2020, PLoS Comput Biol 16:e1007974, S1 Data",
        ),
        checked_input(
            FENYVES_S5_PATH,
            "expression signs: Fenyves et al. 2020, PLoS Comput Biol 16:e1007974, S5 Data",
        ),
        checked_input(
            SIGN_TABLE_PATH,
            "the per-connection sign procedure, and the physiology citation table as committed "
            "values",
        ),
        checked_input(
            LAYER_PATH,
            "the neuromodulator layer: ligands, producers, couples, links and drives, placed from "
            "Bentley et al. 2016 and Ripoll-Sánchez et al. 2023 (CC BY 4.0)",
        ),
        checked_input(
            RECEPTOR_COUPLINGS_PATH,
            "the receptor couplings: each fitted couple's receptor, with its G-protein coupling and "
            "cited sign",
        ),
    ]


def _named(mapping: dict[str, dict[str, Any]], key: str) -> list[dict[str, Any]]:
    """Turn a mapping keyed by name into records carrying the name first, in the mapping's order."""
    clashing = [name for name, entry in mapping.items() if key in entry]
    if clashing:
        msg = f"entries {clashing} already have a {key!r} field, which the name would replace"
        raise ValueError(msg)
    return [{key: name, **entry} for name, entry in mapping.items()]


def build_export(*, commit: str, dirty: bool) -> dict[str, Any]:
    """Build the export from the Emmons 2024 connectome, its signs and the layer's files."""
    inputs = _inputs()
    connectome = load_emmons_2024_hermaphrodite()
    signs = per_connection_signs(connectome)
    layer = json.loads(LAYER_PATH.read_text(encoding="utf-8"))
    couplings = json.loads(RECEPTOR_COUPLINGS_PATH.read_text(encoding="utf-8"))
    chemical = []
    for synapse in connectome.chemical_synapses:
        sign = signs[(synapse.pre, synapse.post)]
        chemical.append(
            {
                "pre": synapse.pre,
                "post": synapse.post,
                "sections": synapse.weight,
                "sign": sign.sign,
                "signSource": sign.source,
                "citation": sign.citation,
            },
        )
    return {
        "schema": SCHEMA,
        "provenance": {
            "exporter": "scripts/export_wormturn.py",
            "nematodeCommit": commit,
            "nematodeDirty": dirty,
            "inputs": inputs,
        },
        "neurons": ew.neuron_records(connectome),
        "chemical": chemical,
        "gap": ew.gap_records(connectome),
        "signCitations": [
            {"key": key, "citation": citation} for key, citation in PHYSIOLOGY_CITATIONS.items()
        ],
        "ligands": _named(layer["ligands"], "name"),
        "couples": layer["couples"],
        "links": layer["links"],
        "drives": layer["drives"],
        "receptorCouplingRule": [
            {"coupling": coupling, "sign": sign} for coupling, sign in couplings["rule"].items()
        ],
        "receptorCouplings": _named(couplings["receptors"], "receptor"),
    }


def render(export: dict[str, Any]) -> str:
    """Serialise ``export`` deterministically, one record per line."""
    return ew.render(export, _RECORD_LISTS)


def main(argv: list[str] | None = None) -> int:
    """Write the export; return the exit code."""
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--out", type=Path, required=True, help="where to write the JSON file")
    ap.add_argument(
        "--allow-dirty",
        action="store_true",
        help="export from a tree with uncommitted changes, and record that it was dirty",
    )
    args = ap.parse_args(argv)

    commit, dirty = ew.git_state()
    if dirty and not args.allow_dirty:
        print(
            "error: tracked files have uncommitted changes, so the recorded commit would not "
            "describe this export; commit them first, or pass --allow-dirty",
            file=sys.stderr,
        )
        return 1

    export = build_export(commit=commit, dirty=dirty)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_bytes(render(export).encode("utf-8"))
    sources = {"physiology": 0, "expression": 0, "rule": 0, "none": 0}
    for record in export["chemical"]:
        sources[record["signSource"]] += 1
    print(
        f"wrote {args.out}: {len(export['neurons'])} neurons, {len(export['chemical'])} chemical "
        f"synapses ({', '.join(f'{n} {source}' for source, n in sources.items())}), "
        f"{len(export['gap'])} gap junctions, {len(export['ligands'])} ligands, "
        f"{len(export['couples'])} couples, {len(export['receptorCouplings'])} receptor couplings, "
        f"from {commit[:12]}{' (dirty tree)' if dirty else ''}",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
