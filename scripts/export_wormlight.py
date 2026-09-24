#!/usr/bin/env python
"""Export the hermaphrodite connectome as one versioned JSON file for Wormlight.

Wormlight (https://github.com/chrisjz/wormlight) runs the *C. elegans* connectome as a graded,
conductance-based neural model that drives a simulated body. It takes its wiring, neuron classes and
release identities from this repository rather than curating its own. This script writes them out
under the schema ``wormlight.connectome/1``:

- ``neurons``: name, class, release identities (primary first), and ``ruleSign``, the sign
  ``TRANSMITTER_SIGN`` gives the primary identity, or null;
- ``muscles``: the 95 body wall muscles, quadrant by quadrant, each from head to tail;
- ``chemical``: directed neuron-to-neuron synapses, as EM serial-section counts;
- ``gap``: gap junctions, each pair once with ``a < b``;
- ``neuromuscular``: synapses from neurons onto body wall muscles;
- ``provenance``: the commit exported from, whether tracked files had uncommitted changes, and the
  path, SHA256 and role of every file the values come from.

The wiring is the Emmons 2024 release of the Cook et al. 2019 matrices (CC BY 4.0). The output is
deterministic, with one record per line, so the same commit gives the same bytes and a re-export
diffs line by line. The script refuses a tree whose tracked files have uncommitted changes, because
the commit it records would not describe what it exported; ``--allow-dirty`` exports anyway and
records that the tree was dirty.

Usage::

    uv run python scripts/export_wormlight.py --out connectome.v1.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from quantumnematode.connectome.loader import (
    EMMONS_2024_HERMAPHRODITE_PATH,
    load_emmons_2024_hermaphrodite,
    load_emmons_2024_neuromuscular,
)
from quantumnematode.connectome.muscles import BODY_WALL_MUSCLES
from quantumnematode.connectome.neurons import NEURON_CO_TRANSMITTERS
from quantumnematode.connectome.neurotransmitters import ATLAS_PATH, sign_for

SCHEMA = "wormlight.connectome/1"
REPO_ROOT = Path(__file__).resolve().parents[1]
NEURON_TABLE_PATH = (
    REPO_ROOT / "packages" / "quantum-nematode" / "quantumnematode" / "connectome" / "neurons.py"
)

# Top-level keys, in the order they are written. Everything after the provenance is a list written
# one record per line.
_RECORD_LISTS = ("neurons", "muscles", "chemical", "gap", "neuromuscular")
_KEYS = ("schema", "provenance", *_RECORD_LISTS)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _input(path: Path, role: str) -> dict[str, str]:
    return {
        "file": path.resolve().relative_to(REPO_ROOT).as_posix(),
        "sha256": _sha256(path),
        "role": role,
    }


def git_state(repo: Path = REPO_ROOT) -> tuple[str, bool]:
    """Return the commit checked out in ``repo`` and whether tracked files have changes.

    Unlike the experiment tracker's git helpers, which report a clean tree when git fails, this
    raises: the export's provenance is only worth recording if it is known.
    """

    def git(*args: str) -> str:
        return subprocess.run(  # noqa: S603 — fixed argv, no shell
            ["git", "-C", str(repo), *args],  # noqa: S607
            check=True,
            capture_output=True,
            text=True,
        ).stdout

    commit = git("rev-parse", "HEAD").strip()
    dirty = bool(git("status", "--porcelain", "--untracked-files=no").strip())
    return commit, dirty


def build_export(*, commit: str, dirty: bool) -> dict[str, Any]:
    """Build the export from the Emmons 2024 connectome and the neuron table."""
    connectome = load_emmons_2024_hermaphrodite()
    neuromuscular = load_emmons_2024_neuromuscular()
    neurons = [
        {
            "name": neuron.name,
            "class": neuron.cell_class,
            "transmitters": [
                *([neuron.neurotransmitter] if neuron.neurotransmitter else []),
                *NEURON_CO_TRANSMITTERS.get(neuron.name, ()),
            ],
            "ruleSign": sign_for(neuron.neurotransmitter),
        }
        for neuron in connectome.neurons.values()
    ]
    return {
        "schema": SCHEMA,
        "provenance": {
            "exporter": "scripts/export_wormlight.py",
            "nematodeCommit": commit,
            "nematodeDirty": dirty,
            "inputs": [
                _input(
                    EMMONS_2024_HERMAPHRODITE_PATH,
                    "chemical synapses, gap junctions and neuromuscular connections: Cook et al. "
                    "2019 as released in Emmons 2024, PLoS Biology 22:e3002939, S1 File (CC BY 4.0)",
                ),
                _input(
                    NEURON_TABLE_PATH,
                    "neuron classes, and release identities as committed values generated from "
                    "the atlas below",
                ),
                _input(
                    ATLAS_PATH,
                    "release identities: Wang et al. 2024, eLife 13:RP95402, Supplementary File 2",
                ),
            ],
        },
        "neurons": neurons,
        "muscles": list(BODY_WALL_MUSCLES),
        "chemical": [
            {"pre": synapse.pre, "post": synapse.post, "sections": synapse.weight}
            for synapse in connectome.chemical_synapses
        ],
        "gap": [
            {"a": junction.neuron_a, "b": junction.neuron_b, "sections": junction.weight}
            for junction in connectome.gap_junctions
        ],
        "neuromuscular": [
            {"pre": junction.pre, "muscle": junction.muscle, "sections": junction.weight}
            for junction in neuromuscular
        ],
    }


def render(export: dict[str, Any]) -> str:
    """Serialise ``export`` deterministically, one record per line."""
    if tuple(export) != _KEYS:
        msg = f"export keys are {list(export)}, expected {list(_KEYS)} in that order"
        raise ValueError(msg)
    provenance = json.dumps(export["provenance"], indent=2).replace("\n", "\n  ")
    lines = ["{", f'  "schema": {json.dumps(export["schema"])},', f'  "provenance": {provenance},']
    for index, key in enumerate(_RECORD_LISTS):
        records = export[key]
        lines.append(f'  "{key}": [')
        lines.extend(
            f"    {json.dumps(record)}{',' if position < len(records) - 1 else ''}"
            for position, record in enumerate(records)
        )
        lines.append("  ]," if index < len(_RECORD_LISTS) - 1 else "  ]")
    lines.append("}")
    return "\n".join(lines) + "\n"


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

    commit, dirty = git_state()
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
    print(
        f"wrote {args.out}: {len(export['neurons'])} neurons, {len(export['chemical'])} chemical "
        f"synapses, {len(export['gap'])} gap junctions, {len(export['neuromuscular'])} "
        f"neuromuscular connections onto {len(export['muscles'])} muscles, from "
        f"{commit[:12]}{' (dirty tree)' if dirty else ''}",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
