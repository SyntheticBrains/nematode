#!/usr/bin/env python
"""Write the vendored atlas's release identities into the committed neuron table.

The 302-entry classification table is a checked-in literal — it asserts its own size at import
and is the project's canonical neuron reference — so the transmitter values live in the source,
not in a spreadsheet read at import time. This script is what puts them there; a test re-derives
them from the atlas and asserts equality, so the committed values can never drift from the file
they came from.

Usage::

    uv run python scripts/generate_neuron_transmitters.py            # rewrite in place
    uv run python scripts/generate_neuron_transmitters.py --check    # exit 1 if out of date
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

from quantumnematode.connectome.neurons import NEURON_CLASSIFICATION
from quantumnematode.connectome.neurotransmitters import read_atlas_transmitters

TABLE = (
    Path(__file__).resolve().parents[1]
    / "packages"
    / "quantum-nematode"
    / "quantumnematode"
    / "connectome"
    / "neurons.py"
)
_ENTRY = re.compile(
    r'^(?P<indent>\s*)"(?P<name>[A-Za-z0-9_]+)": \("(?P<cls>\w+)", (?P<nt>[^)]+)\),$',
)


def render(text: str, transmitters: dict[str, str | None]) -> str:
    """Return ``text`` with every table entry's transmitter replaced by the atlas's."""
    out: list[str] = []
    seen: set[str] = set()
    for line in text.splitlines(keepends=True):
        match = _ENTRY.match(line.rstrip("\n"))
        if match is None or match.group("name") not in NEURON_CLASSIFICATION:
            out.append(line)
            continue
        name = match.group("name")
        seen.add(name)
        identity = transmitters.get(name)
        rendered = "None" if identity is None else f'"{identity}"'
        out.append(f'{match.group("indent")}"{name}": ("{match.group("cls")}", {rendered}),\n')
    missing = set(NEURON_CLASSIFICATION) - seen
    if missing:
        msg = f"{len(missing)} table entries were not matched by the entry pattern: {sorted(missing)[:5]}"
        raise ValueError(msg)
    return "".join(out)


def main(argv: list[str] | None = None) -> int:
    """Rewrite or check the committed table; return the exit code."""
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--check", action="store_true", help="exit 1 if the table is out of date")
    args = ap.parse_args(argv)

    transmitters = read_atlas_transmitters()
    current = TABLE.read_text()
    updated = render(current, transmitters)
    if args.check:
        if updated != current:
            print(
                f"error: {TABLE} is out of date; run this script without --check",
                file=sys.stderr,
            )
            return 1
        print("neuron transmitters are up to date")
        return 0
    TABLE.write_text(updated)
    filled = sum(1 for v in transmitters.values() if v)
    print(
        f"wrote {filled} release identities into {TABLE} ({len(transmitters) - filled} without one)",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
