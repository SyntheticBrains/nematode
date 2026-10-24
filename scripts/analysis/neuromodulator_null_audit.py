#!/usr/bin/env python
r"""The receptor-placement null's matched-statistics table, and the data layer's agreement table.

``audit`` builds the data-layer brain once (``*_lawns_datalayer_s1_primitives.yml``) for its
neurons, sensors, motor neurons and chemical wiring, then compares the data layer with 64 null
draws (seeds 7501-7564), in the model's own units:

- each receptor gene's cell count (kept exactly by the null) and the share of its cells a class of
  unique size pins in place;
- the per-neuron receptor load, the number of rows on each neuron: mean, 99th percentile and
  maximum, over all couples and over fixed couples;
- the per-neuron fixed-sign balance, the sum of the fixed couples' signs on each neuron: its mean
  absolute value and maximum;
- the couples on ALA, CAN, PVD, AVA and AVB;
- per ligand, the share of its receptor cells that are its producers' chemical-synapse partners;
- the shortcut audit's modulator term (``sensory_motor_hops.modulator_shortcuts``).

``agreement`` lists every link the hand-placed circuit verified and whether each source has it.

Usage::

    uv run python scripts/analysis/neuromodulator_null_audit.py audit \
        --out docs/experiments/logbooks/supporting/106-neuromodulator-layer/null_audit.json
    uv run python scripts/analysis/neuromodulator_null_audit.py agreement \
        --out docs/experiments/logbooks/supporting/106-neuromodulator-layer/circuit_agreement.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from quantumnematode.brain.arch.connectome_ppo import ConnectomePPOBrain, ConnectomePPOBrainConfig
from quantumnematode.brain.arch.dtypes import DeviceType
from quantumnematode.connectome.neuromodulator_layer import (
    LAYER_PATH,
    LayerSpec,
    receptor_placement_null,
)
from quantumnematode.connectome.neuromodulators import class_members as circuit_members
from quantumnematode.connectome.neuron_classes import class_members, neuron_classes
from quantumnematode.connectome.neurons import NEURON_CLASSIFICATION
from quantumnematode.utils.config_loader import load_simulation_config

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "analysis"))
import sensory_motor_hops as hops  # noqa: E402  # pyright: ignore[reportMissingImports]

DESCRIPTION = "The receptor-placement null's audit, and the data layer's circuit agreement."
CONFIG = (
    REPO
    / "configs/scenarios/foraging"
    / "connectomeppo_small_continuous2d_fick_adaptive_klinotaxis_lawns_datalayer_s1_primitives.yml"
)
CIRCUIT = REPO / "data" / "neuromodulators" / "roaming_dwelling_circuit.json"
SEEDS = range(7501, 7565)
NAMED = ("ALA", "CAN", "PVD", "AVA", "AVB")


def _brain() -> ConnectomePPOBrain:
    container = load_simulation_config(str(CONFIG)).brain
    assert container is not None  # noqa: S101 - the config names a brain
    config = container.config
    assert isinstance(config, ConnectomePPOBrainConfig)  # noqa: S101 - and it is the connectome
    return ConnectomePPOBrain(config=config, device=DeviceType.CPU)


def _cells_by_gene(spec: LayerSpec) -> dict[str, set[str]]:
    out: dict[str, set[str]] = {}
    for row in spec.rows:
        if row.kind == "couple" and row.receptor:
            out.setdefault(row.receptor, set()).update(row.cells)
    return out


def _loads(spec: LayerSpec, names: list[str]) -> dict[str, np.ndarray]:
    index = {n: i for i, n in enumerate(names)}
    load = np.zeros(len(names))
    fixed = np.zeros(len(names))
    balance = np.zeros(len(names))
    for row in spec.rows:
        if row.kind != "couple":
            continue
        for cell in row.cells:
            load[index[cell]] += 1
            if row.sign is not None:
                fixed[index[cell]] += 1
                balance[index[cell]] += row.sign
    return {"load": load, "fixed": fixed, "balance": balance}


def _summary(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(values.mean()),
        "p99": float(np.percentile(values, 99)),
        "max": float(values.max()),
    }


def statistics(spec: LayerSpec, brain: ConnectomePPOBrain) -> dict[str, Any]:
    """Return one layer's matched statistics on the brain's neurons."""
    topology = brain.topology
    names = list(topology.neuron_names)
    loads = _loads(spec, names)
    chem = (topology.m_chem != 0).numpy()
    partners = chem | chem.T
    by_gene = _cells_by_gene(spec)
    overlap = {}
    for k, ligand in enumerate(spec.ligands):
        producers = [topology._idx[c] for c in spec.producers[k]]
        near = partners[producers].any(axis=0)
        cells = {
            c for row in spec.rows if row.kind == "couple" and row.ligand == k for c in row.cells
        }
        if cells:
            overlap[ligand] = float(np.mean([near[topology._idx[c]] for c in cells]))
    classes = neuron_classes(names)
    named = {
        group: int(sum(loads["load"][names.index(n)] for n in names if classes[n] == group))
        for group in NAMED
    }
    return {
        "cells_per_gene": {g: len(c) for g, c in sorted(by_gene.items())},
        "load_all": _summary(loads["load"]),
        "load_fixed": _summary(loads["fixed"]),
        "fixed_balance": {
            "mean_abs": float(np.abs(loads["balance"]).mean()),
            "max_abs": float(np.abs(loads["balance"]).max()),
        },
        "named_couples": named,
        "partner_overlap": overlap,
        "partner_overlap_mean": float(np.mean(list(overlap.values()))),
        "shortcuts": hops.modulator_shortcuts(topology, spec),
    }


def pinned_shares(spec: LayerSpec, names: list[str]) -> dict[str, float]:
    """Return each receptor gene's share of cells sitting in a class whose size no other shares."""
    classes = neuron_classes(names)
    members = class_members(classes)
    sizes = [len(m) for m in members.values()]
    unique = {c for c, m in members.items() if sizes.count(len(m)) == 1}
    return {
        gene: float(np.mean([classes[c] in unique for c in cells]))
        for gene, cells in sorted(_cells_by_gene(spec).items())
    }


def audit() -> dict[str, Any]:
    """Compare the data layer with its null draws."""
    brain = _brain()
    spec = brain.topology.modulator_spec
    names = list(brain.topology.neuron_names)
    data = statistics(spec, brain)
    draws = [statistics(receptor_placement_null(spec, seed, names), brain) for seed in SEEDS]
    assert all(d["cells_per_gene"] == data["cells_per_gene"] for d in draws)  # noqa: S101

    def spread(path: list[str]) -> dict[str, float]:
        values = []
        for d in draws:
            value: Any = d
            for key in path:
                value = value[key]
            values.append(float(value))
        return {
            "mean": float(np.mean(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
        }

    rows = {
        "load_all.mean": ["load_all", "mean"],
        "load_all.p99": ["load_all", "p99"],
        "load_all.max": ["load_all", "max"],
        "load_fixed.p99": ["load_fixed", "p99"],
        "load_fixed.max": ["load_fixed", "max"],
        "fixed_balance.mean_abs": ["fixed_balance", "mean_abs"],
        "fixed_balance.max_abs": ["fixed_balance", "max_abs"],
        "partner_overlap_mean": ["partner_overlap_mean"],
        "shortcuts.couples": ["shortcuts", "couples"],
        "shortcuts.motor_cells": ["shortcuts", "motor_cells"],
    } | {f"named_couples.{n}": ["named_couples", n] for n in NAMED}
    table = {}
    for name, path in rows.items():
        value: Any = data
        for key in path:
            value = value[key]
        table[name] = {"data": float(value), "null": spread(path)}
    return {
        "description": DESCRIPTION,
        "config": CONFIG.name,
        "seeds": f"{SEEDS.start}-{SEEDS.stop - 1}",
        "does_not_preserve": [
            "which classes carry each receptor",
            "the per-neuron receptor load and sign balance (reported, not held)",
            "receptors' overlap with their ligand's producers and those producers' synaptic partners",
        ],
        "cells_per_gene_kept_exactly": True,
        "pinned_share": pinned_shares(spec, names),
        "table": table,
        "data_partner_overlap": data["partner_overlap"],
    }


def agreement() -> dict[str, Any]:
    """List each link the hand-placed circuit verified, and whether each source places it."""
    circuit = json.loads(CIRCUIT.read_text())
    layer = json.loads(LAYER_PATH.read_text())

    def sources_for(cells: dict[str, list[str]], neuron_class: str) -> list[str]:
        # The circuit's names are classes or their left/right halves (SIAV is SIAVL and SIAVR), read
        # as the hand-placed loader reads them.
        members = circuit_members(neuron_class, sorted(NEURON_CLASSIFICATION))
        return sorted({t for c in members if c in cells for t in cells[c]})

    def couple(ligand: str, receptor: str) -> dict[str, list[str]]:
        found = next(
            (c for c in layer["couples"] if c["ligand"] == ligand and c["receptor"] == receptor),
            None,
        )
        return found["cells"] if found else {}

    serotonin = layer["ligands"]["serotonin"]["producers"]
    mod1 = couple("serotonin", "mod-1")
    pdfr1 = couple("pdf-1", "pdfr-1") | couple("pdf-2", "pdfr-1")
    links = [
        ("serotonin released by", c, serotonin) for c in circuit["serotonin"]["release"]["neurons"]
    ]
    links += [("MOD-1 on", c, mod1) for c in circuit["serotonin"]["receptor"]["neurons"]]
    for neuron_class in circuit["pdf"]["release"]["neurons"]:
        links += [
            ("PDF-1 released by", neuron_class, layer["ligands"]["pdf-1"]["producers"]),
            ("PDF-2 released by", neuron_class, layer["ligands"]["pdf-2"]["producers"]),
        ]
    links += [("PDFR-1 on", c, pdfr1) for c in circuit["pdf"]["receptor"]["neurons"]]
    links.append(("PDFR-1 on (PDF's link to)", "NSM", pdfr1))
    return {
        "description": (
            "Each link the hand-placed circuit verified (data/neuromodulators/roaming_dwelling_circuit.json), "
            "and the sources that place it in the data layer. An empty list means neither source does. "
            "Disagreements are reported, not patched."
        ),
        "links": [
            {"link": f"{kind} {neuron_class}", "sources": sources_for(cells, neuron_class)}
            for kind, neuron_class, cells in links
        ],
    }


def main() -> int:
    """CLI: write the audit or the agreement table."""
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument("step", choices=("audit", "agreement"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    record = audit() if args.step == "audit" else agreement()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(record, indent=1) + "\n")
    print(json.dumps(record.get("table", record.get("links")), indent=1)[:4000])
    return 0


if __name__ == "__main__":
    sys.exit(main())
