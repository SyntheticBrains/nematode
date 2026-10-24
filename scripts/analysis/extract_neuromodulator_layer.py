#!/usr/bin/env python
r"""Derive the connectome's neuromodulator layer from Bentley 2016 and Ripoll-Sánchez 2023.

Three subcommands, run in order:

``fetch``
    Download both sources into a working directory (gitignored ``build/`` by default) and check
    them: figshare's published MD5 for each Ripoll-Sánchez file, and the SHA-256 of Bentley's S1
    Dataset, which PLOS publishes without a checksum. Every file's SHA-256 is written to
    ``fetched.json``.

``synonyms``
    Map every gene name the two sources use to its WormBase public name, through UniProt's
    *C. elegans* entries (their gene names come from WormBase; WormBase's site refuses scripted
    queries). Writes ``data/neuromodulators/gene_synonyms.json``.

``build``
    Write ``data/neuromodulators/layer.json``:

    - **monoamine producers**, Bentley's synthesis-marker rows (tph-1, cat-2, tbh-1, tdc-1) with its
      excluded rows dropped; the reuptake markers (dat-1, mod-5) are not producers;
    - **monoamine couples**, Bentley's receptor rows, each couple's ligand from the row's
      ``monoamine`` column, excluded rows and the provisional dop-5 and dop-6 dropped;
    - **peptide couples**, the union of Ripoll-Sánchez's 92 (producer and receptor sets recovered
      from each long-range individual network, which must be their outer product) and Bentley's 22,
      under canonical gene names, every cell tagged with its sources and every couple with its EC50;
    - the **drive** of food ingestion onto NSM;
    - each neuron's body areas (Ripoll-Sánchez), for a later mid-range model;
    - the names the connectome lacks, and the provenance.

Usage::

    uv run python scripts/analysis/extract_neuromodulator_layer.py fetch --dir build/neuromodulators
    uv run python scripts/analysis/extract_neuromodulator_layer.py synonyms \
        --dir build/neuromodulators --out data/neuromodulators/gene_synonyms.json
    uv run python scripts/analysis/extract_neuromodulator_layer.py build --dir build/neuromodulators \
        --synonyms data/neuromodulators/gene_synonyms.json \
        --signs data/neuromodulators/receptor_signs.json --out data/neuromodulators/layer.json
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import re
import sys
import urllib.request
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from quantumnematode.connectome.neurons import CANONICAL_NAME_ALIASES, NEURON_CLASSIFICATION

DESCRIPTION = "Derive the neuromodulator layer from Bentley 2016 and Ripoll-Sánchez 2023."

BENTLEY = "bentley2016"
RIPOLL = "ripollsanchez2023"
BENTLEY_URL = (
    "https://journals.plos.org/ploscompbiol/article/file?type=supplementary"
    "&id=10.1371/journal.pcbi.1005283.s004"
)
FIGSHARE_API = "https://api.figshare.com/v2/articles/{}/files?page_size=500"
RIPOLL_INDIVIDUAL = 24187119
RIPOLL_PAIRS = 24517180
RIPOLL_NEURONS = 24187104
LONG_RANGE = re.compile(r"^Individual_LR\d+_neuropeptide_network(\d{3})_long_range_model\.csv$")

MONOAMINE_SYNTHESIS = {
    "serotonin": "tph-1",
    "dopamine": "cat-2",
    "octopamine": "tbh-1",
    "tyramine": "tdc-1",
}
PROVISIONAL_RECEPTORS = {"dop-5", "dop-6"}
INGESTION_DRIVE = {
    "name": "ingestion",
    "cells": ["NSML", "NSMR"],
    "source": "Rhoades et al. 2019, Cell 176:85 (NSM senses ingested food via DEL-3/DEL-7)",
}
EC50_LIMIT_NM = 500.0
_EC50 = re.compile(r"([0-9.]+)\s*(nM|μM|uM|µM)?", re.IGNORECASE)
_UNIPROT = (
    "https://rest.uniprot.org/uniprotkb/search?query=gene_exact:{}+AND+organism_id:6239"
    "&fields=gene_names,gene_primary&format=tsv"
)


# ----------------------------------------------------------------------------------------------
# Names
# ----------------------------------------------------------------------------------------------


def neuron_name(raw: str) -> str:
    """Return a source's neuron name in the project's canonical form (DA01 → DA1, aliases applied)."""
    name = raw.strip()
    name = re.sub(r"^(AS|DA|DB|DD|VA|VB|VC|VD)0+(\d)", r"\1\2", name)
    return CANONICAL_NAME_ALIASES.get(name, name)


def ec50_upper_nm(text: str) -> float | None:
    """Return the upper end of an EC50 range in nM ("0.4-9 μM" → 9000), or None if unreadable."""
    values = []
    unit_default = "nM"
    matches = list(_EC50.finditer(text.replace(",", ".")))
    if not matches:
        return None
    # A range's unit can sit on its last number only ("1-8 μM").
    last_unit = next((m.group(2) for m in reversed(matches) if m.group(2)), unit_default)
    for m in matches:
        unit = m.group(2) or last_unit
        value = float(m.group(1)) * (1000.0 if unit.lower() in {"μm", "um", "µm"} else 1.0)
        values.append(value)
    return max(values)


# ----------------------------------------------------------------------------------------------
# Fetch
# ----------------------------------------------------------------------------------------------


def _get(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "quantum-nematode-extraction"})  # noqa: S310
    with urllib.request.urlopen(request, timeout=120) as response:  # noqa: S310 - fixed https URLs
        return response.read()


def fetch(directory: Path) -> dict[str, Any]:
    """Download both sources into ``directory`` and check them; return the fetch record."""
    directory.mkdir(parents=True, exist_ok=True)
    record: dict[str, Any] = {"files": {}}
    bentley = _get(BENTLEY_URL)
    (directory / "bentley2016_s1.zip").write_bytes(bentley)
    record["files"]["bentley2016_s1.zip"] = {
        "url": BENTLEY_URL,
        "bytes": len(bentley),
        "sha256": hashlib.sha256(bentley).hexdigest(),
    }
    ripoll = directory / "ripollsanchez2023"
    ripoll.mkdir(exist_ok=True)
    for article in (RIPOLL_PAIRS, RIPOLL_NEURONS, RIPOLL_INDIVIDUAL):
        for item in json.loads(_get(FIGSHARE_API.format(article))):
            name = item["name"]
            if article == RIPOLL_INDIVIDUAL and not LONG_RANGE.match(name):
                continue
            data = _get(item["download_url"])
            if hashlib.md5(data).hexdigest() != item["computed_md5"]:  # noqa: S324 - figshare's own checksum
                msg = f"{name}: MD5 does not match figshare's"
                raise ValueError(msg)
            (ripoll / name).write_bytes(data)
            record["files"][f"ripollsanchez2023/{name}"] = {
                "url": item["download_url"],
                "figshare_article": article,
                "bytes": len(data),
                "md5": item["computed_md5"],
                "sha256": hashlib.sha256(data).hexdigest(),
            }
    (directory / "fetched.json").write_text(json.dumps(record, indent=1) + "\n")
    return {"files": len(record["files"])}


# ----------------------------------------------------------------------------------------------
# Source readers
# ----------------------------------------------------------------------------------------------


def _bentley_tables(directory: Path) -> dict[str, list[dict[str, str]]]:
    tables: dict[str, list[dict[str, str]]] = {}
    with zipfile.ZipFile(directory / "bentley2016_s1.zip") as archive:
        for info in archive.infolist():
            if info.filename.endswith(".tsv") and "expression_data/" in info.filename:
                text = archive.read(info).decode("utf-8")
                tables[Path(info.filename).stem] = list(
                    csv.DictReader(io.StringIO(text), delimiter="\t"),
                )
    return tables


def ripoll_couples(directory: Path) -> list[dict[str, Any]]:
    """Return Ripoll-Sánchez's couples with producer and receptor sets from the long-range networks.

    Each network must be exactly the outer product of its producer and receptor sets: a producer is
    any row with an edge, a receptor any column with one.
    """
    folder = directory / "ripollsanchez2023"
    pairs = list(
        csv.DictReader(
            (folder / "20230602_Ligand_receptors_interactions_all_EC50_explicit_500nM.csv").open(),
        ),
    )
    networks = {
        int(match.group(1)): p for p in folder.iterdir() if (match := LONG_RANGE.match(p.name))
    }
    if sorted(networks) != list(range(1, len(pairs) + 1)):
        msg = f"{len(networks)} long-range networks for {len(pairs)} couples"
        raise ValueError(msg)
    couples = []
    for index, pair in enumerate(pairs, start=1):
        rows = list(csv.reader(networks[index].open()))
        names = [neuron_name(n) for n in rows[0][1:]]
        matrix = np.array([[float(v) for v in row[1:]] for row in rows[1:]])
        producers, receptors = matrix.any(axis=1), matrix.any(axis=0)
        if not np.array_equal(matrix != 0, np.outer(producers, receptors)):
            msg = (
                f"network {index:03d} ({pair['GeneLigand']} → {pair['GeneGPCR']}) is not an "
                "outer product"
            )
            raise ValueError(msg)
        couples.append(
            {
                "ligand": pair["GeneLigand"],
                "receptor": pair["GeneGPCR"],
                "network": index,
                "producers": [names[i] for i in np.flatnonzero(producers)],
                "cells": [names[i] for i in np.flatnonzero(receptors)],
            },
        )
    return couples


def _areas(directory: Path) -> dict[str, str]:
    rows = list(csv.reader((directory / "ripollsanchez2023" / "neuron_information.csv").open()))
    # The file's header names columns class_neuron_1..302; its one row gives each neuron's areas in
    # the order of the individual networks' columns.
    first = min((directory / "ripollsanchez2023").glob("Individual_LR*network001*.csv"))
    names = [neuron_name(n) for n in next(csv.reader(first.open()))[1:]]
    return dict(zip(names, rows[1], strict=True))


def gene_names(directory: Path) -> list[str]:
    """Return every gene name either source uses, as written there."""
    tables = _bentley_tables(directory)
    names = {r["receptor"] for r in tables["monoamine_receptor_expression"]}
    names |= {r["marker"] for r in tables["monoamine_expression"]}
    names |= {r["neuropeptide"] for r in tables["neuropeptide_expression"]}
    names |= {r["receptor"] for r in tables["neuropeptide_receptor_expression"]}
    for r in tables["neuropeptide_receptor_mapping"]:
        names |= {r["neuropeptide"], r["receptor"]}
    for couple in ripoll_couples(directory):
        names |= {couple["ligand"], couple["receptor"]}
    return sorted(names, key=str.lower)


# ----------------------------------------------------------------------------------------------
# Synonyms
# ----------------------------------------------------------------------------------------------


def synonyms(directory: Path) -> dict[str, Any]:
    """Map each source gene name to its WormBase public name through UniProt."""
    table: dict[str, dict[str, Any]] = {}
    for raw in gene_names(directory):
        query = raw.lower()
        rows = _get(_UNIPROT.format(query)).decode("utf-8").strip().split("\n")[1:]
        primaries = {line.split("\t")[1] for line in rows if line}
        matching = {
            line.split("\t")[1]
            for line in rows
            if line and query in line.split("\t")[0].lower().split()
        }
        # A gene's own public name wins; a name only listed as some gene's synonym must name one.
        verified = True
        if query in primaries:
            canonical = query
        elif len(matching) == 1:
            canonical = matching.pop()
        elif not rows:
            # UniProt has no entry naming it: kept as written, and recorded as unverified.
            canonical, verified = query, False
        else:
            msg = f"{raw}: UniProt gives {sorted(matching or primaries)} as its WormBase name"
            raise ValueError(msg)
        table[raw] = {"canonical": canonical, "renamed": canonical != query, "verified": verified}
    return {
        "description": (
            "Every gene name Bentley 2016 and Ripoll-Sánchez 2023 use, mapped to its WormBase public "
            "name through UniProt's C. elegans entries (organism 6239), whose gene names WormBase "
            "supplies. A name that is a gene's public name keeps it; a name that is only a synonym "
            "must be listed by exactly one gene."
        ),
        "source": "https://rest.uniprot.org (queried by gene_exact)",
        "names": table,
    }


# ----------------------------------------------------------------------------------------------
# Build
# ----------------------------------------------------------------------------------------------


def _tag(target: dict[str, set[str]], cells: list[str], source: str, skipped: set[str]) -> None:
    for cell in cells:
        if cell in NEURON_CLASSIFICATION:
            target[cell].add(source)
        else:
            skipped.add(cell)


def build(directory: Path, synonym_path: Path, signs_path: Path) -> dict[str, Any]:  # noqa: C901 - one pass per source table
    """Return the layer: ligands, couples, drives, areas, skipped names and provenance."""
    names = json.loads(synonym_path.read_text())["names"]
    signs = json.loads(signs_path.read_text())["receptors"]

    def canon(raw: str) -> str:
        return names[raw]["canonical"]

    tables = _bentley_tables(directory)
    skipped: set[str] = set()
    producers: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    kinds: dict[str, str] = {}
    cells: dict[tuple[str, str], dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    ec50: dict[tuple[str, str], dict[str, str]] = defaultdict(dict)
    for row in tables["monoamine_expression"]:
        if row["excluded"] != "1" and MONOAMINE_SYNTHESIS.get(row["monoamine"]) == row["marker"]:
            _tag(producers[row["monoamine"]], [neuron_name(row["neuron"])], BENTLEY, skipped)
            kinds[row["monoamine"]] = "monoamine"
    for row in tables["monoamine_receptor_expression"]:
        receptor = canon(row["receptor"])
        if row["excluded"] == "1" or receptor in PROVISIONAL_RECEPTORS:
            continue
        _tag(cells[(row["monoamine"], receptor)], [neuron_name(row["neuron"])], BENTLEY, skipped)
    peptide_cells: dict[str, list[str]] = defaultdict(list)
    for row in tables["neuropeptide_expression"]:
        peptide_cells[canon(row["neuropeptide"])].append(neuron_name(row["neuron"]))
    receptor_cells: dict[str, list[str]] = defaultdict(list)
    for row in tables["neuropeptide_receptor_expression"]:
        receptor_cells[canon(row["receptor"])].append(neuron_name(row["neuron"]))
    for row in tables["neuropeptide_receptor_mapping"]:
        ligand, receptor = canon(row["neuropeptide"]), canon(row["receptor"])
        kinds[ligand] = "peptide"
        _tag(producers[ligand], peptide_cells[ligand], BENTLEY, skipped)
        _tag(cells[(ligand, receptor)], receptor_cells[receptor], BENTLEY, skipped)
        ec50[(ligand, receptor)][BENTLEY] = row["EC50"]
    seen: dict[tuple[str, str], int] = {}
    for couple in ripoll_couples(directory):
        key = (canon(couple["ligand"]), canon(couple["receptor"]))
        if key in seen:
            msg = f"{key} appears twice in Ripoll-Sánchez's list (networks {seen[key]} and {couple['network']})"
            raise ValueError(msg)
        seen[key] = couple["network"]
        kinds[key[0]] = "peptide"
        _tag(producers[key[0]], couple["producers"], RIPOLL, skipped)
        _tag(cells[key], couple["cells"], RIPOLL, skipped)
        ec50[key][RIPOLL] = "<= 500 nM (the source's inclusion criterion)"
    couples = []
    for (ligand, receptor), placed in sorted(cells.items()):
        sources = sorted({s for tags in placed.values() for s in tags})
        upper = (
            ec50_upper_nm(ec50[(ligand, receptor)].get(BENTLEY, ""))
            if BENTLEY in ec50[(ligand, receptor)]
            else None
        )
        within = RIPOLL in ec50[(ligand, receptor)] or (
            upper is not None and upper <= EC50_LIMIT_NM
        )
        entry = signs.get(receptor) if kinds[ligand] == "monoamine" else None
        fixed = entry is not None and entry.get("sign") in (-1, 1)
        couples.append(
            {
                "id": f"{ligand}:{receptor}",
                "ligand": ligand,
                "receptor": receptor,
                "cells": {c: sorted(s) for c, s in sorted(placed.items())},
                "sources": sources,
                "ec50": ec50.get((ligand, receptor), {}),
                "within_500nM": bool(within) if kinds[ligand] == "peptide" else True,
                "sign": int(entry["sign"]) if fixed and entry is not None else None,
                "sign_class": "fixed" if fixed else "fitted",
            },
        )
    ligands = {
        name: {"kind": kinds[name], "producers": {c: sorted(s) for c, s in sorted(placed.items())}}
        for name, placed in sorted(producers.items())
    }
    return {
        "description": (
            "The connectome's neuromodulator layer, placed from Bentley et al. 2016 and "
            "Ripoll-Sánchez et al. 2023 (both CC BY 4.0). Every producer and receptor cell carries "
            "the sources that place it. Peptide couples are the union of both sources under WormBase "
            "names, with no common potency cut; `within_500nM` marks the couples the ec50_500nM "
            "sensitivity keeps."
        ),
        "ligands": ligands,
        "couples": couples,
        "links": [],
        "drives": [INGESTION_DRIVE],
        "areas": _areas(directory),
        "skipped_names": sorted(skipped),
        "fetched": json.loads((directory / "fetched.json").read_text()),
    }


def main() -> int:
    """CLI: run one step of the extraction."""
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    sub = parser.add_subparsers(dest="step", required=True)
    f = sub.add_parser("fetch")
    f.add_argument("--dir", type=Path, required=True)
    s = sub.add_parser("synonyms")
    s.add_argument("--dir", type=Path, required=True)
    s.add_argument("--out", type=Path, required=True)
    b = sub.add_parser("build")
    b.add_argument("--dir", type=Path, required=True)
    b.add_argument("--synonyms", type=Path, required=True)
    b.add_argument("--signs", type=Path, required=True)
    b.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.step == "fetch":
        print(json.dumps(fetch(args.dir), indent=2))
    elif args.step == "synonyms":
        record = synonyms(args.dir)
        args.out.write_text(json.dumps(record, indent=1, ensure_ascii=False) + "\n")
        renamed = {k: v["canonical"] for k, v in record["names"].items() if v["renamed"]}
        unverified = [k for k, v in record["names"].items() if not v["verified"]]
        summary = {"names": len(record["names"]), "renamed": renamed, "unverified": unverified}
        print(json.dumps(summary, indent=2, ensure_ascii=False))
    else:
        layer = build(args.dir, args.synonyms, args.signs)
        args.out.write_text(json.dumps(layer, indent=1, ensure_ascii=False) + "\n")
        print(
            json.dumps(
                {
                    "ligands": len(layer["ligands"]),
                    "couples": len(layer["couples"]),
                    "fixed": sum(c["sign_class"] == "fixed" for c in layer["couples"]),
                    "within_500nM": sum(c["within_500nM"] for c in layer["couples"]),
                    "skipped_names": layer["skipped_names"],
                },
                indent=2,
            ),
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
