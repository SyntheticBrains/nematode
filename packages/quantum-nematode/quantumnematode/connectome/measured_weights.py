"""Measured synaptic weights from a fitted whole-brain model, joined onto a connectome.

The vendored table under ``data/connectome/`` holds a signed weight for each directed neuron pair a
connectome-constrained linear dynamical system was allowed to use, fitted to optogenetic whole-brain
calcium imaging. Three properties of that table decide how it can be used here:

* **It is not in this model's units.** The values are coefficients of a 2 Hz dynamics matrix on
  calcium signals. A consumer that wants them on the same footing as a random draw has to rescale
  them; nothing in this module does.
* **It was fitted on a different connectome.** Its mask unions chemical and gap-junction edges from
  other reconstructions, and is not typed by connection. Joining it to a connectome is therefore a
  name-keyed lookup whose coverage has to be reported rather than assumed:
  :func:`coverage` does that.
* **It covers part of the nervous system.** Its neurons are those the imaging recorded, which leaves
  most motor neurons out. Head scope below means the edges whose endpoints both lie in the table's
  own neuron set; the table has no diagonal, so self-loops inside that scope are counted apart
  rather than as misses.

The file is small plain text stored byte-for-byte, so it is read directly with its digest checked,
and a file whose digest differs is refused.
"""

from __future__ import annotations

import csv
import functools
import hashlib
import io
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING

from quantumnematode.connectome.neurons import NEURON_CLASSIFICATION

if TYPE_CHECKING:  # pragma: no cover - import-time typing only
    from collections.abc import Mapping

    from quantumnematode.connectome.model import Connectome

MEASURED_WEIGHTS_FILENAME = "creamer_lds_2026_model_weights.csv"
MEASURED_WEIGHTS_PATH = (
    Path(__file__).resolve().parents[4] / "data" / "connectome" / MEASURED_WEIGHTS_FILENAME
)
MEASURED_WEIGHTS_SHA256 = "f452b88461aa90d414fd652e246e11302293d207510b9f4c94bf9a9a8098924c"
_COLUMNS = ("presynaptic cell", "postsynaptic cell", "weight")

Edge = tuple[str, str]


class MeasuredWeightsError(ValueError):
    """The measured-weight file is missing, altered, or not what this module reads."""


def read_measured_weights(path: Path = MEASURED_WEIGHTS_PATH) -> Mapping[Edge, float]:
    """Read the table, refusing a file whose digest differs from the recorded one.

    Returns a read-only mapping from ``(pre, post)`` to the signed weight. Every name is checked
    against the canonical classification, and a pair listed twice is refused rather than resolved
    by whichever row came last.
    """
    if not path.is_file():
        msg = f"measured-weight table not found at {path}"
        raise MeasuredWeightsError(msg)
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != MEASURED_WEIGHTS_SHA256:
        msg = (
            f"{path.name} has SHA256 {digest}, not the recorded {MEASURED_WEIGHTS_SHA256}; "
            "the file differs from the one vendored and its values cannot be trusted as that "
            "table's"
        )
        raise MeasuredWeightsError(msg)

    reader = csv.DictReader(io.StringIO(raw.decode("utf-8")))
    if tuple(reader.fieldnames or ()) != _COLUMNS:
        msg = f"{path.name} has columns {reader.fieldnames}, expected {list(_COLUMNS)}"
        raise MeasuredWeightsError(msg)

    table: dict[Edge, float] = {}
    for row in reader:
        pre, post = row["presynaptic cell"], row["postsynaptic cell"]
        for name in (pre, post):
            if name not in NEURON_CLASSIFICATION:
                msg = f"{path.name} names {name!r}, which is not a canonical neuron"
                raise MeasuredWeightsError(msg)
        if (pre, post) in table:
            msg = f"{path.name} lists {(pre, post)} more than once"
            raise MeasuredWeightsError(msg)
        table[pre, post] = float(row["weight"])
    return MappingProxyType(table)


@functools.cache
def measured_weights() -> Mapping[Edge, float]:
    """Return the vendored table, read once per process."""
    return read_measured_weights()


@dataclass(frozen=True)
class MeasuredCoverage:
    """How a measured table lands on one connectome's chemical synapses.

    Attributes
    ----------
    covered
        Chemical edges that carry a measured value.
    full_scope
        Every chemical edge in the connectome.
    head_scope
        Chemical edges whose endpoints both lie in the table's neuron set.
    head_self_loops
        Self-loops inside head scope, which a table with no diagonal cannot cover.
    gap_junction_only
        Table entries that fall on a gap junction and on no chemical edge. Gap junctions are
        undirected, so either orientation matches.
    no_connection
        Table entries that fall on no connection in this connectome at all.
    """

    covered: frozenset[Edge]
    full_scope: frozenset[Edge]
    head_scope: frozenset[Edge]
    head_self_loops: frozenset[Edge]
    gap_junction_only: frozenset[Edge]
    no_connection: frozenset[Edge]

    @property
    def coverable_head_scope(self) -> frozenset[Edge]:
        """Head-scope edges the table could cover in principle: head scope without self-loops."""
        return self.head_scope - self.head_self_loops


def coverage(table: Mapping[Edge, float], connectome: Connectome) -> MeasuredCoverage:
    """Join the table to a connectome by name and report where it lands."""
    chemical = frozenset((s.pre, s.post) for s in connectome.chemical_synapses)
    gap = {(g.neuron_a, g.neuron_b) for g in connectome.gap_junctions}
    gap |= {(b, a) for a, b in gap}
    names = {name for edge in table for name in edge}

    head = frozenset(e for e in chemical if e[0] in names and e[1] in names)
    listed = frozenset(table)
    return MeasuredCoverage(
        covered=listed & chemical,
        full_scope=chemical,
        head_scope=head,
        head_self_loops=frozenset(e for e in head if e[0] == e[1]),
        gap_junction_only=frozenset(e for e in listed - chemical if e in gap),
        no_connection=frozenset(e for e in listed - chemical if e not in gap),
    )
