"""The serotonin and PDF roaming/dwelling circuit, mapped onto a connectome's neurons.

The circuit file names neuron classes; a class's left and right members both release or both carry
the receptor. A class absent from the connectome is reported and skipped, never invented.

* **Serotonin** is released by NSM and acts through MOD-1, a serotonin-gated chloride channel, so
  its sign is inhibitory.
* **PDF** is released by AVB, PVP and SIAV and acts through PDFR-1, which raises cAMP, so its sign
  is excitatory. PDF also keeps NSM inactive. NSM has no synapse from any somatic neuron, so that
  inhibition is extrasynaptic or polysynaptic and is carried as its own link.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Sequence

logger = logging.getLogger(__name__)

CIRCUIT_PATH = (
    Path(__file__).resolve().parents[4]
    / "data"
    / "neuromodulators"
    / "roaming_dwelling_circuit.json"
)


@dataclass(frozen=True)
class ModulatorCircuit:
    """The circuit as neuron names present in one connectome.

    Attributes
    ----------
    serotonin_release, pdf_release
        The cells whose activity releases each modulator.
    mod1, pdfr1
        The cells carrying each receptor.
    mod1_sign, pdfr1_sign
        Each receptor's sign: -1 inhibitory, +1 excitatory.
    pdf_inhibits
        The cells PDF inhibits outside its receptor's synaptic targets.
    timescale_s
        The concentrations' decay timescale, in worm-seconds.
    skipped
        The member names the circuit cites that the connectome lacks.
    """

    serotonin_release: tuple[str, ...]
    pdf_release: tuple[str, ...]
    mod1: tuple[str, ...]
    pdfr1: tuple[str, ...]
    mod1_sign: int
    pdfr1_sign: int
    pdf_inhibits: tuple[str, ...]
    timescale_s: float
    skipped: tuple[str, ...]


def class_members(neuron_class: str, names: Sequence[str]) -> list[str]:
    """Return a class's members in ``names``: its left and right cells, or the class itself."""
    present = set(names)
    pair = [f"{neuron_class}{side}" for side in ("L", "R") if f"{neuron_class}{side}" in present]
    if pair:
        return pair
    return [neuron_class] if neuron_class in present else []


def load_circuit(names: Sequence[str], path: Path = CIRCUIT_PATH) -> ModulatorCircuit:
    """Map the circuit file's classes onto the neurons in ``names``.

    Raises
    ------
    ValueError
        If a modulator is left with no release cell or no receptor cell, so its effect could not
        exist on this connectome.
    """
    data: dict[str, Any] = json.loads(path.read_text())
    skipped: list[str] = []

    def resolve(classes: Sequence[str]) -> tuple[str, ...]:
        members: list[str] = []
        for neuron_class in classes:
            found = class_members(neuron_class, names)
            if not found:
                skipped.append(neuron_class)
            members.extend(found)
        return tuple(members)

    serotonin, pdf = data["serotonin"], data["pdf"]
    circuit = ModulatorCircuit(
        serotonin_release=resolve(serotonin["release"]["neurons"]),
        pdf_release=resolve(pdf["release"]["neurons"]),
        mod1=resolve(serotonin["receptor"]["neurons"]),
        pdfr1=resolve(pdf["receptor"]["neurons"]),
        mod1_sign=int(serotonin["receptor"]["sign"]),
        pdfr1_sign=int(pdf["receptor"]["sign"]),
        pdf_inhibits=resolve(pdf["inhibits_release_of"]["neurons"]),
        timescale_s=float(data["timescale_s"]["value"]),
        skipped=tuple(skipped),
    )
    for label, cells in (
        ("serotonin release", circuit.serotonin_release),
        ("PDF release", circuit.pdf_release),
        ("MOD-1", circuit.mod1),
        ("PDFR-1", circuit.pdfr1),
    ):
        if not cells:
            msg = f"the connectome has none of the {label} cells the circuit names"
            raise ValueError(msg)
    if skipped:
        logger.warning("Neuromodulator circuit: classes absent from the connectome: %s", skipped)
    return circuit
