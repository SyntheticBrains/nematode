"""The compact encoder: the connectome brain as a function of a few hundred named parameters.

With the encoder on, the connectome's wiring is fixed from data and only its interfaces, a handful
of wiring scales and its neuromodulation are free. A chemical edge's weight is its grounded sign
times its synapse count, normalised by the root sum of squares of the counts onto its postsynaptic
neuron, under one scale for excitatory and one for inhibitory edges. An edge with no grounded sign
takes a signed gain shared by its presynaptic neuron's transmitter group instead. Gap junctions
keep their strength-normalised data weights under one scale.

The free parameters, in the fixed order a vector carries them (:func:`encoder_entries`):

- **wiring**: the excitatory, inhibitory and gap scales, and the unsigned groups' gains;
- **sensory**: per amphid chemosensory class, a concentration and a dC/dt gain shared by its left
  and right neurons and a lateral gain applied with opposite signs to them (the lateral sense is
  right minus left); and a food-contact gain onto each of CEP, ADE and PDE, the dopaminergic
  mechanosensors of bacteria;
- **readout**: the pooled motor readout and one bias per output;
- **noise**: the action log-SDs;
- **modulator**: the neuromodulator layer's strengths, one per row, or tied per ligand;
- **timescale**: one decay timescale for the monoamine pools and one for the neuropeptide pools.

Every parameter has a range (its :class:`EncoderEntry`), so a fitter writes a vector in the unit
box and :func:`write_encoder` maps it into the brain in place. The wiring and sensory compositions
broadcast over leading axes, so a batch of candidates composes in one call.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np
import torch
from torch import nn

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from quantumnematode.connectome.model import Connectome
    from quantumnematode.connectome.neuromodulator_layer import LayerSpec

# The amphid's chemosensory classes; each is a left/right pair named class + "L" and class + "R".
AMPHID_CLASSES: tuple[str, ...] = (
    "AWC",
    "AWA",
    "ASE",
    "AWB",
    "ASK",
    "ADF",
    "ASI",
    "ASJ",
    "ASG",
    "ADL",
    "ASH",
)
# The dopaminergic mechanosensors of bacteria, whose activity slows the worm on food.
CONTACT_CLASSES: dict[str, tuple[str, ...]] = {
    "CEP": ("CEPDL", "CEPDR", "CEPVL", "CEPVR"),
    "ADE": ("ADEL", "ADER"),
    "PDE": ("PDEL", "PDER"),
}
# The classes among the encoded inputs that sense through the cGMP-gated channel TAX-4, whose loss
# removes their input.
TAX4_CLASSES: tuple[str, ...] = ("AWC", "AWB", "ASE", "ASG", "ASI", "ASJ", "ASK")
# An edge without a grounded sign is grouped by its presynaptic neuron's transmitter.
TRANSMITTER_GROUPS: tuple[str, ...] = ("dopamine", "serotonin", "octopamine", "none")
_GROUP_OF_TRANSMITTER: dict[str | None, str] = {
    "DA": "dopamine",
    "5-HT": "serotonin",
    "octopamine": "octopamine",
    None: "none",
}
MONOAMINES: frozenset[str] = frozenset({"dopamine", "octopamine", "serotonin", "tyramine"})
# The sensory features, in the order the brain's food features carry them.
SENSORY_FEATURES: tuple[str, ...] = ("concentration", "lateral", "derivative", "contact")
_CHEMICAL_FEATURES: tuple[str, ...] = ("concentration", "lateral", "derivative")
# The primitives' five outputs and the four pooled motor classes the readout reads.
OUTPUTS: tuple[str, ...] = ("speed", "curvature", "reversal", "omega", "delta")
POOLS: tuple[str, ...] = ("VB", "DB", "VA", "DA")
TIMESCALE_POOLS: tuple[str, ...] = ("monoamine", "neuropeptide")
Tying = Literal["per_couple", "per_ligand"]
Kind = Literal["linear", "log"]

# Scales of 0.5-0.75 leave the settled network graded; at 1.5 more than half of it saturates.
_WIRING_RANGE = (0.0, 1.5)
_GROUP_RANGE = (-2.0, 2.0)
_SENSORY_RANGE = (-5.0, 5.0)
_READOUT_RANGE = (-3.0, 3.0)
_NOISE_RANGE = (-4.0, 1.0)
_SIGNED_STRENGTH_RANGE = (-3.0, 3.0)
_FIXED_STRENGTH_RANGE = (0.001, 3.0)
_TIMESCALE_RANGE_S = (10.0, 3000.0)


@dataclass(frozen=True)
class EncoderEntry:
    """One encoder parameter: its name, its block, its range and how the unit box maps onto it."""

    name: str
    block: str
    low: float
    high: float
    kind: Kind = "linear"

    def value(self, unit: float) -> float:
        """Return the parameter's value at a unit-box coordinate."""
        if self.kind == "log":
            return math.exp(math.log(self.low) + unit * (math.log(self.high) - math.log(self.low)))
        return self.low + unit * (self.high - self.low)

    def unit(self, value: float) -> float:
        """Return a value's unit-box coordinate."""
        if self.kind == "log":
            span = math.log(self.high) - math.log(self.low)
            return (math.log(value) - math.log(self.low)) / span
        return (value - self.low) / (self.high - self.low)


def _row_fixed(spec: LayerSpec, index: int) -> bool:
    """Return whether a layer row's sign is fixed, so its strength is a positive magnitude."""
    return spec.rows[index].sign is not None


def modulator_groups(spec: LayerSpec, tying: Tying) -> list[tuple[str, bool]]:
    """Return the modulator parameters' names and whether each is a fixed-sign strength.

    Per couple there is one per layer row. Per ligand the rows are tied by ligand and by whether
    their sign is fixed, a ligand with rows of both kinds carrying one parameter for each, and each
    drive keeps its own.
    """
    if tying == "per_couple":
        return [(spec.rows[i].name, _row_fixed(spec, i)) for i in range(len(spec.rows))]
    groups: list[tuple[str, bool]] = []
    for i in range(len(spec.rows)):
        group = (_tied_name(spec, i), _row_fixed(spec, i))
        if group not in groups:
            groups.append(group)
    return groups


def _tied_name(spec: LayerSpec, index: int) -> str:
    row = spec.rows[index]
    if row.kind == "drive":
        return f"drive.{spec.drives[row.drive]}"
    kind = "fixed" if row.sign is not None else "fitted"
    return f"{kind}.{spec.ligands[row.ligand]}"


def encoder_entries(spec: LayerSpec, tying: Tying = "per_couple") -> list[EncoderEntry]:
    """Return every encoder parameter, in the fixed order a vector carries them."""
    entries = [
        EncoderEntry(f"wiring.{name}", "wiring", *_WIRING_RANGE)
        for name in ("excitatory", "inhibitory", "gap")
    ]
    entries += [
        EncoderEntry(f"wiring.unsigned.{group}", "wiring", *_GROUP_RANGE)
        for group in TRANSMITTER_GROUPS
    ]
    entries += [
        EncoderEntry(f"sensory.{cls}.{feature}", "sensory", *_SENSORY_RANGE)
        for cls in AMPHID_CLASSES
        for feature in _CHEMICAL_FEATURES
    ]
    entries += [
        EncoderEntry(f"sensory.{cls}.contact", "sensory", *_SENSORY_RANGE)
        for cls in CONTACT_CLASSES
    ]
    entries += [
        EncoderEntry(f"readout.{output}.{pool}", "readout", *_READOUT_RANGE)
        for output in OUTPUTS
        for pool in POOLS
    ]
    entries += [
        EncoderEntry(f"readout.{output}.bias", "readout", *_READOUT_RANGE) for output in OUTPUTS
    ]
    entries += [EncoderEntry(f"noise.{output}", "noise", *_NOISE_RANGE) for output in OUTPUTS]
    # A fixed-sign strength is a positive magnitude, read on a log scale: the box's centre leaves
    # the pools' held currents weak, so they do not swamp the sensory input on food.
    entries += [
        EncoderEntry(f"modulator.{name}", "modulator", *_FIXED_STRENGTH_RANGE, kind="log")
        if fixed
        else EncoderEntry(f"modulator.{name}", "modulator", *_SIGNED_STRENGTH_RANGE)
        for name, fixed in modulator_groups(spec, tying)
    ]
    entries += [
        EncoderEntry(f"timescale.{pool}", "timescale", *_TIMESCALE_RANGE_S, kind="log")
        for pool in TIMESCALE_POOLS
    ]
    return entries


def encoder_parameters(spec: LayerSpec, tying: Tying = "per_couple") -> list[str]:
    """Return the encoder parameters' names, in the order a vector carries them."""
    return [entry.name for entry in encoder_entries(spec, tying)]


def box_centre(spec: LayerSpec, tying: Tying = "per_couple") -> np.ndarray:
    """Return the unit box's centre, every coordinate 0.5."""
    return np.full(len(encoder_entries(spec, tying)), 0.5)


def tax4_vector(
    spec: LayerSpec,
    z: Sequence[float] | np.ndarray,
    tying: Tying = "per_couple",
) -> np.ndarray:
    """Return a vector's tax-4 encoding: the tax-4 classes' sensory gains at 0, all else kept."""
    out = np.asarray(z, dtype=np.float64).copy()
    for k, entry in enumerate(encoder_entries(spec, tying)):
        if entry.block == "sensory" and entry.name.split(".")[1] in TAX4_CLASSES:
            out[k] = entry.unit(0.0)
    return out


# ---------------------------------------------------------------------------------------------
# The compositions, broadcasting over leading axes
# ---------------------------------------------------------------------------------------------


def compose_chemical(
    scales: torch.Tensor,
    gains: torch.Tensor,
    positive: torch.Tensor,
    negative: torch.Tensor,
    grouped: torch.Tensor,
) -> torch.Tensor:
    """Return the chemical weights from the two sign scales and the unsigned groups' gains.

    ``scales`` is ``(..., 2)``, excitatory then inhibitory, and ``gains`` is ``(..., G)``.
    ``positive`` and ``negative`` are the ``(N, N)`` normalised counts of the excitatory and the
    inhibitory edges, the second negated, and ``grouped`` is ``(G, N, N)``, each unsigned group's
    normalised counts. Returns ``(..., N, N)``.
    """
    return (
        scales[..., 0, None, None] * positive
        + scales[..., 1, None, None] * negative
        + torch.einsum("...g,gij->...ij", gains, grouped)
    )


def compose_injection(
    gains: torch.Tensor,
    features: torch.Tensor,
    neurons: torch.Tensor,
    signs: torch.Tensor,
    n_neurons: int,
) -> torch.Tensor:
    """Return the ``(..., F, N)`` map from the sensory features to each neuron's input.

    ``gains`` is ``(..., E)``, one gain per injection entry; entry ``e`` places
    ``signs[e] * gains[..., e]`` at feature ``features[e]`` and neuron ``neurons[e]``.
    """
    leading = gains.shape[:-1]
    flat = torch.zeros(
        (*leading, len(SENSORY_FEATURES) * n_neurons),
        dtype=gains.dtype,
        device=gains.device,
    )
    flat = flat.index_add(-1, features * n_neurons + neurons, gains * signs)
    return flat.reshape(*leading, len(SENSORY_FEATURES), n_neurons)


def _injection_layout(
    neuron_index: Mapping[str, int],
) -> tuple[list[int], list[int], list[float], list[int]]:
    """Return each injection entry's feature, neuron, sign and sensory gain."""
    features: list[int] = []
    neurons: list[int] = []
    signs: list[float] = []
    gains: list[int] = []
    gain = 0
    for cls in AMPHID_CLASSES:
        for feature in _CHEMICAL_FEATURES:
            for side in ("L", "R"):
                features.append(SENSORY_FEATURES.index(feature))
                neurons.append(neuron_index[f"{cls}{side}"])
                # The lateral sense is right minus left, so it drives the two sides oppositely.
                signs.append(-1.0 if feature == "lateral" and side == "L" else 1.0)
                gains.append(gain)
            gain += 1
    for members in CONTACT_CLASSES.values():
        for name in members:
            features.append(SENSORY_FEATURES.index("contact"))
            neurons.append(neuron_index[name])
            signs.append(1.0)
            gains.append(gain)
        gain += 1
    return features, neurons, signs, gains


def _softplus_inverse(strength: torch.Tensor) -> torch.Tensor:
    """Return the value whose softplus is ``strength``, computed in double precision."""
    return torch.log(torch.expm1(strength.double())).float()


# ---------------------------------------------------------------------------------------------
# The module
# ---------------------------------------------------------------------------------------------


class CompactEncoder(nn.Module):
    """The encoder's fixed data and its own free parameters, for one connectome topology.

    Built from the topology's own wiring, so a rewired null carries its own counts, its own
    normalisations and its edges' signs. ``chem_sign`` is each edge's grounded sign, 0 where it has
    none. The readout, the log-SDs and the per-couple modulator strengths stay on the topology; this
    module holds the wiring scales, the group gains, the sensory gains, the tied modulator strengths
    and the two timescales.
    """

    chem_positive: torch.Tensor
    chem_negative: torch.Tensor
    chem_grouped: torch.Tensor
    sensory_features: torch.Tensor
    sensory_neurons: torch.Tensor
    sensory_signs: torch.Tensor
    sensory_gain_index: torch.Tensor
    row_group: torch.Tensor
    group_fixed: torch.Tensor
    monoamine_mask: torch.Tensor
    timescales_s: torch.Tensor

    def __init__(
        self,
        connectome: Connectome,
        neuron_index: Mapping[str, int],
        chem_sign: torch.Tensor,
        spec: LayerSpec,
        tying: Tying = "per_couple",
    ) -> None:
        super().__init__()
        n = len(neuron_index)
        self.n_neurons = n
        self.tying: Tying = tying
        self.spec = spec
        self.entries = encoder_entries(spec, tying)
        counts = np.zeros((n, n), dtype=np.float64)
        for syn in connectome.chemical_synapses:
            counts[neuron_index[syn.pre], neuron_index[syn.post]] = float(syn.weight)
        norm = np.sqrt((counts**2).sum(axis=0))
        base = np.divide(
            counts,
            norm[None, :],
            out=np.zeros_like(counts),
            where=norm[None, :] > 0,
        )
        sign = chem_sign.detach().cpu().numpy().astype(np.int64)
        grouped = np.zeros((len(TRANSMITTER_GROUPS), n, n))
        names = {i: name for name, i in neuron_index.items()}
        for i, j in zip(*np.nonzero((sign == 0) & (counts > 0)), strict=True):
            transmitter = connectome.neurons[names[int(i)]].neurotransmitter
            if transmitter not in _GROUP_OF_TRANSMITTER:
                msg = (
                    f"edge {names[int(i)]} -> {names[int(j)]} has no grounded sign, and its "
                    f"presynaptic transmitter {transmitter!r} is in no unsigned group"
                )
                raise ValueError(msg)
            grouped[TRANSMITTER_GROUPS.index(_GROUP_OF_TRANSMITTER[transmitter]), i, j] = base[i, j]
        self.register_buffer("chem_positive", _float(np.where(sign > 0, base, 0.0)))
        self.register_buffer("chem_negative", _float(np.where(sign < 0, -base, 0.0)))
        self.register_buffer("chem_grouped", _float(grouped))
        features, neurons, signs, gains = _injection_layout(neuron_index)
        self.n_sensory = gains[-1] + 1
        self.register_buffer("sensory_features", torch.tensor(features, dtype=torch.long))
        self.register_buffer("sensory_neurons", torch.tensor(neurons, dtype=torch.long))
        self.register_buffer("sensory_signs", torch.tensor(signs, dtype=torch.float32))
        self.register_buffer("sensory_gain_index", torch.tensor(gains, dtype=torch.long))
        # Placeholders until a vector is written; nothing here draws a random number.
        self.wiring = nn.Parameter(torch.ones(3))
        self.unsigned = nn.Parameter(torch.zeros(len(TRANSMITTER_GROUPS)))
        self.sensory = nn.Parameter(torch.zeros(self.n_sensory))
        if tying == "per_ligand":
            groups = modulator_groups(spec, tying)
            index = {group: k for k, group in enumerate(groups)}
            row_group = [
                index[_tied_name(spec, i), _row_fixed(spec, i)] for i in range(len(spec.rows))
            ]
            self.register_buffer("row_group", torch.tensor(row_group, dtype=torch.long))
            self.register_buffer(
                "group_fixed",
                torch.tensor([fixed for _, fixed in groups], dtype=torch.bool),
            )
            # A fixed group's strength is the softplus of its value, as the per-couple rows' are.
            self.tied_strength = nn.Parameter(torch.zeros(len(groups)))
        monoamine = [name in MONOAMINES for name in spec.ligands]
        self.register_buffer("monoamine_mask", torch.tensor(monoamine, dtype=torch.bool))
        self.register_buffer("timescales_s", torch.full((2,), 60.0))

    def chemical(self) -> torch.Tensor:
        """Return the ``(N, N)`` chemical weights, indexed ``[pre, post]``."""
        return compose_chemical(
            self.wiring[:2],
            self.unsigned,
            self.chem_positive,
            self.chem_negative,
            self.chem_grouped,
        )

    def gap_scale(self) -> torch.Tensor:
        """Return the gap junctions' scale."""
        return self.wiring[2]

    def injection(self) -> torch.Tensor:
        """Return the ``(F, N)`` map from the four sensory features to each neuron's input."""
        return compose_injection(
            self.sensory.index_select(0, self.sensory_gain_index),
            self.sensory_features,
            self.sensory_neurons,
            self.sensory_signs,
            self.n_neurons,
        )

    def row_strengths(self) -> torch.Tensor:
        """Return each layer row's strength under per-ligand tying, in row order."""
        tied = torch.where(
            self.group_fixed,
            nn.functional.softplus(self.tied_strength),
            self.tied_strength,
        )
        return tied.index_select(0, self.row_group)

    def decay(self, step_seconds: float) -> np.ndarray:
        """Return each ligand's per-step decay, ``exp(-step / timescale)``, in the layer's order."""
        timescale = torch.where(self.monoamine_mask, self.timescales_s[0], self.timescales_s[1])
        return np.exp(-step_seconds / timescale.cpu().double().numpy()).astype(np.float32)

    def own_parameters(self) -> list[nn.Parameter]:
        """Return the learnable tensors this module holds."""
        params = [self.wiring, self.unsigned, self.sensory]
        if self.tying == "per_ligand":
            params.append(self.tied_strength)
        return params


def _float(array: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(array.astype(np.float32))


# ---------------------------------------------------------------------------------------------
# Writing and reading a vector
# ---------------------------------------------------------------------------------------------


def _fixed_and_fitted_rows(spec: LayerSpec) -> tuple[list[int], list[int]]:
    fixed = [i for i in range(len(spec.rows)) if _row_fixed(spec, i)]
    fitted = [i for i in range(len(spec.rows)) if not _row_fixed(spec, i)]
    return fixed, fitted


def write_encoder(brain: object, z: Sequence[float] | np.ndarray) -> None:
    """Write a unit-box vector into an encoded connectome brain, in place.

    The encoder, the topology's readout, readout bias, log-SDs and modulator strengths take the
    vector's values, and the brain's per-step ligand decay follows the two timescales.
    """
    topology = brain.topology  # type: ignore[attr-defined]
    encoder: CompactEncoder = topology.encoder
    entries = encoder.entries
    z = np.asarray(z, dtype=np.float64)
    if z.shape != (len(entries),):
        msg = f"the encoder takes {len(entries)} parameters, got a vector of shape {z.shape}"
        raise ValueError(msg)
    values = np.array([entry.value(float(u)) for entry, u in zip(entries, z, strict=True)])
    cursor = 0

    def take(count: int) -> torch.Tensor:
        nonlocal cursor
        chunk = torch.tensor(values[cursor : cursor + count], dtype=torch.float32)
        cursor += count
        return chunk

    n_out, n_pool = len(OUTPUTS), len(POOLS)
    with torch.no_grad():
        encoder.wiring.copy_(take(3))
        encoder.unsigned.copy_(take(len(TRANSMITTER_GROUPS)))
        encoder.sensory.copy_(take(encoder.n_sensory))
        topology.readout.copy_(take(n_out * n_pool).reshape(n_out, n_pool))
        topology.readout_bias.copy_(take(n_out))
        topology.log_std.copy_(take(n_out))
        if encoder.tying == "per_couple":
            strengths = take(len(encoder.spec.rows))
            fixed, fitted = _fixed_and_fitted_rows(encoder.spec)
            topology.modulator_layer_rho.copy_(_softplus_inverse(strengths[fixed]))
            topology.modulator_layer_w.copy_(strengths[fitted])
        else:
            strengths = take(len(encoder.group_fixed))
            device = encoder.group_fixed.device
            encoder.tied_strength.copy_(
                torch.where(
                    encoder.group_fixed,
                    _softplus_inverse(strengths).to(device),
                    strengths.to(device),
                ),
            )
        encoder.timescales_s.copy_(
            torch.tensor(values[cursor : cursor + len(TIMESCALE_POOLS)], dtype=torch.float32),
        )
    brain._modulator_decay = encoder.decay(brain.config.modulator_step_seconds)  # type: ignore[attr-defined]  # noqa: SLF001 - the brain's own decay


def read_encoder(brain: object) -> np.ndarray:
    """Return the unit-box vector an encoded connectome brain carries."""
    topology = brain.topology  # type: ignore[attr-defined]
    encoder: CompactEncoder = topology.encoder
    # Every tensor comes to the CPU before it is widened: some devices hold no float64.
    with torch.no_grad():
        if encoder.tying == "per_couple":
            fixed, fitted = _fixed_and_fitted_rows(encoder.spec)
            strengths = torch.zeros(len(encoder.spec.rows), dtype=torch.float64)
            rho = topology.modulator_layer_rho.detach().cpu().double()
            strengths[fixed] = nn.functional.softplus(rho)
            strengths[fitted] = topology.modulator_layer_w.detach().cpu().double()
        else:
            tied = encoder.tied_strength.detach().cpu().double()
            strengths = torch.where(encoder.group_fixed.cpu(), nn.functional.softplus(tied), tied)
        parts = [
            encoder.wiring,
            encoder.unsigned,
            encoder.sensory,
            topology.readout.reshape(-1),
            topology.readout_bias,
            topology.log_std,
            strengths,
            encoder.timescales_s,
        ]
        values = np.concatenate([p.detach().cpu().double().numpy().ravel() for p in parts])
    return np.array(
        [entry.unit(float(v)) for entry, v in zip(encoder.entries, values, strict=True)],
    )
