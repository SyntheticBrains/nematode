#!/usr/bin/env python
r"""C.2 fitted to the real wild type and Scheer's three mutants by a staged procedure, and its reads.

**The controller** is the encoded connectome under the procedure's options
(``c2_encoder_probe.PROCEDURE_SETTINGS``: per-ligand tying, 137 parameters, the soft-capped readout
with pool centres, the graded box and inhibition floor, PDFR-1's and SER-5's cited signs, slow
speed noise), on the grading arm whose on-food reversals follow the worm's run speed
(``benchmark_grade.ARMS["connectome_c2_following"]``).

**The objective**, against a truth (the bands' centres, or a recovery truth's observation):

- the definedness-first ranking score on the stage's families;
- each fitted mutant's term, the RMS of 10·asinh(z/10) over the statistics the encoding table
  grades it on (:data:`GRADED`), each weighing n_f / n_m, so the mutants together are half;
- 500 for a saturated network, and 300·(1 - s/0.5) for one whose speed or reversal output spans
  s < 0.5 between its 5th and 95th percentiles, both read on the wild type's simulation;
- in the final stage, 10·Σ Φ((abs(z) - bar) / SD), each statistic's chance of failing a
  population, with SD measured by the noise step.

**The procedure** (:data:`STAGES`), sep-CMA at population 48 and 200 animals a candidate after
stage 1, which runs at population 24 and 64 animals:

1. the 77 parameters carrying senses to outputs from the graded box's centre, sigma0 0.2, on D_fit's
   families, 150 generations;
2. **2a**, all 137 from stage 1's answer with the speed noise's timescale at 1 s, sigma0 0.1, on D_fit's
   families, 300 generations;
3. **2b**, all six families with the mutants, from 2a's search state, 200 generations;
4. the noise step: 2b's answer on 30 populations, each statistic's SD between them;
5. **the final stage**, 2b's objective plus the failure probability, from 2b's search state, 50
   generations.

Each stage runs its full count; the final stage's plateau is read and reported.

**The first half is started up to six times** (:data:`STARTS`), in sequence, each from the box's
centre with its own CMA seeds and animals. A start is dropped at 2a's generation 60 when its mean
time on the lawn over the readouts at generations 40, 50 and 60 is below 0.95
(:func:`lawn_holding`). The first start that holds the lawn runs 2a to its end and the procedure
continues from it. A run in which no start holds the lawn has no answer.

**The reads.** The pilot's go rule, F.0b-ii (one truth, the pilot's answer), the registered verdict
(the wild type, tph-1 and tax-4 each in at least 3 of 5 populations), the held-out reading on
Scheer's held-out dates beside W's minimum and a band-cohort reference, the restart spread, and a
tax-4 leave-one-out as description.

Subcommands (each guarded by launch record 133 as registered; ``manifest``,
``speed-noise-control`` and ``smoke`` precede it)::

    uv run python scripts/analysis/c2_fit.py speed-noise-control
    uv run python scripts/analysis/c2_fit.py manifest
    uv run python scripts/analysis/c2_fit.py pilot
    uv run python scripts/analysis/c2_fit.py recovery-preflight
    uv run python scripts/analysis/c2_fit.py recovery
    uv run python scripts/analysis/c2_fit.py run
    uv run python scripts/analysis/c2_fit.py grade
    uv run python scripts/analysis/c2_fit.py read
    uv run python scripts/analysis/c2_fit.py loo
    uv run python scripts/analysis/c2_fit.py smoke    # execution check, discarded
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import hashlib
import json
import math
import pickle
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import benchmark_grade as bg
import body_kinematics_eval as harness
import c2_encoder_probe as probe
import cma_recovery as cr
import cma_recovery_v2 as v2
import npe_recovery as nr
import npe_region_check as rc

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Sequence

DESCRIPTION = "C.2 fitted to the wild type and its mutants by a staged procedure, and its reads."
_REPO = Path(__file__).resolve().parents[2]
_LOGBOOKS = _REPO / "docs" / "experiments" / "logbooks" / "supporting"
SUPPORT = _LOGBOOKS / "133-c2-fit"
LAUNCH_PATH = SUPPORT / "launch.md"
MANIFEST_PATH = SUPPORT / "manifest.json"
SPEED_NOISE_PATH = SUPPORT / "speed_noise_control.json"
PILOT_PATH = SUPPORT / "pilot.json"
RECOVERY_PREFLIGHT_PATH = SUPPORT / "recovery_preflight.json"
RECOVERY_PATH = SUPPORT / "recovery.json"
FIT_PATH = SUPPORT / "fit.json"
LOO_PATH = SUPPORT / "loo.json"
ENCODINGS_PATH = _LOGBOOKS / "113-c1-fit" / "encodings.md"
# Logbook 131's last answer, which passes 10 of 10 under the speed-following reversal.
ANSWER_131_PATH = _LOGBOOKS / "131-c2-clean-start" / "probes" / "hinge4.json"
W_MINIMUM_PATH = _LOGBOOKS / "124-benchmark-v4" / "read.json"
BUILD = _REPO / "build" / "c2-fit"
FORAGING = _REPO / "configs" / "scenarios" / "foraging"

ARM = "connectome_c2_following"
TYING = "per_ligand"
SETTINGS: dict[str, Any] = probe.PROCEDURE_SETTINGS
OPTIONS: dict[str, Any] = probe.PROCEDURE_OPTIONS
# The encoded brain's behaviour does not depend on its seed: every tensor it reads is written.
BRAIN_SEED = 1
ANIMALS = 200
POPSIZE = 48
MUTANTS: tuple[str, ...] = ("tph-1", "pdfr-1", "tax-4")
KNOCKOUT_GENES: dict[str, tuple[str, ...]] = probe.KNOCKOUT_GENES
# The encoding table's graded statistics (113-c1-fit/encodings.md); every other lawn difference is
# recorded and reported, never scored. tph-1's leaving is left out: its band reference is unreliable.
GRADED: dict[str, tuple[str, ...]] = {
    "tph-1": ("roaming_fraction",),
    "pdfr-1": ("roaming_fraction", "exits_per_on_lawn_hour"),
    "tax-4": ("roaming_fraction", "exits_per_on_lawn_hour"),
}
LAWN_DIFFERENCES = ("roaming_fraction", "exits_per_on_lawn_hour")
# The mutants the verdict reads: the two whose directions are resolved.
VERDICT_MUTANTS: tuple[str, ...] = ("tph-1", "tax-4")
FIT_FAMILIES: tuple[str, ...] = ("small_lawn", "uniform_lawn", "kinematics", "activity")
FIT_ASSAYS: tuple[str, ...] = ("small_lawn", "uniform_lawn")
W_FAMILIES: tuple[str, ...] = ("small_lawn", "kinematics", "activity")
STAGE1_BLOCKS: tuple[str, ...] = ("wiring", "sensory", "readout", "noise")

SATURATED = 0.99
SATURATED_SHARE = 0.5
SATURATION_PENALTY = 500.0
SPREAD_FLOOR = 0.5
RESPONSIVE_PENALTY = 300.0
RECORDED_ANIMALS = 8
FAILURE_WEIGHT = 10.0
SD_FLOOR = 0.05
TIMESCALE_AT_2A_S = 1.0
READOUT_EVERY = 10
KILL_READOUTS = 3
HOLDS_LAWN = 0.75
# The first half's starts: stage 1's size, how many, and the lawn-holding reading that keeps one.
STAGE1_ANIMALS = 64
STAGE1_POPSIZE = 24
STARTS = 6
HOLD_READOUTS: tuple[int, ...] = (40, 50, 60)
HOLD_BAR = 0.95
LAWN_STATISTIC = "small_lawn/time_on_lawn"
FIRST_HALF: tuple[str, ...] = ("1", "2a")
D_FIT_KILL = 2.5
VERDICT_POPULATIONS = 5
PASS_NEEDED = 3
ROBERTS_REVERSE_SHARE = 0.173

TRUTH_BASE = 940_000_000
PRIOR_BASE = 941_000_000
CENTRE_BASE = 942_000_000
PRIOR_SEED = 5805
PRIOR_DRAWS = 200
TRUTH_REPEATS = 20  # 1-10 the floor, 11-20 the oracle
CENTRE_REPEATS = 10
RECOVERY_RESIMULATIONS = 10
ORACLE_GATE = 0.85
START_GATE = 0.25
SEPARATION = 1.0
GAIN_GATE = 0.75
SPEED_NOISE_BASE = 888_000_000
REFERENCE_SEED = 2048
REFERENCE_POPULATIONS = 15


@dataclasses.dataclass(frozen=True)
class Stage:
    """One stage of the procedure: what it fits, from where, and for how long."""

    name: str
    families: tuple[str, ...] | None  # None: all six
    mutants: bool  # whether the run's fitted mutants enter
    subset: bool  # stage 1's 77 parameters only
    sigma0: float | None  # None: the previous stage's search state
    generations: int
    failure: bool  # whether the failure probability enters


STAGES: tuple[Stage, ...] = (
    Stage(
        "1",
        FIT_FAMILIES,
        mutants=False,
        subset=True,
        sigma0=0.2,
        generations=150,
        failure=False,
    ),
    Stage(
        "2a",
        FIT_FAMILIES,
        mutants=False,
        subset=False,
        sigma0=0.1,
        generations=300,
        failure=False,
    ),
    Stage("2b", None, mutants=True, subset=False, sigma0=None, generations=200, failure=False),
    Stage("final", None, mutants=True, subset=False, sigma0=None, generations=50, failure=True),
)
NOISE_AFTER = "2b"


@dataclasses.dataclass(frozen=True)
class Layout:
    """A run's animal seeds: the stage sub-blocks, each start's, the readout, noise and grade.

    Stage 1 runs at its own size, each start in its own slice of stage 1's sub-block. 2a's
    generations up to the lawn-holding reading (``early_generations``) are each start's own, in the
    ``early`` block; the kept start's later generations sit in 2a's sub-block at their own positions.
    """

    base: int
    animals: int = ANIMALS
    popsize: int = POPSIZE
    stage_stride: int = 4_000_000
    readout: int = 16_000_000
    noise: int = 17_000_000
    grade: int = 18_000_000
    noise_populations: int = 30
    grade_populations: int = 15
    first_animals: int = STAGE1_ANIMALS
    first_popsize: int = STAGE1_POPSIZE
    start_stride: int = 250_000
    early: int = 12_500_000
    early_stride: int = 576_000
    early_generations: int = HOLD_READOUTS[-1]

    def _block(self, start: int) -> list[int]:
        return list(range(start, start + self.animals))

    def popsize_of(self, stage: int) -> int:
        """Return a stage's population: stage 1's own, or the run's."""
        return self.first_popsize if stage == 0 else self.popsize

    def candidate_seeds(
        self,
        stage: int,
        generation: int,
        member: int,
        start: int = 0,
    ) -> list[int]:
        """Return a candidate's animals: stage index, generation from 1, member, and the start."""
        if stage == 0:
            k = self.first_popsize * (generation - 1) + member
            first = self.base + self.start_stride * start + self.first_animals * k
            return list(range(first, first + self.first_animals))
        k = self.popsize * (generation - 1) + member
        if stage == 1 and generation <= self.early_generations:
            own = self.base + self.early + self.early_stride * start
            return self._block(own + self.animals * k)
        return self._block(self.base + self.stage_stride * stage + self.animals * k)

    def readout_seeds(self) -> list[int]:
        """Return the readout animals, the same at every readout."""
        return self._block(self.base + self.readout)

    def noise_seeds(self, population: int) -> list[int]:
        """Return a noise population's animals."""
        return self._block(self.base + self.noise + self.animals * population)

    def grade_seeds(self, population: int) -> list[int]:
        """Return a grade population's animals."""
        return self._block(self.base + self.grade + self.animals * population)

    def spans(self, generations: Sequence[int], starts: int = 1) -> dict[str, tuple[int, int]]:
        """Return each use's first and last animal seed, for checks of disjointness."""
        out = {}
        for a in range(starts):
            last = self.candidate_seeds(0, generations[0], self.first_popsize - 1, a)[-1]
            out[f"stage 0 start {a}"] = (self.candidate_seeds(0, 1, 0, a)[0], last)
            n = min(generations[1], self.early_generations)
            last = self.candidate_seeds(1, n, self.popsize - 1, a)[-1]
            out[f"stage 1 start {a}"] = (self.candidate_seeds(1, 1, 0, a)[0], last)
        for s, n in enumerate(generations):
            if s == 0 or (s == 1 and n <= self.early_generations):
                continue
            g = self.early_generations + 1 if s == 1 else 1
            first = self.candidate_seeds(s, g, 0)[0]
            out[f"stage {s}"] = (first, self.candidate_seeds(s, n, self.popsize - 1)[-1])
        out["readout"] = (self.readout_seeds()[0], self.readout_seeds()[-1])
        last_noise = self.noise_seeds(self.noise_populations - 1)[-1]
        out["noise"] = (self.noise_seeds(0)[0], last_noise)
        out["grade"] = (self.grade_seeds(0)[0], self.grade_seeds(self.grade_populations - 1)[-1])
        return out


@dataclasses.dataclass(frozen=True)
class Run:
    """One run of the procedure: its seeds, its CMA seeds, its starts and the mutants it fits.

    ``hold_bars`` gives one lawn-holding bar per start, so its length is the number of starts.
    The first start takes the run's CMA seeds; each later one takes the next two from
    ``restart_seed``, for stage 1 and 2a.
    """

    name: str
    layout: Layout
    cma_seeds: tuple[int, ...]
    restart_seed: int
    mutants: tuple[str, ...] = MUTANTS
    generations: tuple[int, ...] = tuple(s.generations for s in STAGES)
    kill_rules: bool = False
    hold_readouts: tuple[int, ...] = HOLD_READOUTS
    hold_bars: tuple[float, ...] = (HOLD_BAR,) * STARTS

    @property
    def starts(self) -> int:
        """Return how many times the first half may be started."""
        return len(self.hold_bars)

    def cma_seed(self, stage: int, start: int) -> int:
        """Return a stage's CMA seed: the first half's stages take each start's own."""
        if start == 0 or stage >= len(FIRST_HALF):
            return self.cma_seeds[stage]
        return self.restart_seed + len(FIRST_HALF) * (start - 1) + stage


# The smoke's stage 1 has its own size and its 2a runs past the reading, as the procedure's do.
SMOKE_LAYOUT = Layout(
    99_880_000,
    animals=8,
    popsize=6,
    stage_stride=2_000,
    readout=8_000,
    noise=9_000,
    grade=9_500,
    noise_populations=3,
    grade_populations=2,
    first_animals=6,
    first_popsize=4,
    start_stride=200,
    early=6_500,
    early_stride=500,
    early_generations=10,
)
SMOKE_GENERATIONS = (2, 12, 2, 2)
RUNS: dict[str, Run] = {
    "pilot": Run("pilot", Layout(1_100_000_000), (9531, 9532, 9533, 9534), 9541, kill_rules=True),
    "recovery": Run("recovery", Layout(920_000_000), (5801, 5802, 5803, 5804), 5831),
    "fit": Run("fit", Layout(960_000_000), (5811, 5812, 5813, 5814), 5841),
    "loo": Run(
        "loo",
        Layout(980_000_000),
        (5821, 5822, 5823, 5824),
        5851,
        mutants=("tph-1", "pdfr-1"),
    ),
    # The smoke's first start can never hold the lawn and its second always does, so both paths run.
    "smoke": Run(
        "smoke",
        SMOKE_LAYOUT,
        (9508, 9509, 9510, 9511),
        9561,
        generations=SMOKE_GENERATIONS,
        hold_readouts=(10,),
        hold_bars=(2.0, 0.0),
    ),
}


# ---------------------------------------------------------------------------------------------
# The encoded brain, its guards and a candidate's simulation
# ---------------------------------------------------------------------------------------------

_BRAINS: dict[tuple[Any, ...], Any] = {}
_OBSERVED: dict[str, Any] = {"steps": None, "count": 0, "outputs": []}


def brain(genes: tuple[str, ...] = (), **overrides: object) -> Any:  # noqa: ANN401 - a brain
    """Return this process's procedure brain for a set of knockout genes, built once and reused."""
    merged = {**SETTINGS, **overrides}
    key = (genes, tuple(sorted((k, repr(v)) for k, v in merged.items())))
    if key not in _BRAINS:
        path = FORAGING / f"{bg.ARMS[ARM]}_assay_scheer_lawn.yml"
        built, _, _ = harness.build_brain(
            path,
            BRAIN_SEED,
            {"modulator_knockout_genes": list(genes), **merged},
        )
        _BRAINS[key] = built
    return _BRAINS[key]


def entries() -> list[Any]:
    """Return the procedure encoder's 137 named parameters, in the vector's order."""
    return list(brain().topology.encoder.entries)


def controller(
    z: Sequence[float] | np.ndarray,
    mutant: str | None = None,
) -> tuple[Any, Callable[[Path], Any]]:
    """Write a vector, or a mutant's encoding of it, into its brain; return it and a factory."""
    from quantumnematode.connectome import encoder as enc
    from quantumnematode.evaluation.controllers import BatchedConnectome

    built = brain(KNOCKOUT_GENES.get(mutant or "", ()))
    spec = built.topology.modulator_spec
    vector = np.asarray(z, dtype=float)
    enc.write_encoder(
        built,
        enc.tax4_vector(spec, vector, TYING, **OPTIONS) if mutant == "tax-4" else vector,
    )
    return built, lambda _path: BatchedConnectome(built)


def _record_forward(mean: Any, h: Any, motor: Any) -> None:  # noqa: ANN401 - tensors
    saturated = h.index_select(-1, motor).abs() > SATURATED
    saturated = saturated.reshape(-1, saturated.shape[-1])
    steps = saturated.sum(dim=0).numpy()
    _OBSERVED["steps"] = steps if _OBSERVED["steps"] is None else _OBSERVED["steps"] + steps
    _OBSERVED["count"] += saturated.shape[0]
    _OBSERVED["outputs"].append(mean.reshape(-1, mean.shape[-1])[:RECORDED_ANIMALS].numpy())


@contextlib.contextmanager
def observing(built: Any) -> Iterator[None]:  # noqa: ANN401 - a brain
    """Record the brain's motor saturation and outputs while the wild type is simulated."""
    _OBSERVED.update(steps=None, count=0, outputs=[])
    topology = built.topology
    # The topology's motor-neuron indices into the settled state, which the guards read.
    motor = topology._motor_flat_indices
    topology.forward_observer = lambda mean, h: _record_forward(mean, h, motor)
    try:
        yield
    finally:
        topology.forward_observer = None


def guard_readings() -> dict[str, Any]:
    """Return the share of saturated motor neurons and each output's 5th-to-95th spread."""
    from quantumnematode.connectome import encoder as enc

    count = max(int(_OBSERVED["count"]), 1)
    share = np.zeros(1) if _OBSERVED["steps"] is None else np.asarray(_OBSERVED["steps"]) / count
    outputs = (
        np.concatenate(_OBSERVED["outputs"])
        if _OBSERVED["outputs"]
        else np.zeros((1, len(enc.OUTPUTS)))
    )
    spread = np.percentile(outputs, 95, axis=0) - np.percentile(outputs, 5, axis=0)
    return {
        "saturated": float(np.mean(share > SATURATED_SHARE)),
        "spread": dict(zip(enc.OUTPUTS, map(float, spread), strict=True)),
    }


def _difference(a: float | None, b: float | None) -> float | None:
    if a is None or b is None or not (math.isfinite(a) and math.isfinite(b)):
        return None
    return float(a - b)


def _statistics(graded: dict[str, Any]) -> dict[str, float | None]:
    return {
        f"{family}/{name}": reading["value"]
        for family, readings in graded["families"].items()
        for name, reading in readings.items()
    }


def simulate_point(
    z: Sequence[float],
    seeds: list[int],
    *,
    assays: tuple[str, ...] | None,
    mutants: tuple[str, ...],
    presence: bool = False,
) -> dict[str, Any]:
    """Simulate a vector on these animals: the wild type's statistics and guards, each mutant's Δ.

    The guards read the wild type's simulation only; each mutant then runs the small lawn on the
    same animals, and its differences from that wild type are recorded on both lawn statistics.
    """
    bands = bg.load_bands_v4()
    built, factory = controller(z)
    with observing(built):
        populations, chemotaxis = bg.simulate_v2(ARM, factory, seeds, bands, assays=assays)
    guards = guard_readings()
    graded = bg.grade_populations_v4(populations, chemotaxis, bands, presence=presence)
    clf = bg.classifier(bands)
    wild = bg.family_values("small_lawn", populations["small_lawn"], clf)
    delta: dict[str, dict[str, float | None]] = {}
    for mutant in mutants:
        _, mutated = controller(z, mutant)
        mutant_populations, _ = bg.simulate_v2(ARM, mutated, seeds, bands, assays=("small_lawn",))
        values = bg.family_values("small_lawn", mutant_populations["small_lawn"], clf)
        delta[mutant] = {s: _difference(values.get(s), wild.get(s)) for s in LAWN_DIFFERENCES}
    small_lawn = graded["families"].get("small_lawn", {})
    share = graded.get("reversal_share")
    record: dict[str, Any] = {
        "statistics": _statistics(graded),
        "z": {
            f"{family}/{name}": reading["z"]
            for family, readings in graded["families"].items()
            for name, reading in readings.items()
        },
        "passes": {
            f"{family}/{name}": reading["passes"]
            for family, readings in graded["families"].items()
            for name, reading in readings.items()
        },
        "family_passes": graded["family_passes"],
        "D": graded["D"],
        "D_fit": graded["D_fit"],
        "time_on_lawn": small_lawn.get("time_on_lawn", {}).get("value"),
        "reversal_share": None if share is None else share["value"],
        **guards,
        "delta": delta,
    }
    if presence:
        record |= {
            "wild_type_passes": bool(graded["wild_type_passes"]),
            "local_search_presence": graded.get("local_search_presence"),
            "chemotaxis_presence": graded.get("chemotaxis_presence"),
        }
    return nr._finite(record)


def vector_digest(z: Sequence[float]) -> str:
    """Return a vector's SHA-256 over its float64 bytes, which a stored record is checked on."""
    return hashlib.sha256(np.asarray(z, dtype=np.float64).tobytes()).hexdigest()


def _candidate_job(item: tuple[str, int, int, int, list[float], int]) -> dict[str, Any]:
    """Simulate one candidate of a run's stage: (run, stage, generation, member, vector, start)."""
    name, stage, generation, member, z, first_half_start = item
    run = RUNS[name]
    spec = STAGES[stage]
    seeds = run.layout.candidate_seeds(stage, generation, member, first_half_start)
    start = time.perf_counter()
    record = simulate_point(
        z,
        seeds,
        assays=FIT_ASSAYS if spec.families == FIT_FAMILIES else None,
        mutants=run.mutants if spec.mutants else (),
    )
    return {
        "generation": generation,
        "member": member,
        "vector": vector_digest(z),
        **record,
        "seconds": time.perf_counter() - start,
    }


def _point_job(item: tuple[list[float], list[int], bool]) -> dict[str, Any]:
    """Simulate a vector on every assay and the three mutants: (vector, animals, presence)."""
    z, seeds, presence = item
    return {
        "vector": vector_digest(z),
        **simulate_point(z, seeds, assays=None, mutants=MUTANTS, presence=presence),
    }


# ---------------------------------------------------------------------------------------------
# The objective
# ---------------------------------------------------------------------------------------------


def phi(x: float) -> float:
    """Return the standard normal distribution function at x."""
    return 0.5 * math.erfc(-x / math.sqrt(2.0))


def band_truth(bands: dict[str, Any]) -> dict[str, Any]:
    """Return the bands' centres as a truth: every wild-type statistic and each mutant's Δ."""
    statistics = {}
    for family, stats in bands["families"].items():
        family_bands = bg.family_bands_v2(bands, family, ANIMALS)
        for s in stats:
            statistics[f"{family}/{s}"] = float(family_bands[s].centre)
    delta = {
        m: {s: float(band.centre) for s, band in bg.mutant_bands(bands, m, ANIMALS).items()}
        for m in MUTANTS
    }
    return {"statistics": statistics, "delta": delta}


def mutant_scales(bands: dict[str, Any]) -> dict[str, dict[str, float]]:
    """Return each mutant's difference bands' scales at n = 200."""
    return {
        m: {s: float(band.scale) for s, band in bg.mutant_bands(bands, m, ANIMALS).items()}
        for m in MUTANTS
    }


def mutant_z(truth: dict[str, Any], record: dict[str, Any], scale: float, m: str, s: str) -> float:
    """Return a mutant difference's z against the truth's, capped at 10; undefined at the cap."""
    from quantumnematode.benchmark import grading as g

    value = record.get("delta", {}).get(m, {}).get(s)
    centre = truth["delta"][m].get(s)
    if value is None or centre is None:
        return g.Z_CAP
    return abs(g.z_score(float(value), g.Band(float(centre), scale)))


class Objective:
    """A stage's score of a candidate's record against a truth."""

    def __init__(
        self,
        truth: dict[str, Any],
        bands: dict[str, Any],
        *,
        families: tuple[str, ...] | None,
        mutants: tuple[str, ...],
        sd: dict[str, float] | None = None,
    ) -> None:
        self.truth = truth
        self.scales = v2.band_scales_v2(bands)
        self.mscales = mutant_scales(bands)
        names = tuple(bands["families"]) if families is None else families
        self.family_map = {f: tuple(bands["families"][f]) for f in names}
        self.all_families = {f: tuple(s) for f, s in bands["families"].items()}
        self.mutants = mutants
        self.sd = sd
        self.wild_bar = float(bands["bars"]["wild_type"])
        self.mutant_bars = {m: float(bands["bars"][m]) for m in MUTANTS}

    def distance(self, record: dict[str, Any]) -> float:
        """Return the ranking score, with the mutants weighed in at half where the stage fits them."""
        score = v2.ranking_score(self.truth, record, self.scales, self.family_map)
        if not self.mutants:
            return score
        undefined = math.floor(score / v2.UNDEFINED_RANK)
        dc = score - v2.UNDEFINED_RANK * undefined
        terms = []
        for m in self.mutants:
            zs = [mutant_z(self.truth, record, self.mscales[m][s], m, s) for s in GRADED[m]]
            compressed = [v2.COMPRESSION * math.asinh(z / v2.COMPRESSION) for z in zs]
            terms.append(math.sqrt(float(np.mean(np.square(compressed)))))
        n_f, n_m = len(self.family_map), len(self.mutants)
        w = n_f / n_m
        combined = math.sqrt((n_f * dc**2 + w * sum(t**2 for t in terms)) / (n_f + w * n_m))
        return v2.UNDEFINED_RANK * undefined + combined

    @staticmethod
    def guards(record: dict[str, Any]) -> float:
        """Return the saturation and graded responsiveness guards' penalty."""
        penalty = SATURATION_PENALTY if record["saturated"] > SATURATED_SHARE else 0.0
        least = min(record["spread"]["speed"], record["spread"]["reversal"])
        if least < SPREAD_FLOOR:
            penalty += RESPONSIVE_PENALTY * (1.0 - least / SPREAD_FLOOR)
        return penalty

    def failure(self, record: dict[str, Any]) -> float:
        """Return 10 x the summed chance that each defined statistic fails a population."""
        if self.sd is None:
            return 0.0
        total = 0.0
        for key, centre in self.truth["statistics"].items():
            value = record["statistics"].get(key)
            if key.split("/")[0] not in self.all_families or value is None or centre is None:
                continue
            z = (float(value) - float(centre)) / self.scales[key]
            total += phi((abs(z) - self.wild_bar) / self.sd[key])
        for m in self.mutants:
            for s in GRADED[m]:
                z = mutant_z(self.truth, record, self.mscales[m][s], m, s)
                total += phi((z - self.mutant_bars[m]) / self.sd[f"{m}/{s}"])
        return FAILURE_WEIGHT * total

    def __call__(self, record: dict[str, Any]) -> float:
        """Return the candidate's score: the distance, the guards and the failure probability."""
        return self.distance(record) + self.guards(record) + self.failure(record)


def noise_sd(
    populations: list[dict[str, Any]],
    truth: dict[str, Any],
    bands: dict[str, Any],
    mutants: tuple[str, ...],
) -> dict[str, float]:
    """Return each statistic's SD of z between populations, floored at 0.05.

    A statistic defined in fewer than two populations takes the floor.
    """
    scales, mscales = v2.band_scales_v2(bands), mutant_scales(bands)

    def floored(zs: list[float]) -> float:
        if len(zs) < 2:
            return SD_FLOOR
        return max(float(np.std(zs, ddof=1)), SD_FLOOR)

    out = {}
    for key, centre in truth["statistics"].items():
        zs = [
            (float(p["statistics"][key]) - float(centre)) / scales[key]
            for p in populations
            if p["statistics"].get(key) is not None
        ]
        out[key] = floored(zs)
    for m in mutants:
        for s in GRADED[m]:
            zs = [
                mutant_z(truth, p, mscales[m][s], m, s)
                for p in populations
                if p.get("delta", {}).get(m, {}).get(s) is not None
            ]
            out[f"{m}/{s}"] = floored(zs)
    return out


# ---------------------------------------------------------------------------------------------
# The grade and the verdict
# ---------------------------------------------------------------------------------------------


def mutant_reading(
    mutant: str,
    delta: dict[str, float | None],
    bands: dict[str, Any],
) -> dict[str, Any]:
    """Grade a mutant's differences on the encoding table's statistics, the others reported."""
    from quantumnematode.benchmark import grading as g

    mutant_bands = bg.mutant_bands(bands, mutant, ANIMALS)
    values = {s: math.nan if (d := delta.get(s)) is None else float(d) for s in LAWN_DIFFERENCES}
    graded = g.grade(
        {s: values[s] for s in GRADED[mutant]},
        {s: mutant_bands[s] for s in GRADED[mutant]},
        float(bands["bars"][mutant]),
        directions=True,
    )
    statistics = {}
    for s in LAWN_DIFFERENCES:
        z = g.z_score(values[s], mutant_bands[s])
        statistics[s] = {
            "delta": delta.get(s),
            "z": z,
            "graded": s in GRADED[mutant],
            "passes": graded[s].passes if s in graded else None,
        }
    return {"statistics": statistics, "passes": all(r.passes for r in graded.values())}


def grade_record(record: dict[str, Any], bands: dict[str, Any]) -> dict[str, Any]:
    """Attach each mutant's reading to a graded population's record."""
    return {
        **record,
        "mutants": {m: mutant_reading(m, record["delta"][m], bands) for m in record["delta"]},
    }


def verdict(grades: list[dict[str, Any]]) -> dict[str, Any]:
    """Read the verdict on the first five populations: the wild type, tph-1 and tax-4 each in 3."""
    used = grades[:VERDICT_POPULATIONS]
    wild = sum(bool(g["wild_type_passes"]) for g in used)
    mutants = {m: sum(bool(g["mutants"][m]["passes"]) for g in used) for m in MUTANTS}
    passes = wild >= PASS_NEEDED and all(mutants[m] >= PASS_NEEDED for m in VERDICT_MUTANTS)
    return {
        "populations": len(used),
        "wild_type_passes": wild,
        "mutant_passes": mutants,
        "passes": bool(passes),
        "diagnosis": diagnosis(used),
    }


def no_answer(reason: str) -> dict[str, Any]:
    """Return the verdict of a run that reached no answer: it fails, for the reason given."""
    return {
        "populations": 0,
        "wild_type_passes": 0,
        "mutant_passes": dict.fromkeys(MUTANTS, 0),
        "passes": False,
        "no_answer": reason,
    }


def diagnosis(grades: list[dict[str, Any]]) -> dict[str, Any]:
    """Return each failing statistic's z per population, presence's failures and each mutant's."""
    statistics: dict[str, list[float | None]] = {}
    for i, record in enumerate(grades):
        for key, ok in record["passes"].items():
            if ok is False:
                statistics.setdefault(key, [None] * len(grades))[i] = record["z"][key]
    presence = {
        name: sum(record.get(name) is not None and not record[name]["present"] for record in grades)
        for name in ("local_search_presence", "chemotaxis_presence")
    }
    mutants = {m: sum(not record["mutants"][m]["passes"] for record in grades) for m in MUTANTS}
    return {"statistics": statistics, "presence_failures": presence, "mutant_failures": mutants}


def pass_rates(grades: list[dict[str, Any]]) -> dict[str, Any]:
    """Return the wild type's and each mutant's pass share over every graded population."""
    return {
        "populations": len(grades),
        "wild_type": float(np.mean([bool(g["wild_type_passes"]) for g in grades])),
        **{m: float(np.mean([bool(g["mutants"][m]["passes"]) for g in grades])) for m in MUTANTS},
    }


# ---------------------------------------------------------------------------------------------
# The held-out reading and the restart spread
# ---------------------------------------------------------------------------------------------


def _family_statistics(record: dict[str, Any]) -> dict[str, dict[str, float | None]]:
    out: dict[str, dict[str, float | None]] = {}
    for key, value in record["statistics"].items():
        family, name = key.split("/", 1)
        out.setdefault(family, {})[name] = value
    return out


def _readings(
    a: dict[str, float | None],
    b: dict[str, float | None],
    scales: dict[str, float],
) -> dict[str, Any]:
    from quantumnematode.benchmark import grading as g

    out = {}
    for s, scale in scales.items():
        d = _difference(a.get(s), b.get(s))
        band = g.Band(0.0, scale)
        value = math.nan if d is None else d
        out[s] = g.Reading(value, band, g.z_score(value, band), passes=True)
    return out


def tallies(tracks: Any, rows: np.ndarray, clf: Any) -> dict[str, dict[int, np.ndarray]]:  # noqa: ANN401
    """Return the small lawn's lawn, kinematic and activity tallies by date for these rows."""
    import benchmark_bands as bb
    import benchmark_v2_controls as v2c

    lawn, kin = v2c._tallies_by_date(tracks, rows, clf)
    activity = bb.per_date_activity_tallies(tracks, bb.WILD_TYPE, rows=rows)
    return {"small_lawn": lawn, "kinematics": kin, "activity": activity}


def statistics_of(per_date: dict[str, dict[int, np.ndarray]], dates: Sequence[int]) -> dict:
    """Return the small lawn's, kinematics' and activity's statistics pooled over these dates."""
    from quantumnematode.benchmark import grading as g

    pooled = {f: np.vstack([per_date[f][d] for d in dates]).sum(axis=0) for f in per_date}
    return {
        "small_lawn": g.lawn_statistics(pooled["small_lawn"]),
        "kinematics": g.kinematic_statistics(pooled["kinematics"]),
        "activity": g.activity_statistics(pooled["activity"]),
    }


def held_out_distances(
    per_date: dict[str, dict[int, np.ndarray]],
    simulated: list[dict[str, Any]],
    reference: list[dict[str, Any]],
    bands: dict[str, Any],
    *,
    splits: int | None = None,
) -> dict[str, Any]:
    """Return the real halves', the answer's and the band reference's distances to a held-out half.

    W's rule (``benchmark_v4_controls.w_minimum_v4``): 1,000 splits of the dates into halves (seed
    2034), each scaled by PD1074's band scales at the half's size rounded to 25, on the small lawn,
    kinematics and the graded activity statistics.
    """
    import benchmark_v2_controls as v2c
    from quantumnematode.benchmark import grading as g

    family_map = bg._v2_family_map(bands)
    graded = tuple(bands["activity_graded"])
    dates = sorted(per_date["small_lawn"])
    builder = bg._builder()
    rng = np.random.default_rng(v2c.W_FITTED_SEED)
    sims = [_family_statistics(r) for r in simulated]
    refs = [_family_statistics(r) for r in reference]
    real, answer, band = [], [], []
    for _ in range(v2c.W_SPLITS if splits is None else splits):
        order = rng.permutation(len(dates))
        sides = [
            [dates[k] for k in order[: len(dates) // 2]],
            [dates[k] for k in order[len(dates) // 2 :]],
        ]
        halves = [statistics_of(per_date, side) for side in sides]
        counts = [sum(len(per_date["small_lawn"][d]) for d in side) for side in sides]
        step = v2c.W_SIZE_STEP
        size = max(step, round(min(counts) / step) * step)
        small, kinematic = builder.small_lawn(size), bg._kinematic_bands_at(size)
        activity = bg.family_bands_v2(bands, "activity", size)
        scales = {
            "small_lawn": {s: small[s].scale for s in g.LAWN_STATISTICS},
            "kinematics": {s: kinematic[s].scale for s in g.KINEMATIC_STATISTICS},
            "activity": {s: activity[s].scale for s in graded},
        }

        def distance(a: dict, b: dict, scales: dict = scales) -> float:
            readings = {f: _readings(a[f], b[f], scales[f]) for f in W_FAMILIES}
            return g.distance(readings, list(W_FAMILIES), family_map)

        real.append(distance(halves[0], halves[1]))
        answer.extend(distance(s, h) for s in sims for h in halves)
        band.extend(distance(r, h) for r in refs for h in halves)

    def summary(values: list[float]) -> dict[str, Any]:
        if not values:
            return {"n": 0, "median": None, "p95": None}
        return {
            "n": len(values),
            "median": float(np.median(values)),
            "p95": float(np.percentile(values, v2c.W_QUANTILE)),
        }

    return {
        "dates": len(dates),
        "animals": int(sum(len(v) for v in per_date["small_lawn"].values())),
        "real_halves": summary(real),
        "answer": summary(answer),
        "band_reference": summary(band),
    }


def date_sets(tracks: Any) -> dict[str, np.ndarray]:  # noqa: ANN401 - the Scheer store
    """Return the held-out sets: N2's unread held-out dates, and PD1074's held-out dates."""
    import benchmark_bands as bb
    import benchmark_v2_controls as v2c
    from quantumnematode.benchmark import references as refs

    n2 = tracks.arm(v2c.N2_ARM)
    unread = v2c.n2_unread_dates(tracks)
    pd = tracks.arm(bb.WILD_TYPE)
    return {
        "N2_unread": n2[np.isin(tracks.date[n2], unread) & (tracks.side[n2] == refs.HELD_OUT)],
        "PD1074_held_out": pd[tracks.side[pd] == refs.HELD_OUT],
    }


def band_reference(
    tracks: Any,  # noqa: ANN401 - the Scheer store
    clf: Any,  # noqa: ANN401 - the classifier
    populations: int = REFERENCE_POPULATIONS,
) -> list[dict[str, Any]]:
    """Return populations of 200 animals drawn from PD1074's band cohorts, as records."""
    import benchmark_bands as bb
    from quantumnematode.benchmark import references as refs

    rows = tracks.arm(bb.WILD_TYPE)
    band_rows = rows[tracks.side[rows] == refs.BANDS]
    rng = np.random.default_rng(REFERENCE_SEED)
    out = []
    for _ in range(populations):
        picked = np.sort(rng.choice(band_rows, size=ANIMALS, replace=False))
        per_date = tallies(tracks, picked, clf)
        stats = statistics_of(per_date, sorted(per_date["small_lawn"]))
        out.append(
            {"statistics": {f"{f}/{s}": v for f, vs in stats.items() for s, v in vs.items()}},
        )
    return out


def held_out_reading(
    simulated: list[dict[str, Any]],
    *,
    sets: dict[str, np.ndarray] | None = None,
    splits: int | None = None,
    reference_populations: int = REFERENCE_POPULATIONS,
) -> dict[str, Any]:
    """Read the answer's graded populations on each held-out date set, beside both references."""
    from quantumnematode.benchmark import references as refs

    bands = bg.load_bands_v4()
    clf = bg.classifier(bands)
    tracks = refs.load_scheer_tracks()
    reference = band_reference(tracks, clf, reference_populations)
    out = {}
    for name, rows in (date_sets(tracks) if sets is None else sets).items():
        per_date = tallies(tracks, rows, clf)
        out[name] = held_out_distances(per_date, simulated, reference, bands, splits=splits)
    return out


def restart_spread(
    first: list[dict[str, Any]],
    second: list[dict[str, Any]],
    bands: dict[str, Any],
) -> dict[str, Any]:
    """Return the median capped D between two answers' populations g, on W's families and all six."""
    scales = v2.band_scales_v2(bands)
    full = {f: tuple(s) for f, s in bands["families"].items()}
    w = {f: full[f] for f in W_FAMILIES}
    pairs = list(zip(first, second, strict=False))
    return {
        "populations": len(pairs),
        "w_families": float(
            np.median([v2.predictive_distance_v2(a, b, scales, w, capped=True) for a, b in pairs]),
        ),
        "all_six": float(
            np.median(
                [v2.predictive_distance_v2(a, b, scales, full, capped=True) for a, b in pairs],
            ),
        ),
    }


# ---------------------------------------------------------------------------------------------
# F.0b-ii: the recovery distance, the preflight and the gate
# ---------------------------------------------------------------------------------------------


def recovery_distance(
    truth: dict[str, Any],
    record: dict[str, Any],
    bands: dict[str, Any],
) -> float:
    """Return the registered capped D on the six families, the mutants in at half on their capped z."""
    scales, mscales = v2.band_scales_v2(bands), mutant_scales(bands)
    family_map = {f: tuple(s) for f, s in bands["families"].items()}
    wild = v2.predictive_distance_v2(truth, record, scales, family_map, capped=True)
    terms = [
        math.sqrt(
            float(np.mean([mutant_z(truth, record, mscales[m][s], m, s) ** 2 for s in GRADED[m]])),
        )
        for m in MUTANTS
    ]
    n_f, n_m = len(family_map), len(MUTANTS)
    w = n_f / n_m
    return math.sqrt((n_f * wild**2 + w * sum(t**2 for t in terms)) / (n_f + w * n_m))


def gain(prior: float, d: float, floor: float) -> float:
    """Return the gain fraction, (prior D - D) / (prior D - floor)."""
    return rc.gain_fraction(prior, d, floor)


def definedness_flips(observation: dict[str, Any], repeats: list[dict[str, Any]]) -> list[str]:
    """Return the statistics whose definedness differs between the observation and any repeat."""

    def defined(record: dict[str, Any]) -> set[str]:
        wild = {k for k, v in record["statistics"].items() if v is not None}
        mutant = {
            f"{m}/{s}" for m, d in record["delta"].items() for s, v in d.items() if v is not None
        }
        return wild | mutant

    seen = defined(observation)
    return sorted(set().union(*(seen ^ defined(r) for r in repeats)))


def preflight_record(
    observation: dict[str, Any],
    repeats: list[dict[str, Any]],
    prior_runs: list[dict[str, Any]],
    centre_runs: list[dict[str, Any]],
    bands: dict[str, Any],
) -> dict[str, Any]:
    """Return the recovery's pre-fit checks: ``repeats`` are re-simulations 1-20, in order."""
    distance = lambda r: recovery_distance(observation, r, bands)  # noqa: E731
    floor = float(np.median([distance(r) for r in repeats[:10]]))
    oracle = float(np.median([distance(r) for r in repeats[10:20]]))
    prior = float(np.median([distance(r) for r in prior_runs]))
    centre = float(np.median([distance(r) for r in centre_runs]))
    reference = distance(band_truth(bands))
    flips = definedness_flips(observation, repeats[:10])
    # The procedure keeps only a start that holds the lawn, so the truth it recovers must.
    lawn = observation["statistics"].get(LAWN_STATISTIC) or 0.0
    checks = {
        "separated": prior - floor >= SEPARATION,
        "oracle": gain(prior, oracle, floor) >= ORACLE_GATE,
        "centre": gain(prior, centre, floor) <= START_GATE,
        "stable": not flips,
        "holds_lawn": lawn >= HOLD_BAR,
    }
    return {
        "floor": floor,
        "observation_time_on_lawn": lawn,
        "oracle_D": oracle,
        "prior_D": prior,
        "centre_D": centre,
        "oracle_gain": gain(prior, oracle, floor),
        "centre_gain": gain(prior, centre, floor),
        "wild_type_reference_D": reference,
        "wild_type_reference_gain": gain(prior, reference, floor),
        "gate_line_D": prior - GAIN_GATE * (prior - floor),
        "definedness_flips": flips,
        "checks": checks,
        "passed": all(checks.values()),
        "limit": (
            "The truth sits beside the wild type: a record at the bands' centres gains "
            "wild_type_reference_gain, so the gate cannot tell recovering the truth from fitting "
            "the wild type. It tests reaching the region from the box centre, and the spread."
        ),
    }


def recovery_gate(d: float, preflight: dict[str, Any]) -> dict[str, Any]:
    """Return the gate: the gain at the answer's median D, at least 0.75."""
    value = gain(preflight["prior_D"], d, preflight["floor"])
    return {"D": d, "gain": value, "passes": value >= GAIN_GATE}


# ---------------------------------------------------------------------------------------------
# Running the procedure
# ---------------------------------------------------------------------------------------------


def _write_json(path: Path, record: dict[str, Any]) -> None:
    cr._write_json(path, record)


def _write_pickle(path: Path, state: dict[str, Any]) -> None:
    cr._write_bytes(path, pickle.dumps(state))


def subset_indices(names: list[Any]) -> list[int]:
    """Return stage 1's parameter indices: the wiring, sensory, readout and noise blocks, centres."""
    return [
        i
        for i, e in enumerate(names)
        if e.block in STAGE1_BLOCKS or e.name.startswith("readout.centre.")
    ]


def carried(strategy: Any, x0: list[float], seed: int, popsize: int) -> Any:  # noqa: ANN401
    """Return a new sep-CMA strategy at x0 with a previous strategy's sigma and per-parameter steps."""
    import cma

    sigma = float(strategy.sigma)
    stds = np.asarray(strategy.stds, dtype=float) / sigma
    options = {
        "CMA_diagonal": True,
        "CMA_stds": stds.tolist(),
        "bounds": [0.0, 1.0],
        "popsize": popsize,
        "seed": seed,
        "verbose": -9,
    }
    return cma.CMAEvolutionStrategy(x0, sigma, options)


def fresh(x0: list[float], sigma0: float, seed: int, popsize: int) -> Any:  # noqa: ANN401
    """Return a new sep-CMA strategy at x0, isotropic at sigma0."""
    import cma

    options = {
        "CMA_diagonal": True,
        "bounds": [0.0, 1.0],
        "popsize": popsize,
        "seed": seed,
        "verbose": -9,
    }
    return cma.CMAEvolutionStrategy(x0, sigma0, options)


def settings_of(run: Run, truth: dict[str, Any]) -> dict[str, Any]:
    """Return what a run's saved state must match on resume."""
    return {
        "run": run.name,
        "layout": dataclasses.asdict(run.layout),
        "cma_seeds": list(run.cma_seeds),
        "restart_seed": run.restart_seed,
        "mutants": list(run.mutants),
        "generations": list(run.generations),
        "hold_readouts": list(run.hold_readouts),
        "hold_bars": list(run.hold_bars),
        "stages": [dataclasses.asdict(s) for s in STAGES],
        "arm": ARM,
        "settings": {k: repr(v) for k, v in SETTINGS.items()},
        "truth": hashlib.sha256(json.dumps(truth, sort_keys=True).encode()).hexdigest(),
    }


def _readout_job(item: tuple[str, list[float], list[int]]) -> tuple[str, dict[str, Any]]:
    """Simulate the readout's wild type (every assay, guards) or one mutant's small lawn."""
    kind, z, seeds = item
    bands = bg.load_bands_v4()
    if kind == "wild":
        return kind, simulate_point(z, seeds, assays=None, mutants=())
    clf = bg.classifier(bands)
    _, factory = controller(z, kind)
    populations, _ = bg.simulate_v2(ARM, factory, seeds, bands, assays=("small_lawn",))
    return kind, bg.family_values("small_lawn", populations["small_lawn"], clf)


def readout(pool: Any, z: list[float], seeds: list[int], truth: dict[str, Any]) -> dict:  # noqa: ANN401
    """Grade a stage's mean on the readout animals: every statistic, the guards, every mutant."""
    bands = bg.load_bands_v4()
    parts = dict(pool.map(_readout_job, [(k, z, seeds) for k in ("wild", *MUTANTS)]))
    wild = parts["wild"]
    # The wild type's small-lawn values on the readout animals, which each mutant is read against.
    wild_small = {s: wild["statistics"].get(f"small_lawn/{s}") for s in LAWN_DIFFERENCES}
    wild["delta"] = {
        m: {s: _difference(parts[m].get(s), wild_small.get(s)) for s in LAWN_DIFFERENCES}
        for m in MUTANTS
    }
    record = grade_record(wild, bands)
    record["D_truth"] = recovery_distance(truth, record, bands)
    return nr._finite(record)


def _stored(path: Path) -> dict[tuple[int, int], dict[str, Any]]:
    if not path.exists():
        return {}
    v2._drop_torn_line(path)
    out = {}
    for line in path.read_text().splitlines():
        record = json.loads(line)
        out[record["generation"], record["member"]] = record
    return out


def _generation(  # noqa: PLR0913 - the pool, the run, the stage, its generation, store and start
    pool: Any,  # noqa: ANN401 - the pool
    run: Run,
    stage: int,
    generation: int,
    candidates: list[list[float]],
    path: Path,
    first_half_start: int = 0,
) -> list[dict[str, Any]]:
    """Simulate a generation's candidates, reusing any stored for the same vectors."""
    stored = _stored(path)
    out: dict[int, dict[str, Any]] = {}
    todo = []
    for m, z in enumerate(candidates):
        record = stored.get((generation, m))
        if record is None:
            todo.append((run.name, stage, generation, m, z, first_half_start))
        elif record["vector"] != vector_digest(z):
            msg = f"{path}: generation {generation} member {m} is stored at another vector"
            raise ValueError(msg)
        else:
            out[m] = record
    if todo:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a") as handle:
            for record in pool.map(_candidate_job, todo):
                handle.write(json.dumps(nr._finite(record), allow_nan=False) + "\n")
                handle.flush()
                out[record["member"]] = record
    return [out[m] for m in range(len(candidates))]


def _kill(stage: Stage, readouts: dict[int, dict[str, Any]]) -> str | None:
    """Return the pilot's kill reason after stage 1 or 2a, from the mean of the last three readouts."""
    last = [readouts[g] for g in sorted(readouts)[-KILL_READOUTS:]]
    if stage.name == "1":
        spread = float(
            np.mean([min(r["spread"]["speed"], r["spread"]["reversal"]) for r in last]),
        )
        lawn = float(np.mean([r["time_on_lawn"] or 0.0 for r in last]))
        if spread < SPREAD_FLOOR or lawn < HOLDS_LAWN:
            return f"stage 1: mean spread {spread:.2f} (floor 0.5), on the lawn {lawn:.2f} (0.75)"
    if stage.name == "2a":
        d_fit = float(np.mean([r["D_fit"] for r in last]))
        if d_fit > D_FIT_KILL:
            return f"2a: mean D_fit {d_fit:.2f} above 2.5"
    return None


def checkpoints(stage: Stage, readouts: dict[int, dict[str, Any]]) -> dict[str, Any]:
    """Return a stage's checkpoint reading, which only the pilot stops on."""
    return {"stage": stage.name, "kill": _kill(stage, readouts)}


def lawn_holding(
    readouts: dict[int, dict[str, Any]],
    generations: Sequence[int],
    bar: float,
) -> dict[str, Any]:
    """Return a start's lawn-holding reading: its mean time on the lawn over 2a's given readouts."""
    lawn = float(np.mean([readouts[g]["time_on_lawn"] or 0.0 for g in generations]))
    return {
        "generations": list(generations),
        "time_on_lawn": lawn,
        "bar": bar,
        # Readouts all at the bar hold, whatever the rounding of their mean.
        "holds": lawn >= bar - 1e-12,
    }


def _start_record(state: dict[str, Any], run: Run) -> dict[str, Any]:
    return {
        "start": state["start"],
        "cma_seeds": [run.cma_seed(s, state["start"]) for s in range(len(FIRST_HALF))],
    }


def _start_json(start: dict[str, Any]) -> dict[str, Any]:
    """Return a start's record with its readouts keyed as JSON keys them."""
    out = dict(start)
    if "readouts" in out:
        out["readouts"] = {
            name: {str(g): r for g, r in stage.items()} for name, stage in out["readouts"].items()
        }
    return out


def _drop(
    state: dict[str, Any],
    run: Run,
    holding: dict[str, Any],
    mean: list[float],
) -> None:
    """Set a start aside with everything it produced, and return the run to the box's centre.

    ``holding`` is the lawn-holding reading that drops it, and ``mean`` the search's mean there.
    """
    lawn, bar = holding["time_on_lawn"], holding["bar"]
    reason = f"2a: on the lawn {lawn:.3f} at generations {holding['generations']} (bar {bar})"
    record: dict[str, Any] = {
        **_start_record(state, run),
        "kept": False,
        "dropped": reason,
        "holding": holding,
        "mean_at_drop": mean,
    }
    for key in ("answers", "histories", "readouts", "checkpoints", "seconds"):
        record[key] = {n: state[key].pop(n) for n in FIRST_HALF if n in state[key]}
    state["starts"].append(record)
    state["start"] += 1
    state["stage"], state["es"] = 0, None
    state["x"] = [0.5] * len(state["x"])
    state.pop("previous", None)
    if state["start"] >= run.starts:
        state["killed"] = f"no start of {run.starts} held the lawn"


def noise_step(pool: Any, run: Run, z: list[float], path: Path) -> list[dict[str, Any]]:  # noqa: ANN401
    """Simulate 2b's answer on the run's noise populations, reusing any stored for that vector."""
    stored = {}
    if path.exists():
        for line in path.read_text().splitlines():
            record = json.loads(line)
            stored[record["population"]] = record
    todo = [
        p
        for p in range(run.layout.noise_populations)
        if p not in stored or stored[p]["vector"] != vector_digest(z)
    ]
    if any(p in stored for p in todo):
        msg = f"{path}: a noise population is stored at another vector"
        raise ValueError(msg)
    if todo:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a") as handle:
            jobs = [(z, run.layout.noise_seeds(p), False) for p in todo]
            for p, record in zip(todo, pool.map(_point_job, jobs), strict=True):
                stored[p] = {"population": p, **record}
                handle.write(json.dumps(nr._finite(stored[p]), allow_nan=False) + "\n")
                handle.flush()
    return [stored[p] for p in range(run.layout.noise_populations)]


def procedure(  # noqa: C901, PLR0912, PLR0915 - the stages in order, each saved
    run: Run,
    truth: dict[str, Any],
    pool: Any,  # noqa: ANN401 - the pool
) -> dict[str, Any]:
    """Run the procedure to its answer, resuming from the saved state; return its record."""
    from quantumnematode.connectome import encoder as enc

    state_dir = BUILD / run.name
    state_path = state_dir / "state.pkl"
    expected = settings_of(run, truth)
    bands = bg.load_bands_v4()
    names = entries()
    if state_path.exists():
        state = pickle.loads(state_path.read_bytes())  # noqa: S301 - this run's own state
        if state["settings"] != expected:
            msg = f"{state_path} was saved with other settings; remove it to start again"
            raise RuntimeError(msg)
    else:
        state = {
            "settings": expected,
            "stage": 0,
            "x": [0.5] * len(names),
            "es": None,
            "histories": {},
            "readouts": {},
            "answers": {},
            "checkpoints": {},
            "noise": None,
            "seconds": {},
            "killed": None,
            "start": 0,
            "starts": [],
            "kept": None,
        }
    timescale = names.index(next(e for e in names if e.name == enc.SPEED_NOISE_ENTRY))
    subset = subset_indices(names)
    hold_at = run.hold_readouts[-1]
    if any(g % READOUT_EVERY and g != run.generations[1] for g in run.hold_readouts):
        msg = f"{run.name}: a lawn-holding generation of {run.hold_readouts} has no readout"
        raise ValueError(msg)
    if hold_at > run.generations[1] or hold_at != run.layout.early_generations:
        msg = f"{run.name}: the lawn-holding reading at {hold_at} does not match 2a or its seeds"
        raise ValueError(msg)
    while state["stage"] < len(STAGES) and state["killed"] is None:
        s = state["stage"]
        stage = STAGES[s]
        generations = run.generations[s]
        idx = subset if stage.subset else list(range(len(names)))
        if state["es"] is None:
            x = list(state["x"])
            if stage.name == "2a":
                x[timescale] = names[timescale].unit(TIMESCALE_AT_2A_S)
                state["x"] = x
            x0 = [x[i] for i in idx]
            seed = run.cma_seed(s, state["start"])
            if stage.sigma0 is not None:
                state["es"] = fresh(x0, stage.sigma0, seed, run.layout.popsize_of(s))
            else:
                state["es"] = carried(state["previous"], x0, seed, run.layout.popsize_of(s))
            state["histories"][stage.name] = {
                "medians": [],
                "best": [],
                "sigmas": [],
                "failure": [],
            }
            state["readouts"][stage.name] = {}
            state["seconds"][stage.name] = 0.0
            _write_pickle(state_path, state)
        es = state["es"]
        history = state["histories"][stage.name]
        readouts = state["readouts"][stage.name]
        sd = state["noise"]["sd"] if stage.failure else None
        objective = Objective(
            truth,
            bands,
            families=stage.families,
            mutants=run.mutants if stage.mutants else (),
            sd=sd,
        )

        def full(point: Sequence[float], x: list[float] = state["x"], idx: list[int] = idx) -> list:
            out = list(x)
            for j, i in enumerate(idx):
                out[i] = float(np.clip(point[j], 0.0, 1.0))
            return out

        if 0 not in readouts:
            readouts[0] = readout(
                pool,
                full(es.result.xfavorite),
                run.layout.readout_seeds(),
                truth,
            )
            _write_pickle(state_path, state)
        first_half = stage.name in FIRST_HALF
        store = (
            f"start_{state['start']}_stage_{stage.name}" if first_half else f"stage_{stage.name}"
        )
        not_holding = None
        while True:
            done = len(history["medians"])
            if stage.name == FIRST_HALF[-1] and state["kept"] is None and done >= hold_at:
                # The start's lawn-holding reading, taken once, at its last readout's generation.
                holding = lawn_holding(readouts, run.hold_readouts, run.hold_bars[state["start"]])
                if not holding["holds"]:
                    not_holding = holding
                    break
                state["kept"] = state["start"]
                state["starts"].append(
                    {
                        **_start_record(state, run),
                        "kept": True,
                        "dropped": None,
                        "holding": holding,
                    },
                )
                _write_pickle(state_path, state)
            if done >= generations:
                break
            g = done + 1
            began = time.time()
            points = es.ask()
            candidates = [full(p) for p in points]
            records = _generation(
                pool,
                run,
                s,
                g,
                candidates,
                state_dir / f"{store}.jsonl",
                state["start"] if first_half else 0,
            )
            scores = [objective(r) for r in records]
            es.tell(points, scores)
            best = int(np.argmin(scores))
            history["medians"].append(float(np.median(scores)))
            history["best"].append(float(scores[best]))
            history["sigmas"].append(float(es.sigma))
            history["failure"].append(objective.failure(records[best]))
            if g % READOUT_EVERY == 0 or g == generations:
                readouts[g] = readout(
                    pool,
                    full(es.result.xfavorite),
                    run.layout.readout_seeds(),
                    truth,
                )
            state["seconds"][stage.name] += time.time() - began
            print(
                f"{run.name} start {state['start']} {stage.name} gen {g}: median {history['medians'][-1]:.2f} "
                f"best {history['best'][-1]:.2f} sigma {es.sigma:.4f}",
                flush=True,
            )
            _write_pickle(state_path, state)
        answer = full(es.result.xfavorite)
        if not_holding is not None:
            _drop(state, run, not_holding, answer)
            _write_pickle(state_path, state)
            continue
        state["answers"][stage.name] = answer
        state["checkpoints"][stage.name] = checkpoints(stage, readouts)
        state["x"], state["previous"], state["es"] = answer, es, None
        if run.kill_rules and state["checkpoints"][stage.name]["kill"]:
            state["killed"] = state["checkpoints"][stage.name]["kill"]
        if stage.name == NOISE_AFTER and state["killed"] is None:
            populations = noise_step(pool, run, answer, state_dir / "noise.jsonl")
            state["noise"] = {
                "sd": noise_sd(populations, truth, bands, run.mutants),
                "populations": len(populations),
            }
        state["stage"] = s + 1
        _write_pickle(state_path, state)
    final = state["histories"].get("final", {}).get("medians", [])
    return {
        "run": run.name,
        "head": head(),
        "manifest": manifest_digest(),
        "killed": state["killed"],
        "start": state["start"],
        "kept": state["kept"],
        "starts": [_start_json(a) for a in state["starts"]],
        "answer": state["answers"].get("final"),
        "answers": state["answers"],
        "histories": state["histories"],
        "readouts": {k: {str(g): r for g, r in v.items()} for k, v in state["readouts"].items()},
        "checkpoints": state["checkpoints"],
        "noise": state["noise"],
        "plateau_generation": cr.plateau_generation(final) if final else None,
        "seconds": state["seconds"],
    }


def grade(pool: Any, run: Run, z: list[float], populations: int) -> list[dict[str, Any]]:  # noqa: ANN401
    """Grade an answer on the run's first ``populations`` grade populations, with presence."""
    bands = bg.load_bands_v4()
    jobs = [(z, run.layout.grade_seeds(g), True) for g in range(populations)]
    return [
        nr._finite({"population": g, **grade_record(r, bands)})
        for g, r in enumerate(pool.map(_point_job, jobs))
    ]


# ---------------------------------------------------------------------------------------------
# Slow speed noise's positive control
# ---------------------------------------------------------------------------------------------


def answer_131() -> list[float]:
    """Return Logbook 131's last answer, in the procedure encoder's order."""
    return list(json.loads(ANSWER_131_PATH.read_text())["z"])


def _without_noise(z: Sequence[float]) -> list[float]:
    """Return a vector without its speed-noise timescale, the pool centres kept in place."""
    from quantumnematode.connectome import encoder as enc

    names = [e.name for e in entries()]
    out = list(z)
    del out[names.index(enc.SPEED_NOISE_ENTRY)]
    return out


def _speed_noise_point(tau_s: float | None) -> dict[str, Any]:
    """Grade the answer at a timescale, or without the OU noise, on the control's animals."""
    from quantumnematode.connectome import encoder as enc
    from quantumnematode.evaluation.controllers import BatchedConnectome

    z = answer_131()
    seeds = list(range(SPEED_NOISE_BASE, SPEED_NOISE_BASE + ANIMALS))
    bands = bg.load_bands_v4()
    if tau_s is None:
        built = brain((), encoder_speed_noise="none")
        enc.write_encoder(built, np.asarray(_without_noise(z)))
        factory = lambda _path: BatchedConnectome(built)  # noqa: E731
    else:
        names = entries()
        timescale = names.index(next(e for e in names if e.name == enc.SPEED_NOISE_ENTRY))
        z[timescale] = names[timescale].unit(tau_s)
        _, factory = controller(z)
    populations, chemotaxis = bg.simulate_v2(ARM, factory, seeds, bands)
    graded = bg.grade_populations_v4(populations, chemotaxis, bands, presence=False)
    return nr._finite(
        {
            "tau_s": tau_s,
            "kinematics": {k: v["value"] for k, v in graded["families"]["kinematics"].items()},
        },
    )


def speed_median_before_cap(z: Sequence[float]) -> dict[str, float]:
    """Return the speed output's median over one reference episode, and before the soft cap.

    The soft cap, 2·tanh(x/2), bounds the output inside ±2, so the control's responsiveness bound of
    2 reads the value before it, 2·atanh(m/2).
    """
    import torch
    from quantumnematode.connectome import encoder as enc
    from quantumnematode.evaluation.reference import run_reference
    from quantumnematode.utils.config_loader import load_simulation_config

    built, _ = controller(z)
    path = FORAGING / f"{bg.ARMS[ARM]}_assay_scheer_lawn.yml"
    config = load_simulation_config(str(path))
    seed = SPEED_NOISE_BASE + ANIMALS
    episode = run_reference(config, built, seed, probe.SPEED_NOISE_REFERENCE_STEPS)
    states = torch.from_numpy(np.asarray(episode.features, dtype=np.float32))
    topology = built.topology
    with torch.no_grad():
        mean, _ = topology.forward_with_hidden_batched(
            states[:, : topology.n_food_features],
            modulators=built._unpack_modulators(states),
        )
    capped = mean[:, 0].double().numpy()
    limit = enc.READOUT_CAP * (1.0 - 1e-12)
    before = enc.READOUT_CAP * np.arctanh(np.clip(capped, -limit, limit) / enc.READOUT_CAP)
    return {"capped": float(np.median(capped)), "before_cap": float(np.median(before))}


def _speed_noise_control(workers: int) -> int:
    if SPEED_NOISE_PATH.exists():
        print(f"{SPEED_NOISE_PATH} exists: the control is read once", file=sys.stderr)
        return 1
    points: list[float | None] = [None, *probe.SPEED_NOISE_TAUS_S]
    with ProcessPoolExecutor(
        max_workers=min(workers, len(points)),
        initializer=nr._worker_setup,
    ) as pool:
        graded = list(pool.map(_speed_noise_point, points))
    bands = bg.load_bands_v4()
    scales = {k: v["scale"] for k, v in bands["bands"][str(ANIMALS)]["kinematics"].items()}
    median = speed_median_before_cap(answer_131())
    reading = probe.read_speed_noise_control(
        [(row["tau_s"], row["kinematics"]) for row in graded],
        scales,
        median["before_cap"],
    )
    record = {
        "description": (
            "Slow speed noise's positive control, under its registered rule "
            "(c2_encoder_probe.read_speed_noise_control), on Logbook 131's answer under the "
            "procedure settings and the speed-following arm: the answer at each timescale and "
            "without the OU noise, on the same 200 animals. The responsiveness bound reads the "
            "speed output before the soft cap."
        ),
        "head": head(),
        "seeds": f"{SPEED_NOISE_BASE:,} + i, i < {ANIMALS}; reference {SPEED_NOISE_BASE + ANIMALS:,}",
        "answer": str(ANSWER_131_PATH.relative_to(_REPO)),
        "speed_median": median,
        **reading,
        "points": graded,
    }
    _write_json(SPEED_NOISE_PATH, record)
    keys = ("speed_responsive", "best_tau_s", "passes_rise", "passes_nested", "passes")
    print(json.dumps(nr._finite({k: record[k] for k in keys}), indent=2))
    return 0 if record["passes"] else 1


# ---------------------------------------------------------------------------------------------
# The guards: the launch record's parts, the manifest and the stages
# ---------------------------------------------------------------------------------------------

MANIFEST_FILES: tuple[str, ...] = (
    "scripts/analysis/c2_fit.py",
    "scripts/analysis/c2_encoder_probe.py",
    "scripts/analysis/benchmark_grade.py",
    "scripts/analysis/benchmark_bands.py",
    "scripts/analysis/benchmark_v2_controls.py",
    "scripts/analysis/benchmark_v3_controls.py",
    "scripts/analysis/benchmark_v4_controls.py",
    "scripts/analysis/body_kinematics_eval.py",
    "scripts/analysis/cma_recovery.py",
    "scripts/analysis/cma_recovery_v2.py",
    "scripts/analysis/npe_recovery.py",
    "scripts/analysis/npe_region_check.py",
    *(
        f"configs/scenarios/foraging/{bg.ARMS[ARM]}_assay_{p}.yml"
        for p in ("scheer_lawn", "uniform_lawn_scheer", "food_removal_lopezcruz", "chemotaxis_chen")
    ),
    "data/benchmark/bands_v4.json",
    "data/locomotion/primitives.json",
    "data/neuromodulators/layer.json",
    "data/neuromodulators/receptor_signs.json",
    "data/neuromodulators/receptor_couplings.json",
    "docs/experiments/logbooks/supporting/113-c1-fit/encodings.md",
)
PACKAGE = "packages/quantum-nematode/quantumnematode"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def manifest(root: Path = _REPO) -> dict[str, Any]:
    """Return the digests of every file a run reads: the scripts, configs, data and the package."""
    package = hashlib.sha256()
    for path in sorted((root / PACKAGE).rglob("*.py")):
        package.update(str(path.relative_to(root)).encode())
        package.update(_sha256(path).encode())
    return {
        "files": {rel: _sha256(root / rel) for rel in MANIFEST_FILES},
        "package": package.hexdigest(),
    }


def manifest_digest() -> str:
    """Return the current manifest's own digest, which every record states."""
    return hashlib.sha256(json.dumps(manifest(), sort_keys=True).encode()).hexdigest()


def manifest_matches(path: Path = MANIFEST_PATH, root: Path = _REPO) -> bool:
    """Return whether the registered manifest exists and every digest in it holds now."""
    if not path.exists():
        return False
    return json.loads(path.read_text()) == manifest(root)


def head() -> str:
    """Return the commit the working tree is at."""
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],  # noqa: S607
        cwd=_REPO,
        capture_output=True,
        check=False,
    )
    return result.stdout.decode().strip()


def committed(path: Path) -> bool:
    """Return whether a record is tracked and unchanged against HEAD."""
    rel = str(path.relative_to(_REPO))
    tracked = subprocess.run(  # noqa: S603 - fixed argv
        ["git", "ls-files", "--error-unmatch", rel],  # noqa: S607
        cwd=_REPO,
        capture_output=True,
        check=False,
    )
    unchanged = subprocess.run(  # noqa: S603 - fixed argv
        ["git", "diff", "--quiet", "HEAD", "--", rel],  # noqa: S607
        cwd=_REPO,
        capture_output=True,
        check=False,
    )
    return tracked.returncode == 0 and unchanged.returncode == 0


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text()) if path.exists() else {}


def _part(name: str) -> bool:
    return LAUNCH_PATH.exists() and f"\n## Part {name}" in LAUNCH_PATH.read_text()


PART_A_BOUND: tuple[Path, ...] = (ENCODINGS_PATH, MANIFEST_PATH, SPEED_NOISE_PATH)


def guard(stage: str) -> str | None:  # noqa: C901, PLR0911, PLR0912 - one refusal per condition
    """Return why a stage may not run, or None when it may."""
    if not manifest_matches():
        return "the manifest is missing, or a file it binds has changed"
    if not bg.v4_is_valid():
        return "benchmark v4's bands are not the validated ones"
    if not cr.launch_committed(PART_A_BOUND, launch_path=LAUNCH_PATH):
        return "Part A, the manifest or the speed-noise control is uncommitted or changed"
    if not _json(SPEED_NOISE_PATH).get("passes"):
        return "slow speed noise's positive control has not passed"
    if stage == "pilot":
        return None
    part_b = (*PART_A_BOUND, PILOT_PATH)
    if not (_part("B") and cr.launch_committed(part_b, launch_path=LAUNCH_PATH)):
        return "Part B, with the pilot's record, is uncommitted or changed"
    if not _json(PILOT_PATH).get("go"):
        return "the pilot's go rule failed"
    if stage == "recovery-preflight":
        return None
    if stage == "recovery":
        if not committed(RECOVERY_PREFLIGHT_PATH):
            return "the recovery's preflight record is uncommitted or changed"
        if not _json(RECOVERY_PREFLIGHT_PATH).get("passed"):
            return "the recovery's preflight has not passed"
        return None
    part_c = (*part_b, RECOVERY_PREFLIGHT_PATH, RECOVERY_PATH)
    if not (_part("C") and cr.launch_committed(part_c, launch_path=LAUNCH_PATH)):
        return "Part C, with the recovery's records, is uncommitted or changed"
    if not _json(RECOVERY_PATH).get("gate", {}).get("passes"):
        return "F.0b-ii's gate has not passed"
    return None


# ---------------------------------------------------------------------------------------------
# The stages
# ---------------------------------------------------------------------------------------------


def _pool(workers: int) -> ProcessPoolExecutor:
    return ProcessPoolExecutor(max_workers=workers, initializer=nr._worker_setup)


def _pilot(workers: int) -> int:
    run = RUNS["pilot"]
    bands = bg.load_bands_v4()
    with _pool(workers) as pool:
        record = procedure(run, band_truth(bands), pool)
        grades = (
            []
            if record["killed"]
            else grade(pool, run, record["answer"], run.layout.grade_populations)
        )
    reading = verdict(grades) if grades else None
    go = bool(reading and reading["passes"])
    out = {
        "description": "C.2's readiness pilot: the registered procedure on discarded seeds.",
        **record,
        "verdict": reading,
        "go": go,
        "pass_rates": pass_rates(grades) if grades else None,
        "grades": grades,
    }
    _write_json(PILOT_PATH, out)
    print(json.dumps(nr._finite({"killed": record["killed"], "verdict": reading, "go": go})))
    return 0 if go else 1


def _recovery_preflight(workers: int) -> int:
    if RECOVERY_PREFLIGHT_PATH.exists():
        print(f"{RECOVERY_PREFLIGHT_PATH} exists: the preflight is read once", file=sys.stderr)
        return 1
    truth = _json(PILOT_PATH)["answer"]
    n = len(truth)
    prior = np.random.default_rng(PRIOR_SEED).random((PRIOR_DRAWS, n))

    def seeds(base: int, q: int) -> list[int]:
        return list(range(base + ANIMALS * q, base + ANIMALS * (q + 1)))

    jobs = [(truth, seeds(TRUTH_BASE, q), False) for q in range(TRUTH_REPEATS + 1)]
    jobs += [(list(map(float, prior[j])), seeds(PRIOR_BASE, j), False) for j in range(PRIOR_DRAWS)]
    jobs += [([0.5] * n, seeds(CENTRE_BASE, q), False) for q in range(CENTRE_REPEATS)]
    with _pool(workers) as pool:
        runs = list(pool.map(_point_job, jobs))
    truth_runs = runs[: TRUTH_REPEATS + 1]
    prior_runs = runs[TRUTH_REPEATS + 1 : TRUTH_REPEATS + 1 + PRIOR_DRAWS]
    centre_runs = runs[TRUTH_REPEATS + 1 + PRIOR_DRAWS :]
    bands = bg.load_bands_v4()
    record = {
        "description": "F.0b-ii's preflight: the truth's observation and its checks.",
        "head": head(),
        "manifest": manifest_digest(),
        "truth": truth,
        "observation": truth_runs[0],
        **preflight_record(truth_runs[0], truth_runs[1:], prior_runs, centre_runs, bands),
    }
    _write_json(RECOVERY_PREFLIGHT_PATH, record)
    (BUILD / "recovery_preflight_runs.json").write_text(
        json.dumps(nr._finite({"truth": truth_runs, "prior": prior_runs, "centre": centre_runs}))
        + "\n",
    )
    keys = ("floor", "oracle_gain", "prior_D", "centre_gain", "wild_type_reference_gain", "passed")
    print(json.dumps(nr._finite({k: record[k] for k in keys}), indent=2))
    return 0 if record["passed"] else 1


RECOVERY_DESCRIPTION = "F.0b-ii: the registered procedure recovering the pilot's answer."


def _recovery(workers: int) -> int:
    preflight = _json(RECOVERY_PREFLIGHT_PATH)
    observation = preflight["observation"]
    run = RUNS["recovery"]
    bands = bg.load_bands_v4()
    truth = {"statistics": observation["statistics"], "delta": observation["delta"]}
    with _pool(workers) as pool:
        record = procedure(run, truth, pool)
        if record["answer"] is None:
            gate = {"D": None, "gain": None, "passes": False, "no_answer": record["killed"]}
            _write_json(
                RECOVERY_PATH,
                {"description": RECOVERY_DESCRIPTION, **record, "gate": gate},
            )
            print(json.dumps(gate))
            return 1
        answers = grade(pool, run, record["answer"], RECOVERY_RESIMULATIONS)
    d = float(np.median([recovery_distance(truth, a, bands) for a in answers]))
    block = {}
    for e, value, target in zip(entries(), record["answer"], preflight["truth"], strict=True):
        block.setdefault(e.block, []).append(abs(value - target))
    out = {
        "description": RECOVERY_DESCRIPTION,
        **record,
        "gate": recovery_gate(d, preflight),
        "wild_type_reference_gain": preflight["wild_type_reference_gain"],
        "distance_from_truth": {b: float(np.mean(v)) for b, v in block.items()},
        "answer_grade_against_bands": verdict(answers),
        "resimulations": answers,
    }
    _write_json(RECOVERY_PATH, out)
    print(json.dumps(nr._finite(out["gate"])))
    return 0 if out["gate"]["passes"] else 1


def _run(workers: int) -> int:
    bands = bg.load_bands_v4()
    with _pool(workers) as pool:
        record = procedure(RUNS["fit"], band_truth(bands), pool)
    _write_json(BUILD / "fit_procedure.json", record)
    return 0


def _grade(workers: int) -> int:
    record = _json(BUILD / "fit_procedure.json")
    if record.get("killed"):
        print(f"nothing to grade, the fit has no answer: {record['killed']}", file=sys.stderr)
        return 0
    if record.get("answer") is None:
        print("the registered fit has not reached its answer", file=sys.stderr)
        return 1
    with _pool(workers) as pool:
        grades = grade(pool, RUNS["fit"], record["answer"], RUNS["fit"].layout.grade_populations)
    _write_json(BUILD / "fit_grades.json", {"head": head(), "grades": grades})
    return 0


def _read() -> int:
    record = _json(BUILD / "fit_procedure.json")
    if record.get("killed"):
        # No start held the lawn: the fit has no answer, and that is its verdict.
        out = {"description": DESCRIPTION, **record, "verdict": no_answer(record["killed"])}
        _write_json(FIT_PATH, out)
        print(json.dumps(nr._finite({"verdict": out["verdict"]})))
        return 0
    grades = _json(BUILD / "fit_grades.json")["grades"]
    pilot = _json(PILOT_PATH)
    bands = bg.load_bands_v4()
    out = {
        "description": DESCRIPTION,
        **record,
        "verdict": verdict(grades),
        "in_sample": "every graded statistic and both resolved mutants are fitted",
        "pass_rates": pass_rates(grades),
        "held_out": held_out_reading(grades),
        "restart_spread": restart_spread(pilot["grades"], grades, bands),
        "w_minimum": _json(W_MINIMUM_PATH).get("w_minimum", {}),
        "reversal_share_reference": ROBERTS_REVERSE_SHARE,
        "grades": grades,
    }
    _write_json(FIT_PATH, out)
    print(json.dumps(nr._finite({"verdict": out["verdict"], "pass_rates": out["pass_rates"]})))
    return 0


LOO_DESCRIPTION = "C.2's tax-4 leave-one-out, as description: tax-4 never scored."


def _loo(workers: int) -> int:
    run = RUNS["loo"]
    bands = bg.load_bands_v4()
    with _pool(workers) as pool:
        record = procedure(run, band_truth(bands), pool)
        if record["answer"] is None:
            _write_json(
                LOO_PATH,
                {"description": LOO_DESCRIPTION, **record, "tax4_predicted": None},
            )
            print(json.dumps({"tax4_predicted": None, "no_answer": record["killed"]}))
            return 0
        grades = grade(pool, run, record["answer"], run.layout.grade_populations)
    fit = _json(FIT_PATH)
    out = {
        "description": LOO_DESCRIPTION,
        **record,
        "tax4_predicted": pass_rates(grades)["tax-4"],
        "tax4_fitted": fit.get("pass_rates", {}).get("tax-4"),
        "verdict": verdict(grades),
        "pass_rates": pass_rates(grades),
        "grades": grades,
    }
    _write_json(LOO_PATH, out)
    print(json.dumps(nr._finite({k: out[k] for k in ("tax4_predicted", "tax4_fitted")})))
    return 0


def _smoke(workers: int) -> int:
    """Every stage, a dropped start and a kept one, the noise step, a grade, the preflight, the reading."""
    import shutil

    from quantumnematode.benchmark import references as refs

    run = RUNS["smoke"]
    shutil.rmtree(BUILD / run.name, ignore_errors=True)
    bands = bg.load_bands_v4()
    truth = band_truth(bands)
    with _pool(workers) as pool:
        record = procedure(run, truth, pool)
        if record["answer"] is None:
            print(f"the smoke reached no answer: {record['killed']}", file=sys.stderr)
            return 1
        grades = grade(pool, run, record["answer"], run.layout.grade_populations)
    pre = preflight_record(grades[0], [grades[1]] * 20, [truth], [grades[1]], bands)
    tracks = refs.load_scheer_tracks()
    import benchmark_bands as bb

    rows = tracks.arm(bb.WILD_TYPE)
    band_rows = rows[tracks.side[rows] == refs.BANDS]
    reading = held_out_reading(
        grades,
        sets={"band_cohorts": band_rows},
        splits=3,
        reference_populations=2,
    )
    summary = {
        "starts": [
            {k: a[k] for k in ("start", "kept", "dropped", "holding")} for a in record["starts"]
        ],
        "medians": {k: v["medians"] for k, v in record["histories"].items()},
        "noise_sd_keys": len(record["noise"]["sd"]),
        "verdict": verdict(grades),
        "preflight_floor": pre["floor"],
        "held_out": reading,
    }
    print(json.dumps(nr._finite(summary), indent=2))
    ran = {s.name: n for s, n in zip(STAGES, run.generations, strict=True)}
    ok = (
        all(len(v["medians"]) == ran[k] for k, v in record["histories"].items())
        and len(grades) == 2
    )
    # The first start is dropped at its lawn-holding reading with its two stages, the second kept.
    dropped, kept = record["starts"]
    ok = ok and record["kept"] == 1 and kept["kept"] and not dropped["kept"]
    reading = run.hold_readouts[-1]
    lengths = [len(dropped["histories"][n]["medians"]) for n in FIRST_HALF]
    ok = ok and lengths == [ran[FIRST_HALF[0]], reading]
    return 0 if ok else 1


def _manifest() -> int:
    _write_json(MANIFEST_PATH, manifest())
    print(f"wrote {MANIFEST_PATH.relative_to(_REPO)}: {manifest_digest()}")
    return 0


def main(argv: list[str] | None = None) -> int:
    """CLI: the control, the manifest, the pilot, F.0b-ii, the fit, its reads and a smoke."""
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    stages = (
        "speed-noise-control",
        "manifest",
        "pilot",
        "recovery-preflight",
        "recovery",
        "run",
        "grade",
        "read",
        "loo",
        "smoke",
    )
    parser.add_argument("stage", choices=stages)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args(argv)
    unguarded: dict[str, Callable[[], int]] = {
        "speed-noise-control": lambda: _speed_noise_control(args.workers),
        "manifest": _manifest,
        "smoke": lambda: _smoke(args.workers),
    }
    if args.stage in unguarded:
        return unguarded[args.stage]()
    reason = guard(args.stage)
    if reason is not None:
        print(f"refused: {reason}", file=sys.stderr)
        return 1
    guarded: dict[str, Callable[[], int]] = {
        "pilot": lambda: _pilot(args.workers),
        "recovery-preflight": lambda: _recovery_preflight(args.workers),
        "recovery": lambda: _recovery(args.workers),
        "run": lambda: _run(args.workers),
        "grade": lambda: _grade(args.workers),
        "read": _read,
        "loo": lambda: _loo(args.workers),
    }
    return guarded[args.stage]()


if __name__ == "__main__":
    sys.exit(main())
