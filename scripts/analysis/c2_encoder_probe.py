#!/usr/bin/env python
r"""C.2's compact encoder checked before F.0b-ii registers: its cost, a slow decay, its readiness.

Description on discarded seeds; nothing here is a registered verdict. Every stage runs the encoded
connectome (``benchmark_grade.ARMS["connectome_c2"]``) at ``per_couple`` tying, one brain per
wiring and mutant in each worker, each candidate's vector written into it.

**cost** times one generation, 16 simulations of 200 animals on 16 workers, on D_fit's two assays
and on all four, twice each so the second shows brain reuse. It profiles one simulation of each in
process, splitting the time between the settling, the action noise, the rest of the evaluator,
record conversion, measurement and grading, and times the settling step at a whole generation's
3,200 animals in one process on CPU and on MPS.

**feasibility** runs a hand-set vector on López-Cruz's food removal: the wiring scales at zero, the
ingestion drive exciting NSM, and one peptide couple from NSM, nlp-54 through TRHR-1 onto 11 of the
12 VA motor neurons, read by the reversal output, its pool's timescale 540 s. The post-removal onset
rate's decay constant, 1/gamma from the benchmark's own fit, is set against 6-12 min. The vector's
magnitudes were set by trial on seeds 54,100,000 and up; the reported run is on 54,000,000.

**pilot** runs one sep-CMA restart (CMA 5985) for 150 generations in the rescaled box, its answer
re-simulated twice every 25 generations on D_fit's assays. In the first box (wiring scales 0-3),
restart 0's answer stalled at a ranking score of about 10.6 from generation 50 to 325, its start
saturating more than half the network; the readiness restarts run only if the pilot beats 10.6.

**pilot-uncompressed** repeats the pilot (CMA 5986) ranking candidates on definedness first and then
the uncapped weighted RMS of z, since the ranking score's compression, 10·asinh(z/10), leaves a
statistic far from its band (time on the lawn at z about -120) almost no slope. It passes when its
best checkpoint by the registered ranking score beats 10.6 with its worm on the lawn more than the
stall's 26% of the time.

**reduced** asks whether the encoder can hold the lawn at all: it fits only the lawn-holding subset
(the wiring scales, the food-contact gains, the speed and reversal readout rows with their biases,
and those two log-SDs; :data:`REDUCED_NAMES`), every other parameter at the box centre, on the
uncompressed score for 150 generations (CMA 5987). The worm holds the lawn when the best
checkpoint's median time on it reaches 0.75. It did not (0.18): forward passes showed contact can
slow the worm, and settings that do held the lawn (0.92 of the assay on it) only with the fixed-sign
modulator strengths at their floor, the box centre's 1.5 swamping the sensory input on food.

**pilot-log** is the third pilot (CMA 5988), with the fixed-sign strengths on a log scale over the
same range, centred at 0.055, on the registered ranking score; it passes as the second pilot does,
and the readiness restarts run only once it has. It did not (10.69), though random settings of the
lawn subset at the new box's centre hold the lawn (0.93).

**pilot-staged** fits in two stages: the lawn subset from the box centre on the uncompressed score
(CMA 5989), which must hold the lawn, then every parameter from its answer at sigma0 0.1 on the
registered ranking score (CMA 5990); it passes as the second pilot does. Stage 1 reached the best
answer of any fit (score 10.0, 0.34 of the assay on the lawn) without holding the lawn at 0.75;
**stage2-anyway** runs stage 2 from it regardless, recorded as a deviation from that rule; it held
the lawn (0.99 of the assay on it, score 4.1 at generation 125). **stage2-extend** resumes it to
500 generations to see where it converges. **sizing** writes the W sizing record from the cost
pilot and the staged pilot.

**readiness** runs up to 3 sep-CMA restarts (CMA 5980-5982) from the box centre on v3's ranking
score against the band centres on D_fit's families, one after another, stopping once 2 are ready.
Each runs through the cap of 1,000 generations: the plateau rule (recorded with every history, with
its guard on every median of the last 40 generations below 100) stopped the first box's restart 0
at generation 48 while its answer had stopped improving far from the bands, so each restart's
answer is instead re-simulated twice at every 50th generation to show when it converges. Each
answer is graded with presence on 5 populations, on all five families and the three mutants'
differences from their wild type; an answer is ready when D_fit's three families pass in at least
3 of 5. If exactly one of the three is ready, a warm restart from it runs (sigma0 0.1, CMA 5983).

**read** writes the readiness record and the W sizing record from the stored histories and grades.

**speed-noise** is slow speed noise's positive control (Logbook 123). It grades the stage-2 answer on
the wiring it was fitted on, with independent noise and with slow noise at timescales 0.25, 2, 8.7,
30, 100 and 300 s, on the same animals. It passes when the roaming speed's 90th percentile rises by
at least 3 band scales above its value at 0.25 s at some timescale, and at 0.25 s every kinematics
statistic is within 2 band scales of independent noise.

Seeds: the restarts at 40,000,000 + 200·(20,000·r + 16·(n - 1) + m) + i; the warm restart at
55,000,000 + 200·(16·(n - 1) + m) + i; the grading at 52,000,000 + 200·g + i, every restart on the same five
populations and its mutants on their wild type's animals; the cost pilot at 53,000,000; the feasibility check at 54,000,000; the
smoke at 59,000,000; the first box's early reads at 61,000,000; the pilot at 62,000,000 and its
trajectory at 63,000,000 + 200·(2g + q) + i; the restarts' trajectories at 64,000,000 +
200·(2,100·r + 2g + q) + i; the second pilot at 66,000,000 and its trajectory at 67,000,000; the reachability probe at
68,000,000 and its trajectory at 69,000,000; the existence tests at 70,000,000 and 70,100,000; the
third pilot at 71,000,000 and its trajectory at 72,000,000; the staged pilot's stage 1 at 73,000,000
(trajectory 74,000,000) and stage 2 at 75,000,000 (trajectory 76,000,000); stage 2's resumed
generations 151-500 at 80,000,000 + 200·(k - 2,400) + i and their checkpoints at 82,000,000; the
stage-2 answer's diagnoses at 88,000,000, 88,200,000 and 88,299,999; the speed-noise control at
88,100,000.

Usage::

    uv run python scripts/analysis/c2_encoder_probe.py cost
    uv run python scripts/analysis/c2_encoder_probe.py feasibility
    uv run python scripts/analysis/c2_encoder_probe.py pilot
    uv run python scripts/analysis/c2_encoder_probe.py pilot-uncompressed
    uv run python scripts/analysis/c2_encoder_probe.py reduced
    uv run python scripts/analysis/c2_encoder_probe.py pilot-log
    uv run python scripts/analysis/c2_encoder_probe.py pilot-staged
    uv run python scripts/analysis/c2_encoder_probe.py stage2-anyway
    uv run python scripts/analysis/c2_encoder_probe.py stage2-extend
    uv run python scripts/analysis/c2_encoder_probe.py speed-noise
    uv run python scripts/analysis/c2_encoder_probe.py sizing
    uv run python scripts/analysis/c2_encoder_probe.py readiness
    uv run python scripts/analysis/c2_encoder_probe.py read
    uv run python scripts/analysis/c2_encoder_probe.py smoke

**reproduce** converts the probe procedure's passing answer (Logbook 129's ``grade_nudged.json``)
into the package's encoding, entry by entry by name (:func:`package_vector`), with the procedure's
options (:data:`PROCEDURE_SETTINGS`). It checks that the package's forward computes what the probe's
patched forward did: every output within 1e-5 on one reference episode's recorded states on
Scheer's small lawn and one on food removal, and every step's primitive the same. The answer's
five-population grade, on the probe's own populations, is reported beside the probe's, not gated.
"""

from __future__ import annotations

import argparse
import copy
import cProfile
import functools
import itertools
import json
import math
import os
import pstats
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import benchmark_grade as bg
import body_kinematics_eval as harness
import cma_recovery as cr
import cma_recovery_v2 as v2
import npe_recovery as nr

if TYPE_CHECKING:
    from collections.abc import Callable

DESCRIPTION = "C.2's encoder before F.0b-ii: its cost, a slow decay's feasibility, its readiness."
_REPO = Path(__file__).resolve().parents[2]
SUPPORT = _REPO / "docs" / "experiments" / "logbooks" / "supporting" / "122-c2-encoder"
BUILD = _REPO / "build" / "c2-encoder"
FORAGING = _REPO / "configs" / "scenarios" / "foraging"
ARM = "connectome_c2"
TYING = "per_couple"
# The encoded brain's behaviour does not depend on its seed: every tensor it reads is written.
BRAIN_SEED = 1
ANIMALS = nr.ANIMALS
FIT_ASSAYS = ("small_lawn", "uniform_lawn")
MUTANTS = ("tph-1", "pdfr-1", "tax-4")
KNOCKOUT_GENES: dict[str, tuple[str, ...]] = {"tph-1": ("tph-1",), "pdfr-1": ("pdfr-1",)}

READINESS_BASE = 40_000_000
RESTART_BLOCK = 20_000
READINESS_CMA = 5980
RESTARTS = 3
PASSES_WANTED = 2
CAP = 1_000
GUARD = 100.0
WARM_BASE = 55_000_000
WARM_CMA = 5983
WARM_SIGMA0 = 0.1
WARM_RESTART = RESTARTS
GRADE_BASE = 52_000_000
GRADE_POPULATIONS = 5
PASS_NEEDED = 3
BOUND_TOLERANCE = 0.01

COST_BASE = 53_000_000
COST_SPREAD = 0.1
COST_GENERATIONS = 2
COMPUTE_ANIMALS = 3_200
COMPUTE_REPEATS = 30

FEASIBILITY_BASE = 54_000_000
FEASIBILITY_TRIAL_BASE = 54_100_000
# The trials that set the hand-set vector's magnitudes: trial t on population t at the trial base.
FEASIBILITY_TRIALS = 20
FEASIBILITY_WINDOW_MIN = (6.0, 12.0)
SMOKE_BASE = 59_000_000
# The measured restart's trajectory: its answer at every 50th generation, re-simulated twice.
# The old box's early reads of its stalled restart, kept with the stall's record.
PEEK_BASE = 61_000_000
# The pilot in the rescaled box, its answer read every 25 generations.
PILOT_BASE = 62_000_000
PILOT_CMA = 5985
PILOT_CAP = 150
PILOT_EVERY = 25
PILOT_TRAJECTORY_BASE = 63_000_000
# The stalled restart's answer score in the old box, which the pilot must beat.
STALL_SCORE = 10.6
# The first box's stalled answer's time on the small lawn (generation 300, its diagnosis).
STALL_TIME_ON_LAWN = 0.261
# The second pilot, ranking its candidates on the uncompressed score, and its trajectory.
UNCOMPRESSED_PILOT_BASE = 66_000_000
UNCOMPRESSED_PILOT_CMA = 5986
UNCOMPRESSED_PILOT_TRAJECTORY_BASE = 67_000_000
# Above any weighted RMS of uncapped z a physically possible record reaches.
UNCOMPRESSED_UNDEFINED_RANK = 1_000_000.0
# The reachability probe: only the lawn-holding subset is fitted, every other parameter held at the
# box centre, on the uncompressed score.
REDUCED_NAMES: tuple[str, ...] = (
    "wiring.excitatory",
    "wiring.inhibitory",
    "wiring.gap",
    "sensory.CEP.contact",
    "sensory.ADE.contact",
    "sensory.PDE.contact",
    *(
        f"readout.{output}.{pool}"
        for output in ("speed", "reversal")
        for pool in ("VB", "DB", "VA", "DA")
    ),
    "readout.speed.bias",
    "readout.reversal.bias",
    "noise.speed",
    "noise.reversal",
)
REDUCED_BASE = 68_000_000
REDUCED_CMA = 5987
REDUCED_TRAJECTORY_BASE = 69_000_000
# The worm holds the lawn when the best checkpoint's median time on it reaches this (real: 0.98).
HOLDS_LAWN = 0.75
# The existence tests of contact-slowing settings, at the centre's and the floor's fixed strengths.
EXISTENCE_BASES = (70_000_000, 70_100_000, 70_200_000)
# The third pilot, in the box with log-scaled fixed-sign strengths, on the registered ranking score.
LOG_PILOT_BASE = 71_000_000
LOG_PILOT_CMA = 5988
LOG_PILOT_TRAJECTORY_BASE = 72_000_000
# The staged pilot: stage 1 fits the lawn subset from the box centre on the uncompressed score and
# must hold the lawn; stage 2 fits all parameters from its answer on the registered ranking score.
STAGE1_BASE = 73_000_000
STAGE1_CMA = 5989
STAGE1_TRAJECTORY_BASE = 74_000_000
STAGE2_BASE = 75_000_000
STAGE2_CMA = 5990
STAGE2_TRAJECTORY_BASE = 76_000_000
STAGE2_SIGMA0 = 0.1
# Stage 2 resumed from its 150th generation to here, to see where the staged fit converges. The
# resumed generations draw their animals from a block of their own, starting at generation 151.
STAGE2_EXTENDED_CAP = 500
STAGE2_EXTENSION_BASE = 80_000_000
STAGE2_EXTENSION_TRAJECTORY_BASE = 82_000_000
# Each readiness restart's answer at every 50th generation, re-simulated twice.
TRAJECTORY_BASE = 64_000_000
TRAJECTORY_EVERY = 50
TRAJECTORY_REPEATS = 2
TRAJECTORY_BLOCK = 2_100
SMOKE_CMA = 5984

READINESS_PATH = SUPPORT / "readiness.json"
COST_PATH = SUPPORT / "cost.json"
FEASIBILITY_PATH = SUPPORT / "feasibility.json"
W_SIZING_PATH = SUPPORT / "w_sizing.json"
PILOT_PATH = SUPPORT / "readiness" / "pilot.json"
UNCOMPRESSED_PILOT_PATH = SUPPORT / "readiness" / "pilot_uncompressed.json"
REDUCED_PATH = SUPPORT / "readiness" / "reduced.json"
LOG_PILOT_PATH = SUPPORT / "readiness" / "pilot_logbox.json"
STAGED_PILOT_PATH = SUPPORT / "readiness" / "pilot_staged.json"
SPEED_NOISE_PATH = (
    _REPO
    / "docs"
    / "experiments"
    / "logbooks"
    / "supporting"
    / "123-c2-speed-noise"
    / "positive_control.json"
)
# The stage-2 answer's diagnoses (Logbook 123's supporting records), each read once: its variants on
# 200 animals, its actions on four episodes, and its runs moved mid-range (one reference episode
# and 200 animals). Declared so no stage reuses them.
DIAGNOSIS_BLOCKS = ((88_000_000, 88_000_200), (88_200_000, 88_200_004), (88_299_999, 88_300_200))
# Slow speed noise's positive control: the stage-2 answer on its own wiring, one population, the
# same animals at every timescale and with independent noise.
SPEED_NOISE_BASE = 88_100_000
SPEED_NOISE_TAUS_S = (0.25, 2.0, math.sqrt(0.25 * 300.0), 30.0, 100.0, 300.0)
# The bar: the roaming speed's 90th percentile rises by this many band scales at some timescale,
# and at the shortest every kinematics statistic is within this many of independent noise.
SPEED_NOISE_RISE_SCALES = 3.0
SPEED_NOISE_NESTED_SCALES = 2.0
# The control is read only on an answer whose speed output responds: its pre-squash median over one
# reference episode, on the animal after the control's own, within this bound of zero.
SPEED_NOISE_RESPONSIVE = 2.0
SPEED_NOISE_REFERENCE_SEED = SPEED_NOISE_BASE + ANIMALS
SPEED_NOISE_REFERENCE_STEPS = 1_800


# ---------------------------------------------------------------------------------------------
# The encoded brains, one per wiring and mutant in each process
# ---------------------------------------------------------------------------------------------

_BRAINS: dict[tuple[Any, ...], Any] = {}
# Brain settings a stage applies to every brain this process builds, part of the cache key.
_BRAIN_SETTINGS: dict[str, Any] = {}


def encoded_brain(genes: tuple[str, ...] = (), **overrides: object) -> Any:  # noqa: ANN401 - a brain
    """Return this process's encoded brain for a set of knockout genes and wiring overrides.

    Built once and reused: a candidate's vector is written into it, and the reused brain acts as a
    fresh build with that vector does.
    """
    merged = {**_BRAIN_SETTINGS, **overrides}
    key = (genes, tuple(sorted((k, repr(v)) for k, v in merged.items())))
    if key not in _BRAINS:
        path = FORAGING / f"{bg.ARMS[ARM]}_assay_scheer_lawn.yml"
        settings = {"modulator_knockout_genes": list(genes), "encoder_modulator_tying": TYING}
        brain, _, _ = harness.build_brain(path, BRAIN_SEED, {**settings, **merged})
        _BRAINS[key] = brain
    return _BRAINS[key]


def controller(z: list[float] | np.ndarray, mutant: str | None = None) -> Callable[[Path], Any]:
    """Write a vector, or a mutant's encoding of it, into its brain; return a controller factory."""
    from quantumnematode.connectome import encoder as enc
    from quantumnematode.evaluation.controllers import BatchedConnectome

    brain = encoded_brain(KNOCKOUT_GENES.get(mutant or "", ()))
    spec = brain.topology.modulator_spec
    enc.write_encoder(brain, enc.tax4_vector(spec, z, TYING) if mutant == "tax-4" else z)
    return lambda _path: BatchedConnectome(brain)


@functools.cache
def layer_spec() -> Any:  # noqa: ANN401 - a layer spec
    """Return the wild type's data layer, which names the encoder's parameters."""
    from quantumnematode.connectome.loader import load_emmons_2024_hermaphrodite
    from quantumnematode.connectome.neuromodulator_layer import load_data_layer

    return load_data_layer(sorted(load_emmons_2024_hermaphrodite().neurons))


@functools.cache
def entries(*, speed_noise: bool = False) -> list[Any]:
    """Return the encoder's parameter table at this probe's tying, with or without slow noise."""
    from quantumnematode.connectome import encoder as enc

    return enc.encoder_entries(layer_spec(), TYING, speed_noise=speed_noise)


def population_seeds(base: int, k: int) -> list[int]:
    """Return population k's animals at a seed base."""
    return [base + ANIMALS * k + i for i in range(ANIMALS)]


# ---------------------------------------------------------------------------------------------
# Simulation, grading and the objective
# ---------------------------------------------------------------------------------------------


def _statistics(graded: dict[str, Any]) -> dict[str, float | None]:
    return {
        f"{family}/{name}": reading["value"]
        for family, readings in graded["families"].items()
        for name, reading in readings.items()
    }


def simulate(k: int, z: list[float], seed_base: int) -> dict[str, Any]:
    """Run simulation k at z on D_fit's two assays; return its graded statistics."""
    bands = bg.load_bands_v3()
    start = time.perf_counter()
    populations, _ = bg.simulate_v2(
        ARM,
        controller(z),
        population_seeds(seed_base, k),
        bands,
        assays=FIT_ASSAYS,
    )
    graded = bg.grade_populations_v3(populations, None, bands, presence=False)
    return nr._finite(
        {
            "k": k,
            "z": list(map(float, z)),
            "statistics": _statistics(graded),
            "D_fit": graded["D_fit"],
            "seconds": time.perf_counter() - start,
        },
    )


def simulate_all(k: int, z: list[float], seed_base: int) -> dict[str, Any]:
    """Run simulation k at z on all four assays; return its graded statistics."""
    bands = bg.load_bands_v3()
    start = time.perf_counter()
    populations, chemotaxis = bg.simulate_v2(
        ARM,
        controller(z),
        population_seeds(seed_base, k),
        bands,
    )
    graded = bg.grade_populations_v3(populations, chemotaxis, bands, presence=False)
    return nr._finite(
        {
            "k": k,
            "z": list(map(float, z)),
            "statistics": _statistics(graded),
            "D": graded["D"],
            "D_fit": graded["D_fit"],
            "seconds": time.perf_counter() - start,
        },
    )


def mutant_reading(
    mutant: str,
    wild: dict[str, float],
    values: dict[str, float],
    bands: dict[str, Any],
    n: int,
) -> dict[str, Any]:
    """Grade a mutant's difference from its wild type on its v3 bands at n, with its direction."""
    from quantumnematode.benchmark import grading as g

    delta = {s: values[s] - wild[s] for s in g.MUTANT_LAWN_STATISTICS}
    mutant_bands = bg.mutant_bands(bands, mutant, n)
    readings = g.grade(delta, mutant_bands, float(bands["bars"][mutant]), directions=True)
    return {
        "statistics": {
            s: {
                "delta": r.value,
                "z": r.z,
                "scale": r.band.scale,
                "direction": mutant_bands[s].direction,
                "passes": r.passes,
            }
            for s, r in readings.items()
        },
        "passes": all(r.passes for r in readings.values()),
    }


def grade_answer(k: int, z: list[float], base: int) -> dict[str, Any]:
    """Grade an answer on population k with presence: all five families and the three mutants."""
    return {"k": k, "base": base, **grade_seeds(z, population_seeds(base, k))}


def grade_seeds(z: list[float] | np.ndarray, seeds: list[int]) -> dict[str, Any]:
    """Grade a vector on these animals with presence: all five families and the three mutants.

    Each mutant runs the small lawn on its wild type's animals, and its difference from that wild
    type is graded on its band.
    """
    bands = bg.load_bands_v3()
    clf = bg.classifier(bands)
    populations, chemotaxis = bg.simulate_v2(ARM, controller(z), seeds, bands)
    graded = bg.grade_populations_v3(populations, chemotaxis, bands, presence=True)
    wild = bg.family_values("small_lawn", populations["small_lawn"], clf)
    mutants = {}
    for mutant in MUTANTS:
        mutated, _ = bg.simulate_v2(
            ARM,
            controller(z, mutant),
            seeds,
            bands,
            assays=("small_lawn",),
        )
        values = bg.family_values("small_lawn", mutated["small_lawn"], clf)
        mutants[mutant] = mutant_reading(mutant, wild, values, bands, len(seeds))
    share = graded["reversal_share"]
    fit_passes = all(graded["family_passes"][f] for f in bands["fit_families"])
    return nr._finite(
        {
            "statistics": _statistics(graded),
            "families": {
                family: {
                    name: {"value": r["value"], "z": r["z"], "passes": r["passes"]}
                    for name, r in readings.items()
                }
                for family, readings in graded["families"].items()
            },
            "family_passes": graded["family_passes"],
            "fit_passes": fit_passes,
            "local_search_presence": graded.get("local_search_presence"),
            "chemotaxis_presence": graded.get("chemotaxis_presence"),
            "wild_type_passes": bool(graded["wild_type_passes"]),
            "mutants": mutants,
            "D": graded["D"],
            "D_fit": graded["D_fit"],
            "reversal_share": None if share is None else share["value"],
        },
    )


class FitObjective:
    """v3's ranking score against the band centres at n = 200, on D_fit's families; picklable."""

    def __init__(self) -> None:
        bands = bg.load_bands_v3()
        self.scales = v2.band_scales_v2(bands)
        self.family_map = {f: tuple(bands["families"][f]) for f in bands["fit_families"]}
        centres = {}
        for family, stats in self.family_map.items():
            family_bands = bg.family_bands_v2(bands, family, ANIMALS)
            for s in stats:
                centres[f"{family}/{s}"] = float(family_bands[s].centre)
        self.truth = {"statistics": centres}

    def __call__(self, record: dict[str, Any]) -> float:
        """Return one simulation's ranking score against the band centres."""
        return v2.ranking_score(self.truth, record, self.scales, self.family_map)


def uncompressed_score(
    truth: dict[str, Any],
    draw: dict[str, Any],
    scales: dict[str, float],
    family_map: dict[str, tuple[str, ...]],
) -> float:
    """Return definedness first, then the weighted RMS of the uncapped z.

    The score is 1,000,000 x u + D, u the statistics undefined in exactly one of the two records and
    D the weighted RMS (families equal, split equally within each) of each defined statistic's z on
    the bands' scales, neither capped nor compressed, a statistic undefined in both counting 0. Far
    from the bands a statistic's term grows linearly, so progress on it is rewarded at full weight.
    """
    from quantumnematode.benchmark import grading as g

    readings: dict[str, dict[str, Any]] = {f: {} for f in family_map}
    undefined = 0
    for family, stats in family_map.items():
        for s in stats:
            key = f"{family}/{s}"
            a, b = truth["statistics"].get(key), draw["statistics"].get(key)
            if (a is None) != (b is None):
                undefined += 1
                term = 0.0
            elif a is None:
                term = 0.0
            else:
                term = (float(b) - float(a)) / scales[key]
            readings[family][s] = g.Reading(0.0, g.Band(0.0, scales[key]), term, passes=True)
    return UNCOMPRESSED_UNDEFINED_RANK * undefined + g.distance(
        readings,
        list(family_map),
        family_map,
    )


class UncompressedObjective(FitObjective):
    """The uncompressed score against the band centres on D_fit's families; picklable."""

    def __call__(self, record: dict[str, Any]) -> float:
        """Return one simulation's uncompressed score against the band centres."""
        return uncompressed_score(self.truth, record, self.scales, self.family_map)


def readiness_stop(medians: list[float], sigma: float) -> bool:
    """Return whether a restart stops: the guarded plateau, every median of its window below 100.

    The plateau rule compares the last two 20-generation windows; a median of 100 or more in either
    means a candidate in the generation's better half still leaves a statistic undefined.
    """
    window = medians[-2 * cr.PLATEAU_WINDOW :]
    return v2.guarded_stop(medians, sigma) and max(window) < GUARD


def start_centre(seed: int) -> Any:  # noqa: ANN401 - a cma strategy
    """Return sep-CMA at the encoder box's centre, σ₀ 0.2."""
    import cma

    options = {
        "CMA_diagonal": True,
        "bounds": [0.0, 1.0],
        "popsize": cr.POPSIZE,
        "seed": seed,
        "verbose": -9,
    }
    return cma.CMAEvolutionStrategy([cr.X0] * len(entries()), cr.SIGMA0, options)


def embed(x: list[float] | np.ndarray) -> list[float]:
    """Return the full vector: the reduced subset's values in place, every other at the centre."""
    names = [e.name for e in entries()]
    z = [cr.X0] * len(names)
    for name, value in zip(REDUCED_NAMES, x, strict=True):
        z[names.index(name)] = float(value)
    return z


def simulate_reduced(k: int, x: list[float], seed_base: int) -> dict[str, Any]:
    """Run simulation k of the reduced subset at x, the rest at the centre, on D_fit's assays."""
    return {**simulate(k, embed(x), seed_base), "x": list(map(float, x))}


def start_reduced(seed: int) -> Any:  # noqa: ANN401 - a cma strategy
    """Return sep-CMA at the reduced subset's centre, sigma0 0.2."""
    import cma

    options = {
        "CMA_diagonal": True,
        "bounds": [0.0, 1.0],
        "popsize": cr.POPSIZE,
        "seed": seed,
        "verbose": -9,
    }
    return cma.CMAEvolutionStrategy([cr.X0] * len(REDUCED_NAMES), cr.SIGMA0, options)


def start_stage2(seed: int) -> Any:  # noqa: ANN401 - a cma strategy
    """Return sep-CMA over every parameter at stage 1's answer, sigma0 0.1."""
    import cma

    x0 = json.loads((BUILD / "readiness" / "staged" / "stage1_x0.json").read_text())["x0"]
    options = {
        "CMA_diagonal": True,
        "bounds": [0.0, 1.0],
        "popsize": cr.POPSIZE,
        "seed": seed,
        "verbose": -9,
    }
    return cma.CMAEvolutionStrategy(x0, STAGE2_SIGMA0, options)


def start_warm(seed: int) -> Any:  # noqa: ANN401 - a cma strategy
    """Return sep-CMA at the one passing restart's answer, σ₀ 0.1."""
    import cma

    x0 = json.loads((BUILD / "readiness" / "warm_x0.json").read_text())["x0"]
    options = {
        "CMA_diagonal": True,
        "bounds": [0.0, 1.0],
        "popsize": cr.POPSIZE,
        "seed": seed,
        "verbose": -9,
    }
    return cma.CMAEvolutionStrategy(x0, WARM_SIGMA0, options)


# ---------------------------------------------------------------------------------------------
# The cost pilot
# ---------------------------------------------------------------------------------------------


def cost_points() -> list[list[float]]:
    """Return a generation's 16 points: the box centre plus a small spread, clipped to the box."""
    rng = np.random.default_rng(COST_BASE)
    n = len(entries())
    return [
        np.clip(cr.X0 + COST_SPREAD * rng.standard_normal(n), 0.0, 1.0).tolist()
        for _ in range(cr.POPSIZE)
    ]


# Where a simulation's time goes: each part's functions, by module suffix and name. Settling is the
# network's forward pass; noise the per-animal action draws; the evaluator the whole batched run,
# environment included, settling and noise among it.
_PARTS: dict[str, tuple[tuple[str, str], ...]] = {
    "evaluator_run": (("evaluation/batched.py", "run"),),
    "settling": (("brain/arch/connectome_ppo.py", "forward_with_hidden_batched"),),
    "modulator_release": (("brain/arch/connectome_ppo.py", "modulator_release_batched"),),
    "action_noise": (("evaluation/controllers.py", "_sample"),),
    "record_conversion": (("evaluation/batched.py", "to_behaviour_steps"),),
    "measurement": (
        ("benchmark/measurement.py", "measure_window"),
        ("benchmark/measurement.py", "measure"),
    ),
    "grading": (("analysis/benchmark_grade.py", "_grade_measured"),),
    "brain_write": (("connectome/encoder.py", "write_encoder"),),
}


def _profile(function: Callable[[int, list[float], int], Any], k: int, z: list[float]) -> dict:
    profiler = cProfile.Profile()
    start = time.perf_counter()
    profiler.runcall(function, k, z, COST_BASE)
    total = time.perf_counter() - start
    stats = pstats.Stats(profiler)
    table: dict[tuple[str, int, str], tuple[Any, ...]] = stats.stats  # type: ignore[attr-defined]
    parts = {
        part: float(
            sum(
                entry[3]
                for (file, _, name), entry in table.items()
                if any(file.endswith(suffix) and name == fn for suffix, fn in functions)
            ),
        )
        for part, functions in _PARTS.items()
    }
    top = sorted(table.items(), key=lambda item: item[1][2], reverse=True)[:15]
    return {
        "seconds": total,
        "parts_cumulative_seconds": parts,
        "top_by_own_time": [
            {"function": f"{Path(file).name}:{line}:{name}", "own_seconds": float(entry[2])}
            for (file, line, name), entry in top
        ],
    }


def _compute_path(z: list[float]) -> dict[str, Any]:
    """Time the settling step at a generation's 3,200 animals on CPU (1 and all threads) and MPS."""
    import torch
    from quantumnematode.connectome import encoder as enc

    brain = encoded_brain()
    enc.write_encoder(brain, z)
    generator = torch.Generator().manual_seed(COST_BASE)
    food = torch.rand(COMPUTE_ANIMALS, brain.topology.n_food_features, generator=generator)
    modulators = torch.rand(COMPUTE_ANIMALS, brain._n_concentrations + 1, generator=generator)

    def timed(topology: Any, device: str) -> tuple[float, Any]:  # noqa: ANN401 - a topology
        f, m = food.to(device), modulators.to(device)
        times = []
        mean = None
        with torch.no_grad():
            for repeat in range(COMPUTE_REPEATS + 5):
                start = time.perf_counter()
                mean, _ = topology.forward_with_hidden_batched(f, modulators=m)
                if device == "mps":
                    torch.mps.synchronize()
                if repeat >= 5:
                    times.append(time.perf_counter() - start)
        assert mean is not None  # noqa: S101 - the loop ran
        return float(np.median(times)), mean.cpu()

    threads = os.cpu_count() or 1
    out: dict[str, Any] = {"animals": COMPUTE_ANIMALS, "repeats": COMPUTE_REPEATS}
    torch.set_num_threads(1)
    out["cpu_1_thread_seconds"], reference = timed(brain.topology, "cpu")
    torch.set_num_threads(threads)
    out[f"cpu_{threads}_threads_seconds"], _ = timed(brain.topology, "cpu")
    if torch.backends.mps.is_available():
        on_mps = copy.deepcopy(brain.topology).to("mps")
        out["mps_seconds"], mean = timed(on_mps, "mps")
        out["mps_max_abs_difference_from_cpu"] = float((mean - reference).abs().max())
    else:
        out["mps_seconds"] = None
    torch.set_num_threads(1)
    return out


def _cost(workers: int) -> int:
    """Time a generation on D_fit's assays and on all four, profile both, time the compute path."""
    points = cost_points()
    record: dict[str, Any] = {"description": DESCRIPTION + " The cost pilot.", "workers": workers}
    with ProcessPoolExecutor(max_workers=workers, initializer=nr._worker_setup) as pool:
        for label, function, offset in (("fit", simulate, 0), ("all", simulate_all, 1_000)):
            generations = []
            for generation in range(COST_GENERATIONS):
                ks = [offset + cr.POPSIZE * generation + m for m in range(cr.POPSIZE)]
                start = time.perf_counter()
                results = list(pool.map(function, ks, points, [COST_BASE] * cr.POPSIZE))
                generations.append(
                    {
                        "wall_seconds": time.perf_counter() - start,
                        "simulation_seconds_median": float(
                            np.median([r["seconds"] for r in results]),
                        ),
                    },
                )
            record[label] = {"assays": list(FIT_ASSAYS) if label == "fit" else "all four"}
            record[label]["generations"] = generations
            print(label, json.dumps(generations), flush=True)
            _write(COST_PATH, record)
    nr._worker_setup()
    record["profile"] = {
        "fit": _profile(simulate, 2_000, points[0]),
        "all": _profile(simulate_all, 2_001, points[0]),
    }
    _write(COST_PATH, record)
    record["compute_path"] = _compute_path(points[0])
    reused = record["fit"]["generations"][-1]["wall_seconds"]
    record["projection_hours_per_restart"] = {
        f"{gens}_generations": reused * gens / 3600.0 for gens in (100, 300, 500, 1_000)
    }
    _write(COST_PATH, record)
    print(json.dumps({k: record[k] for k in ("compute_path", "projection_hours_per_restart")}))
    return 0


# ---------------------------------------------------------------------------------------------
# The feasibility of a slow decay
# ---------------------------------------------------------------------------------------------

# The hand-set vector, in parameter values. Unnamed parameters sit at zero where their range allows
# it and at their floor otherwise (the fixed-sign strengths).
FEASIBILITY_VALUES: dict[str, float] = {
    "readout.reversal.VA": 3.0,
    "readout.speed.bias": -1.0,
    "readout.reversal.bias": -1.0,
    "readout.omega.bias": -3.0,
    "readout.delta.bias": -3.0,
    "noise.speed": -4.0,
    "noise.curvature": 1.0,
    "noise.reversal": -4.0,
    "noise.omega": -4.0,
    "noise.delta": -4.0,
    "modulator.ingestion": 3.0,
    "modulator.nlp-54:trhr-1": 3.0,
    "timescale.monoamine": 60.0,
    "timescale.neuropeptide": 540.0,
}


def feasibility_vector(values: dict[str, float] | None = None) -> np.ndarray:
    """Return the hand-set vector: the named values, every other parameter at zero or its floor."""
    values = FEASIBILITY_VALUES if values is None else values
    table = entries()
    unknown = sorted(set(values) - {e.name for e in table})
    if unknown:
        msg = f"no encoder parameters named {unknown}"
        raise KeyError(msg)
    by_name = {e.name: e for e in table}
    outside = sorted(
        f"{name}={value} (range {by_name[name].low}-{by_name[name].high})"
        for name, value in values.items()
        if not by_name[name].low <= value <= by_name[name].high
    )
    if outside:
        msg = f"hand-set values outside their parameters' ranges: {outside}"
        raise ValueError(msg)
    out = []
    for e in table:
        value = values.get(e.name, min(max(0.0, e.low), e.high))
        out.append(e.unit(value))
    return np.clip(np.asarray(out), 0.0, 1.0)


def feasibility_run(z: np.ndarray, base: int) -> dict[str, Any]:
    """Run López-Cruz's food removal at z on 200 animals; return the onset rate's decay."""
    from quantumnematode.benchmark import grading as g

    bands = bg.load_bands_v3()
    seeds = population_seeds(base, 0)
    populations, _ = bg.simulate_v2(ARM, controller(z), seeds, bands, assays=("local_search",))
    statistics = g.simulated_search(populations["local_search"])
    gamma = statistics["gamma_per_min"]
    tau = None if gamma is None or not np.isfinite(gamma) or gamma <= 0 else 1.0 / gamma
    low, high = FEASIBILITY_WINDOW_MIN
    return nr._finite(
        {
            "base": base,
            "statistics": statistics,
            "decay_constant_min": tau,
            "within_window": tau is not None and low <= tau <= high,
        },
    )


def _feasibility() -> int:
    nr._worker_setup()
    z = feasibility_vector()
    result = feasibility_run(z, FEASIBILITY_BASE)
    record = {
        "description": DESCRIPTION + " The feasibility of a ~9-min post-removal decay.",
        "values": FEASIBILITY_VALUES,
        "vector": z.tolist(),
        "window_min": list(FEASIBILITY_WINDOW_MIN),
        "trial_seed_base": FEASIBILITY_TRIAL_BASE,
        **result,
    }
    _write(FEASIBILITY_PATH, record)
    print(json.dumps({k: record[k] for k in ("decay_constant_min", "within_window")}))
    return 0


# ---------------------------------------------------------------------------------------------
# The readiness probe
# ---------------------------------------------------------------------------------------------


def _restart_dir(r: int) -> Path:
    return BUILD / "readiness" / f"r{r}"


def _grades_path() -> Path:
    return BUILD / "readiness" / "grades.json"


def ready(grades: list[dict[str, Any]]) -> bool:
    """Return whether an answer is ready: D_fit's three families pass in at least 3 of 5."""
    return sum(bool(g["fit_passes"]) for g in grades) >= PASS_NEEDED


def trajectory_k(restart: int, generation: int, repeat: int) -> int:
    """Return the simulation index of a restart's answer at a generation, re-simulated."""
    return TRAJECTORY_BLOCK * restart + TRAJECTORY_REPEATS * generation + repeat


def _fit(  # noqa: PLR0913 - one restart's settings
    pool: ProcessPoolExecutor,
    *,
    cma_seed: int,
    seed_base: int,
    cap: int,
    state_dir: Path,
    factory: Callable[[int], Any],
    objective: Callable[[dict[str, Any]], float] | None = None,
    simulate_fn: Callable[[int, list[float], int], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Run, or resume, one restart through its cap; the plateau rule is recorded but not used."""
    return cr.run_fit(
        {},
        fit=0,
        cma_seed=cma_seed,
        seed_base=seed_base,
        cap=cap,
        use_rule=False,
        state_dir=state_dir,
        pool=pool,
        scales={},
        objective=FitObjective() if objective is None else objective,
        simulate=simulate if simulate_fn is None else simulate_fn,
        stop=readiness_stop,
        strategy_factory=factory,
    )


def trajectory(  # noqa: PLR0913 - the history, the pool and how to re-simulate it
    history: dict[str, Any],
    pool: ProcessPoolExecutor,
    *,
    base: int,
    restart: int,
    every: int,
    simulate_fn: Callable[[int, list[float], int], dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Re-simulate a restart's answer at every ``every``-th generation twice on D_fit's assays."""
    generations = list(range(every, len(history["answers"]) + 1, every))
    jobs = [(g, q) for g in generations for q in range(TRAJECTORY_REPEATS)]
    records = list(
        pool.map(
            simulate if simulate_fn is None else simulate_fn,
            [trajectory_k(restart, g, q) for g, q in jobs],
            [history["answers"][g - 1] for g, _ in jobs],
            [base] * len(jobs),
        ),
    )
    objective, uncompressed = FitObjective(), UncompressedObjective()
    points = []
    for g in generations:
        mine = [r for (gen, _), r in zip(jobs, records, strict=True) if gen == g]
        points.append(
            {
                "generation": g,
                "median": history["medians"][g - 1],
                "sigma": history["sigmas"][g - 1],
                "answer_score": [objective(r) for r in mine],
                "answer_uncompressed_score": [uncompressed(r) for r in mine],
                "answer_D_fit": [r["D_fit"] for r in mine],
                "answer_time_on_lawn": [r["statistics"]["small_lawn/time_on_lawn"] for r in mine],
            },
        )
    return points


def converged_generation(points: list[dict[str, Any]], tolerance: float = 0.05) -> int | None:
    """Return the first checkpoint whose answer score is within 5% of the restart's best."""
    scores = [float(np.median(p["answer_score"])) for p in points]
    if not scores:
        return None
    best = min(scores)
    return next(
        p["generation"] for p, x in zip(points, scores, strict=True) if x <= best * (1 + tolerance)
    )


def _pilot(workers: int) -> int:
    """One restart in the rescaled box for 150 generations, its answer read every 25."""
    with ProcessPoolExecutor(max_workers=workers, initializer=nr._worker_setup) as pool:
        history = _fit(
            pool,
            cma_seed=PILOT_CMA,
            seed_base=PILOT_BASE,
            cap=PILOT_CAP,
            state_dir=BUILD / "readiness" / "pilot",
            factory=start_centre,
        )
        points = trajectory(history, pool, base=PILOT_TRAJECTORY_BASE, restart=0, every=PILOT_EVERY)
    best = min(float(np.median(p["answer_score"])) for p in points)
    record = {
        "description": DESCRIPTION + " The pilot in the rescaled box.",
        "wiring_range": next([e.low, e.high] for e in entries() if e.name == "wiring.excitatory"),
        "stall_score": STALL_SCORE,
        "best_answer_score": best,
        "improves_past_stall": best < STALL_SCORE,
        "points": points,
        "medians": history["medians"],
        "sigmas": history["sigmas"],
        "answer": history["answers"][-1],
    }
    _write(PILOT_PATH, record)
    print(json.dumps({k: record[k] for k in ("best_answer_score", "improves_past_stall")}))
    return 0


def _uncompressed_pilot(workers: int) -> int:
    """Run the second pilot: 150 generations ranking candidates on the uncompressed score.

    It passes when its best checkpoint by the registered ranking score beats the stall's 10.6 and
    that checkpoint's worm spends more of the small lawn's assay on the lawn than the stall's 0.261.
    """
    with ProcessPoolExecutor(max_workers=workers, initializer=nr._worker_setup) as pool:
        history = _fit(
            pool,
            cma_seed=UNCOMPRESSED_PILOT_CMA,
            seed_base=UNCOMPRESSED_PILOT_BASE,
            cap=PILOT_CAP,
            state_dir=BUILD / "readiness" / "pilot_uncompressed",
            factory=start_centre,
            objective=UncompressedObjective(),
        )
        points = trajectory(
            history,
            pool,
            base=UNCOMPRESSED_PILOT_TRAJECTORY_BASE,
            restart=0,
            every=PILOT_EVERY,
        )
    best = min(points, key=lambda p: float(np.median(p["answer_score"])))
    best_score = float(np.median(best["answer_score"]))
    best_time = float(np.median(best["answer_time_on_lawn"]))
    record = {
        "description": DESCRIPTION + " The second pilot, on the uncompressed score.",
        "stall_score": STALL_SCORE,
        "stall_time_on_lawn": STALL_TIME_ON_LAWN,
        "best_generation": best["generation"],
        "best_answer_score": best_score,
        "best_time_on_lawn": best_time,
        "passes": best_score < STALL_SCORE and best_time > STALL_TIME_ON_LAWN,
        "points": points,
        "medians": history["medians"],
        "sigmas": history["sigmas"],
        "answer": history["answers"][-1],
    }
    _write(UNCOMPRESSED_PILOT_PATH, record)
    keys = ("best_generation", "best_answer_score", "best_time_on_lawn", "passes")
    print(json.dumps({k: record[k] for k in keys}))
    return 0


def _reduced(workers: int) -> int:
    """Run the reachability probe: the lawn-holding subset alone, 150 generations, read every 25."""
    with ProcessPoolExecutor(max_workers=workers, initializer=nr._worker_setup) as pool:
        history = _fit(
            pool,
            cma_seed=REDUCED_CMA,
            seed_base=REDUCED_BASE,
            cap=PILOT_CAP,
            state_dir=BUILD / "readiness" / "reduced",
            factory=start_reduced,
            objective=UncompressedObjective(),
            simulate_fn=simulate_reduced,
        )
        points = trajectory(
            history,
            pool,
            base=REDUCED_TRAJECTORY_BASE,
            restart=0,
            every=PILOT_EVERY,
            simulate_fn=simulate_reduced,
        )
    best = max(points, key=lambda p: float(np.median(p["answer_time_on_lawn"])))
    best_time = float(np.median(best["answer_time_on_lawn"]))
    record = {
        "description": DESCRIPTION + " The reachability probe: the lawn-holding subset alone.",
        "names": list(REDUCED_NAMES),
        "holds_lawn_at": HOLDS_LAWN,
        "best_generation": best["generation"],
        "best_time_on_lawn": best_time,
        "best_answer_score": float(np.median(best["answer_score"])),
        "holds_lawn": best_time >= HOLDS_LAWN,
        "points": points,
        "medians": history["medians"],
        "sigmas": history["sigmas"],
        "answer": history["answers"][-1],
        "answer_full": embed(history["answers"][-1]),
    }
    _write(REDUCED_PATH, record)
    keys = ("best_generation", "best_time_on_lawn", "best_answer_score", "holds_lawn")
    print(json.dumps({k: record[k] for k in keys}))
    return 0


def _log_pilot(workers: int) -> int:
    """Run the third pilot: 150 generations with log-scaled fixed-sign strengths, ranking score.

    It passes when its best checkpoint by the registered ranking score beats the stall's 10.6 and
    that checkpoint's worm spends more of the small lawn's assay on the lawn than the stall's 0.261.
    """
    with ProcessPoolExecutor(max_workers=workers, initializer=nr._worker_setup) as pool:
        history = _fit(
            pool,
            cma_seed=LOG_PILOT_CMA,
            seed_base=LOG_PILOT_BASE,
            cap=PILOT_CAP,
            state_dir=BUILD / "readiness" / "pilot_logbox",
            factory=start_centre,
        )
        points = trajectory(
            history,
            pool,
            base=LOG_PILOT_TRAJECTORY_BASE,
            restart=0,
            every=PILOT_EVERY,
        )
    best = min(points, key=lambda p: float(np.median(p["answer_score"])))
    best_score = float(np.median(best["answer_score"]))
    best_time = float(np.median(best["answer_time_on_lawn"]))
    record = {
        "description": DESCRIPTION + " The third pilot, fixed-sign strengths on a log scale.",
        "stall_score": STALL_SCORE,
        "stall_time_on_lawn": STALL_TIME_ON_LAWN,
        "best_generation": best["generation"],
        "best_answer_score": best_score,
        "best_time_on_lawn": best_time,
        "passes": best_score < STALL_SCORE and best_time > STALL_TIME_ON_LAWN,
        "points": points,
        "medians": history["medians"],
        "sigmas": history["sigmas"],
        "answer": history["answers"][-1],
    }
    _write(LOG_PILOT_PATH, record)
    keys = ("best_generation", "best_answer_score", "best_time_on_lawn", "passes")
    print(json.dumps({k: record[k] for k in keys}))
    return 0


def _summary(points: list[dict[str, Any]]) -> dict[str, Any]:
    best = min(points, key=lambda p: float(np.median(p["answer_score"])))
    return {
        "best_generation": best["generation"],
        "best_answer_score": float(np.median(best["answer_score"])),
        "best_time_on_lawn": float(np.median(best["answer_time_on_lawn"])),
        "final_time_on_lawn": float(np.median(points[-1]["answer_time_on_lawn"])),
    }


def _staged_pilot(workers: int) -> int:
    """Run the staged pilot: the lawn subset first, then every parameter from its answer.

    Stage 1 must hold the lawn (its final checkpoint's median time on it at least 0.75) for stage 2
    to run. The pilot passes when stage 2's best checkpoint by the registered ranking score beats
    the stall's 10.6 with the worm on the lawn more than the stall's 0.261.
    """
    root = BUILD / "readiness" / "staged"
    with ProcessPoolExecutor(max_workers=workers, initializer=nr._worker_setup) as pool:
        first = _fit(
            pool,
            cma_seed=STAGE1_CMA,
            seed_base=STAGE1_BASE,
            cap=PILOT_CAP,
            state_dir=root / "stage1",
            factory=start_reduced,
            objective=UncompressedObjective(),
            simulate_fn=simulate_reduced,
        )
        first_points = trajectory(
            first,
            pool,
            base=STAGE1_TRAJECTORY_BASE,
            restart=0,
            every=PILOT_EVERY,
            simulate_fn=simulate_reduced,
        )
        record: dict[str, Any] = {
            "description": DESCRIPTION + " The staged pilot.",
            "names_stage1": list(REDUCED_NAMES),
            "stage1": {
                **_summary(first_points),
                "points": first_points,
                "seconds_per_generation_median": float(np.median(first["seconds"])),
            },
        }
        record["stage1"]["holds_lawn"] = record["stage1"]["final_time_on_lawn"] >= HOLDS_LAWN
        start = embed(first["answers"][-1])
        record["stage1"]["answer_full"] = start
        _write(root / "stage1_x0.json", {"x0": start})
        if record["stage1"]["holds_lawn"]:
            second = _fit(
                pool,
                cma_seed=STAGE2_CMA,
                seed_base=STAGE2_BASE,
                cap=PILOT_CAP,
                state_dir=root / "stage2",
                factory=start_stage2,
            )
            second_points = trajectory(
                second,
                pool,
                base=STAGE2_TRAJECTORY_BASE,
                restart=0,
                every=PILOT_EVERY,
            )
            record["stage2"] = {
                **_summary(second_points),
                "points": second_points,
                "answer": second["answers"][-1],
                "seconds_per_generation_median": float(np.median(second["seconds"])),
            }
            summary = record["stage2"]
            record["passes"] = (
                summary["best_answer_score"] < STALL_SCORE
                and summary["best_time_on_lawn"] > STALL_TIME_ON_LAWN
            )
        else:
            record["passes"] = False
    _write(STAGED_PILOT_PATH, record)
    print(
        json.dumps(
            {
                "stage1_holds_lawn": record["stage1"]["holds_lawn"],
                "stage1_final_time_on_lawn": record["stage1"]["final_time_on_lawn"],
                "stage2": {
                    k: v for k, v in record.get("stage2", {}).items() if k.startswith("best")
                },
                "passes": record["passes"],
            },
        ),
    )
    return 0


def _stage2_anyway(workers: int) -> int:
    """Run the staged pilot's stage 2 from stage 1's answer, although stage 1 missed its lawn bar.

    A deviation from the pilot's stated rule, recorded as one: stage 1 reached the best answer of
    any fit (below the stall's score) without holding the lawn at 0.75.
    """
    root = BUILD / "readiness" / "staged"
    record = json.loads(STAGED_PILOT_PATH.read_text())
    with ProcessPoolExecutor(max_workers=workers, initializer=nr._worker_setup) as pool:
        second = _fit(
            pool,
            cma_seed=STAGE2_CMA,
            seed_base=STAGE2_BASE,
            cap=PILOT_CAP,
            state_dir=root / "stage2",
            factory=start_stage2,
        )
        second_points = trajectory(
            second,
            pool,
            base=STAGE2_TRAJECTORY_BASE,
            restart=0,
            every=PILOT_EVERY,
        )
    record["stage2"] = {
        **_summary(second_points),
        "points": second_points,
        "answer": second["answers"][-1],
        "ran_although_stage1_missed_its_bar": True,
        "seconds_per_generation_median": float(np.median(second["seconds"])),
    }
    summary = record["stage2"]
    record["stage2_would_pass"] = (
        summary["best_answer_score"] < STALL_SCORE
        and summary["best_time_on_lawn"] > STALL_TIME_ON_LAWN
    )
    _write(STAGED_PILOT_PATH, record)
    print(
        json.dumps(
            {
                **{k: v for k, v in summary.items() if k.startswith(("best", "final"))},
                "stage2_would_pass": record["stage2_would_pass"],
            },
        ),
    )
    return 0


def _extend_stage2(workers: int) -> int:
    """Resume stage 2 from its last generation to 500, its answer read every 25 generations.

    The saved history's cap stop is cleared, and the fit resumes from its checkpointed strategy under
    the same simulation, start and seeds, so its first 150 generations are those already run.
    """
    root = BUILD / "readiness" / "staged" / "stage2"
    history_path = root / "fit0.json"
    saved = json.loads(history_path.read_text())
    if saved["stop"] is not None and saved["stop"]["generation"] < STAGE2_EXTENDED_CAP:
        saved["stop"] = None
        history_path.write_text(json.dumps(saved))
    record = json.loads(STAGED_PILOT_PATH.read_text())
    # Stage 2 first stopped at its cap; a resumed extension keeps the same seed mapping.
    resumed_from = PILOT_CAP
    # Generation g's member m is simulation k = 16(g - 1) + m, so the first resumed generation's
    # animals start at the extension's own base.
    seed_base = STAGE2_EXTENSION_BASE - ANIMALS * cr.POPSIZE * resumed_from
    with ProcessPoolExecutor(max_workers=workers, initializer=nr._worker_setup) as pool:
        second = _fit(
            pool,
            cma_seed=STAGE2_CMA,
            seed_base=seed_base,
            cap=STAGE2_EXTENDED_CAP,
            state_dir=root,
            factory=start_stage2,
        )
        points = trajectory(
            second,
            pool,
            base=STAGE2_EXTENSION_TRAJECTORY_BASE,
            restart=0,
            every=PILOT_EVERY,
        )
    record["stage2_extended"] = {
        **_summary(points),
        "cap": STAGE2_EXTENDED_CAP,
        "resumed_from_generation": resumed_from,
        "seeds": (
            f"generations 1-{resumed_from} at {STAGE2_BASE:,} + 200k + i; generations "
            f"{resumed_from + 1}-{STAGE2_EXTENDED_CAP} at {STAGE2_EXTENSION_BASE:,} + "
            f"200(k - {cr.POPSIZE * resumed_from:,}) + i; checkpoints at "
            f"{STAGE2_EXTENSION_TRAJECTORY_BASE:,} + 200(2g + q) + i"
        ),
        "converged_generation": converged_generation(points),
        "points": points,
        "medians": second["medians"],
        "sigmas": second["sigmas"],
        "answer": second["answers"][-1],
        "seconds_per_generation_median": float(np.median(second["seconds"])),
    }
    _write(STAGED_PILOT_PATH, record)
    summary = record["stage2_extended"]
    print(json.dumps({k: v for k, v in summary.items() if k.startswith(("best", "final", "conv"))}))
    return 0


def _restart(
    r: int,
    pool: ProcessPoolExecutor,
    grades: dict[str, list[dict[str, Any]]],
) -> None:
    """Run one readiness restart through the cap, grade its answer and read its trajectory."""
    warm = r == WARM_RESTART
    history = _fit(
        pool,
        cma_seed=WARM_CMA if warm else READINESS_CMA + r,
        seed_base=WARM_BASE if warm else READINESS_BASE + ANIMALS * RESTART_BLOCK * r,
        cap=CAP,
        state_dir=_restart_dir(r),
        factory=start_warm if warm else start_centre,
    )
    # The trajectory first: a restart's grades mark it done, and the reading needs both.
    path = _restart_dir(r) / "trajectory.json"
    if not path.exists():
        points = trajectory(history, pool, base=TRAJECTORY_BASE, restart=r, every=TRAJECTORY_EVERY)
        _write(path, {"restart": r, "points": points})
    if str(r) not in grades:
        answer = history["answers"][-1]
        # Every restart on the same populations, so the spread compares restarts on one set of
        # animals.
        ks = list(range(GRADE_POPULATIONS))
        grades[str(r)] = list(
            pool.map(grade_answer, ks, [answer] * len(ks), [GRADE_BASE] * len(ks)),
        )
        _write(_grades_path(), grades)


def _readiness(workers: int) -> int:
    """Run the readiness restarts in turn, each through the cap, until 2 are ready."""
    if not (LOG_PILOT_PATH.exists() and json.loads(LOG_PILOT_PATH.read_text())["passes"]):
        print("the log-box pilot has not passed; no readiness restart runs", file=sys.stderr)
        return 1
    path = _grades_path()
    grades: dict[str, list[dict[str, Any]]] = json.loads(path.read_text()) if path.exists() else {}
    with ProcessPoolExecutor(max_workers=workers, initializer=nr._worker_setup) as pool:
        for r in range(RESTARTS):
            if sum(ready(grades[q]) for q in grades if int(q) < RESTARTS) >= PASSES_WANTED:
                break
            _restart(r, pool, grades)
        passing = [int(q) for q in grades if int(q) < RESTARTS and ready(grades[q])]
        if len(passing) == 1 and all(str(r) in grades for r in range(RESTARTS)):
            x0 = json.loads((_restart_dir(passing[0]) / "fit0.json").read_text())["answers"][-1]
            warm = BUILD / "readiness" / "warm_x0.json"
            if not warm.exists():
                _write(warm, {"from_restart": passing[0], "x0": x0})
            _restart(WARM_RESTART, pool, grades)
    return 0


def _parameters_at_bounds(answer: list[float]) -> list[str]:
    return [
        e.name
        for e, u in zip(entries(), answer, strict=True)
        if u <= BOUND_TOLERANCE or u >= 1.0 - BOUND_TOLERANCE
    ]


def _held_out_distance(a: dict[str, Any], b: dict[str, Any], bands: dict[str, Any]) -> float:
    held_out = [f for f in bands["families"] if f not in bands["fit_families"]]
    family_map = {f: tuple(bands["families"][f]) for f in held_out}
    return v2.predictive_distance_v2(a, b, v2.band_scales_v2(bands), family_map, capped=True)


def _mutant_distance(a: dict[str, Any], b: dict[str, Any], mutant: str) -> float:
    """Return two answers' RMS z-difference in a mutant's differences, on its band scales."""
    stats = a["mutants"][mutant]["statistics"]
    terms = [
        ((stats[s]["delta"] - b["mutants"][mutant]["statistics"][s]["delta"]) / stats[s]["scale"])
        ** 2
        for s in stats
        if stats[s]["delta"] is not None
        and b["mutants"][mutant]["statistics"][s]["delta"] is not None
    ]
    return float(np.sqrt(np.mean(terms))) if terms else float("nan")


def spread(grades: dict[int, list[dict[str, Any]]], bands: dict[str, Any]) -> dict[str, Any]:
    """Return the restart-to-restart spread of the held-out families and the mutant differences.

    Each pair of restarts is compared population by population, the same animals in both, and the
    pair's value is the median over the grading populations.
    """
    pairs = []
    for i, j in itertools.combinations(sorted(grades), 2):
        a, b = grades[i], grades[j]
        pairs.append(
            {
                "restarts": [i, j],
                "held_out_D": float(
                    np.median([_held_out_distance(x, y, bands) for x, y in zip(a, b, strict=True)]),
                ),
                "mutant_z": {
                    m: float(
                        np.nanmedian(
                            [_mutant_distance(x, y, m) for x, y in zip(a, b, strict=True)],
                        ),
                    )
                    for m in MUTANTS
                },
            },
        )
    return {"pairs": pairs}


def _read() -> int:
    """Write the readiness record and the W sizing record."""
    from quantumnematode.connectome import encoder as enc

    bands = bg.load_bands_v3()
    grades = {int(r): g for r, g in json.loads(_grades_path().read_text()).items()}
    table = entries()
    timescales = [i for i, e in enumerate(table) if e.block == "timescale"]
    restarts = []
    for r in sorted(grades):
        history = json.loads((_restart_dir(r) / "fit0.json").read_text())
        answer = history["answers"][-1]
        points = json.loads((_restart_dir(r) / "trajectory.json").read_text())["points"]
        restarts.append(
            {
                "restart": r,
                "warm": r == WARM_RESTART,
                "stop": history["stop"],
                "final_median": history["medians"][-1],
                "converged_generation": converged_generation(points),
                "trajectory": points,
                "seconds_per_generation_median": float(np.median(history["seconds"])),
                "ready": ready(grades[r]),
                "fit_passing_populations": int(sum(bool(g["fit_passes"]) for g in grades[r])),
                "family_passing_populations": {
                    f: int(sum(bool(g["family_passes"][f]) for g in grades[r]))
                    for f in bands["families"]
                },
                "mutant_passing_populations": {
                    m: int(sum(bool(g["mutants"][m]["passes"]) for g in grades[r])) for m in MUTANTS
                },
                "reversal_share": [g["reversal_share"] for g in grades[r]],
                "D_fit": [g["D_fit"] for g in grades[r]],
                "D": [g["D"] for g in grades[r]],
                "parameters_at_a_bound": _parameters_at_bounds(answer),
                "timescales_s": {table[i].name: table[i].value(answer[i]) for i in timescales},
                "answer": answer,
            },
        )
    passing = [x["restart"] for x in restarts if x["ready"]]
    record = {
        "description": DESCRIPTION + " The readiness probe.",
        "tying": TYING,
        "parameters": len(table),
        "names": enc.encoder_parameters(layer_spec(), TYING),
        "ready_restarts": passing,
        "ready": len(passing) >= PASSES_WANTED,
        "restarts": restarts,
        "spread": spread(grades, bands),
        "w_minimum": json.loads(
            (SUPPORT.parent / "120-benchmark-v3" / "controls_v3.json").read_text(),
        )["w_minimum"],
        "reversal_share_reference": bands["reported"]["reversal_share"],
    }
    _write(READINESS_PATH, record)
    _write(W_SIZING_PATH, w_sizing(record))
    print(json.dumps({k: record[k] for k in ("ready_restarts", "ready")}))
    return 0


# W's arms as planned: the wild type and five nulls; and its planned draws per arm.
W_ARMS = 6
W_DRAWS = (8, 16)


def _sizing() -> int:
    """Write the W sizing record from the cost pilot and the staged pilot: measured, and not yet.

    The readiness restarts did not run, so the restart-to-restart spread, the tying and the
    generations a fit needs to converge are not measured; the record says so beside what is.
    """
    cost = json.loads(COST_PATH.read_text())
    staged = json.loads(STAGED_PILOT_PATH.read_text())
    extended = staged["stage2_extended"]
    pace = {
        "stage1": staged["stage1"]["seconds_per_generation_median"],
        "stage2": extended["seconds_per_generation_median"],
    }
    fit_hours = {
        f"stage2_to_{cap}": (PILOT_CAP * pace["stage1"] + cap * pace["stage2"]) / 3600.0
        for cap in (STAGE2_EXTENDED_CAP, CAP)
    }
    record = {
        "description": DESCRIPTION + " What F.0b-ii's and W's registrations are sized from.",
        "measured": {
            "cost_pilot_generation_seconds": {
                "d_fit_assays": cost["fit"]["generations"][-1]["wall_seconds"],
                "all_four_assays": cost["all"]["generations"][-1]["wall_seconds"],
            },
            "staged_fit_generation_seconds": pace,
            "compute_path": cost["compute_path"],
            "compute_path_decision": "CPU: MPS speeds only the settling, and the rest of a "
            "simulation would then run serially in one process",
            "staged_fit": {
                "stage1_generations": PILOT_CAP,
                "stage2_generations": STAGE2_EXTENDED_CAP,
                "stage2_best_generation": extended["best_generation"],
                "stage2_best_answer_score": extended["best_answer_score"],
                "converged_by_cap": extended["converged_generation"] < STAGE2_EXTENDED_CAP,
            },
            "hours_per_staged_fit": fit_hours,
        },
        "projection_hours_for_w": {
            f"{W_ARMS}_arms_x_{draws}_draws": {
                key: W_ARMS * draws * hours for key, hours in fit_hours.items()
            }
            for draws in W_DRAWS
        },
        "not_measured": [
            "the generations a staged fit needs to converge (stage 2 improved at its cap of 500)",
            (
                "the restart-to-restart spread of the held-out families and mutant differences, "
                "against W's minimum (2.97 fitted, 2.05 held-out)"
            ),
            "the tying (per_couple against per_ligand)",
            "the draws per arm",
        ],
    }
    _write(W_SIZING_PATH, record)
    print(json.dumps(record["projection_hours_for_w"], indent=1))
    return 0


def w_sizing(readiness: dict[str, Any]) -> dict[str, Any]:
    """Return what F.0b-ii's and W's registrations size from: generations, pace, spread, path."""
    cost = json.loads(COST_PATH.read_text()) if COST_PATH.exists() else None
    converged = [x["converged_generation"] for x in readiness["restarts"] if not x["warm"]]
    pairs = readiness["spread"]["pairs"]
    held_out = [p["held_out_D"] for p in pairs]
    minimum = readiness["w_minimum"]["held_out"]["minimum_95th"]
    return nr._finite(
        {
            "description": DESCRIPTION + " What F.0b-ii's and W's registrations are sized from.",
            "tying": TYING,
            "converged_generation_per_restart": converged,
            "cap": CAP,
            "seconds_per_generation_d_fit": [
                x["seconds_per_generation_median"] for x in readiness["restarts"]
            ],
            "cost_pilot": None
            if cost is None
            else {
                "fit_generation_seconds": cost["fit"]["generations"][-1]["wall_seconds"],
                "all_generation_seconds": cost["all"]["generations"][-1]["wall_seconds"],
                "compute_path": cost["compute_path"],
            },
            "held_out_spread_D": held_out,
            "held_out_minimum": minimum,
            "held_out_spread_within_minimum": bool(held_out)
            and float(np.median(held_out)) <= minimum,
            "mutant_spread_z": [p["mutant_z"] for p in pairs],
            "ready_restarts": readiness["ready_restarts"],
        },
    )


# ---------------------------------------------------------------------------------------------
# The smoke, and writing
# ---------------------------------------------------------------------------------------------


def _smoke(workers: int) -> int:
    """Two generations of a readiness restart and one graded population, on discarded seeds."""
    import shutil

    state = BUILD / "smoke"
    shutil.rmtree(state, ignore_errors=True)
    with ProcessPoolExecutor(max_workers=workers, initializer=nr._worker_setup) as pool:
        history = cr.run_fit(
            {},
            fit=0,
            cma_seed=SMOKE_CMA,
            seed_base=SMOKE_BASE,
            cap=2,
            use_rule=True,
            state_dir=state,
            pool=pool,
            scales={},
            objective=FitObjective(),
            simulate=simulate,
            stop=readiness_stop,
            strategy_factory=start_centre,
        )
        (graded,) = pool.map(grade_answer, [100], [history["answers"][-1]], [SMOKE_BASE])
    print(
        json.dumps(
            {
                "medians": history["medians"],
                "fit_passes": graded["fit_passes"],
                "D": graded["D"],
                "mutants": {m: graded["mutants"][m]["passes"] for m in MUTANTS},
            },
        ),
    )
    return 0 if len(history["medians"]) == 2 and graded["D_fit"] is not None else 1


def _write(path: Path, record: Any) -> None:  # noqa: ANN401 - any JSON-shaped record
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(nr._finite(record), indent=2, allow_nan=False) + "\n")


def _speed_noise_point(tau_s: float | None) -> dict[str, Any]:
    """Grade the stage-2 answer on its own wiring with slow speed noise at ``tau_s``, or without."""
    staged = json.loads(STAGED_PILOT_PATH.read_text())
    z = list(staged["stage2_extended"]["answer"])
    _BRAIN_SETTINGS["sign_override_exclude"] = ["lopezcruz2019"]
    if tau_s is not None:
        _BRAIN_SETTINGS["encoder_speed_noise"] = "ou"
        z.append(entries(speed_noise=True)[-1].unit(tau_s))
    seeds = [SPEED_NOISE_BASE + i for i in range(ANIMALS)]
    return {"tau_s": tau_s, "graded": grade_seeds(z, seeds)}


def _speed_median(z: list[float]) -> float:
    """Return the answer's speed output's pre-squash median over one reference episode."""
    import torch
    from quantumnematode.connectome import encoder as enc
    from quantumnematode.evaluation.reference import run_reference
    from quantumnematode.utils.config_loader import load_simulation_config

    _BRAIN_SETTINGS["sign_override_exclude"] = ["lopezcruz2019"]
    brain = encoded_brain()
    enc.write_encoder(brain, z)
    path = FORAGING / f"{bg.ARMS[ARM]}_assay_scheer_lawn.yml"
    config = load_simulation_config(str(path))
    episode = run_reference(config, brain, SPEED_NOISE_REFERENCE_SEED, SPEED_NOISE_REFERENCE_STEPS)
    states = torch.from_numpy(np.asarray(episode.features, dtype=np.float32))
    topology = brain.topology
    with torch.no_grad():
        mean, _ = topology.forward_with_hidden_batched(
            states[:, : topology.n_food_features],
            modulators=brain._unpack_modulators(states),
        )
    return float(mean[:, 0].median())


def _defined(value: float | None) -> bool:
    return value is not None and math.isfinite(value)


def read_speed_noise_control(
    kinematics: list[tuple[float | None, dict[str, float | None]]],
    scales: dict[str, float],
    speed_median: float,
) -> dict[str, Any]:
    """Read the positive control from each point's kinematics: independent noise first.

    A rise or a nested difference that needs an undefined statistic is undefined, and fails its
    check; the control passes only on an answer whose speed output responds.
    """
    (_, independent), *slow = kinematics
    shortest = slow[0][1]

    def scaled(a: float | None, b: float | None, name: str) -> float | None:
        if not (_defined(a) and _defined(b)):
            return None
        return (cast("float", a) - cast("float", b)) / scales[name]

    name = "roaming_speed_p90"
    rise = {str(tau): scaled(k[name], shortest[name], name) for tau, k in slow}
    nested = {
        stat: (None if (d := scaled(shortest[stat], value, stat)) is None else abs(d))
        for stat, value in independent.items()
    }
    finite = {tau: v for tau, v in rise.items() if v is not None}
    best = max(finite, key=lambda tau: finite[tau]) if finite else None
    at_best = None if best is None else dict(slow[list(rise).index(best)][1])
    responsive = abs(speed_median) <= SPEED_NOISE_RESPONSIVE
    passes_rise = best is not None and finite[best] >= SPEED_NOISE_RISE_SCALES
    passes_nested = all(v is not None and v <= SPEED_NOISE_NESTED_SCALES for v in nested.values())
    return {
        "speed_pre_squash_median": speed_median,
        "speed_responsive": responsive,
        "roaming_p90_rise_scales": rise,
        "best_tau_s": None if best is None else float(best),
        "dwelling_p90_at_best": None if at_best is None else at_best.get("dwelling_speed_p90"),
        "nested_scales": nested,
        "passes_rise": passes_rise,
        "passes_nested": passes_nested,
        "passes": bool(responsive and passes_rise and passes_nested),
    }


def _speed_noise_control(workers: int) -> int:
    """Grade the stage-2 answer at each timescale and read the positive control's bar."""
    from concurrent.futures import ProcessPoolExecutor

    points: list[float | None] = [None, *SPEED_NOISE_TAUS_S]
    with ProcessPoolExecutor(max_workers=min(workers, len(points))) as pool:
        graded = list(pool.map(_speed_noise_point, points))
    bands = bg.load_bands_v3()
    scales = {k: v["scale"] for k, v in bands["bands"][str(ANIMALS)]["kinematics"].items()}
    kinematics = [
        (row["tau_s"], {k: v["value"] for k, v in row["graded"]["families"]["kinematics"].items()})
        for row in graded
    ]
    staged = json.loads(STAGED_PILOT_PATH.read_text())
    reading = read_speed_noise_control(
        kinematics,
        scales,
        _speed_median(list(staged["stage2_extended"]["answer"])),
    )
    record = {
        "description": (
            "Description: slow speed noise's positive control. C.2's stage-2 answer on the wiring it "
            "was fitted on (the López-Cruz overrides excluded), graded with presence and the mutants "
            f"on {ANIMALS} animals of seed block {SPEED_NOISE_BASE:,}, the same animals with "
            "independent noise and at each timescale. It passes when the answer's speed output "
            f"responds (its pre-squash median over one {SPEED_NOISE_REFERENCE_STEPS}-step episode, "
            f"seed {SPEED_NOISE_REFERENCE_SEED:,}, within ±{SPEED_NOISE_RESPONSIVE:g}), the roaming "
            f"speed's 90th percentile rises by at least {SPEED_NOISE_RISE_SCALES:g} band scales "
            "above its value at the shortest timescale at some timescale, and at the shortest every "
            f"kinematics statistic is within {SPEED_NOISE_NESTED_SCALES:g} band scales of "
            "independent noise. An undefined statistic fails the check that needs it."
        ),
        "seeds": f"{SPEED_NOISE_BASE:,} + i, i < {ANIMALS}; reference {SPEED_NOISE_REFERENCE_SEED:,}",
        "taus_s": list(SPEED_NOISE_TAUS_S),
        **reading,
        "points": graded,
    }
    SPEED_NOISE_PATH.parent.mkdir(parents=True, exist_ok=True)
    SPEED_NOISE_PATH.write_text(json.dumps(nr._finite(record), indent=2, allow_nan=False) + "\n")
    shown = ("speed_pre_squash_median", "roaming_p90_rise_scales", "nested_scales", "passes")
    print(json.dumps({k: record[k] for k in shown}))
    return 0 if record["passes"] else 1


# ---------------------------------------------------------------------------------------------
# The fitting procedure's options, and the probe answer's reproduction
# ---------------------------------------------------------------------------------------------

# The procedure's encoder: per-ligand tying, the soft-capped readout with pool centres, the graded
# box with an inhibition floor, PDFR-1's and SER-5's cited signs, and slow speed noise.
PROCEDURE_SETTINGS: dict[str, Any] = {
    "encoder_modulator_tying": "per_ligand",
    "encoder_readout": "soft_cap",
    "encoder_pool_centres": True,
    "encoder_box": "graded",
    "encoder_inhibition_floor": 0.09,
    "encoder_speed_noise": "ou",
    "modulator_receptor_signs": ["pdfr-1", "ser-5"],
}
PROCEDURE_OPTIONS: dict[str, Any] = {
    "speed_noise": True,
    "box": "graded",
    "inhibition_floor": 0.09,
    "pool_centres": True,
}
PROBE_ANSWER_PATH = (
    _REPO
    / "docs/experiments/logbooks/supporting/129-c2-wild-type-and-mutants/probes/grade_nudged.json"
)
REPRODUCE_PATH = (
    _REPO / "docs/experiments/logbooks/supporting/130-c2-encoder-procedure/reproduce.json"
)
REPRODUCE_GRADE_BASE = 795_000_000
REPRODUCE_REFERENCE_SEED = 99_860_000
REPRODUCE_STEPS = {"scheer_lawn": 2_400, "food_removal_lopezcruz": 5_400}
REPRODUCE_TOLERANCE = 1e-5
_PROBE_CAP = 2.0
_PROBE_WIRING = ("wiring.excitatory", "wiring.inhibitory", "wiring.gap")


def _probe_layout() -> tuple[list[Any], Any]:
    """Return the probe scripts' base entries (per-ligand, default layer) and their timescale."""
    from quantumnematode.connectome import encoder as enc

    base = enc.encoder_entries(layer_spec(), "per_ligand")
    tau = enc.encoder_entries(layer_spec(), "per_ligand", speed_noise=True)[-1]
    return base, tau


def probe_values(probe_z: list[float] | np.ndarray) -> dict[str, float]:
    """Return each value a probe-layout vector writes, by entry name.

    The probe scripts' vectors carry the base entries under per-ligand tying on the default layer,
    with the three wiring scales on [0, 0.75] and the sensory gains on [-2, 2] by their own map,
    then the four pool centres (-1 + 2u), then the speed noise's timescale.
    """
    from quantumnematode.connectome import encoder as enc

    base, tau = _probe_layout()
    n = len(base)
    values: dict[str, float] = {}
    for entry, u in zip(base, probe_z[:n], strict=True):
        if entry.name in _PROBE_WIRING:
            values[entry.name] = 0.75 * float(u)
        elif entry.name.startswith("sensory."):
            values[entry.name] = -2.0 + 4.0 * float(u)
        else:
            values[entry.name] = entry.value(float(u))
    for pool, u in zip(enc.POOLS, probe_z[n : n + len(enc.POOLS)], strict=True):
        values[f"readout.centre.{pool}"] = -1.0 + 2.0 * float(u)
    values[enc.SPEED_NOISE_ENTRY] = tau.value(float(probe_z[n + len(enc.POOLS)]))
    return values


def package_vector(probe_z: list[float] | np.ndarray, entries: list[Any]) -> np.ndarray:
    """Map a probe-layout vector onto a package encoder's entries, entry by entry by name.

    A couple signed at load from a cited receptor coupling is named
    ``modulator.fixed.<ligand>.<receptor>`` in the package; in the probe it was the ligand's fitted
    group, ``modulator.fitted.<ligand>``, held non-negative by a bound. Its value moves onto the
    fixed-strength log range, raised to that range's floor if below it.
    """
    values = probe_values(probe_z)
    out = []
    for entry in entries:
        if entry.name in values:
            value = values[entry.name]
        else:
            kind, ligand = entry.name.split(".")[1:3]
            if kind != "fixed":
                msg = f"no probe value for {entry.name}"
                raise KeyError(msg)
            value = max(values[f"modulator.fitted.{ligand}"], entry.low)
        out.append(min(1.0, max(0.0, entry.unit(value))))
    return np.asarray(out)


def _probe_brain(probe_z: list[float] | np.ndarray) -> Any:  # noqa: ANN401 - a brain
    """Return the probe scripts' brain: per-ligand tying and slow noise, the probe vector written."""
    import torch
    from quantumnematode.connectome import encoder as enc

    brain = encoded_brain((), encoder_modulator_tying="per_ligand", encoder_speed_noise="ou")
    base, _ = _probe_layout()
    n = len(base)
    units = []
    for entry, u in zip(base, probe_z[:n], strict=True):
        if entry.name in _PROBE_WIRING:
            units.append(float(u) * 0.75 / 1.5)
        elif entry.name.startswith("sensory."):
            units.append((-2.0 + 4.0 * float(u) + 5.0) / 10.0)
        else:
            units.append(float(u))
    units.append(float(probe_z[n + len(enc.POOLS)]))
    enc.write_encoder(brain, units)
    centres = np.asarray(probe_z[n : n + len(enc.POOLS)], dtype=np.float32)
    brain.topology.probe_centres = torch.from_numpy(-1.0 + 2.0 * centres)
    return brain


def _probe_forward(self: Any, h: Any, modulators: Any) -> tuple[Any, Any]:  # noqa: ANN401 - the topology and its tensors
    """Settle and read out as the probe scripts' patched forward did: centres, readout, then cap."""
    import torch
    from quantumnematode.brain.arch import connectome_ppo as cp

    encoder = self.encoder
    recurrent = encoder.chemical() + self.g_gap * encoder.gap_scale()
    held = self._held_current(modulators)
    for _ in range(self.forward_pass_depth):
        preact = h @ recurrent
        if held is not None:
            preact = preact + held
        h = cp._settling_tanh(preact)
    pooled = self._pool_motor(h) - self.probe_centres
    mean = pooled @ self.readout.T + self.readout_bias
    return _PROBE_CAP * torch.tanh(mean / _PROBE_CAP), h


def _with_forward(probe: bool) -> Any:  # noqa: ANN401, FBT001 - a context manager; the probe's or the package's
    """Return a context in which the topology's encoded forward is the probe's when ``probe``."""
    import contextlib

    from quantumnematode.brain.arch import connectome_ppo as cp

    @contextlib.contextmanager
    def swapped() -> Any:  # noqa: ANN401 - a generator context
        original = cp.ConnectomeTopology._forward_encoded
        if probe:
            cp.ConnectomeTopology._forward_encoded = _probe_forward  # type: ignore[method-assign]
        try:
            yield
        finally:
            cp.ConnectomeTopology._forward_encoded = original  # type: ignore[method-assign]

    return swapped()


def _episode(brain: Any, protocol: str, *, probe: bool) -> Any:  # noqa: ANN401 - a brain and its episode
    """Run one reference episode, with the probe's forward in place when ``probe``."""
    from quantumnematode.evaluation.reference import run_reference
    from quantumnematode.utils.config_loader import load_simulation_config

    config = load_simulation_config(str(FORAGING / f"{bg.ARMS[ARM]}_assay_{protocol}.yml"))
    with _with_forward(probe):
        return run_reference(config, brain, REPRODUCE_REFERENCE_SEED, REPRODUCE_STEPS[protocol])


def _forward_outputs(brain: Any, states: Any, *, probe: bool) -> np.ndarray:  # noqa: ANN401 - a brain and its states
    """Return the pre-squash outputs a brain's forward gives on recorded states."""
    import torch

    topology = brain.topology
    with _with_forward(probe), torch.no_grad():
        mean, _ = topology.forward_with_hidden_batched(
            states[:, : topology.n_food_features],
            modulators=brain._unpack_modulators(states),
        )
    return mean.numpy()


def reproduce_forwards(probe_z: list[float] | np.ndarray) -> dict[str, Any]:
    """Compare the package's forward with the probe's on the probe answer, episode by episode."""
    import torch
    from quantumnematode.connectome import encoder as enc

    probe = _probe_brain(probe_z)
    package = encoded_brain((), **PROCEDURE_SETTINGS)
    enc.write_encoder(package, package_vector(probe_z, list(package.topology.encoder.entries)))
    out: dict[str, Any] = {}
    for protocol in REPRODUCE_STEPS:
        theirs = _episode(probe, protocol, probe=True)
        ours = _episode(package, protocol, probe=False)
        states = torch.from_numpy(np.asarray(theirs.features, dtype=np.float32))
        diff = np.abs(
            _forward_outputs(package, states, probe=False)
            - _forward_outputs(probe, states, probe=True),
        )
        same = [
            str(a.primitive) == str(b.primitive)
            for a, b in zip(theirs.behaviour, ours.behaviour, strict=True)
        ]
        out[protocol] = {
            "steps": len(states),
            "max_output_difference": float(diff.max()),
            "primitives_identical": all(same),
            "first_differing_step": None if all(same) else same.index(False),
        }
    out["passes"] = all(
        r["max_output_difference"] <= REPRODUCE_TOLERANCE and r["primitives_identical"]
        for r in out.values()
    )
    return out


def _procedure_controller(z: np.ndarray, mutant: str | None = None) -> Callable[[Path], Any]:
    """Write a package vector, or a mutant's encoding of it, into a procedure brain."""
    from quantumnematode.connectome import encoder as enc
    from quantumnematode.evaluation.controllers import BatchedConnectome

    brain = encoded_brain(KNOCKOUT_GENES.get(mutant or "", ()), **PROCEDURE_SETTINGS)
    spec = brain.topology.modulator_spec
    tax4 = mutant == "tax-4"
    enc.write_encoder(
        brain,
        enc.tax4_vector(spec, z, "per_ligand", **PROCEDURE_OPTIONS) if tax4 else z,
    )
    return lambda _path: BatchedConnectome(brain)


def _reproduce_wild(item: tuple[int, list[float]]) -> tuple[int, dict[str, Any]]:
    g, z = item
    bands = bg.load_bands_v4()
    seeds = population_seeds(REPRODUCE_GRADE_BASE, g)
    controller_of = _procedure_controller(np.asarray(z))
    populations, chemotaxis = bg.simulate_v2(ARM, controller_of, seeds, bands)
    graded = bg.grade_populations_v4(populations, chemotaxis, bands, presence=True)
    clf = bg.classifier(bands)
    return g, nr._finite(
        {
            "wild_type_passes": graded["wild_type_passes"],
            "D": graded["D"],
            "fails": {
                f"{f}/{n}": r["z"]
                for f, rs in graded["families"].items()
                for n, r in rs.items()
                if not r["passes"]
            },
            "small_lawn_values": bg.family_values("small_lawn", populations["small_lawn"], clf),
        },
    )


def _reproduce_mutant(
    item: tuple[int, str, list[float], dict[str, float]],
) -> tuple[int, str, bool]:
    g, mutant, z, wild = item
    bands = bg.load_bands_v4()
    clf = bg.classifier(bands)
    seeds = population_seeds(REPRODUCE_GRADE_BASE, g)
    populations, _ = bg.simulate_v2(
        ARM,
        _procedure_controller(np.asarray(z), mutant),
        seeds,
        bands,
        assays=("small_lawn",),
    )
    values = bg.family_values("small_lawn", populations["small_lawn"], clf)
    return g, mutant, bool(mutant_reading(mutant, wild, values, bands, ANIMALS)["passes"])


def _reproduce(workers: int) -> int:
    """Reproduce the probe answer in the package: the forward parity, then a five-population grade."""
    probe_record = json.loads(PROBE_ANSWER_PATH.read_text())
    probe_z = probe_record["answer"]
    forwards = reproduce_forwards(probe_z)
    print(json.dumps(forwards, indent=2), flush=True)
    package = encoded_brain((), **PROCEDURE_SETTINGS)
    z = package_vector(probe_z, list(package.topology.encoder.entries)).tolist()
    with ProcessPoolExecutor(max_workers=workers, initializer=nr._worker_setup) as pool:
        wilds = dict(pool.map(_reproduce_wild, [(g, z) for g in range(GRADE_POPULATIONS)]))
        jobs = [(g, m, z, wilds[g]["small_lawn_values"]) for g in wilds for m in MUTANTS]
        for g, mutant, passes in pool.map(_reproduce_mutant, jobs):
            wilds[g].setdefault("mutants", {})[mutant] = passes
    theirs = probe_record["grade"]["populations"]
    grade = {
        str(g): {
            "package": {k: v for k, v in w.items() if k != "small_lawn_values"},
            "probe": {
                "wild_type_passes": theirs[str(g)]["wild_type_passes"],
                "D": theirs[str(g)]["D"],
                "mutants": {m: r["passes"] for m, r in theirs[str(g)]["mutants"].items()},
            },
        }
        for g, w in sorted(wilds.items())
    }
    record = {
        "description": (
            "Description: the probe procedure's passing answer (Logbook 129, grade_nudged.json) "
            "converted into the package encoding. The forward parity is the gate; the "
            "five-population grade, on the probe's own populations, is reported beside the probe's."
        ),
        "settings": PROCEDURE_SETTINGS,
        "forwards": forwards,
        "grade": grade,
        "package_wild_type_passing": sum(bool(w["wild_type_passes"]) for w in wilds.values()),
        "probe_wild_type_passing": sum(bool(t["wild_type_passes"]) for t in theirs.values()),
    }
    _write(REPRODUCE_PATH, record)
    print(
        f"forward parity {forwards['passes']} | wild type: package "
        f"{record['package_wild_type_passing']} of 5, probe {record['probe_wild_type_passing']}",
        flush=True,
    )
    return 0 if forwards["passes"] else 1


def main(argv: list[str] | None = None) -> int:
    """CLI: the cost pilot, the feasibility check, the readiness probe, its reading, a smoke."""
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument(
        "stage",
        choices=(
            "cost",
            "feasibility",
            "pilot",
            "pilot-uncompressed",
            "reduced",
            "pilot-log",
            "pilot-staged",
            "stage2-anyway",
            "stage2-extend",
            "speed-noise",
            "sizing",
            "readiness",
            "read",
            "smoke",
            "reproduce",
        ),
    )
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args(argv)
    stages: dict[str, Callable[[], int]] = {
        "cost": lambda: _cost(args.workers),
        "feasibility": _feasibility,
        "pilot": lambda: _pilot(args.workers),
        "pilot-uncompressed": lambda: _uncompressed_pilot(args.workers),
        "reduced": lambda: _reduced(args.workers),
        "pilot-log": lambda: _log_pilot(args.workers),
        "pilot-staged": lambda: _staged_pilot(args.workers),
        "stage2-anyway": lambda: _stage2_anyway(args.workers),
        "stage2-extend": lambda: _extend_stage2(args.workers),
        "speed-noise": lambda: _speed_noise_control(args.workers),
        "sizing": _sizing,
        "readiness": lambda: _readiness(args.workers),
        "read": _read,
        "smoke": lambda: _smoke(args.workers),
        "reproduce": lambda: _reproduce(args.workers),
    }
    return stages[args.stage]()


if __name__ == "__main__":
    sys.exit(main())
