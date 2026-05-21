"""Co-evolution loop orchestrator.

Composes two `EvolutionLoop`-shaped sides (prey + predator) under an
alternating-schedule controller. Key design choices:

- **Optimiser:** Both sides use `CMAESOptimizer(diagonal=True)`
  (sep-CMA-ES). TPE-style optimisers are rejected because the weight
  encoders return unbounded `genome_bounds`, which TPE cannot sample.
- **Predator gen-0 bootstrap:** Either heuristic-imitation pretrain
  (arm A) or cold-start zeros (arm B). The pretrain runs inside
  `__init__` on arm-A construction; cost is amortised across the
  `K_per_block * generation_pairs` generations of the run.
- **Asymmetric fitness + inheritance:** Prey side uses
  `LearnedPerformanceFitness` + `LamarckianInheritance` (large policy
  space, K-train inner loop matters); predator side uses
  `PredatorEpisodicKillRate` + `NoInheritance` (small policy space,
  CMA-ES outer-loop weight gradient suffices). Asymmetry intentional
  and pinned in `__init__` (NOT YAML-configurable — making it so
  invites footguns like predator using `LearnedPerformanceFitness`
  blowing the compute budget).
- **Prey gen-0 warmstart:** Optional warmstart genome JSON loaded
  from `prey_gen0_seed_path` and passed to `CMAESOptimizer(x0=...)`.
  When the path is None, the prey side starts from `x0=zeros`.
- **Schema validation:** YAML schema lives at
  `quantumnematode.utils.config_loader.CoevolutionConfig` with model
  validators enforcing all of the above at load time.

The orchestrator ships in two phases: this module ships the loop
itself (alternating schedule, HoF opposition sampling, generality
probe schema, checkpoint/resume); the campaign-level integration
(opposition end-to-end weight injection, multiprocessing pool
dispatch, smoke config) is layered on top of these primitives via
the call-site sim_config patching documented below.

Opposition injection (call-site responsibility)
-----------------------------------------------
The per-evaluation pattern samples opposition via
`hof.mix_with_pop(rng, opposing_population, frac_hof=0.3)` and the
opposition genomes are recorded on the candidate's evaluation. The
final `sim_config` patching that translates opposition genomes into
env-side opponents (decoded predator brains installed on
`Predator.brain` slots, decoded prey brains exposed via
`sim_config.multi_agent.agents`) is the campaign integration layer's
responsibility. The loop itself calls `fitness.evaluate(...)` with
the configured `sim_config` and records opposition for lineage; the
caller is expected to patch `sim_config` per-evaluation when the
integrated path is wired.

Champion history schema
-----------------------
After each K-block ends, the training-side block elite is appended to
`champion_history` as a dict ``{"genome_id": str, "generation": int,
"k_block_index": int, "fitness": float, "params": list[float]}``.
Serialised at checkpoint time as JSON via `params.tolist()`; restored
via `np.asarray(d["params"], dtype=np.float32)` per the convention
shared with `Genome.params`. Post-hoc analysis tooling walks this
history for cycling and escalation detection.
"""

from __future__ import annotations

import csv
import json
import logging
import tempfile
import time
from contextlib import suppress
from dataclasses import dataclass, field
from multiprocessing import Pool
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np

from quantumnematode.brain.weights import save_weights
from quantumnematode.evolution.encoders import (
    LSTMPPOEncoder,
    build_birth_metadata,
)
from quantumnematode.evolution.fitness import LearnedPerformanceFitness
from quantumnematode.evolution.genome import Genome, genome_id_for
from quantumnematode.evolution.hall_of_fame import HallOfFame
from quantumnematode.evolution.inheritance import (
    LamarckianInheritance,
    NoInheritance,
)
from quantumnematode.evolution.lineage import LineageTracker
from quantumnematode.evolution.predator_encoders import MLPPPOPredatorEncoder
from quantumnematode.evolution.predator_fitness import PredatorEpisodicKillRate
from quantumnematode.optimizers.evolutionary import CMAESOptimizer
from quantumnematode.utils.config_loader import (
    AgentConfig,
    MultiAgentConfig,
)

if TYPE_CHECKING:
    from multiprocessing.pool import Pool as PoolType

    from quantumnematode.brain.arch._brain import Brain
    from quantumnematode.evolution.encoders import GenomeEncoder
    from quantumnematode.evolution.fitness import FitnessFunction
    from quantumnematode.evolution.inheritance import InheritanceStrategy
    from quantumnematode.utils.config_loader import (
        CoevolutionConfig,
        EnvironmentConfig,
        EvolutionConfig,
        SimulationConfig,
    )

# Re-use `EvolutionLoop`'s worker init + 12-tuple worker entry point so
# the dispatch ABI stays single-source. Both functions are top-level in
# `loop.py` so they pickle cleanly across the Pool's worker boundary
# (spawn on macOS, fork on Linux per Python's `multiprocessing.Pool`
# default; both require picklable worker entry points).
from quantumnematode.evolution.loop import (
    _evaluate_in_worker,
    _init_worker,
)

logger = logging.getLogger(__name__)


# Default Hall-of-Fame capacity (8 entries). With the 70% current-pop
# / 30% HoF mix, this preserves live signal while preventing
# catastrophic forgetting of strong past champions.
DEFAULT_HOF_CAPACITY = 8

# Checkpoint format version. Bumped when the on-disk shape changes;
# resume rejects mismatched versions to prevent silent state corruption.
# Distinct from `EvolutionLoop.CHECKPOINT_VERSION` — they evolve
# independently because the shapes differ (single-population pickle vs
# 4-file co-evolution split).
CHECKPOINT_VERSION = 1

# Repo-root-anchored path to the prey held-out bundle (committed
# in-repo so a fresh checkout can run the campaign reproducibly).
# Resolved from this file's location rather than the cwd, so the loop
# works regardless of where the campaign driver was launched. The
# number of `.parents[N]` hops matches the pattern in
# `validation/datasets.py` (4 parents: this file → evolution/ →
# quantumnematode/ → packages/quantum-nematode/ → repo root). The
# campaign driver may override `CoevolutionLoop._PREY_HELD_OUT_BUNDLE_DIR`
# (e.g. for multi-bundle ablations), and tests likewise.
# Unified prey reference bundle: same source genomes serve as both
# the prey gen-0 warmstart anchor (per-seed `seed_{N}.json` consumed
# by `_load_prey_warmstart`) AND the held-out probe opponents
# (consumed by `_load_held_out_prey_bundle`). The two roles share the
# same provenance JSONs by design — the curation script writes one
# bundle dir; both loaders read from it.
_DEFAULT_PREY_HELD_OUT_BUNDLE_DIR = (
    Path(__file__).resolve().parents[4] / "configs" / "evolution" / "coevolution_warmstart_prey"
)


# Prey-side probe env override. The probe measures whether the prey
# elite generalises to opponents it never saw in training. For that
# diagnostic to be informative the probe env must be DIFFICULTY-CALIBRATED
# to the substrate's known-survivable range — otherwise hard-env training
# automatically nukes the diagnostic (every prey scores 0.0 regardless
# of training quality; substrate ceiling, not overfitting).
#
# These values were calibrated by running the trained held-out prey
# bundle through frozen-weight `EpisodicSuccessRate` across a grid of
# heuristic predator radii:
#   - count=2, speed=0.5, grid=20 -> mean fitness 0.531 across 24 cells
#     (8 radius specs x 3 unique genomes x 10 episodes). Range 0.07-0.93.
#   - count=4, speed=1.0, grid=16 (high-pressure variant) -> mean
#     fitness 0.000 across all 24 cells (substrate ceiling - no
#     klinotaxis-LSTMPPO prey can survive).
# The pilot env is therefore the lower-bound discriminative env. Probe
# uses it regardless of training env.
PROBE_ENV_PREDATOR_COUNT = 2
PROBE_ENV_PREDATOR_SPEED = 0.5
PROBE_ENV_GRID_SIZE = 20


# ---------------------------------------------------------------------------
# Side state
# ---------------------------------------------------------------------------


@dataclass
class _SideState:
    """Per-side state bundle (prey or predator).

    Holds everything that lives "on one side" of the co-evolution loop:
    encoder, fitness function, current optimizer, inheritance strategy,
    HoF buffer, latest population, and the unbounded champion history.

    Construction is split between :meth:`CoevolutionLoop.__init__`
    (which builds the encoder + fitness + inheritance + HoF + initial
    optimizer) and :meth:`CoevolutionLoop._rebuild_optimizer` (which
    re-constructs the optimizer at every K-block transition).
    """

    name: Literal["prey", "predator"]
    encoder: GenomeEncoder
    fitness: FitnessFunction
    optimizer: CMAESOptimizer
    inheritance: InheritanceStrategy
    evolution_config: EvolutionConfig
    output_dir: Path
    hof: HallOfFame
    # Filled in once the first K-block runs.
    population: list[Genome] = field(default_factory=list)
    champion_history: list[dict[str, Any]] = field(default_factory=list)
    # Tracks the highest generation index this side has trained
    # through (across all of its K-blocks). Distinct from the loop's
    # K-block counter so per-side lineage rows can be numbered
    # contiguously even though the side trains in K-block bursts.
    generation: int = 0
    # Genome IDs from the most recently completed generation on this
    # side; used to populate `parent_ids` for the next generation's
    # children. Mirrors `EvolutionLoop._prev_generation_ids`.
    prev_generation_ids: list[str] = field(default_factory=list)
    # Eager lineage tracker (one per side; lives at
    # `{output_dir}/{side}/lineage.csv`). Constructed in
    # `_build_{prey,predator}_state` so the CSV header exists from
    # __init__ time onward, matching `EvolutionLoop`'s shape.
    # Typed `Optional` only so the dataclass `__init__` can accept
    # `None` for tests that bypass the standard construction path;
    # production runs always populate it.
    lineage: LineageTracker | None = None
    # Selected parents from the most recently completed generation on
    # this side. Drives per-child inheritance dispatch
    # (`_resolve_per_child_inheritance`): under Lamarckian, child idx i
    # inherits from `selected_parent_ids[inheritance.assign_parent(i)]`
    # via the parent's pre-saved checkpoint. Empty until the first
    # `select_parents` call after K-block 0's generation 0 completes.
    # Mirrors `EvolutionLoop._selected_parent_ids` (per-side analogue).
    selected_parent_ids: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# CoevolutionLoop
# ---------------------------------------------------------------------------


class CoevolutionLoop:
    """Two-population alternating-schedule co-evolution orchestrator.

    Composes two `_SideState` bundles (prey + predator) and drives them
    through `generation_pairs * 2` K-blocks of training, alternating
    between sides. The opposing side's population is FROZEN during a
    K-block (no `optimizer.tell()` on that side); only the training
    side advances.

    Construction
    ------------
    `__init__` performs gen-0 setup for both sides:

    - **Prey:** Loads an optional warmstart genome from
      `coevolution_config.prey_gen0_seed_path`; the genome's
      `params` list is passed as `CMAESOptimizer(x0=...)`. When the
      path is None, the optimiser starts from `x0=zeros` (matching
      the existing cold-start convention).
    - **Predator (arm A):** When
      `coevolution_config.predator_gen0_bootstrap == "heuristic_imitation_pretrain"`,
      runs `pretrain_against_heuristic` inside `__init__` and seeds
      the optimizer with the pretrained weight vector via `x0=...`.
      Cost (~30s) is amortised across the K=10 x `generation_pairs`
      generations of the run.
    - **Predator (arm B):** When
      `coevolution_config.predator_gen0_bootstrap == "cold_start"`,
      seeds the optimizer with `x0=zeros`.

    Fitness is hardcoded per side (NOT YAML-configurable): prey gets
    `LearnedPerformanceFitness`; predator gets
    `PredatorEpisodicKillRate`.

    Inheritance: prey is hardcoded to
    `LamarckianInheritance(elite_count=1)` (single-elite-broadcast,
    matching the upstream Lamarckian-LSTMPPO substrate). Predator
    inheritance is YAML-configurable via
    `predator_evolution.inheritance` and accepts `none` (default;
    legacy frozen-weight contract) or `lamarckian` (predator PPO
    inner-loop path; requires `PredatorEpisodicKillRate.evaluate` to thread
    `warm_start_path_override` + `weight_capture_path`, which it does,
    and brings up the predator brain's PPO inner-loop via the
    multi-agent runner's per-step `learn(reward, episode_done)`
    hook).

    Parameters
    ----------
    sim_config
        Full simulation config. Must have `sim_config.coevolution`
        populated with a validated `CoevolutionConfig`.
    output_dir
        Where to write per-side `lineage.csv` (under `output_dir/prey/`
        and `output_dir/predator/`) plus the top-level
        `coevolution_state.json` checkpoint (lands in subsequent commit).
        Created if it doesn't exist.
    rng
        Numpy `Generator` for non-optimiser RNG (e.g. per-K-block CMA
        seed derivation). Independent of the per-side optimizer seeds.
    log_level
        Forwarded to worker init; matches `EvolutionLoop`'s convention.
    """

    # Class-level attribute so tests + future subclasses can override
    # the bundle directory without monkey-patching module globals.
    # Defaults to the repo-root-anchored path so the loop works
    # regardless of cwd. The campaign driver may override this when
    # running multi-bundle ablations (e.g. seed-specific bundles).
    _PREY_HELD_OUT_BUNDLE_DIR: Path = _DEFAULT_PREY_HELD_OUT_BUNDLE_DIR

    def __init__(
        self,
        sim_config: SimulationConfig,
        output_dir: Path,
        rng: np.random.Generator,
        *,
        log_level: int = logging.WARNING,
    ) -> None:
        if sim_config.coevolution is None:
            msg = (
                "CoevolutionLoop requires sim_config.coevolution to be set. "
                "Either populate the YAML coevolution: block or use "
                "EvolutionLoop for single-population runs."
            )
            raise ValueError(msg)
        self.sim_config = sim_config
        self.coevolution_config: CoevolutionConfig = sim_config.coevolution
        self.output_dir = output_dir
        self.rng = rng
        self.log_level = log_level

        # Immutable run-level seed used by `_derive_optimizer_seed`. Drawn
        # once from the master rng at construction time so the per-K-block
        # optimiser seed is a pure function of `(run_seed, side,
        # k_block_index)` — independent of how many other rng draws have
        # happened in between. This keeps optimiser-seed derivation
        # reproducible regardless of the order of unrelated rng consumers
        # (held-out construction, eval seeds, pretrain seeds).
        self._run_seed: int = int(rng.integers(0, 2**31 - 1))

        self.output_dir.mkdir(parents=True, exist_ok=True)
        # Per-side subdirs for lineage CSVs (matches `EvolutionLoop`'s
        # output_dir convention so single-population analysis tooling
        # reuses unchanged).
        (self.output_dir / "prey").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "predator").mkdir(parents=True, exist_ok=True)

        # Build per-side bundles. Construction order matters: encoder
        # first (drives genome_dim), then fitness, then inheritance,
        # then optimizer (needs genome_dim).
        self.prey = self._build_prey_state()
        self.predator = self._build_predator_state()

        # Probe-time fitness function for prey side (Option 1
        # probe-semantics fix). The training-time `LearnedPerformanceFitness`
        # runs K PPO train episodes against the held-out opponent
        # before L frozen eval, which pushes the elite's policy in
        # unhelpful directions when the opponent class differs from
        # what the prey trained against (the case for heuristic-radius
        # held-out opponents vs MLPPPO training opponents). The probe
        # uses `EpisodicSuccessRate` (frozen-weight L eval only) instead,
        # giving the scientifically correct "what did the prey learn?"
        # measurement. Settable as an attribute so tests can stub it
        # without setting up a full RewardConfig + env-realistic env.
        # Local import to avoid circular: `fitness` imports from
        # `inheritance` which imports from `_predator_brain_factory`
        # which imports from this module's environment.
        from quantumnematode.evolution.fitness import EpisodicSuccessRate

        self._prey_probe_fitness: Any = EpisodicSuccessRate()

        # Alternating-schedule controller state. `_k_block_index`
        # counts completed K-blocks (0 = before first K-block ran);
        # `_current_side` is the side currently training (flips at
        # each K-block boundary).
        self._k_block_index: int = 0
        self._current_side: Literal["prey", "predator"] = self.coevolution_config.start_side

        # Per-K-block mean-fitness history per side, used by the
        # rebalance heuristic. Each side accumulates one entry
        # per completed K-block; the rebalance check at K-block end
        # compares the most-recent K-block means across sides. The
        # buffers are unbounded but small (one float per K-block; 6
        # K-blocks at pilot scale).
        self._k_block_mean_fitness: dict[Literal["prey", "predator"], list[float]] = {
            "prey": [],
            "predator": [],
        }
        # Note: when the rebalance condition fires, the dominant side
        # gets an extra K-block — implemented by NOT flipping
        # `_current_side` at the next K-block boundary. The check
        # consumes `_k_block_mean_fitness` history directly inside
        # `_evaluate_rebalance`; no separate counter state is needed
        # for the single-shot freeze the spec mandates.

        # Generality-probe state. Held-out opponent sets are constructed
        # once at __init__ and never mutated thereafter — opposite of
        # the live `population` and `hof` which both grow / churn during
        # the run. Stored separately per side because the construction
        # rules differ (prey loads from a committed JSON bundle;
        # predator builds heuristic-radius variants).
        held_out_seed = int(self.rng.integers(0, 2**31 - 1))
        self._held_out_rng = np.random.default_rng(seed=held_out_seed)
        self._prey_held_out: list[Genome] = self._load_held_out_prey_bundle()
        self._predator_held_out_specs: list[tuple[int, int]] = self._build_held_out_predator_specs()
        self._probe_csv_path = self.output_dir / "generality_probe.csv"
        # Initialise the probe CSV with its header row at __init__ so
        # post-hoc inspection works even if the loop crashes before any
        # probe fires. RESUME-SAFE: write the header only when the file
        # doesn't already exist; on resume the prior run's rows are
        # preserved so the aggregator's totals reflect the full
        # multi-resume campaign rather than just the post-resume slice.
        if not self._probe_csv_path.exists():
            with self._probe_csv_path.open("w", newline="") as fh:
                writer = csv.writer(fh)
                writer.writerow(["generation", "side", "opponent_index", "fitness"])

        # Wall-time instrumentation. One CSV row per evaluation +
        # per-generation aggregate rows. Captured master-side via
        # `time.perf_counter()` brackets so the worker ABI
        # (`_evaluate_in_worker` returning float) stays stable. The
        # campaign aggregator reads this for the wall-time
        # reconciliation row. Schema:
        #   - scope: "evaluation" or "generation"
        #   - side: "prey" or "predator"
        #   - generation: int (the side's generation counter at write
        #     time; for "evaluation" rows, the gen the eval belongs to)
        #   - index: int (genome index within the gen for "evaluation"
        #     rows; population_size for "generation" rows)
        #   - parallel_workers: int (effective workers for that batch;
        #     1 for sequential, side.evolution_config.parallel_workers
        #     otherwise)
        #   - wall_seconds: float
        # RESUME-SAFE: same header-only-on-fresh-create pattern as
        # the probe CSV above, so the multi-day full run's
        # walltime.csv survives resumes intact.
        self._walltime_csv_path = self.output_dir / "walltime.csv"
        if not self._walltime_csv_path.exists():
            with self._walltime_csv_path.open("w", newline="") as fh:
                writer = csv.writer(fh)
                writer.writerow(
                    [
                        "scope",
                        "side",
                        "generation",
                        "index",
                        "parallel_workers",
                        "wall_seconds",
                    ],
                )

    # ------------------------------------------------------------------
    # Per-side construction
    # ------------------------------------------------------------------

    def _build_prey_state(self) -> _SideState:
        """Construct the prey side's `_SideState` with gen-0 warmstart.

        The prey side's gen-0 elite is loaded from a warmstart genome
        JSON (typically a lamarckian-LSTMPPO elite from a prior
        single-population campaign). When `prey_gen0_seed_path` is
        None, falls back to `x0=zeros` — useful for tests and
        non-warmstart pilots.
        """
        cfg = self.coevolution_config
        encoder = LSTMPPOEncoder()
        fitness = LearnedPerformanceFitness()
        inheritance = LamarckianInheritance(elite_count=1)
        hof = HallOfFame(capacity=DEFAULT_HOF_CAPACITY, replacement="quality")
        evolution_cfg = cfg.prey_evolution

        genome_dim = encoder.genome_dim(self.sim_config)
        x0 = self._load_prey_warmstart(cfg.prey_gen0_seed_path, genome_dim)
        optimizer = self._build_optimizer(
            num_params=genome_dim,
            x0=x0,
            evolution_config=evolution_cfg,
            seed=self._derive_optimizer_seed(side="prey", k_block_index=0),
        )

        prey_dir = self.output_dir / "prey"
        return _SideState(
            name="prey",
            encoder=encoder,
            fitness=fitness,
            optimizer=optimizer,
            inheritance=inheritance,
            evolution_config=evolution_cfg,
            output_dir=prey_dir,
            hof=hof,
            # Eager lineage construction (matches `EvolutionLoop` shape).
            # The CSV header is written at construction time; if the
            # side never trains in this run (e.g. test that exercises
            # only one side) the file exists but stays at the header
            # row, which is harmless.
            lineage=LineageTracker(prey_dir / "lineage.csv"),
        )

    def _build_predator_state(self) -> _SideState:
        """Construct the predator side's `_SideState` with gen-0 bootstrap.

        Per the bootstrap-arm config: arm A pretrains via `pretrain_against_heuristic`
        (cost amortised across the run); arm B cold-starts at zeros.
        Pretrain runs inside `__init__` rather than as a pre-computed
        bundle so a fresh checkout can run the campaign without an
        artefact dependency.

        Predator-side inheritance: default `none` (frozen-weight kill-rate
        evolution; CMA-ES outer loop owns the weight gradient). When
        `predator_evolution.inheritance: lamarckian` is set in YAML,
        `LamarckianInheritance(elite_count=1)` is constructed and the
        co-evolution loop wires `warm_start_path_override` + `weight_capture_path`
        through `PredatorEpisodicKillRate.evaluate`'s new kwargs. The
        learning-enabled flag on the predator brain is derived from the
        inheritance kind in the kwargs path — when either inheritance
        kwarg is set, the brain is built with `enable_learning=True` and
        the multi-agent runner's per-step `predator.brain.learn(reward,
        episode_done)` hook fires during evaluation.
        """
        cfg = self.coevolution_config
        encoder = MLPPPOPredatorEncoder()
        fitness = PredatorEpisodicKillRate()
        # Predator inheritance is now YAML-configurable (was previously
        # hardcoded to `NoInheritance()`). `lamarckian` requires the
        # predator brain to support PPO inner-loop training; `none`
        # preserves the legacy frozen-weight contract.
        # Baldwin is not a meaningful predator-side strategy without
        # the prey's hyperparam-evolution machinery, and is rejected
        # here.
        if cfg.predator_evolution.inheritance == "lamarckian":
            inheritance = LamarckianInheritance(elite_count=1)
        elif cfg.predator_evolution.inheritance == "none":
            inheritance = NoInheritance()
        else:
            msg = (
                "Predator-side inheritance must be one of "
                "{'none', 'lamarckian'}; got "
                f"{cfg.predator_evolution.inheritance!r}."
            )
            raise ValueError(msg)
        hof = HallOfFame(capacity=DEFAULT_HOF_CAPACITY, replacement="quality")
        evolution_cfg = cfg.predator_evolution

        genome_dim = encoder.genome_dim(self.sim_config)
        if cfg.predator_gen0_bootstrap == "heuristic_imitation_pretrain":
            x0 = self._pretrain_predator_x0(genome_dim)
        else:
            # arm B: cold-start zeros.
            x0 = [0.0] * genome_dim
        optimizer = self._build_optimizer(
            num_params=genome_dim,
            x0=x0,
            evolution_config=evolution_cfg,
            seed=self._derive_optimizer_seed(side="predator", k_block_index=0),
        )

        predator_dir = self.output_dir / "predator"
        return _SideState(
            name="predator",
            encoder=encoder,
            fitness=fitness,
            optimizer=optimizer,
            inheritance=inheritance,
            evolution_config=evolution_cfg,
            output_dir=predator_dir,
            hof=hof,
            lineage=LineageTracker(predator_dir / "lineage.csv"),
        )

    # ------------------------------------------------------------------
    # Gen-0 init helpers
    # ------------------------------------------------------------------

    def _load_prey_warmstart(
        self,
        path: Path | None,
        genome_dim: int,
    ) -> list[float]:
        """Load a warmstart genome's `params` list, or fall back to zeros.

        File format: `{"genome_id": str, "generation": int,
        "fitness": float, "params": list[float], "brain_config":
        {...}}`. Only `params` is read here; the rest is for
        provenance + aggregator introspection.

        Raises
        ------
        ValueError
            If the file is missing, the params length doesn't match
            `genome_dim`, or the JSON is malformed. Fail-fast at
            construction time — better than silent zero-init when the
            user thought they were warmstarting.
        """
        if path is None:
            logger.info(
                "Prey gen-0: no warmstart path configured; falling back to "
                "x0=zeros (cold-start prey).",
            )
            return [0.0] * genome_dim
        if not path.exists():
            msg = (
                f"prey_gen0_seed_path={path!r} does not exist. "
                "Either provide a valid warmstart genome JSON or set "
                "the field to null in the YAML to fall back to "
                "cold-start zeros."
            )
            raise ValueError(msg)
        with path.open("r") as fh:
            data = json.load(fh)
        if "params" not in data:
            msg = (
                f"prey warmstart {path!r} is missing the required "
                "'params' field; expected schema is "
                "{genome_id, generation, fitness, params: list[float], brain_config}."
            )
            raise ValueError(msg)
        params = data["params"]
        if not isinstance(params, list) or len(params) != genome_dim:
            msg = (
                f"prey warmstart {path!r} has params of length "
                f"{len(params) if isinstance(params, list) else 'N/A'}, "
                f"expected {genome_dim}. The genome_dim is determined by "
                "the prey encoder; the warmstart bundle MUST match the "
                "current prey brain config (`actor_hidden_dim`, "
                "`critic_hidden_dim`, `lstm_hidden_dim`, sensory_modules, "
                "etc.). Re-curate the bundle for the current brain shape "
                "by running `scripts/campaigns/curate_coevolution_prey_bundles.py` "
                "with the matching source-campaign YAML."
            )
            raise ValueError(msg)
        logger.info(
            "Prey gen-0: loaded warmstart genome %s (gen=%s, fitness=%s) from %s",
            data.get("genome_id", "<unknown>"),
            data.get("generation", "<unknown>"),
            data.get("fitness", "<unknown>"),
            path,
        )
        return [float(v) for v in params]

    def _pretrain_predator_x0(self, genome_dim: int) -> list[float]:
        """Run heuristic-imitation pretrain and return flattened weights.

        Constructs the student brain via the same predator factory used
        at runtime (`instantiate_predator_brain_from_sim_config`) so the
        pretrained architecture honours
        `sim_config.environment.predators.brain_config.extra` overrides
        (`actor_hidden_dim`, `critic_hidden_dim`, `num_hidden_layers`).
        Trains it with `pretrain_against_heuristic`, then flattens the
        weights via the same `_select_genome_components` /
        `_flatten_components` path the encoder uses so the resulting
        `x0` is byte-equivalent to what the encoder would produce.

        Cost ~30s at the default 50-batch budget; runs once per
        `CoevolutionLoop` construction (NOT per K-block).
        """
        # Local imports — pretrain helper pulls torch + the heuristic
        # brain machinery; defer until the bootstrap path is actually
        # selected so cold-start runs don't pay the import cost.
        from quantumnematode.env._predator_brain_pretrain import (
            pretrain_against_heuristic,
        )
        from quantumnematode.env.predator_brain import HeuristicPredatorBrain
        from quantumnematode.evolution._predator_brain_factory import (
            instantiate_predator_brain_from_sim_config,
        )
        from quantumnematode.evolution.encoders import (
            _flatten_components,
            _select_genome_components,
        )

        logger.info(
            "Predator gen-0: running heuristic-imitation pretrain bootstrap...",
        )
        seed = int(self.rng.integers(0, 2**31 - 1))
        student = instantiate_predator_brain_from_sim_config(
            self.sim_config,
            seed=seed,
        )
        teacher = HeuristicPredatorBrain()
        pretrain_against_heuristic(student, teacher, seed=seed)

        # Flatten to a list[float] suitable for CMAESOptimizer(x0=...).
        components = _select_genome_components(student.get_weight_components())
        params, _ = _flatten_components(components)
        flat = params.tolist()
        if len(flat) != genome_dim:
            msg = (
                f"predator pretrain produced {len(flat)} weight floats but "
                f"encoder reports genome_dim={genome_dim}. The pretrain "
                "helper must produce a brain whose encoder round-trip matches "
                "the encoder's expected dimension; if these have drifted, "
                "verify MLPPPOPredatorEncoder vs MLPPPOPredatorBrain wire-up "
                "(or that brain_config.extra overrides flow through both)."
            )
            raise ValueError(msg)
        logger.info(
            "Predator gen-0: pretrain complete; encoded into %d floats for x0.",
            len(flat),
        )
        return flat

    # ------------------------------------------------------------------
    # Held-out opponent construction
    # ------------------------------------------------------------------

    def _load_held_out_prey_bundle(self) -> list[Genome]:
        """Load held-out prey genomes from `configs/evolution/coevolution_warmstart_prey/*.json`.

        The held-out probe loader reads from the SAME bundle directory
        as the prey warmstart (`_load_prey_warmstart`): same source
        genomes, two roles. Per-seed warmstart anchors (`seed_{N}.json`,
        consumed at gen-0 init) and held-out probe opponents (consumed
        every `generality_probe_every` gens) are byte-identical files;
        unifying the bundle dir avoids duplicating ~5 MB of JSON in-repo.

        Each JSON file in the bundle directory is expected to follow the
        warmstart fixture format (`{genome_id, generation, fitness,
        params: list[float], brain_config}`). Files whose `params`
        length doesn't match the prey encoder's `genome_dim` are
        skipped with a warning rather than crashing the loop.

        When the directory is empty or missing the prey-side probe is
        a no-op for this run (logged as a one-time warning at
        __init__) — the probe still fires for the predator side, and
        CI smoke runs work without the bundle. The production bundle
        (typically curated from prior single-population elite genomes)
        is shipped as a separate artefact, decoupled from this loader.

        Sampling: when `held_out_size > len(bundle)` we sample WITH
        replacement; when `held_out_size <= len(bundle)` WITHOUT
        replacement. Both via `_held_out_rng.choice` so the same
        master seed reproduces the held-out set.
        """
        cfg = self.coevolution_config
        # Bundle path is repo-root-anchored via the class attribute
        # `_PREY_HELD_OUT_BUNDLE_DIR` (resolved from this file's
        # location, NOT from cwd, so the loop works regardless of
        # where the campaign driver was launched). Tests + the
        # campaign driver can override the class attribute to point
        # at a different bundle for ablations.
        bundle_dir = self._PREY_HELD_OUT_BUNDLE_DIR
        if not bundle_dir.is_dir():
            logger.warning(
                "Prey held-out bundle missing at %s; prey-side probe will be a "
                "no-op for this run. Provide a curated bundle when running the full campaign.",
                bundle_dir,
            )
            return []

        bundle_paths = sorted(bundle_dir.glob("*.json"))
        if not bundle_paths:
            logger.warning(
                "Prey held-out bundle dir %s is empty; prey-side probe will be a no-op.",
                bundle_dir,
            )
            return []

        expected_dim = self.prey.encoder.genome_dim(self.sim_config)
        loaded: list[Genome] = []
        for p in bundle_paths:
            try:
                with p.open("r") as fh:
                    data = json.load(fh)
                params = data["params"]
            except (json.JSONDecodeError, KeyError, OSError) as e:
                logger.warning("Skipping malformed prey held-out file %s: %s", p, e)
                continue
            if not isinstance(params, list) or len(params) != expected_dim:
                logger.warning(
                    "Skipping prey held-out file %s: params length %s != expected %d.",
                    p,
                    len(params) if isinstance(params, list) else "<not a list>",
                    expected_dim,
                )
                continue
            loaded.append(
                Genome(
                    params=np.asarray(params, dtype=np.float32),
                    genome_id=str(data.get("genome_id", p.stem)),
                    parent_ids=[],
                    generation=int(data.get("generation", 0)),
                    birth_metadata=build_birth_metadata(self.sim_config),
                ),
            )

        # Down-sample / up-sample to held_out_size. With-replacement
        # when oversize so the count always matches the configured
        # value; without-replacement when undersize so each held-out
        # opponent is distinct.
        target = cfg.held_out_size
        if not loaded:
            return []
        if target <= len(loaded):
            chosen = self._held_out_rng.choice(len(loaded), size=target, replace=False)
        else:
            chosen = self._held_out_rng.choice(len(loaded), size=target, replace=True)
        return [loaded[int(i)] for i in chosen]

    def _build_held_out_predator_specs(self) -> list[tuple[int, int]]:
        """Build the predator-side held-out heuristic-radius grid.

        Default grid:
        `detection_radius in {4, 6, 8, 10} x damage_radius in {0, 1}`
        = 8 combos at default `held_out_size=8`. Returns a list of
        `(detection_radius, damage_radius)` tuples; the actual
        heuristic-brain instances are constructed lazily inside the
        probe (a tuple is what gets stored / checkpointed).

        Sampling: `held_out_rng.choice` with replacement / without per
        the same convention as the prey loader.
        """
        cfg = self.coevolution_config
        detection_radii = [4, 6, 8, 10]
        damage_radii = [0, 1]
        grid = [(d, dmg) for d in detection_radii for dmg in damage_radii]
        target = cfg.held_out_size
        if target <= len(grid):
            chosen = self._held_out_rng.choice(len(grid), size=target, replace=False)
        else:
            chosen = self._held_out_rng.choice(len(grid), size=target, replace=True)
        return [grid[int(i)] for i in chosen]

    def _reload_prey_held_out_by_ids(self, recorded_ids: list[str]) -> list[Genome]:
        """Reconstruct the prey held-out list on resume by matching genome IDs.

        Used by `_load_checkpoint` to rebuild `_prey_held_out` without
        relying on the post-restore RNG state (which doesn't reproduce
        the construction-time draw — see `_load_checkpoint`'s detailed
        docstring). Walks the same bundle directory `_load_held_out_prey_bundle`
        reads from, builds an index of `{genome_id: Genome}`, and
        returns the list in the order recorded at save time. Failure
        modes:

        - Bundle dir missing: empty `recorded_ids` is the only valid
          shape (the original run had no bundle either); non-empty
          recorded_ids with missing dir means the bundle was deleted
          between save and resume → raise.
        - A recorded ID is not present in the current bundle (file
          was deleted or renamed) → raise. This is the actual
          "bundle drifted" failure mode that was supposed to be
          detected by the cross-check the original code attempted.

        Returns the held-out list in the same order it was saved.
        """
        # Same repo-root-anchored path as `_load_held_out_prey_bundle`.
        bundle_dir = self._PREY_HELD_OUT_BUNDLE_DIR
        if not bundle_dir.is_dir():
            if recorded_ids:
                msg = (
                    f"Resume: prey held-out bundle dir {bundle_dir} is missing "
                    f"but the checkpoint recorded {len(recorded_ids)} held-out "
                    "IDs. The bundle was deleted between save and resume; "
                    "restore it to its prior contents to resume."
                )
                raise ValueError(msg)
            return []

        # Build {genome_id: Genome} from the bundle on disk.
        expected_dim = self.prey.encoder.genome_dim(self.sim_config)
        by_id: dict[str, Genome] = {}
        for p in sorted(bundle_dir.glob("*.json")):
            try:
                with p.open("r") as fh:
                    data = json.load(fh)
            except (json.JSONDecodeError, OSError):
                continue
            params = data.get("params")
            if not isinstance(params, list) or len(params) != expected_dim:
                continue
            gid = str(data.get("genome_id", p.stem))
            by_id[gid] = Genome(
                params=np.asarray(params, dtype=np.float32),
                genome_id=gid,
                parent_ids=[],
                generation=int(data.get("generation", 0)),
                birth_metadata=build_birth_metadata(self.sim_config),
            )

        # Match recorded IDs in order. Any missing ID is a real drift
        # — fail loudly with a diagnostic.
        missing = [gid for gid in recorded_ids if gid not in by_id]
        if missing:
            msg = (
                f"Resume: prey held-out bundle drifted between save and resume. "
                f"The checkpoint recorded {len(recorded_ids)} IDs but {len(missing)} "
                f"are no longer present on disk: {missing}. "
                "Restore the bundle to its prior contents (or accept a fresh "
                "run instead of resuming)."
            )
            raise ValueError(msg)
        return [by_id[gid] for gid in recorded_ids]

    # ------------------------------------------------------------------
    # Per-K-block CMA-ES re-construction
    # ------------------------------------------------------------------

    def _build_optimizer(
        self,
        *,
        num_params: int,
        x0: list[float],
        evolution_config: EvolutionConfig,
        seed: int,
    ) -> CMAESOptimizer:
        """Construct a fresh `CMAESOptimizer(diagonal=True)` for one side.

        Used both at gen-0 (initial construction) and at every K-block
        transition (re-construct rather than reset, since the
        existing optimiser has no public reset method). The fresh
        instance clears stale opposition-conditioned covariance from
        the prior K-block where the opponent was a different
        population.
        """
        return CMAESOptimizer(
            num_params=num_params,
            x0=x0,
            population_size=evolution_config.population_size,
            sigma0=evolution_config.sigma0,
            seed=seed,
            diagonal=evolution_config.cma_diagonal,
        )

    def _derive_optimizer_seed(
        self,
        *,
        side: Literal["prey", "predator"],
        k_block_index: int,
    ) -> int:
        """Derive a per-K-block CMA-ES seed deterministically.

        Pure function of `(self._run_seed, side, k_block_index)`. Does
        NOT consume `self.rng` — `_run_seed` is captured once at
        construction time so two K-blocks (across sides or generations)
        get distinct optimiser samples while a re-run with the same
        master seed reproduces the same trajectory regardless of
        unrelated rng draws elsewhere in the loop.
        """
        # Side-specific offset salts the seed so prey K-block 0 and
        # predator K-block 0 are distinct streams.
        side_offset = 0 if side == "prey" else 1
        # Mix in the k_block_index so every block gets a fresh draw.
        salt = f"{side}-{k_block_index}".encode()
        # Fast deterministic hash: fold the salt's first 8 bytes into a
        # 31-bit integer and XOR with the immutable run seed plus
        # block/side mixing. Avoids importing hashlib for what's a
        # low-stakes seed mix.
        salt_int = int.from_bytes(salt[:8].ljust(8, b"\0"), "big") & 0x7FFFFFFF
        return (self._run_seed ^ salt_int ^ (k_block_index * 31) ^ (side_offset * 17)) & 0x7FFFFFFF

    def _rebuild_optimizer(self, side: _SideState) -> None:
        """Re-construct `side.optimizer` at a K-block transition.

        Carries over the side's current best (= last K-block's elite)
        as the new `x0` so the optimizer continues from the right
        starting point; `sigma0` resets to the configured value (the
        re-construction is the equivalent of "reset" since CMA's `cma`
        library exposes no public reset method).

        Skip-condition: when `side.evolution_config.persist_cma_across_kblocks`
        is True, the rebuild is a no-op — the existing CMA-ES study
        continues with its accumulated covariance + mean across K-block
        boundaries. This is intended for the predator side, where the
        wide-search-distribution restart compounds the slow-convergence
        problem on the ~700-dim weight space.
        """
        if side.evolution_config.persist_cma_across_kblocks:
            logger.info(
                "K-block %d: %s optimizer NOT rebuilt "
                "(persist_cma_across_kblocks=True); continuing existing study "
                "with accumulated covariance.",
                self._k_block_index,
                side.name,
            )
            return
        # Use the most-recent champion as x0 if we have one; otherwise
        # the side hasn't run any K-block yet so fall back to the
        # current optimizer's mean (CMA-ES exposes the running mean
        # via `_es.mean`).
        if side.champion_history:
            x0 = list(side.champion_history[-1]["params"])
        else:
            # CMA's running mean is the best gen-0 anchor when no
            # champion has been recorded yet.
            x0 = list(side.optimizer._es.mean)  # noqa: SLF001 — cma library API
        side.optimizer = self._build_optimizer(
            num_params=len(x0),
            x0=x0,
            evolution_config=side.evolution_config,
            seed=self._derive_optimizer_seed(
                side=side.name,
                k_block_index=self._k_block_index,
            ),
        )
        logger.info(
            "K-block %d: rebuilt %s optimizer (sep-CMA-ES, sigma0=%.4f, x0 from %s).",
            self._k_block_index,
            side.name,
            side.evolution_config.sigma0,
            "champion" if side.champion_history else "running mean",
        )

    # ------------------------------------------------------------------
    # Rebalance heuristic
    # ------------------------------------------------------------------

    def _evaluate_rebalance(
        self,
        training_side: _SideState,
        opposing_side: _SideState,  # noqa: ARG002 — accepted for symmetry + future use
    ) -> bool:
        """Return True if the current side SHALL flip; False to grant an extra K-block.

        Mitigation for the "one side dominates → other's gradient
        saturates" failure mode: when the `rebalance_threshold` knob
        is set and a side's K-block-mean fitness drops below
        `rebalance_threshold * opposing_side_mean`
        for >= 3 consecutive K-blocks, freeze the dominant side for an
        extra K-block (the saturated side keeps training to recover).

        Disabled by default (`rebalance_threshold is None`); pilot may
        enable if domination is observed.

        The check fires AFTER K-block end (so we have at least one
        K-block of fitness data on the just-trained side); needs >= 3
        K-blocks of history on EACH side to evaluate the
        "consecutively below threshold" predicate. Until both sides
        have that history, the heuristic is a no-op (returns True =
        flip normally).

        Returns
        -------
        bool
            True → flip `_current_side` per the standard alternating
            schedule. False → keep `_current_side` (the saturated side
            gets an extra K-block).
        """
        threshold = self.coevolution_config.rebalance_threshold
        if threshold is None:
            return True

        # Both sides need at least 3 K-blocks of history before we can
        # apply the "≥3 consecutive" rule. Until then, flip normally.
        min_history = 3
        prey_history = self._k_block_mean_fitness["prey"]
        predator_history = self._k_block_mean_fitness["predator"]
        if len(prey_history) < min_history or len(predator_history) < min_history:
            return True

        # Pair up the most-recent 3 K-blocks per side (the comparison
        # is cross-side, so we look at each side's last 3 means).
        recent_prey = prey_history[-min_history:]
        recent_predator = predator_history[-min_history:]
        # Saturation predicate: a side is "saturated" if every one of
        # its last 3 K-block means is below `threshold * opposing_mean`
        # (using the corresponding K-block on the opposing side as the
        # comparison point). Both directions checked: prey saturated
        # OR predator saturated would each trigger a freeze of the
        # OTHER side.
        prey_saturated = all(
            p < threshold * pr for p, pr in zip(recent_prey, recent_predator, strict=True)
        )
        predator_saturated = all(
            p < threshold * pr for p, pr in zip(recent_predator, recent_prey, strict=True)
        )

        # Freeze the dominant side: the side OPPOSITE the saturated
        # one keeps training (so the saturated side has an extra
        # block of opposition to learn against).
        #
        # Note on one-block lag: the rebalance only fires when the
        # SATURATED side is also the just-trained side. If the
        # dominant side just trained, we flip to the saturated side
        # normally — the rebalance kicks in on the NEXT K-block
        # boundary (after the saturated side has trained once with
        # the now-frozen dominant side). This is by design: the
        # check is keyed on `training_side.name` because that's the
        # side whose K-block we just observed, and the freeze applies
        # to the next iteration of the same side's training. A
        # one-block delay is acceptable since the saturation predicate
        # already requires >=3 consecutive K-blocks of evidence.
        if prey_saturated and training_side.name == "prey":
            return False  # don't flip - prey gets an extra K-block
        # `predator` saturated branch handled symmetrically.
        return not (predator_saturated and training_side.name == "predator")

    # ------------------------------------------------------------------
    # Checkpoint / resume
    # ------------------------------------------------------------------

    def _save_checkpoint(self) -> None:
        """Persist full loop state across five files (per-side pickles + 2 JSON + RNG pickle).

        Output layout:

        - `{output_dir}/prey/checkpoint.pkl`: per-side pickle containing
          optimizer + population + prev_generation_ids + generation +
          champion_history + k_block_index + inheritance literal.
          Reuses the existing `EvolutionLoop` checkpoint shape so
          single-population resume tooling can introspect.
        - `{output_dir}/predator/checkpoint.pkl`: same shape.
        - `{output_dir}/coevolution_state.json`: top-level JSON
          containing co-evolution-specific state: K-block index,
          alternating-schedule cursor, both HoFs (via `HallOfFame.to_dict`),
          held-out predator specs, prey held-out genome IDs, and
          per-side K-block-mean fitness history (rebalance heuristic).
        - `{output_dir}/champion_history.json`: top-level JSON read by
          aggregator tooling. `{prey: list[dict], predator: list[dict]}`
          plus a `k_block_index` field for cross-file consistency.
        - `{output_dir}/coevolution_rng.pkl`: RNG state for the
          master `rng` and the held-out RNG, plus `k_block_index` as
          the canonical cross-file consistency value. Pickled because
          numpy bit_generator state has nested arrays that don't JSON
          natively. Written LAST: its presence signals "checkpoint
          complete"; its absence triggers `_load_checkpoint` to
          refuse to resume (treats partial writes as never-happened).

        Crash recovery design:

        - Each file is tmp+rename atomic — individual files never
          observe a torn write.
        - The RNG pickle is written LAST. If the process is killed
          between any of the earlier writes, the RNG pickle is
          missing and `_load_checkpoint` refuses to resume.
        - `k_block_index` is embedded in EVERY file as a cross-file
          consistency check. `_load_checkpoint` uses the RNG-pickle
          value as canonical (since it was written last) and
          validates the other four against it. Mismatch implies a
          torn save where the kill landed mid-write across files;
          resume refuses with a diagnostic naming the divergent file.
        """
        # Write order is load-bearing for crash recovery: the four
        # non-RNG files are written first; `coevolution_rng.pkl` is
        # written LAST. If the process is killed between any of the
        # earlier writes, the RNG pickle is missing and `_load_checkpoint`
        # refuses to resume (treats absence as "incomplete checkpoint").
        # Each file is also tmp+rename atomic so individual files
        # never observe a torn write.
        #
        # Cross-file consistency: `k_block_index` is embedded in EVERY
        # file (per-side pickles + JSON state + champion_history JSON +
        # RNG pickle) so `_load_checkpoint` can detect partial-write
        # divergence (e.g. prey pickle at K-block 4, predator pickle at
        # K-block 3 because the kill landed mid-write) by mismatching
        # the `k_block_index` fields across files.

        # Per-side pickles. The optimizer carries ALL CMA-ES adaptive
        # state (covariance, sigma, mean, generation counter inside the
        # cma library); pickling it directly is the same approach
        # `EvolutionLoop._save_checkpoint` uses.
        for side in (self.prey, self.predator):
            payload = {
                "checkpoint_version": CHECKPOINT_VERSION,
                # Cross-file consistency check: `_load_checkpoint`
                # verifies BOTH per-side pickles agree on this value
                # AND that the JSON's `k_block_index` matches.
                "k_block_index": self._k_block_index,
                "side": side.name,
                "optimizer": side.optimizer,
                "population_params": [g.params.tolist() for g in side.population],
                "population_genome_ids": [g.genome_id for g in side.population],
                "prev_generation_ids": list(side.prev_generation_ids),
                "selected_parent_ids": list(side.selected_parent_ids),
                "generation": side.generation,
                "champion_history": list(side.champion_history),
                # Match `EvolutionLoop._save_checkpoint`'s shape: persist
                # the inheritance literal so resume can compare against
                # the resolved current value and reject mid-run drift.
                # In practice `CoevolutionConfig._validate_invariants`
                # hardcodes inheritance per side at YAML load time, so
                # a mid-run config edit is rejected before reaching
                # the loop — but the field travels with the per-side
                # checkpoint anyway, matching single-population
                # tooling's expectations.
                "inheritance": side.evolution_config.inheritance,
            }
            self._atomic_pickle_write(
                side.output_dir / "checkpoint.pkl",
                payload,
            )

        # Top-level JSON for human-readable co-evolution state.
        coevo_state = {
            "checkpoint_version": CHECKPOINT_VERSION,
            "k_block_index": self._k_block_index,
            "current_side": self._current_side,
            "prey_hof": self.prey.hof.to_dict(),
            "predator_hof": self.predator.hof.to_dict(),
            "predator_held_out_specs": [list(spec) for spec in self._predator_held_out_specs],
            "prey_held_out_ids": [g.genome_id for g in self._prey_held_out],
            "k_block_mean_fitness": {
                "prey": list(self._k_block_mean_fitness["prey"]),
                "predator": list(self._k_block_mean_fitness["predator"]),
            },
        }
        self._atomic_json_write(
            self.output_dir / "coevolution_state.json",
            coevo_state,
        )

        # Top-level champion_history.json — the aggregator tooling
        # reads this file. Format: `{prey: list[dict], predator: list[dict]}`
        # where each dict is a champion_history entry as documented in
        # the module docstring (`{genome_id, generation, k_block_index,
        # fitness, params: list[float]}`). Already JSON-serialisable
        # since the per-K-block append in `_run_one_k_block` calls
        # `params.tolist()`.
        champion_payload = {
            # Embed `k_block_index` for cross-file consistency check.
            # Aggregator tooling reads `prey` + `predator` only; the
            # extra field is ignored by aggregator code.
            "k_block_index": self._k_block_index,
            "prey": list(self.prey.champion_history),
            "predator": list(self.predator.champion_history),
        }
        self._atomic_json_write(
            self.output_dir / "champion_history.json",
            champion_payload,
        )

        # RNG state pickle (separate file because numpy bit_generator
        # state is awkward to JSON). Written LAST: its presence
        # signals "checkpoint complete"; its absence triggers
        # `_load_checkpoint` to refuse to resume (treats partial
        # writes as never having happened).
        rng_payload = {
            # Embed `k_block_index` for cross-file consistency check —
            # since the RNG pickle is the last-written file, this is
            # the canonical "expected k_block_index" against which the
            # other four files are validated at load time.
            "k_block_index": self._k_block_index,
            "master_rng_state": self.rng.bit_generator.state,
            "held_out_rng_state": self._held_out_rng.bit_generator.state,
            # Persist `_run_seed` so post-resume calls to
            # `_derive_optimizer_seed` produce the same per-K-block
            # optimiser seeds the original run would have. Construction
            # captures it from the input rng with a one-time draw, but
            # that draw consumes state — re-running construction on
            # resume would advance the post-restore rng past where we
            # want it.
            "run_seed": self._run_seed,
        }
        self._atomic_pickle_write(
            self.output_dir / "coevolution_rng.pkl",
            rng_payload,
        )

    def _load_checkpoint(self) -> None:  # noqa: C901, PLR0912, PLR0915
        """Restore full loop state from the five checkpoint files.

        Complexity note: C901 / PLR0912 / PLR0915 ruff thresholds are
        intentionally exceeded — the method is a sequence of fail-fast
        validation steps (5 file existence checks, 5 cross-file
        consistency checks, version comparisons, restoration assignments).
        Splitting into helpers would fragment the resume contract and
        make the failure modes harder to follow at the call site.

        Inverse of `_save_checkpoint`. Reads per-side pickles to
        restore optimizer + population + champion_history; reads the
        top-level JSON to restore K-block index, side cursor, HoFs;
        reads the RNG pickle to restore master + held-out RNG state.

        Resume invariants verified (in order):

        1. RNG pickle exists. The RNG pickle is written LAST in
           `_save_checkpoint`; its absence implies the prior save was
           interrupted before completing → refuse to resume rather
           than recover from inconsistent intermediate state.
        2. Each per-side checkpoint's `checkpoint_version` matches
           `CHECKPOINT_VERSION`; mismatch raises `ValueError`.
        3. **Cross-file `k_block_index` consistency**: the RNG
           pickle's `k_block_index` is the canonical value (it's
           written last so disagreement implies a torn save). All
           four other files (per-side pickles, state JSON,
           champion_history JSON) MUST agree. Mismatch raises with
           a diagnostic naming the offending file → operator can
           hand-recover by deleting the divergent file and
           re-running with `resume=True` (which will then fail at
           the version check below for the absent file, prompting
           a fresh-start decision).
        4. The two per-side files agree on `inheritance` literal
           with the resolved current value (mid-run inheritance
           changes rejected, matching `EvolutionLoop`'s shape).

        Held-out sets are NOT re-serialised in detail — the prey
        bundle is reconstructed BY ID from
        `configs/evolution/coevolution_warmstart_prey/*.json`
        (`_reload_prey_held_out_by_ids`) so the recorded order is
        preserved without depending on RNG state.
        """
        import pickle

        # Load RNG pickle FIRST so we have the canonical
        # `k_block_index` to cross-check the other files. Absence
        # signals an incomplete save (RNG pickle is written last);
        # refuse to resume rather than recover from inconsistent
        # intermediate state.
        rng_path = self.output_dir / "coevolution_rng.pkl"
        if not rng_path.exists():
            msg = (
                f"CoevolutionLoop._load_checkpoint: RNG state missing at "
                f"{rng_path}. Either the prior save was interrupted "
                "before completing (refusing to resume from inconsistent "
                "state) or the file was deleted manually. Restore the "
                "file from a backup or start a fresh run."
            )
            raise FileNotFoundError(msg)
        with rng_path.open("rb") as fh:
            rng_payload = pickle.load(fh)  # noqa: S301 — trusted local file
        canonical_k_block_index = rng_payload.get("k_block_index")
        if canonical_k_block_index is None:
            msg = (
                "coevolution_rng.pkl missing required `k_block_index` field. "
                "This field is mandatory for cross-file consistency checking; "
                "refusing to resume from a checkpoint without it."
            )
            raise ValueError(msg)

        for side in (self.prey, self.predator):
            ckpt_path = side.output_dir / "checkpoint.pkl"
            if not ckpt_path.exists():
                msg = (
                    f"CoevolutionLoop._load_checkpoint: per-side checkpoint "
                    f"missing at {ckpt_path}. Resume requires both per-side "
                    "files plus coevolution_state.json + coevolution_rng.pkl."
                )
                raise FileNotFoundError(msg)
            with ckpt_path.open("rb") as fh:
                payload = pickle.load(fh)  # noqa: S301 — trusted local file
            version = payload.get("checkpoint_version")
            if version != CHECKPOINT_VERSION:
                msg = (
                    f"Per-side checkpoint version mismatch on {side.name}: "
                    f"expected {CHECKPOINT_VERSION}, got {version}. "
                    "Refusing to resume."
                )
                raise ValueError(msg)
            # Cross-file `k_block_index` consistency: the per-side
            # pickle's value MUST match the canonical RNG pickle.
            # Mismatch implies a torn save (one side's pickle was
            # written before the kill, the other after), which would
            # silently corrupt the alternating-schedule cursor on
            # resume.
            side_k_block = payload.get("k_block_index")
            if side_k_block != canonical_k_block_index:
                msg = (
                    f"Per-side checkpoint k_block_index mismatch on {side.name}: "
                    f"checkpoint={side_k_block!r}, canonical (from RNG pickle)="
                    f"{canonical_k_block_index!r}. This indicates a torn save "
                    "(the prior run was interrupted between writing this "
                    "side's pickle and writing the RNG pickle). Recover by "
                    "deleting the divergent file and accepting a fresh run."
                )
                raise ValueError(msg)
            # Cross-check inheritance literal. `CoevolutionConfig` already
            # rejects mid-run inheritance changes at YAML load time, so
            # this is belt-and-braces — but matching `EvolutionLoop`'s
            # shape avoids surprising future maintainers.
            checkpoint_inheritance = payload.get("inheritance")
            current_inheritance = side.evolution_config.inheritance
            if checkpoint_inheritance is not None and checkpoint_inheritance != current_inheritance:
                msg = (
                    f"Per-side checkpoint inheritance mismatch on {side.name}: "
                    f"checkpoint={checkpoint_inheritance!r}, "
                    f"current={current_inheritance!r}. Mid-run inheritance "
                    "changes are not supported."
                )
                raise ValueError(msg)
            side.optimizer = payload["optimizer"]
            side.generation = int(payload["generation"])
            side.prev_generation_ids = list(payload["prev_generation_ids"])
            # `selected_parent_ids` was added after the initial
            # checkpoint format; tolerate older checkpoints by
            # defaulting to an empty list.
            side.selected_parent_ids = list(payload.get("selected_parent_ids", []))
            side.champion_history = list(payload["champion_history"])
            # Re-build genome instances from population_params +
            # population_genome_ids; birth_metadata is reconstructed
            # from the current sim_config (encoder shape_map is
            # determined by the encoder + sim_config, so re-running
            # `build_birth_metadata` here gives a fresh template that
            # matches what the loop would produce in a non-resume run).
            side.population = [
                Genome(
                    params=np.asarray(params, dtype=np.float32),
                    genome_id=gid,
                    parent_ids=[],
                    generation=side.generation,
                    birth_metadata=build_birth_metadata(self.sim_config),
                )
                for params, gid in zip(
                    payload["population_params"],
                    payload["population_genome_ids"],
                    strict=True,
                )
            ]

        coevo_path = self.output_dir / "coevolution_state.json"
        if not coevo_path.exists():
            msg = (
                f"CoevolutionLoop._load_checkpoint: top-level state missing "
                f"at {coevo_path}. Both per-side checkpoints loaded but the "
                "co-evolution-specific state file is required."
            )
            raise FileNotFoundError(msg)
        with coevo_path.open("r") as fh:
            coevo = json.load(fh)
        if coevo.get("checkpoint_version") != CHECKPOINT_VERSION:
            msg = (
                f"coevolution_state.json version mismatch: expected "
                f"{CHECKPOINT_VERSION}, got {coevo.get('checkpoint_version')!r}. "
                "Refusing to resume."
            )
            raise ValueError(msg)
        # Cross-file `k_block_index` consistency: the JSON's value
        # MUST match the canonical RNG-pickle value loaded at the top.
        json_k_block = coevo.get("k_block_index")
        if json_k_block != canonical_k_block_index:
            msg = (
                f"coevolution_state.json k_block_index mismatch: "
                f"json={json_k_block!r}, canonical (from RNG pickle)="
                f"{canonical_k_block_index!r}. This indicates a torn save "
                "(JSON written before kill, RNG pickle written after a "
                "subsequent successful save, or the JSON is stale). "
                "Recover by deleting the divergent file."
            )
            raise ValueError(msg)
        self._k_block_index = int(coevo["k_block_index"])
        self._current_side = coevo["current_side"]
        self.prey.hof = HallOfFame.from_dict(coevo["prey_hof"])
        self.predator.hof = HallOfFame.from_dict(coevo["predator_hof"])
        self._predator_held_out_specs = [tuple(spec) for spec in coevo["predator_held_out_specs"]]
        # Optional rebalance-history field (added in checkpoint v1; kept
        # `.get` to not break older v1 checkpoints that pre-date the
        # rebalance knob — the `if not in payload` would force a
        # checkpoint regen).
        rebalance_state = coevo.get("k_block_mean_fitness")
        if rebalance_state is not None:
            self._k_block_mean_fitness = {
                "prey": list(rebalance_state.get("prey", [])),
                "predator": list(rebalance_state.get("predator", [])),
            }

        # Cross-check `champion_history.json`'s k_block_index too —
        # the file is a write-only artefact for aggregator tooling,
        # never read on resume, but its `k_block_index` field still
        # participates in the cross-file consistency check so a torn
        # save that wrote per-side pickles + state JSON but failed
        # before champion_history.json + RNG pickle is detected.
        champion_path = self.output_dir / "champion_history.json"
        if champion_path.exists():
            with champion_path.open("r") as fh:
                champion_data = json.load(fh)
            champion_k_block = champion_data.get("k_block_index")
            if champion_k_block != canonical_k_block_index:
                msg = (
                    f"champion_history.json k_block_index mismatch: "
                    f"champion_history={champion_k_block!r}, canonical="
                    f"{canonical_k_block_index!r}. Torn save detected; "
                    "delete the divergent file to recover."
                )
                raise ValueError(msg)

        # RNG state already loaded at the top of this method as the
        # canonical k_block_index source — restore the actual
        # generator states now that we've verified the cross-file
        # consistency invariants. MUST happen BEFORE the held-out
        # bundle reload (next block) so any RNG-dependent
        # construction sees the saved state.
        self.rng.bit_generator.state = rng_payload["master_rng_state"]
        self._held_out_rng.bit_generator.state = rng_payload["held_out_rng_state"]
        # Restore the immutable run seed if present. Older checkpoints
        # (pre run-seed introduction) lack the field — fall back to the
        # construction-time draw rather than failing, since legacy
        # checkpoints couldn't have been written with a stable run seed
        # anyway.
        if "run_seed" in rng_payload:
            self._run_seed = int(rng_payload["run_seed"])

        # Re-sample the prey held-out bundle from disk with the
        # restored `_held_out_rng` state. This MUST run AFTER the RNG
        # restore (above): the bundle was originally sampled with a
        # `_held_out_rng` seeded by the master rng's pre-construction
        # state; on resume the construction-time RNGs are stale, so
        # we re-draw with the post-restore state to reproduce the
        # saved subset. The cross-check below then verifies the
        # recorded IDs match the fresh draw — failure indicates the
        # bundle dir on disk has actually drifted (files added /
        # removed) and resume cannot reproduce the saved set.
        #
        # The `_held_out_rng` state at the moment of the original
        # `__init__` sampling was the FRESHLY-SEEDED state (no draws
        # yet). Restoring from the checkpoint gives us the state
        # AFTER the original sampling drew its `held_out_size`
        # entries plus any subsequent draws (probe seeds, etc.). To
        # reproduce the original sample we need the pre-sampling
        # state, which we don't have. Instead we cross-check IDs
        # only — if the bundle dir is byte-identical to the saved run,
        # `_load_held_out_prey_bundle` (called fresh below with the
        # restored RNG) will draw a SUBSET, but it might not match
        # the recorded IDs. The simplest correct path: skip
        # re-sampling entirely on resume and instead reconstruct the
        # held-out genomes BY ID from the bundle dir, matching the
        # recorded order.
        recorded_ids = list(coevo["prey_held_out_ids"])
        self._prey_held_out = self._reload_prey_held_out_by_ids(recorded_ids)
        logger.info(
            "CoevolutionLoop: resumed from checkpoint (k_block=%d, side=%s, "
            "prey gen=%d, predator gen=%d).",
            self._k_block_index,
            self._current_side,
            self.prey.generation,
            self.predator.generation,
        )

    def _atomic_pickle_write(self, target: Path, payload: object) -> None:
        """Write a pickle atomically via tmp file + rename.

        Mirrors `EvolutionLoop._save_checkpoint`'s pattern. Avoids
        torn writes if the process is killed mid-checkpoint.
        """
        import pickle

        tmp_path = target.with_suffix(target.suffix + ".tmp")
        with tmp_path.open("wb") as fh:
            pickle.dump(payload, fh)
        tmp_path.replace(target)

    def _atomic_json_write(self, target: Path, payload: dict[str, Any]) -> None:
        """Write a JSON file atomically via tmp file + rename."""
        tmp_path = target.with_suffix(target.suffix + ".tmp")
        with tmp_path.open("w") as fh:
            json.dump(payload, fh, indent=2)
        tmp_path.replace(target)

    # ------------------------------------------------------------------
    # Run loop (alternating schedule)
    # ------------------------------------------------------------------

    def run(self, *, resume: bool = False) -> None:
        """Run `generation_pairs * 2` K-blocks under the alternating schedule.

        Each K-block trains one side for `K_per_block` generations
        while the opposing side is FROZEN (no `optimizer.tell()`).
        At the end of every K-block:

        1. The training-side block elite is pushed to its `hof` and
           appended to `champion_history` (in `_run_one_k_block`).
        2. Full state is persisted to the four checkpoint files so a
           resume from this point is byte-identical to a non-interrupted
           continuation.
        3. The current side flips.
        4. The just-flipped (now-training) side's optimizer is
           rebuilt.

        Parameters
        ----------
        resume
            When True, load state from the four checkpoint files
            (`{output_dir}/{prey,predator}/checkpoint.pkl`,
            `{output_dir}/coevolution_state.json`,
            `{output_dir}/coevolution_rng.pkl`) before entering the
            loop. The loop then continues from the last checkpointed
            K-block. When False (default), starts fresh from
            `_k_block_index=0`.
        """
        if resume:
            self._load_checkpoint()

        cfg = self.coevolution_config
        total_blocks = cfg.generation_pairs * 2
        logger.info(
            "CoevolutionLoop: %d K-blocks total (%d generation pairs x 2 sides), "
            "K_per_block=%d, start_side=%s, starting at k_block=%d",
            total_blocks,
            cfg.generation_pairs,
            cfg.K_per_block,
            cfg.start_side,
            self._k_block_index,
        )

        while self._k_block_index < total_blocks:
            training_side = self.prey if self._current_side == "prey" else self.predator
            opposing_side = self.predator if self._current_side == "prey" else self.prey

            logger.info(
                "K-block %d / %d: training side=%s, opposing side=%s (frozen)",
                self._k_block_index,
                total_blocks,
                training_side.name,
                opposing_side.name,
            )

            self._run_one_k_block(training_side, opposing_side)

            # K-block end: flip side (or freeze opposing per the
            # rebalance heuristic), rebuild the newly-training side's
            # optimizer, then persist state. Order matters:
            # 1. Increment k_block_index so post-flip state reflects
            #    the *next* block, not the one that just finished.
            # 2. Decide flip via the rebalance heuristic, update
            #    `_current_side` accordingly.
            # 3. Rebuild the next training side's optimizer FIRST so
            #    the freshly-built optimizer state is what gets pickled.
            #    A crash between step 3 and step 4 just falls back to
            #    rebuilding again on resume; a crash AFTER step 4
            #    resumes from the rebuilt optimizer with no extra work.
            # 4. Save the checkpoint.
            self._k_block_index += 1
            should_flip = self._evaluate_rebalance(training_side, opposing_side)
            if should_flip:
                self._current_side = "predator" if self._current_side == "prey" else "prey"
            else:
                # Saturated side gets an extra K-block — `_current_side`
                # stays the same so the next iteration retrains the
                # losing side (against the same frozen dominant side).
                # Logged so the operator can see the rebalance fired.
                logger.info(
                    "K-block %d: rebalance heuristic kept training side=%s "
                    "(this side has been saturated relative to %s for >=3 K-blocks; "
                    "%s stays frozen for an extra K-block).",
                    self._k_block_index,
                    self._current_side,
                    opposing_side.name,
                    opposing_side.name,
                )
            if self._k_block_index < total_blocks:
                next_training_side = self.prey if self._current_side == "prey" else self.predator
                self._rebuild_optimizer(next_training_side)
            self._save_checkpoint()

        logger.info("CoevolutionLoop: run complete.")

    def _run_one_k_block(  # noqa: C901, PLR0912, PLR0915
        self,
        training_side: _SideState,
        opposing_side: _SideState,
    ) -> None:
        """Run `K_per_block` generations of `training_side` vs frozen `opposing_side`.

        Complexity note: C901 / PLR0912 / PLR0915 thresholds are
        intentionally exceeded — this method is the canonical
        ask → eval → tell → inheritance pipeline (mirroring
        `EvolutionLoop.run`'s structure). Splitting it would fragment
        the per-generation data flow that downstream readers (and the
        spec scenarios) reason about.

        Per-generation flow:

        1. `optimizer.ask()` produces population_size flat float vectors.
        2. Wrap each into a `Genome` with `genome_id_for(...)`; record
           on `training_side.population` for opposition lookup.
        3. For each candidate, build the opposition set via
           `opposing_side.hof.mix_with_pop(rng, opposing.population,
           frac_hof=0.3)` (per spec "70/30 Mixture During Evaluation").
           Empty-HoF fallback returns all-from-pop.
        4. Evaluate the candidate against the opposition. The
           call-site sim_config patching that injects decoded
           opponents is the campaign integration layer's
           responsibility — see module docstring "Opposition
           injection (call-site responsibility)".
        5. `optimizer.tell(solutions, neg_fitnesses)` (negate per
           CMA-ES minimisation convention).
        6. Record lineage row.
        7. Increment `training_side.generation`.

        At K-block end (after the K-th generation):

        - The block elite (highest-fitness genome of the K generations)
          is pushed to `training_side.hof` AND appended to
          `champion_history` per spec "Block Elite Pushed To HoF" +
          "Side State Surface".
        - The opposing side's `population`, `optimizer`, and `hof`
          remain UNTOUCHED throughout (per "Opposing Side Frozen
          During Off-Block").

        Generality probe fires every `coevolution_config.generality_probe_every`
        generations within the block (counted on `training_side.generation`),
        evaluating each side's elite against its held-out set without
        mutating any state (per "Probe Does Not Mutate Population State").
        """
        cfg = self.coevolution_config
        # Lineage tracker is constructed eagerly in
        # `_build_{prey,predator}_state` to match `EvolutionLoop`'s
        # shape. The field type is `LineageTracker | None` (None
        # permitted for tests that bypass the standard construction);
        # explicit None-check narrows the type for the .record() call
        # later in this method.
        if training_side.lineage is None:  # pragma: no cover - standard construction populates
            msg = (
                f"_run_one_k_block: {training_side.name}.lineage is None. "
                "Standard construction (_build_prey_state / _build_predator_state) "
                "populates this; tests that bypass construction must populate "
                "the field manually before calling run()."
            )
            raise RuntimeError(msg)

        # Track the K-block's best so we know what to push to HoF +
        # champion_history at the end. `block_elite_*` are updated each
        # generation when a new best appears.
        block_elite_genome: Genome | None = None
        block_elite_fitness: float = float("-inf")
        # Accumulate all fitnesses across the K-block to compute the
        # K-block mean for the rebalance heuristic.
        block_fitnesses: list[float] = []

        # Optional `multiprocessing.Pool` for parallel candidate evaluation.
        # Pool init mirrors `EvolutionLoop`'s pattern: ignore SIGINT in
        # workers so Ctrl-C propagates through the master process cleanly.
        # When `parallel_workers <= 1` we keep the sequential dispatch path
        # for ease of debugging + test reproducibility.
        pool: PoolType | None = None
        if training_side.evolution_config.parallel_workers > 1:
            pool = Pool(
                processes=training_side.evolution_config.parallel_workers,
                initializer=_init_worker,
                initargs=(self.log_level,),
            )

        try:
            for k_gen in range(cfg.K_per_block):
                global_gen = training_side.generation
                logger.info(
                    "K-block %d (%s) gen %d/%d (side-global gen %d)",
                    self._k_block_index,
                    training_side.name,
                    k_gen,
                    cfg.K_per_block,
                    global_gen,
                )

                solutions = training_side.optimizer.ask()
                parent_ids = list(training_side.prev_generation_ids)

                # Per-generation tmp dir for opposition `.pt` files.
                # Lifetime: must outlive `pool.map`/the sequential
                # comprehension because workers read from it via
                # `load_weights` during fitness evaluation.
                with tempfile.TemporaryDirectory(
                    prefix=f"coevo_gen{global_gen}_",
                ) as tmp_dir_str:
                    tmp_path = Path(tmp_dir_str)
                    eval_args, gen_genomes, inherited_from_per_child = self._build_eval_args(
                        training_side=training_side,
                        opposing_side=opposing_side,
                        solutions=solutions,
                        parent_ids=parent_ids,
                        global_gen=global_gen,
                        tmp_path=tmp_path,
                    )

                    # Dispatch evaluation. The 12-tuple `eval_args` shape
                    # matches `EvolutionLoop._evaluate_in_worker` (12th
                    # slot is `tei_prior_source`; co-evolution always
                    # passes `None`).
                    # Master-side wall-time bracket: per-eval timing for
                    # the sequential path, per-batch timing for the pool
                    # path (worker ABI returns float; we don't extend it
                    # to a tuple here to keep the spec's "ABI does NOT
                    # change for co-evolution" contract intact). Pool-
                    # mode per-eval rows record the *amortised* wall
                    # (batch_wall / population_size) — a per-worker
                    # average rather than a true per-eval measurement,
                    # which is the correct denominator for the
                    # wall-time reconciliation question ("seconds per
                    # episode at parallel_workers=N").
                    eval_walls: list[float] = []
                    gen_start = time.perf_counter()
                    if pool is not None:
                        fitnesses = list(pool.map(_evaluate_in_worker, eval_args))
                        gen_wall = time.perf_counter() - gen_start
                        # Amortise across population_size to recover a
                        # per-eval average. Cast to float to avoid
                        # integer division in case `len(eval_args) == 0`
                        # (empty population — not a real case but
                        # defensive).
                        amortised = gen_wall / len(eval_args) if eval_args else 0.0
                        eval_walls = [amortised] * len(eval_args)
                    else:
                        fitnesses = []
                        for args in eval_args:
                            eval_start = time.perf_counter()
                            fitnesses.append(_evaluate_in_worker(args))
                            eval_walls.append(time.perf_counter() - eval_start)
                        gen_wall = time.perf_counter() - gen_start
                    self._record_walltime(
                        side=training_side,
                        generation=global_gen,
                        eval_walls=eval_walls,
                        gen_wall=gen_wall,
                        parallel_workers=(
                            training_side.evolution_config.parallel_workers
                            if pool is not None
                            else 1
                        ),
                    )

                # Post-eval bookkeeping: lineage rows, block elite,
                # optimiser tell, generation advance.
                for genome, fit, inherited_from in zip(
                    gen_genomes,
                    fitnesses,
                    inherited_from_per_child,
                    strict=True,
                ):
                    training_side.lineage.record(
                        genome,
                        fitness=fit,
                        brain_type=training_side.encoder.brain_name,
                        inherited_from=inherited_from,
                    )
                    # Strict-greater: first-seen high-tier wins on ties
                    # (preserves recency of the original champion in
                    # the K-block).
                    if fit > block_elite_fitness:
                        block_elite_fitness = fit
                        block_elite_genome = genome
                        # Probe-semantics gap-3 (Option A): archive the
                        # K-block elite's POST-PPO-trained weights to a
                        # dedicated dir that is NOT subject to inheritance
                        # GC. The source `.pt` was just saved during this
                        # generation's eval at the canonical inheritance
                        # path; the post-eval GC at the end of this same
                        # generation may remove it (if the genome isn't
                        # selected as a parent for the next gen). Copy
                        # NOW so the probe can find it at K-block end,
                        # whether or not GC has run. Archive is keyed
                        # on `k_block_index` only — one entry per K-block
                        # per side, overwritten on each fitness
                        # improvement within the K-block.
                        self._archive_kblock_elite_checkpoint(
                            training_side,
                            elite_genome=genome,
                            elite_gen=global_gen,
                        )

                # Tell optimiser (CMA-ES minimises; our fitness maximises).
                training_side.optimizer.tell(list(solutions), [-f for f in fitnesses])
                block_fitnesses.extend(fitnesses)

                # Update side-state population + bookkeeping for next gen.
                training_side.population = gen_genomes
                training_side.prev_generation_ids = [g.genome_id for g in gen_genomes]
                training_side.generation += 1

                # Per-side inheritance bookkeeping after each generation.
                # Mirrors `EvolutionLoop.run`'s post-eval block: pick next
                # gen's `selected_parent_ids` via the strategy, then GC
                # stale per-genome inheritance checkpoints to keep disk
                # usage bounded.
                if self._inheritance_records_lineage(training_side):
                    next_selected = training_side.inheritance.select_parents(
                        [g.genome_id for g in gen_genomes],
                        list(fitnesses),
                        global_gen,
                    )
                    training_side.selected_parent_ids = next_selected
                    if self._inheritance_active(training_side):
                        # Drop the previous-gen elites whose children just ran.
                        self._gc_inheritance_dir(training_side, global_gen - 1, [])
                        # Keep only the about-to-be-parents in current gen.
                        self._gc_inheritance_dir(training_side, global_gen, next_selected)
        finally:
            if pool is not None:
                pool.close()
                pool.join()

        # K-block end: push elite to HoF + champion_history per spec.
        # MUST happen BEFORE the probe cadence check so the probe sees
        # the just-completed-block's elite when the cadence fires —
        # otherwise the very first probe at side.generation == K_per_block
        # would observe an empty champion_history and silently no-op.
        if block_elite_genome is not None:
            training_side.hof.push(block_elite_genome, fitness=block_elite_fitness)
            training_side.champion_history.append(
                {
                    "genome_id": block_elite_genome.genome_id,
                    "generation": block_elite_genome.generation,
                    "k_block_index": self._k_block_index,
                    "fitness": float(block_elite_fitness),
                    "params": block_elite_genome.params.tolist(),
                },
            )
            logger.info(
                "K-block %d (%s) end: pushed elite %s (fitness=%.4f) to HoF + champion_history.",
                self._k_block_index,
                training_side.name,
                block_elite_genome.genome_id,
                block_elite_fitness,
            )

        # Record K-block mean fitness for the rebalance heuristic.
        # Empty-block fallback is 0.0 — defensive against the
        # NotImplementedError-stub case + unit tests that may run
        # zero-iteration K-blocks.
        block_mean = float(np.mean(block_fitnesses)) if block_fitnesses else 0.0
        self._k_block_mean_fitness[training_side.name].append(block_mean)

        # Probe cadence check at K-block boundary, AFTER the elite has
        # been pushed to champion_history. Cadence is keyed on
        # `training_side.generation` (the side-global gen counter)
        # rather than `k_gen` so it applies consistently across
        # K-blocks. The just-finished training side's elite is now
        # available; the opposing side's elite (if any) reflects its
        # most-recent K-block's champion.
        if (
            cfg.generality_probe_every > 0
            and training_side.generation > 0
            and training_side.generation % cfg.generality_probe_every == 0
        ):
            self._fire_generality_probe()

    def _record_walltime(
        self,
        *,
        side: _SideState,
        generation: int,
        eval_walls: list[float],
        gen_wall: float,
        parallel_workers: int,
    ) -> None:
        """Append per-eval + per-gen aggregate rows to walltime.csv.

        One row per evaluation (`scope="evaluation"`, `index=child_idx`)
        plus one summary row per generation (`scope="generation"`,
        `index=len(eval_walls)`). Header is written once at __init__
        in `_init_walltime_csv` (header: `scope, side, generation,
        index, parallel_workers, wall_seconds`).

        Pool-mode per-eval rows record the amortised wall (batch_wall
        / population_size) since the worker ABI returns float, not a
        timing tuple. Sequential-mode per-eval rows record the true
        per-eval wall.
        """
        with self._walltime_csv_path.open("a", newline="") as fh:
            writer = csv.writer(fh)
            for idx, wall in enumerate(eval_walls):
                writer.writerow(
                    [
                        "evaluation",
                        side.name,
                        generation,
                        idx,
                        parallel_workers,
                        f"{wall:.6f}",
                    ],
                )
            writer.writerow(
                [
                    "generation",
                    side.name,
                    generation,
                    len(eval_walls),
                    parallel_workers,
                    f"{gen_wall:.6f}",
                ],
            )

    # ------------------------------------------------------------------
    # Opposition + evaluation
    # ------------------------------------------------------------------

    def _build_opposition(
        self,
        opposing_side: _SideState,
    ) -> list[Genome]:
        """Build the opposition list for one candidate evaluation.

        Mix-with-pop: draw
        `opposing_side.evolution_config.population_size` opposition
        genomes via the HoF mix. Empty-pop fallback (very first
        K-block, opposing side has no population yet) returns an
        empty list — `_build_patched_sim_config` handles that by
        treating the genome as un-opposed (degenerate fitness; see
        the patcher's docstring).
        """
        if not opposing_side.population:
            # First K-block of the run: the opposing side hasn't
            # populated its `population` yet (it sits at the gen-0
            # optimiser mean but no `ask()` has fired). Return empty —
            # `_build_patched_sim_config` treats this as the un-opposed
            # base case to bootstrap the first block.
            return []
        # Pin frac_hof=0.3 at the call site rather than relying on
        # `HallOfFame.DEFAULT_FRAC_HOF` so the 70/30 mix contract
        # lives in the loop, not in the buffer's default. If the
        # buffer's default ever drifts (e.g. for an unrelated use),
        # the loop's contract stays correct.
        return opposing_side.hof.mix_with_pop(
            self.rng,
            opposing_side.population,
            frac_hof=0.3,
        )

    # ------------------------------------------------------------------
    # Per-generation eval-args batch construction (master process)
    # ------------------------------------------------------------------

    def _build_eval_args(  # noqa: PLR0913 — kw-only batch builder; each input is orthogonal per-generation state
        self,
        *,
        training_side: _SideState,
        opposing_side: _SideState,
        solutions: list,
        parent_ids: list[str],
        global_gen: int,
        tmp_path: Path,
    ) -> tuple[list[tuple], list[Genome], list[str]]:
        """Build the per-generation `eval_args` batch for pool dispatch.

        For each candidate:

        1. Wrap the optimiser's flat-vector solution in a `Genome`.
        2. Resolve per-child inheritance via the side's
           `InheritanceStrategy` (warm-start path + capture path under
           Lamarckian; empty strings under NoInheritance).
        3. Sample HoF-mixed opposition (per "70/30 Mixture During
           Evaluation" spec scenario), pre-decode each opposition
           genome, and write its weights to a `.pt` under `tmp_path`.
        4. Patch `sim_config` with the opposition `weights_path` keys
           plus the side's per-side `EvolutionConfig` so fitness
           evaluation reads the right per-side fields.
        5. Append the 12-tuple shape consumed by
           `_evaluate_in_worker` (`tei_prior_source` slot is always
           `None` for co-evolution).

        All RNG draws (eval_seed + opposition mix) happen in the master
        process so the loop's RNG-state advancement is deterministic
        regardless of `parallel_workers`.

        Returns
        -------
        tuple
            `(eval_args, gen_genomes, inherited_from_per_child)` —
            `eval_args` for the pool dispatch; `gen_genomes` for
            post-eval lineage rows + block-elite tracking;
            `inherited_from_per_child` for the lineage CSV's
            `inherited_from` column.
        """
        eval_args: list[tuple] = []
        gen_genomes: list[Genome] = []
        inherited_from_per_child: list[str] = []
        for idx, params in enumerate(solutions):
            gid = genome_id_for(global_gen, idx, parent_ids)
            genome = Genome(
                params=np.asarray(params, dtype=np.float32),
                genome_id=gid,
                parent_ids=parent_ids,
                generation=global_gen,
                birth_metadata=build_birth_metadata(self.sim_config),
            )
            gen_genomes.append(genome)

            # HoF-mixed opposition — drawn here in the master process so
            # RNG-state advancement is reproducible regardless of pool
            # parallelism. Empty for the first K-block before the
            # opposing side has populated.
            opposition = self._build_opposition(opposing_side)

            # Per-child inheritance dispatch (mirrors
            # `EvolutionLoop._resolve_per_child_inheritance`).
            (
                parent_warm_start,
                child_capture_path,
                inherited_from,
            ) = self._resolve_per_child_inheritance(training_side, idx, global_gen, gid)
            inherited_from_per_child.append(inherited_from)

            # Per-evaluation seed derivation matches `EvolutionLoop`'s
            # pattern (each child gets a distinct RNG stream).
            eval_seed = int(self.rng.integers(0, 2**31 - 1))

            # Patch `sim_config` (opposition weights + side's
            # EvolutionConfig). `candidate_tag=gid` namespaces the
            # opposition `.pt` filenames so concurrent pool workers
            # within the same per-generation tmp dir do not collide.
            patched_sim_config = self._build_patched_sim_config(
                side=training_side,
                opposition=opposition,
                eval_seed=eval_seed,
                tmp_path=tmp_path,
                candidate_tag=gid,
            )

            eval_args.append(
                (
                    np.asarray(params, dtype=np.float32),
                    patched_sim_config,
                    training_side.encoder,
                    training_side.fitness,
                    training_side.evolution_config.episodes_per_eval,
                    eval_seed,
                    global_gen,
                    idx,
                    parent_ids,
                    parent_warm_start,
                    child_capture_path,
                    # Co-evolution doesn't use the transgenerational
                    # cascade — None preserves the worker-tuple shape
                    # (``tei_prior_source`` slot).
                    None,
                    # Co-evolution does not emit eval_diagnostics.jsonl
                    # — None for ``diagnostics_path`` (13th slot;
                    # mirrors ``EvolutionLoop._evaluate_in_worker``).
                    None,
                ),
            )
        return eval_args, gen_genomes, inherited_from_per_child

    # ------------------------------------------------------------------
    # Per-side inheritance helpers (mirror `EvolutionLoop`'s)
    # ------------------------------------------------------------------

    def _inheritance_active(self, side: _SideState) -> bool:
        """Return True iff the side's strategy uses per-genome weight checkpoints."""
        return side.inheritance.kind() == "weights"

    def _inheritance_records_lineage(self, side: _SideState) -> bool:
        """Return True iff the side's strategy populates `inherited_from`."""
        return side.inheritance.kind() != "none"

    def _resolve_per_child_inheritance(
        self,
        side: _SideState,
        child_idx: int,
        gen: int,
        gid: str,
    ) -> tuple[Path | None, Path | None, str]:
        """Compute one child's `(parent_warm_start, child_capture_path, inherited_from)`.

        Per-side analogue of `EvolutionLoop._resolve_per_child_inheritance`.
        Three-branch switch on the side's strategy `kind()`:

        - `"none"` → `(None, None, "")` — child trains from scratch,
          lineage row records no parent.
        - `"trait"` (Baldwin) → `(None, None, parent_id)` where
          `parent_id` is the prior-gen elite from
          `side.selected_parent_ids` (or empty for gen 0).
        - `"weights"` (Lamarckian) → full
          `(parent_warm_start, child_capture_path, parent_id)` tuple.
          `parent_warm_start` is the parent's pre-saved checkpoint Path
          under the SIDE'S output dir, or None on missing.
        """
        kind = side.inheritance.kind()
        if kind == "none":
            return None, None, ""
        if kind == "trait":
            parent_id = side.selected_parent_ids[0] if side.selected_parent_ids else ""
            return None, None, parent_id
        # Lamarckian path. `checkpoint_path` is keyed off the side's
        # `output_dir`, which the strategy already accepts so the
        # per-side directory layout (`<run>/{side}/inheritance/...`)
        # follows automatically.
        child_capture_path = side.inheritance.checkpoint_path(side.output_dir, gen, gid)
        parent_id = side.inheritance.assign_parent(child_idx, side.selected_parent_ids)
        if parent_id is None:
            return None, child_capture_path, ""
        candidate = side.inheritance.checkpoint_path(side.output_dir, gen - 1, parent_id)
        if candidate is not None and candidate.exists():
            return candidate, child_capture_path, parent_id
        logger.warning(
            "Lamarckian parent checkpoint missing for %s child idx=%d gen=%d "
            "(expected parent_id=%s at %s) — falling back to from-scratch.",
            side.name,
            child_idx,
            gen,
            parent_id,
            candidate,
        )
        return None, child_capture_path, ""

    def _gc_inheritance_dir(
        self,
        side: _SideState,
        generation: int,
        keep_ids: list[str],
    ) -> None:
        """Garbage-collect non-survivor checkpoints in one side's inheritance dir.

        Per-side analogue of `EvolutionLoop._gc_inheritance_dir`. No-op
        when `generation < 0` (e.g. the first GC pass before any
        inheritance file has been written) or when the directory does
        not exist.
        """
        if generation < 0:
            return
        gen_dir = side.output_dir / "inheritance" / f"gen-{generation:03d}"
        if not gen_dir.exists():
            return
        keep = set(keep_ids)
        for path in gen_dir.glob("genome-*.pt"):
            # Mirrors EvolutionLoop._gc_inheritance_dir's ``.tei.pt`` skip:
            # ``Path.stem`` strips only one suffix, so for a substrate file
            # named ``genome-<gid>.tei.pt`` the extracted gid would end in
            # ``.tei`` and never match the keep-set, deleting the substrate.
            # Coevolution does not currently produce ``.tei.pt`` files
            # (no per-side substrate machinery), but the guard is included
            # preemptively so a future composed-coevolution arc cannot
            # silently re-introduce the bug.
            if path.name.endswith(".tei.pt"):
                continue
            gid = path.stem.removeprefix("genome-")
            if gid not in keep:
                path.unlink(missing_ok=True)
        # Cosmetic: drop the dir if it's now empty so the inheritance/
        # tree stays clean during a run.
        with suppress(OSError):
            gen_dir.rmdir()

    def _kblock_archive_path(self, side: _SideState, k_block_index: int) -> Path:
        """Canonical path for a side's K-block-elite weight archive.

        Probe-semantics gap-3 (Option A): the K-block elite's POST-PPO-trained
        weights are archived here, separate from `<side>/inheritance/`
        which is subject to GC. One entry per K-block per side.
        """
        return side.output_dir / "champion_archive" / f"k_block-{k_block_index:03d}.pt"

    def _archive_kblock_elite_checkpoint(
        self,
        side: _SideState,
        *,
        elite_genome: Genome,
        elite_gen: int,
    ) -> None:
        """Copy the elite's `.pt` from inheritance dir to the champion archive.

        Called each time the K-block-elite is updated within a K-block. The
        source `.pt` was just saved during the elite's eval at the canonical
        inheritance path
        `<side>/inheritance/gen-<elite_gen>/genome-<elite_id>.pt`. The
        destination is `<side>/champion_archive/k_block-<k>.pt`, keyed only
        on the loop's current `_k_block_index` (overwriting prior entries
        for the same K-block).

        No-op when:
        - The side's inheritance kind is not "weights" (no `.pt` to archive).
        - The source `.pt` doesn't exist (defensive — shouldn't happen since
          this fires right after the eval that wrote it, but be safe).
        """
        if side.inheritance.kind() != "weights":
            return
        source = side.inheritance.checkpoint_path(
            side.output_dir,
            int(elite_gen),
            str(elite_genome.genome_id),
        )
        if source is None or not source.exists():
            return
        dest = self._kblock_archive_path(side, self._k_block_index)
        dest.parent.mkdir(parents=True, exist_ok=True)
        # `shutil.copyfile` rather than rename: source is still active
        # in the inheritance dir and may be consumed by subsequent
        # Lamarckian warm-start lookups within the same K-block.
        import shutil

        shutil.copyfile(source, dest)

    def _build_patched_sim_config(
        self,
        *,
        side: _SideState,
        opposition: list[Genome],
        eval_seed: int,
        tmp_path: Path,
        candidate_tag: str,
    ) -> SimulationConfig:
        """Build the per-evaluation `sim_config` with opposition + side patches.

        Caller owns the `tmp_path` lifecycle (the opposition `.pt`
        files written here must outlive this call so the worker process
        can read them on `load_weights`). `candidate_tag` namespaces
        the file names so concurrent calls within the same generation
        do not collide; pass the genome_id (or any per-candidate
        unique string).

        Three patches are layered onto `self.sim_config`:

        1. `sim_config.evolution = side.evolution_config` so per-side
           fitness fields (`learn_episodes_per_eval`,
           `eval_episodes_per_eval`) resolve correctly for
           `LearnedPerformanceFitness`.
        2. When `side.name == "prey"`, opposition predator weights flow
           into `environment.predators.brain_config.extra["weights_path"]`
           — all N predator slots load the same opposition genome
           (matching the focal genome's "same brain on every slot"
           semantic). Empty opposition skips this patch and the env
           builds predators with random-init weights.
        3. When `side.name == "predator"`, opposition prey weights flow
           into `multi_agent.agents[i].weights_path` — one slot per
           opposition genome, capped at the
           `MultiAgentConfig._validate_population` upper bound (10).
           Empty opposition pads with random-init prey opponents up to
           the schema's lower bound (2).

        `model_copy(update=...)` produces shallow copies; fitness
        evaluation does not mutate the config so shallow is safe.
        """
        base_patch: dict[str, object] = {"evolution": side.evolution_config}

        opposing_side = self.predator if side.name == "prey" else self.prey
        if side.name == "prey":
            if opposition:
                base_patch["environment"] = self._build_prey_side_environment_patch(
                    opposition=opposition,
                    opposing_side=opposing_side,
                    eval_seed=eval_seed,
                    tmp_path=tmp_path,
                    candidate_tag=candidate_tag,
                )
        else:
            base_patch["multi_agent"] = self._build_predator_side_multi_agent_patch(
                opposition=opposition,
                opposing_side=opposing_side,
                eval_seed=eval_seed,
                tmp_path=tmp_path,
                candidate_tag=candidate_tag,
            )
        return self.sim_config.model_copy(update=base_patch)

    def _build_prey_side_environment_patch(
        self,
        *,
        opposition: list[Genome],
        opposing_side: _SideState,
        eval_seed: int,
        tmp_path: Path,
        candidate_tag: str,
    ) -> EnvironmentConfig:
        """Materialise one opposition predator's weights and patch env.

        Picks the first opposition genome (deterministic across calls
        with the same `opposition` list ordering, which itself is
        seeded via `_build_opposition`). All env predator slots load
        these weights via the env's
        `_build_predator_brain` `extra["weights_path"]` hook.
        """
        # Decode the opposition genome to a fresh predator brain, save its
        # weights to a tmp .pt, and patch the env's predator brain_config
        # to load them on construction.
        opp_genome = opposition[0]
        opp_brain = opposing_side.encoder.decode(
            opp_genome,
            self.sim_config,
            seed=eval_seed,
        )
        # `candidate_tag` namespaces the file so multiple
        # `_build_patched_sim_config` calls within the same per-generation
        # tmp dir do not collide (concurrent pool workers all read from
        # the same dir).
        weights_file = tmp_path / f"opposition_predator_{candidate_tag}.pt"
        # `encoder.decode` annotates the return as `Brain`; concrete
        # predator decoders return `MLPPPOPredatorBrain` which satisfies
        # the separate `PredatorBrain` Protocol. The runtime brain
        # implements `WeightPersistence`; `save_weights` only needs
        # `WeightPersistence` (it accepts `Brain` for back-compat but
        # only consults `get_weight_components`). Cast for the static
        # checker.
        save_weights(cast("Brain", opp_brain), weights_file)

        env_cfg = self.sim_config.environment
        if env_cfg is None or env_cfg.predators is None:
            msg = (
                "Prey-side opposition injection requires sim_config.environment.predators "
                "to be set; got None. The YAML must enable predators with "
                "`brain_config.kind: mlpppo_predator`."
            )
            raise ValueError(msg)
        predators_cfg = env_cfg.predators
        existing_brain_cfg = predators_cfg.brain_config
        if existing_brain_cfg is None:
            msg = (
                "Prey-side opposition injection requires "
                "sim_config.environment.predators.brain_config "
                "to be set with `kind: mlpppo_predator`. "
                "Add the block to the YAML."
            )
            raise ValueError(msg)
        # The env's `_build_predator_brain` dispatcher only honours
        # `extra["weights_path"]` on the `mlpppo_predator` branch —
        # a `heuristic` brain config would silently ignore the
        # injected path and the run would proceed with the default
        # heuristic predator opponent (no opposition signal). Fail
        # fast so a misconfigured YAML surfaces here rather than
        # producing a flat-zero fitness gradient.
        if existing_brain_cfg.kind != "mlpppo_predator":
            msg = (
                "Prey-side opposition injection requires "
                f"sim_config.environment.predators.brain_config.kind == 'mlpppo_predator'; "
                f"got {existing_brain_cfg.kind!r}. The env's `_build_predator_brain` "
                "dispatcher only honours `extra['weights_path']` on the learnable "
                "predator branch."
            )
            raise ValueError(msg)
        new_extra = dict(existing_brain_cfg.extra or {})
        new_extra["weights_path"] = str(weights_file)
        new_brain_cfg = existing_brain_cfg.model_copy(update={"extra": new_extra})
        new_predators_cfg = predators_cfg.model_copy(
            update={"brain_config": new_brain_cfg},
        )
        return env_cfg.model_copy(update={"predators": new_predators_cfg})

    def _build_predator_side_multi_agent_patch(
        self,
        *,
        opposition: list[Genome],
        opposing_side: _SideState,
        eval_seed: int,
        tmp_path: Path,
        candidate_tag: str,
    ) -> MultiAgentConfig:
        """Materialise opposition prey weights and patch multi_agent.

        One env agent slot per opposition genome. Each slot's
        `BrainContainerConfig` mirrors the YAML's top-level
        `sim_config.brain` (the prey brain shape); per-slot
        `weights_path` injects the opposition genome's weights.
        """
        if self.sim_config.brain is None:
            msg = (
                "Predator-side opposition injection requires sim_config.brain "
                "(top-level prey brain block) to be set so opposition prey "
                "agents can construct via the same brain shape."
            )
            raise ValueError(msg)
        # `MultiAgentConfig._validate_population` enforces 2 <= len(agents) <= 10.
        # Cap the opposition list size so a pop=24 prey side (which produces
        # ~24 opposition genomes per evaluation) does not blow the schema cap.
        # Subsample uniformly (preserving the HoF mix order at the head of the
        # list) rather than truncating arbitrarily — the 70/30 mix is set
        # upstream in `_build_opposition`, so the head of the list already
        # reflects the desired prevalence.
        max_agents = 10
        min_agents = 2
        capped_opposition = list(opposition[:max_agents])
        agent_configs: list[AgentConfig] = []
        for idx, opp_genome in enumerate(capped_opposition):
            opp_brain = opposing_side.encoder.decode(
                opp_genome,
                self.sim_config,
                seed=eval_seed,
            )
            weights_file = tmp_path / f"opposition_prey_{candidate_tag}_{idx}.pt"
            save_weights(cast("Brain", opp_brain), weights_file)
            agent_configs.append(
                AgentConfig(
                    id=f"opposition_prey_{idx}",
                    brain=self.sim_config.brain,
                    weights_path=str(weights_file),
                ),
            )
        # Pad up to the schema's lower bound (2 agents) with random-init
        # bootstrap entries when opposition is empty or has only 1 entry.
        # The schema rejects len(agents) < 2; the fitness function
        # tolerates extra prey slots since predator metrics aggregate
        # across slots.
        bootstrap_idx = 0
        while len(agent_configs) < min_agents:
            agent_configs.append(
                AgentConfig(
                    id=f"opposition_prey_bootstrap_{bootstrap_idx}",
                    brain=self.sim_config.brain,
                    weights_path=None,
                ),
            )
            bootstrap_idx += 1
        # Replace `count` (if any) with the explicit per-slot config.
        # `_validate_population` rejects setting both, so we explicitly
        # null out `count`.
        existing_ma = self.sim_config.multi_agent
        if existing_ma is None:
            return MultiAgentConfig(enabled=True, agents=agent_configs)
        return existing_ma.model_copy(
            update={"enabled": True, "count": None, "agents": agent_configs},
        )

    # ------------------------------------------------------------------
    # Generality probe
    # ------------------------------------------------------------------

    def _fire_generality_probe(self) -> None:
        """Evaluate each side's current elite against its held-out set.

        Writes one row per (side, opponent_index) pair to
        `{output_dir}/generality_probe.csv`. Does NOT mutate any
        side's population, optimizer, or hof; the probe runs in
        append-mode against the CSV.

        Per side: the elite is the latest entry in `champion_history`
        (the K-block elite). When a side has no champions yet (first
        K-block hasn't completed), the probe is a no-op for that side.

        Held-out wiring (cross-species yardstick): each side's elite
        is evaluated against the OPPOSING-side held-out set — prey
        elite vs. frozen heuristic predators (`_predator_held_out_specs`),
        predator elite vs. frozen prey genomes (`_prey_held_out`).
        Per-opponent evaluation runs sequentially in the master process
        via `_probe_one_opponent` (probe volume is low — typically
        ~8 evals every 10 gens — so pool dispatch overhead is not
        worth the complication).

        When a side's opposing held-out set is empty (e.g. the prey
        bundle dir was missing at __init__), that side's probe rows
        are skipped silently for the K-block — `_load_held_out_prey_bundle`
        already logs the one-time warning at startup.
        """
        with self._probe_csv_path.open("a", newline="") as fh:
            writer = csv.writer(fh)
            for side in (self.prey, self.predator):
                if not side.champion_history:
                    continue
                elite_record = side.champion_history[-1]
                # Reconstruct the elite genome from the champion_history
                # record. Mirror Genome.params dtype for downstream
                # encoder consumption.
                elite = Genome(
                    params=np.asarray(elite_record["params"], dtype=np.float32),
                    genome_id=str(elite_record["genome_id"]),
                    parent_ids=[],
                    generation=int(elite_record["generation"]),
                    birth_metadata=build_birth_metadata(self.sim_config),
                )
                # Each side faces the OPPOSING-side held-out set: the
                # prey-side probe iterates over `_predator_held_out_specs`
                # (heuristic predator radius variants); the predator-side
                # probe iterates over `_prey_held_out` (frozen
                # lamarckian-LSTMPPO prey genomes from the upstream
                # single-population campaign). The variable names are
                # content-keyed (what's in the collection), not
                # consumer-keyed (which side reads it) — they form
                # cross-species yardsticks for the opposing side's
                # elite.
                if side.name == "prey":
                    held_out_count = len(self._predator_held_out_specs)
                else:
                    held_out_count = len(self._prey_held_out)
                for opp_idx in range(held_out_count):
                    # Probe RNG seed is derived per-fire so the rng
                    # advances deterministically regardless of how many
                    # opponents each held-out set holds.
                    eval_seed = int(self.rng.integers(0, 2**31 - 1))
                    fitness_val = self._probe_one_opponent(
                        side=side,
                        elite=elite,
                        opponent_index=opp_idx,
                        eval_seed=eval_seed,
                    )
                    writer.writerow(
                        [
                            side.generation,
                            side.name,
                            opp_idx,
                            f"{fitness_val:.6f}",
                        ],
                    )

    def _probe_one_opponent(
        self,
        *,
        side: _SideState,
        elite: Genome,
        opponent_index: int,
        eval_seed: int,
    ) -> float:
        """Evaluate `elite` against one held-out opponent.

        Sequential, master-side evaluation. Materialises a per-call
        tmp dir for the opposition `.pt` checkpoint (predator-side
        only — prey-side probe uses heuristic predators with no weight
        injection), patches `sim_config` to install the held-out
        opponent on the opposing-side slot(s), and routes through
        `_evaluate_in_worker` with `warm_start_path_override=None,
        weight_capture_path=None` so the probe does not mutate the
        elite's germline weights. The tmp dir is cleaned up on exit.

        Fitness function selection (Option 1: probe-semantics fix):
        - **Prey side**: forces `self._prey_probe_fitness` (default
          `EpisodicSuccessRate`, frozen-weight, L eval episodes only)
          INSTEAD of the side's training-time
          `LearnedPerformanceFitness`. Rationale: training-time
          `LearnedPerformanceFitness` runs K PPO train episodes against
          the held-out opponent before the L eval — those K episodes
          fine-tune the elite's policy against a totally different
          opponent class (heuristic radius variants vs MLPPPO), pushing
          weights in a direction the prey didn't see during co-evolution
          training. With K=8 episodes against an unfamiliar opponent
          class, the policy consistently degrades to 0.0 by eval phase.
          Frozen-weight eval measures the elite AS-IS — the
          scientifically correct test of "what did the prey learn?"
          matching the post-hoc analysis path. The attribute is settable
          for tests that stub the fitness function to skip real
          episode execution.
        - **Predator side**: keeps `side.fitness`
          (`PredatorEpisodicKillRate`) — already frozen-weight by
          default, no inner-loop train phase, so the probe's measurement
          semantic was already correct on this side.
        """
        with tempfile.TemporaryDirectory(prefix="probe_") as tmp:
            tmp_path = Path(tmp)
            if side.name == "prey":
                patched_sim_config = self._build_prey_side_probe_sim_config(
                    opponent_index=opponent_index,
                )
                # Frozen-weight measurement — see docstring.
                probe_fitness = self._prey_probe_fitness
            else:
                patched_sim_config = self._build_predator_side_probe_sim_config(
                    opponent_index=opponent_index,
                    eval_seed=eval_seed,
                    tmp_path=tmp_path,
                )
                # Predator side already frozen-weight; keep side.fitness.
                probe_fitness = side.fitness

            # Probe-semantics gap-3 fix: when the side uses Lamarckian
            # inheritance, the K-block-elite's genome.params encode the
            # CMA-ES sample (pre-PPO-training weights). The actual
            # co-evolved policy lives in the post-training `.pt`
            # checkpoint. Load it via `warm_start_path_override` so the
            # probe measures what the prey ACTUALLY learned, not the
            # untrained sample. The post-hoc analysis path uses the
            # same checkpoint via the script's `load_weights` call;
            # this fix aligns the in-run probe with that semantic.
            #
            # Lookup order:
            # 1. `champion_archive/k_block-<k>.pt` — the elite's `.pt`
            #    archived at fitness-improvement time during the K-block
            #    (Option A; NOT subject to inheritance GC).
            # 2. `inheritance/gen-<g>/genome-<id>.pt` — the canonical
            #    Lamarckian path, may have been GC'd by now if the
            #    elite wasn't selected as a parent for subsequent gens.
            # 3. None — fall back to raw genome.params via encoder.decode.
            #
            # The frozen-weight fitness functions (`EpisodicSuccessRate`
            # and `PredatorEpisodicKillRate`) both accept
            # `warm_start_path_override` and load the checkpoint into
            # the decoded brain before evaluation.
            warm_start_path: Path | None = None
            if side.inheritance.kind() == "weights":
                # Path 1: champion archive (preferred; gap-3 Option A).
                # Find the K-block index that produced this elite. We
                # iterate champion_history to locate the entry matching
                # the elite's (genome_id, generation) — typically the
                # most-recently-pushed entry.
                archive_k_block_index: int | None = None
                for record in reversed(side.champion_history):
                    if record.get("genome_id") == elite.genome_id and int(
                        record.get("generation", -1),
                    ) == int(elite.generation):
                        archive_k_block_index = int(record.get("k_block_index", -1))
                        break
                if archive_k_block_index is not None and archive_k_block_index >= 0:
                    archive_candidate = self._kblock_archive_path(
                        side,
                        archive_k_block_index,
                    )
                    if archive_candidate.exists():
                        warm_start_path = archive_candidate

                # Path 2 fallback: canonical inheritance path.
                if warm_start_path is None:
                    candidate = side.inheritance.checkpoint_path(
                        side.output_dir,
                        int(elite.generation),
                        str(elite.genome_id),
                    )
                    if candidate is not None and candidate.exists():
                        warm_start_path = candidate

            episodes = self._probe_episode_count(side)
            args = (
                np.asarray(elite.params, dtype=np.float32),
                patched_sim_config,
                side.encoder,
                probe_fitness,
                episodes,
                eval_seed,
                int(elite.generation),
                opponent_index,
                list(elite.parent_ids),
                warm_start_path,  # gap-3 fix: load post-PPO checkpoint when Lamarckian
                None,  # weight_capture_path — probe must not write germline weights
                None,  # tei_prior_source — probes never use the transgenerational substrate
                None,  # diagnostics_path — probes write to a separate probe-output stream
            )
            return float(_evaluate_in_worker(args))

    def _probe_episode_count(self, side: _SideState) -> int:
        """Resolve the per-opponent episode count for the probe.

        Mirrors the training-time `episodes_per_eval` so probe values
        and training values are directly comparable on the same axis.
        Note: under Option 1's probe-semantics fix, the prey-side probe
        uses `EpisodicSuccessRate` (frozen-weight, L eval episodes only,
        no K train phase). The `episodes_per_eval` field maps to L
        cleanly in that mode. For the predator side
        (`PredatorEpisodicKillRate`), `episodes_per_eval` is the total
        episode count (no train/eval split).
        """
        return int(side.evolution_config.episodes_per_eval)

    def _build_prey_side_probe_sim_config(
        self,
        *,
        opponent_index: int,
    ) -> SimulationConfig:
        """Build the per-eval `sim_config` for a prey-side probe.

        Patches `environment.predators` to a HEURISTIC predator at the
        held-out spec's `(detection_radius, damage_radius)`. The
        `brain_config.kind` flips from `"mlpppo_predator"` (training
        config) to `"heuristic"`, and any pre-existing
        `extra["weights_path"]` is dropped — the probe deliberately
        side-steps the learned predator brain so the held-out yardstick
        is independent of co-evolution lineage.

        Also overrides the env-level `count`, `speed`, and `grid_size`
        to the substrate-calibrated probe values (`PROBE_ENV_*` constants
        at module top). The training env may use harder settings (e.g.
        count=4, speed=1.0, grid=16) where no klinotaxis-LSTMPPO prey
        can survive heuristic predators at all; using the training env
        for the probe in that case produces uniformly-zero fitness
        independent of training quality, which is uninformative. The
        calibrated probe env preserves the probe's discriminative range
        (baseline prey score ~0.5 mean across this env's spec grid;
        a co-evolved prey scoring near zero indicates real overfitting,
        not substrate ceiling).

        Also lays the prey-side `evolution_config` onto `sim_config.evolution`
        so `LearnedPerformanceFitness` reads the right per-side budget
        fields (matches `_build_patched_sim_config`'s training-side
        patch).
        """
        spec = self._predator_held_out_specs[opponent_index]
        detection_radius, damage_radius = spec

        env_cfg = self.sim_config.environment
        if env_cfg is None or env_cfg.predators is None:
            msg = (
                "Prey-side probe requires sim_config.environment.predators to be "
                "set; got None. The YAML must enable predators."
            )
            raise ValueError(msg)
        predators_cfg = env_cfg.predators
        existing_brain_cfg = predators_cfg.brain_config
        if existing_brain_cfg is None:
            # Fall back to constructing a minimal heuristic brain config
            # via the existing schema; tests that build a config without
            # a `brain_config` block exercise this path.
            new_brain_cfg = None
        else:
            # Strip any learned-brain `extra["weights_path"]` and force
            # `kind="heuristic"`. `model_copy(update=...)` preserves the
            # other fields so future schema additions don't silently
            # diverge from the training config's heuristic-flavour shape.
            new_brain_cfg = existing_brain_cfg.model_copy(
                update={"kind": "heuristic", "extra": None},
            )
        new_predators_cfg = predators_cfg.model_copy(
            update={
                "enabled": True,
                "count": PROBE_ENV_PREDATOR_COUNT,
                "speed": PROBE_ENV_PREDATOR_SPEED,
                "detection_radius": int(detection_radius),
                "damage_radius": int(damage_radius),
                "brain_config": new_brain_cfg,
            },
        )
        new_env = env_cfg.model_copy(
            update={
                "predators": new_predators_cfg,
                "grid_size": PROBE_ENV_GRID_SIZE,
            },
        )
        return self.sim_config.model_copy(
            update={
                "environment": new_env,
                "evolution": self.prey.evolution_config,
            },
        )

    def _build_predator_side_probe_sim_config(
        self,
        *,
        opponent_index: int,
        eval_seed: int,
        tmp_path: Path,
    ) -> SimulationConfig:
        """Build the per-eval `sim_config` for a predator-side probe.

        Patches `multi_agent.agents` to install one held-out prey
        genome's weights on slot 0 via `weights_path`, padding with
        random-init bootstrap entries up to the schema's
        `MultiAgentConfig._validate_population` minimum (2 agents).
        Mirrors `_build_predator_side_multi_agent_patch`'s shape but
        with `opposition=[held_out_prey_genome]` (single-element
        opposition).

        The held-out prey's weights are decoded via the prey encoder
        and persisted to a tmp `.pt` whose lifecycle the caller owns
        (`_probe_one_opponent`'s `TemporaryDirectory`).
        """
        held_out_genome = self._prey_held_out[opponent_index]
        opp_brain = self.prey.encoder.decode(
            held_out_genome,
            self.sim_config,
            seed=eval_seed,
        )
        weights_file = tmp_path / f"probe_held_out_prey_{opponent_index}.pt"
        save_weights(cast("Brain", opp_brain), weights_file)

        if self.sim_config.brain is None:
            msg = (
                "Predator-side probe requires sim_config.brain (top-level prey "
                "brain block) to be set so held-out prey can construct via the "
                "same brain shape."
            )
            raise ValueError(msg)
        agent_configs = [
            AgentConfig(
                id=f"probe_held_out_prey_{opponent_index}",
                brain=self.sim_config.brain,
                weights_path=str(weights_file),
            ),
        ]
        # Pad to the schema's lower bound (2 agents); the second slot
        # is a random-init bootstrap entry. The predator-vs-held-out
        # signal still dominates because the predator metrics aggregate
        # across slots and the held-out prey is by construction the
        # stronger opponent.
        min_agents = 2
        bootstrap_idx = 0
        while len(agent_configs) < min_agents:
            agent_configs.append(
                AgentConfig(
                    id=f"probe_bootstrap_prey_{bootstrap_idx}",
                    brain=self.sim_config.brain,
                    weights_path=None,
                ),
            )
            bootstrap_idx += 1

        existing_ma = self.sim_config.multi_agent
        if existing_ma is None:
            new_ma = MultiAgentConfig(enabled=True, agents=agent_configs)
        else:
            new_ma = existing_ma.model_copy(
                update={"enabled": True, "count": None, "agents": agent_configs},
            )
        return self.sim_config.model_copy(
            update={
                "multi_agent": new_ma,
                "evolution": self.predator.evolution_config,
            },
        )
