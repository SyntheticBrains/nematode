"""Co-evolution loop orchestrator (PR 3 §6 — M5).

Composes two `EvolutionLoop`-shaped sides (prey + predator) under an
alternating-schedule controller. Per design.md:

- **D2:** Both sides use `CMAESOptimizer(diagonal=True)` (sep-CMA-ES).
  TPE is rejected for unbounded weight encoders.
- **D7:** Predator gen-0 is bootstrapped via either heuristic-imitation
  pretrain (arm A) or cold-start zeros (arm B). The pretrain runs
  inside `__init__` on arm-A construction; cost is amortised across
  the `K_per_block * generation_pairs` generations of the run.
- **D10/D13:** Prey side uses `LearnedPerformanceFitness` +
  `LamarckianInheritance`; predator side uses `PredatorEpisodicKillRate`
  + `NoInheritance`. Asymmetry intentional and pinned in `__init__`
  (NOT YAML-configurable per B20).
- **D12:** Prey gen-0 elite loaded from `prey_gen0_seed_path` (a
  warmstart genome JSON produced by the M3 lamarckian pilot) and passed
  to `CMAESOptimizer(x0=...)`. When the path is None, the prey side
  starts from `x0=zeros` like a cold-start run.
- **D14:** YAML schema lives at
  `quantumnematode.utils.config_loader.CoevolutionConfig` with model
  validators enforcing all of the above at load time.

PR 3 commit 5 shipped the scaffold + alternating schedule + per-K-block
fresh `CMAESOptimizer` re-construction. PR 3 commit 6 (this commit)
adds the per-generation loop body — HoF push at K-block end, HoF-mixed
opposition sampling, generality probe, held-out opponent construction
for both sides. Checkpoint/resume + worker dispatch + full test suite
land in commits 7-8 within this PR.

Opposition injection (deferred to PR 4)
---------------------------------------
The per-evaluation pattern samples opposition via
`hof.mix_with_pop(rng, opposing_population, frac_hof=0.3)` and the
opposition genomes are recorded on the candidate's evaluation. The
final `sim_config` patching that translates opposition genomes into
env-side opponents (decoded predator brains installed on
`Predator.brain` slots, decoded prey brains exposed via
`sim_config.multi_agent.agents`) requires the campaign-level configs
that PR 4 ships. For PR 3 the loop calls `fitness.evaluate(...)` with
the unpatched `sim_config` and records opposition for lineage; full
end-to-end opposition injection is wired in PR 4 alongside the smoke
config that exercises the integrated path.

Champion history schema
-----------------------
After each K-block ends, the training-side block elite is appended to
`champion_history` as a dict ``{"genome_id": str, "generation": int,
"k_block_index": int, "fitness": float, "params": list[float]}``.
Serialised at checkpoint time as JSON via `params.tolist()`; restored
via `np.asarray(d["params"], dtype=np.float32)` per the convention
shared with `Genome.params`. The aggregator (PR 5 task 8.1) walks
this history for cycling and escalation analysis.
"""

from __future__ import annotations

import csv
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

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

if TYPE_CHECKING:
    from quantumnematode.evolution.encoders import GenomeEncoder
    from quantumnematode.evolution.fitness import FitnessFunction
    from quantumnematode.evolution.inheritance import InheritanceStrategy
    from quantumnematode.utils.config_loader import (
        CoevolutionConfig,
        EvolutionConfig,
        SimulationConfig,
    )

logger = logging.getLogger(__name__)


# Default Hall-of-Fame capacity per design.md D3 (8 entries; 70/30 mix
# preserves live signal while preventing forgetting).
DEFAULT_HOF_CAPACITY = 8

# Checkpoint format version. Bumped when the on-disk shape changes;
# resume rejects mismatched versions to prevent silent state corruption.
# Distinct from `EvolutionLoop.CHECKPOINT_VERSION` — they evolve
# independently because the shapes differ (single-population pickle vs
# 4-file co-evolution split).
CHECKPOINT_VERSION = 1


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
    re-constructs the optimizer at every K-block transition per D2).
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
    `__init__` performs gen-0 setup for both sides per D7 + D12:

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

    Fitness is hardcoded per side per D13 + B20 (NOT YAML-configurable):
    prey gets `LearnedPerformanceFitness`; predator gets
    `PredatorEpisodicKillRate`.

    Inheritance is hardcoded per side per D10 + D13: prey gets
    `LamarckianInheritance(elite_count=1)` (single-elite-broadcast,
    matching the M3 substrate); predator gets `NoInheritance()`.

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

        self.output_dir.mkdir(parents=True, exist_ok=True)
        # Per-side subdirs for lineage CSVs (matches `EvolutionLoop`'s
        # output_dir convention so M3 single-population analysis tooling
        # reuses unchanged).
        (self.output_dir / "prey").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "predator").mkdir(parents=True, exist_ok=True)

        # Build per-side bundles. Construction order matters: encoder
        # first (drives genome_dim), then fitness, then inheritance,
        # then optimizer (needs genome_dim).
        self.prey = self._build_prey_state()
        self.predator = self._build_predator_state()

        # Alternating-schedule controller state. `_k_block_index`
        # counts completed K-blocks (0 = before first K-block ran);
        # `_current_side` is the side currently training (flips at
        # each K-block boundary).
        self._k_block_index: int = 0
        self._current_side: Literal["prey", "predator"] = self.coevolution_config.start_side

        # Per-K-block mean-fitness history per side, used by the
        # rebalance heuristic (§6.14). Each side accumulates one entry
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
        # probe fires. Atomic-write friendly (single write, no append).
        with self._probe_csv_path.open("w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(["generation", "side", "opponent_index", "fitness"])

    # ------------------------------------------------------------------
    # Per-side construction
    # ------------------------------------------------------------------

    def _build_prey_state(self) -> _SideState:
        """Construct the prey side's `_SideState` with gen-0 warmstart.

        Per D12: the prey side's gen-0 elite is loaded from a warmstart
        genome JSON (typically an M3 lamarckian-LSTMPPO elite from
        logbook 013). When `prey_gen0_seed_path` is None, falls back
        to `x0=zeros` — useful for tests and non-warmstart pilots.
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

        Per D7: arm A pretrains via `pretrain_against_heuristic`
        (cost amortised across the run); arm B cold-starts at zeros.
        Pretrain runs inside `__init__` rather than as a pre-computed
        bundle so a fresh checkout can run the campaign without an
        artefact dependency.
        """
        cfg = self.coevolution_config
        encoder = MLPPPOPredatorEncoder()
        fitness = PredatorEpisodicKillRate()
        inheritance = NoInheritance()
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

        File format (matches PR 4 task 7.0b's production bundle):
        `{"genome_id": str, "generation": int, "fitness": float,
        "params": list[float], "brain_config": {...}}`. Only `params`
        is read here; the rest is for provenance + aggregator
        introspection.

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
                "Either provide a valid warmstart genome JSON (per PR 4 "
                "task 7.0b's bundle) or set the field to null in the YAML "
                "to fall back to cold-start zeros."
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
                "current prey brain config."
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
        """Run heuristic-imitation pretrain (D7 arm A) and return flattened weights.

        Builds a fresh `MLPPPOPredatorBrain` with the encoder's default
        config, trains it with `pretrain_against_heuristic`, then
        encodes the brain via the `MLPPPOPredatorEncoder`'s
        `WeightPersistence` round-trip to produce a flat weight
        vector suitable for `CMAESOptimizer(x0=...)`.

        Cost ~30s at the default 50-batch budget; runs once per
        `CoevolutionLoop` construction (NOT per K-block).
        """
        # Local imports — pretrain helper pulls torch + the heuristic
        # brain machinery; defer until the arm-A path is actually
        # selected so cold-start runs don't pay the import cost.
        from quantumnematode.env._predator_brain_pretrain import (
            pretrain_against_heuristic,
        )
        from quantumnematode.env.mlpppo_predator_brain import MLPPPOPredatorBrain
        from quantumnematode.env.predator_brain import HeuristicPredatorBrain
        from quantumnematode.evolution.encoders import (
            _flatten_components,
            _select_genome_components,
        )

        logger.info(
            "Predator gen-0: running heuristic-imitation pretrain (D7 arm A)...",
        )
        student = MLPPPOPredatorBrain()
        teacher = HeuristicPredatorBrain()
        seed = int(self.rng.integers(0, 2**31 - 1))
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
                "verify MLPPPOPredatorEncoder vs MLPPPOPredatorBrain wire-up."
            )
            raise ValueError(msg)
        logger.info(
            "Predator gen-0: pretrain complete; encoded into %d floats for x0.",
            len(flat),
        )
        return flat

    # ------------------------------------------------------------------
    # Held-out opponent construction (§6.10)
    # ------------------------------------------------------------------

    def _load_held_out_prey_bundle(self) -> list[Genome]:
        """Load held-out prey genomes from `configs/evolution/coevolution_held_out_prey/*.json`.

        Each JSON file in the bundle directory is expected to follow the
        warmstart fixture format (`{genome_id, generation, fitness,
        params: list[float], brain_config}`). Files whose `params`
        length doesn't match the prey encoder's `genome_dim` are
        skipped with a warning rather than crashing the loop.

        PR 3 ships the loader path; the production bundle (~8 real M3
        elites) is curated in PR 4 task 7.0a. When the directory is
        empty or missing the prey-side probe is a no-op for this run
        (logged as a one-time warning at __init__) — the probe still
        fires for the predator side, and CI smoke runs work without
        the bundle.

        Sampling: when `held_out_size > len(bundle)` we sample WITH
        replacement; when `held_out_size <= len(bundle)` WITHOUT
        replacement. Both via `_held_out_rng.choice` so the same
        master seed reproduces the held-out set.
        """
        cfg = self.coevolution_config
        # Bundle path is fixed by spec ("Held-Out Set Construction"
        # scenario). Resolved relative to the repo root via cwd —
        # tests can patch this attribute by passing a custom
        # `_prey_held_out_dir` override when subclassing for tests.
        bundle_dir = Path("configs/evolution/coevolution_held_out_prey")
        if not bundle_dir.is_dir():
            logger.warning(
                "Prey held-out bundle missing at %s; prey-side probe will be a "
                "no-op for this run. The production bundle ships in PR 4 task 7.0a.",
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

        # Down-sample / up-sample to held_out_size. The spec scenario is
        # explicit about with/without-replacement semantics here.
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

        Default grid per spec scenario "Held-Out Set Construction":
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
        bundle_dir = Path("configs/evolution/coevolution_held_out_prey")
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
    # Per-K-block CMA-ES re-construction (D2)
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
        transition (per D2: re-construct rather than reset, since the
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

        Combines the run's master `rng` state with the side and
        K-block index so two K-blocks (across sides or generations)
        get distinct optimiser samples while a re-run with the same
        master seed reproduces the same trajectory.
        """
        # `rng.integers` with side-specific offset salts the seed so
        # prey K-block 0 and predator K-block 0 are distinct streams.
        side_offset = 0 if side == "prey" else 1
        # Mix in the k_block_index so every block gets a fresh draw.
        salt = f"{side}-{k_block_index}".encode()
        # Fast deterministic hash via numpy: take 32 bits of the
        # `rng.integers` draw XORed with a hash of the salt string.
        # Avoids importing hashlib for what's a low-stakes seed mix.
        salt_int = int.from_bytes(salt[:8].ljust(8, b"\0"), "big") & 0x7FFFFFFF
        master = int(self.rng.integers(0, 2**31 - 1))
        return (master ^ salt_int ^ (k_block_index * 31) ^ (side_offset * 17)) & 0x7FFFFFFF

    def _rebuild_optimizer(self, side: _SideState) -> None:
        """Re-construct `side.optimizer` at a K-block transition (D2).

        Carries over the side's current best (= last K-block's elite)
        as the new `x0` so the optimizer continues from the right
        starting point; `sigma0` resets to the configured value (the
        re-construction is the equivalent of "reset" since CMA's `cma`
        library exposes no public reset method).
        """
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
    # Rebalance heuristic (§6.14)
    # ------------------------------------------------------------------

    def _evaluate_rebalance(
        self,
        training_side: _SideState,
        opposing_side: _SideState,  # noqa: ARG002 — accepted for symmetry + future use
    ) -> bool:
        """Return True if the current side SHALL flip; False to grant an extra K-block.

        Per design.md Open Question 1 + Risk register row "One side
        dominates → other's gradient saturates": when the
        `rebalance_threshold` knob is set and a side's K-block-mean
        fitness drops below `rebalance_threshold * opposing_side_mean`
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
    # Checkpoint / resume (§6.11)
    # ------------------------------------------------------------------

    def _save_checkpoint(self) -> None:
        """Persist full loop state across two per-side files + one top-level JSON.

        Per task 6.11 + spec scenario "Probe Cadence and Output Layout":

        - `{output_dir}/prey/checkpoint.pkl`: per-side pickle containing
          optimizer + population + prev_generation_ids + generation +
          champion_history. Reuses the existing `EvolutionLoop`
          checkpoint shape so M3 single-population resume tooling
          can introspect.
        - `{output_dir}/predator/checkpoint.pkl`: same shape.
        - `{output_dir}/coevolution_state.json`: top-level JSON
          containing co-evolution-specific state: K-block index,
          alternating-schedule cursor, both HoFs (via `HallOfFame.to_dict`),
          held-out predator specs, prey held-out genome IDs (the
          full bundle reloads from disk on resume — no need to
          re-serialise the params).
        - `{output_dir}/coevolution_rng.pkl`: RNG state for the
          master `rng` and the held-out RNG. Pickled separately
          because numpy bit_generator state has nested arrays that
          don't JSON natively.

        Atomic write via tmp file + rename for each output to avoid
        torn writes if the process is killed mid-checkpoint.
        """
        # Per-side pickles. The optimizer carries ALL CMA-ES adaptive
        # state (covariance, sigma, mean, generation counter inside the
        # cma library); pickling it directly is the same approach
        # `EvolutionLoop._save_checkpoint` uses.
        for side in (self.prey, self.predator):
            payload = {
                "checkpoint_version": CHECKPOINT_VERSION,
                "side": side.name,
                "optimizer": side.optimizer,
                "population_params": [g.params.tolist() for g in side.population],
                "population_genome_ids": [g.genome_id for g in side.population],
                "prev_generation_ids": list(side.prev_generation_ids),
                "generation": side.generation,
                "champion_history": list(side.champion_history),
                # Match `EvolutionLoop._save_checkpoint`'s shape: persist
                # the inheritance literal so resume can compare against
                # the resolved current value and reject mid-run drift.
                # In practice `CoevolutionConfig._validate_invariants`
                # hardcodes inheritance per side at YAML load time, so
                # a mid-run config edit is rejected before reaching
                # the loop — but the field travels with the per-side
                # checkpoint anyway, matching M3 single-population
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

        # Top-level champion_history.json per spec scenario "Probe
        # Cadence and Output Layout" — the aggregator (PR 5) reads
        # this file. Format: `{prey: list[dict], predator: list[dict]}`
        # where each dict is a champion_history entry as documented in
        # the module docstring (`{genome_id, generation, k_block_index,
        # fitness, params: list[float]}`). Already JSON-serialisable
        # since the per-K-block append in `_run_one_k_block` calls
        # `params.tolist()`.
        champion_payload = {
            "prey": list(self.prey.champion_history),
            "predator": list(self.predator.champion_history),
        }
        self._atomic_json_write(
            self.output_dir / "champion_history.json",
            champion_payload,
        )

        # RNG state pickle (separate file because numpy bit_generator
        # state is awkward to JSON).
        rng_payload = {
            "master_rng_state": self.rng.bit_generator.state,
            "held_out_rng_state": self._held_out_rng.bit_generator.state,
        }
        self._atomic_pickle_write(
            self.output_dir / "coevolution_rng.pkl",
            rng_payload,
        )

    def _load_checkpoint(self) -> None:
        """Restore full loop state from the four checkpoint files.

        Inverse of `_save_checkpoint`. Reads per-side pickles to
        restore optimizer + population + champion_history; reads the
        top-level JSON to restore K-block index, side cursor, HoFs;
        reads the RNG pickle to restore master + held-out RNG state.

        Resume invariants verified:

        - Each per-side checkpoint's `checkpoint_version` matches
          `CHECKPOINT_VERSION`; mismatch raises `ValueError` to
          prevent silent state corruption.
        - The two per-side files agree on `k_block_index` (cross-
          checked against the top-level JSON).

        Held-out sets are NOT re-serialised in detail — the prey
        bundle is re-read from `configs/evolution/coevolution_held_out_prey/*.json`
        and the recorded genome IDs are checked against the freshly
        loaded set so a resume across a bundle-edit fails loudly.
        Predator held-out specs are restored from the JSON directly
        (they're tiny tuples).
        """
        import pickle

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

        # RNG restore MUST happen BEFORE the held-out re-sample below.
        # The held-out bundle was originally drawn in `__init__` using
        # `_held_out_rng` seeded from `self.rng.integers(...)` — but
        # on this resume path, both RNGs were re-seeded fresh in
        # `__init__` (different state from the saved run). Restoring
        # `self.rng` and `self._held_out_rng` to their checkpointed
        # state lets the bundle re-sample below produce the same
        # subset that was saved.
        rng_path = self.output_dir / "coevolution_rng.pkl"
        if not rng_path.exists():
            msg = (
                f"CoevolutionLoop._load_checkpoint: RNG state missing at "
                f"{rng_path}. All four checkpoint files are required."
            )
            raise FileNotFoundError(msg)
        with rng_path.open("rb") as fh:
            rng_payload = pickle.load(fh)  # noqa: S301 — trusted local file
        self.rng.bit_generator.state = rng_payload["master_rng_state"]
        self._held_out_rng.bit_generator.state = rng_payload["held_out_rng_state"]

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
           rebuilt per D2.

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
            # rebalance heuristic), persist state, then rebuild the
            # newly-training side's optimizer per D2. Order matters:
            # increment k_block_index BEFORE saving so the checkpoint
            # records the post-block state (i.e. resume picks up from
            # the *next* block, not the one that just finished).
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
            self._save_checkpoint()
            if self._k_block_index < total_blocks:
                next_training_side = self.prey if self._current_side == "prey" else self.predator
                self._rebuild_optimizer(next_training_side)

        logger.info("CoevolutionLoop: run complete.")

    def _run_one_k_block(
        self,
        training_side: _SideState,
        opposing_side: _SideState,
    ) -> None:
        """Run `K_per_block` generations of `training_side` vs frozen `opposing_side`.

        Per-generation flow:

        1. `optimizer.ask()` produces population_size flat float vectors.
        2. Wrap each into a `Genome` with `genome_id_for(...)`; record
           on `training_side.population` for opposition lookup.
        3. For each candidate, build the opposition set via
           `opposing_side.hof.mix_with_pop(rng, opposing.population,
           frac_hof=0.3)` (per spec "70/30 Mixture During Evaluation").
           Empty-HoF fallback returns all-from-pop (per "Empty HoF
           Fallback").
        4. Evaluate the candidate against the opposition (PR 4 wires
           opposition into `sim_config`; PR 3 invokes
           `fitness.evaluate(...)` with the unpatched config — see
           module docstring "Opposition injection (deferred to PR 4)").
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
        # K-block mean for the rebalance heuristic (§6.14).
        block_fitnesses: list[float] = []

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
            gen_genomes: list[Genome] = []
            fitnesses: list[float] = []

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

                # HoF-mixed opposition. Even though the actual
                # opposition-injection wiring is deferred to PR 4, we
                # still draw the mix here so the deterministic
                # RNG-state advancement is correct (the loop's RNG must
                # be in the same state at the same generation index
                # whether or not the opposition was actually injected
                # downstream).
                opposition = self._build_opposition(opposing_side)

                # Per-evaluation seed derivation matches EvolutionLoop's
                # pattern (each child gets a distinct stream from the
                # master rng).
                eval_seed = int(self.rng.integers(0, 2**31 - 1))
                fitness_val = self._evaluate_candidate(
                    side=training_side,
                    genome=genome,
                    opposition=opposition,
                    eval_seed=eval_seed,
                )
                fitnesses.append(fitness_val)

                # Lineage row.
                training_side.lineage.record(
                    genome,
                    fitness=fitness_val,
                    brain_type=training_side.encoder.brain_name,
                    inherited_from="",
                )

                # Track block elite. Strict-greater so first-seen
                # high-tier wins on ties (preserves recency of the
                # original champion in the K-block).
                if fitness_val > block_elite_fitness:
                    block_elite_fitness = fitness_val
                    block_elite_genome = genome

            # Tell optimiser (CMA-ES minimises; our fitness maximises).
            training_side.optimizer.tell(list(solutions), [-f for f in fitnesses])
            block_fitnesses.extend(fitnesses)

            # Update side-state population + bookkeeping for next gen.
            training_side.population = gen_genomes
            training_side.prev_generation_ids = [g.genome_id for g in gen_genomes]
            training_side.generation += 1

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
        # most-recent K-block's champion. See spec scenario "Probe
        # Cadence and Output Layout" for the full contract.
        if (
            cfg.generality_probe_every > 0
            and training_side.generation > 0
            and training_side.generation % cfg.generality_probe_every == 0
        ):
            self._fire_generality_probe()

    # ------------------------------------------------------------------
    # Opposition + evaluation (§6.7-§6.8)
    # ------------------------------------------------------------------

    def _build_opposition(
        self,
        opposing_side: _SideState,
    ) -> list[Genome]:
        """Build the opposition list for one candidate evaluation.

        Per spec scenario "70/30 Mixture During Evaluation": draw
        `opposing_side.evolution_config.population_size` opposition
        genomes via the HoF mix. Empty-pop fallback (very first
        K-block, opposing side has no population yet) returns an
        empty list — `_evaluate_candidate` handles that by treating
        the genome as un-opposed (degenerate fitness; see method
        docstring).
        """
        if not opposing_side.population:
            # First K-block of the run: the opposing side hasn't
            # populated its `population` yet (it sits at the gen-0
            # optimiser mean but no `ask()` has fired). Return empty —
            # `_evaluate_candidate` treats this as the un-opposed base
            # case to bootstrap the first block.
            return []
        # Pin frac_hof=0.3 at the call site rather than relying on
        # `HallOfFame.DEFAULT_FRAC_HOF` so the 70/30 contract from
        # design.md D3 + the spec scenario "70/30 Mixture During
        # Evaluation" lives in the loop, not in the buffer's default.
        # If the buffer's default ever drifts (e.g. for an unrelated
        # use), the loop's contract stays correct.
        return opposing_side.hof.mix_with_pop(
            self.rng,
            opposing_side.population,
            frac_hof=0.3,
        )

    def _evaluate_candidate(
        self,
        *,
        side: _SideState,
        genome: Genome,
        opposition: list[Genome],
        eval_seed: int,
    ) -> float:
        """Evaluate one candidate genome against the sampled opposition.

        For PR 3 this dispatches directly to `side.fitness.evaluate(...)`
        with the unpatched `sim_config`. The opposition list is recorded
        but not yet injected into the env (see module docstring
        "Opposition injection (deferred to PR 4)"). PR 4 will:

        1. Pre-decode each opposition genome into its brain instance
           (calling `opposing_side.encoder.decode(...)` against
           `sim_config`).
        2. Patch `sim_config` to point at the decoded brains:
           - Prey-side training: install decoded predator brains on
             `env.predators[i].brain` post-construction (re-using the
             pattern from `predator_fitness._build_env_with_genome_predators`).
           - Predator-side training: populate
             `sim_config.multi_agent.agents` with prey-side
             BrainContainerConfigs whose weights point at the decoded
             prey brains.
        3. Pass the patched `sim_config` to `fitness.evaluate`.

        The opposition argument is documented + accepted here so the
        interface is stable across the PR 3 → PR 4 boundary; only the
        body changes.
        """
        del opposition  # PR 4 will consume; PR 3 records via ask/tell only
        return side.fitness.evaluate(
            genome,
            self.sim_config,
            side.encoder,
            episodes=side.evolution_config.episodes_per_eval,
            seed=eval_seed,
        )

    # ------------------------------------------------------------------
    # Generality probe (§6.9)
    # ------------------------------------------------------------------

    def _fire_generality_probe(self) -> None:
        """Evaluate each side's current elite against its held-out set.

        Writes one row per (side, opponent_index) pair to
        `{output_dir}/generality_probe.csv` per spec scenario "Probe
        Cadence and Output Layout". Does NOT mutate any side's
        population, optimizer, or hof (per "Probe Does Not Mutate
        Population State"); the probe runs in append-mode against
        the CSV.

        Per side: the elite is the latest entry in `champion_history`
        (the K-block elite). When a side has no champions yet (first
        K-block hasn't completed), the probe is a no-op for that side.
        Held-out evaluations themselves still use the deferred-to-PR-4
        wiring — opposition is recorded but not injected end-to-end
        in PR 3 (see `_evaluate_candidate`).
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
                # Per-side held-out set: prey loaded as Genome list;
                # predator stored as (detection_radius, damage_radius)
                # tuples. The actual probe evaluation uses the same
                # deferred-to-PR-4 wiring as training evaluations.
                # Currently the loop records the probe schema (one row
                # per opponent) with a placeholder fitness of NaN —
                # PR 4 plugs in real evaluation here.
                if side.name == "prey":
                    held_out_count = len(self._prey_held_out)
                else:
                    held_out_count = len(self._predator_held_out_specs)
                for opp_idx in range(held_out_count):
                    # Probe RNG seed is derived per-fire so the rng
                    # advances deterministically — PR 4 will consume.
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

        PR 3 records the probe schema with a NaN fitness; PR 4 plugs
        in real evaluation by patching `sim_config` (heuristic-radius
        predator for prey-side probes; M3 elite prey genome for
        predator-side probes) and calling the side's fitness function.
        See `_evaluate_candidate` for the full PR-4 plan.
        """
        del side, elite, opponent_index, eval_seed
        return float("nan")
