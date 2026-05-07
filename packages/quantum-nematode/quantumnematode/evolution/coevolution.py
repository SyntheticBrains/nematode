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

This commit (PR 3 commit 5) ships the scaffold + alternating schedule +
per-K-block fresh `CMAESOptimizer` re-construction. Hall-of-Fame
integration, generality probe, checkpoint/resume, and full test suite
land in subsequent commits within this PR.

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

import json
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

from quantumnematode.evolution.encoders import LSTMPPOEncoder
from quantumnematode.evolution.fitness import LearnedPerformanceFitness
from quantumnematode.evolution.hall_of_fame import HallOfFame
from quantumnematode.evolution.inheritance import (
    LamarckianInheritance,
    NoInheritance,
)
from quantumnematode.evolution.predator_encoders import MLPPPOPredatorEncoder
from quantumnematode.evolution.predator_fitness import PredatorEpisodicKillRate
from quantumnematode.optimizers.evolutionary import CMAESOptimizer

if TYPE_CHECKING:
    from pathlib import Path

    import numpy as np

    from quantumnematode.evolution.encoders import GenomeEncoder
    from quantumnematode.evolution.fitness import FitnessFunction
    from quantumnematode.evolution.genome import Genome
    from quantumnematode.evolution.inheritance import InheritanceStrategy
    from quantumnematode.evolution.lineage import LineageTracker
    from quantumnematode.utils.config_loader import (
        CoevolutionConfig,
        EvolutionConfig,
        SimulationConfig,
    )

logger = logging.getLogger(__name__)


# Default Hall-of-Fame capacity per design.md D3 (8 entries; 70/30 mix
# preserves live signal while preventing forgetting).
DEFAULT_HOF_CAPACITY = 8


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
    # Lazily initialised lineage tracker (one per side; lives at
    # `{output_dir}/{side}/lineage.csv`). Created on first K-block.
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

        return _SideState(
            name="prey",
            encoder=encoder,
            fitness=fitness,
            optimizer=optimizer,
            inheritance=inheritance,
            evolution_config=evolution_cfg,
            output_dir=self.output_dir / "prey",
            hof=hof,
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

        return _SideState(
            name="predator",
            encoder=encoder,
            fitness=fitness,
            optimizer=optimizer,
            inheritance=inheritance,
            evolution_config=evolution_cfg,
            output_dir=self.output_dir / "predator",
            hof=hof,
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
    # Run loop (alternating schedule)
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Run `generation_pairs * 2` K-blocks under the alternating schedule.

        Each K-block trains one side for `K_per_block` generations
        while the opposing side is FROZEN (no `optimizer.tell()`).
        At the end of every K-block:

        1. The training-side block elite is appended to its
           `champion_history` (and pushed to its `hof` in commit 6).
        2. The current side flips.
        3. The just-flipped (now-training) side's optimizer is
           rebuilt per D2.

        HoF integration, opposition sampling, generality probe, and
        checkpoint/resume land in subsequent commits within this PR.
        This commit's `run()` is the controller scaffold — the actual
        per-generation `ask`/`tell`/`evaluate` plumbing is a stub
        marked NotImplemented so a smoke-test of the schedule
        fails loudly until the wiring lands.
        """
        cfg = self.coevolution_config
        total_blocks = cfg.generation_pairs * 2
        logger.info(
            "CoevolutionLoop: %d K-blocks total (%d generation pairs x 2 sides), "
            "K_per_block=%d, start_side=%s",
            total_blocks,
            cfg.generation_pairs,
            cfg.K_per_block,
            cfg.start_side,
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

            # Per-generation evaluation loop lands in commit 6 (HoF +
            # opposition sampling) and commit 7 (worker reuse). Until
            # then this is a structural placeholder: the run method's
            # contract is "drive the schedule"; the body is wired in
            # the next commit.
            self._run_one_k_block(training_side, opposing_side)

            # K-block end: flip side and rebuild the just-flipped (now-
            # training) side's optimizer per D2.
            self._k_block_index += 1
            self._current_side = "predator" if self._current_side == "prey" else "prey"
            if self._k_block_index < total_blocks:
                next_training_side = self.prey if self._current_side == "prey" else self.predator
                self._rebuild_optimizer(next_training_side)

        logger.info("CoevolutionLoop: run complete.")

    def _run_one_k_block(
        self,
        training_side: _SideState,
        opposing_side: _SideState,
    ) -> None:
        """Run `K_per_block` generations of `training_side` against frozen `opposing_side`.

        Stub for this commit (PR 3 §6.3-6.6). Per-generation `ask`/
        `tell`/`evaluate` plumbing — including HoF-mixed opposition
        sampling, generality probe, and worker dispatch — lands in
        commits 6 and 7 within this PR. Defining the method here lets
        the schedule controller call it and unit tests verify the
        K-block boundary semantics in commit 8 even before the body
        is implemented.
        """
        del training_side, opposing_side  # unused until commits 6/7 wire up the body
        msg = (
            "_run_one_k_block body lands in PR 3 commit 6 (HoF + opposition "
            "sampling) and commit 7 (worker reuse). The schedule controller "
            "(this commit) drives the K-block flips; the per-generation "
            "evaluation plumbing is wired separately for review tractability."
        )
        raise NotImplementedError(msg)
