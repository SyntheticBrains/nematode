"""Hall-of-Fame buffer for past champion genomes.

A bounded buffer with a configurable replacement policy. Two roles in the
M5 co-evolution loop:

1. **Co-evolution opposition pool.** When evaluating a candidate on side X,
   draw a fraction of opponents from side Y's HoF (preserves strong past
   champions so they don't get forgotten when Y's current population
   regresses).
2. **Single-population novelty-search primitive.** A general "remember
   the best you've seen" buffer that any optimiser can plug into without
   the co-evolution machinery.

Two replacement policies:

- ``"quality"`` (default): on push at capacity, evict the lowest-fitness
  entry iff the new fitness is strictly greater. Preserves the strongest
  champions seen so far.
- ``"fifo"``: on push at capacity, evict the oldest-pushed entry
  regardless of fitness. Reserved as an ablation for the co-evolution
  paper to test whether quality-based eviction matters.

Checkpoint serialisation via :meth:`HallOfFame.to_dict` / :meth:`from_dict`
preserves capacity, policy, entry order, and every genome's
flat-parameter array. ``params`` round-trip through `list[float]`
(so the dict is JSON-serialisable without a numpy dependency on the
read side).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from quantumnematode.evolution.genome import Genome

if TYPE_CHECKING:
    from collections.abc import Sequence


# Default fraction of opponents drawn from the HoF in `mix_with_pop`. Per
# design.md D3: 70% current pop / 30% HoF. Preserves the live signal as
# the primary driver while preventing forgetting.
DEFAULT_FRAC_HOF = 0.3


@dataclass
class _HoFEntry:
    """Internal entry pairing a genome with its observed fitness.

    Kept module-private so the public surface only exposes :class:`Genome`
    instances; tests that need to inspect fitness can read
    :meth:`HallOfFame.fitnesses`.
    """

    genome: Genome
    fitness: float


class HallOfFame:
    """Bounded buffer of past champion genomes with replacement policy.

    Parameters
    ----------
    capacity
        Maximum number of entries. Must be >= 1.
    replacement
        Eviction policy on push at capacity:

        - ``"quality"`` (default): evict the lowest-fitness entry iff the
          new fitness is strictly greater.
        - ``"fifo"``: evict the oldest-pushed entry regardless.

    Raises
    ------
    ValueError
        If ``capacity < 1`` or ``replacement`` is not one of the
        supported literals.
    """

    def __init__(
        self,
        capacity: int,
        *,
        replacement: Literal["quality", "fifo"] = "quality",
    ) -> None:
        if capacity < 1:
            msg = f"HallOfFame capacity must be >= 1, got {capacity}"
            raise ValueError(msg)
        if replacement not in {"quality", "fifo"}:
            msg = f"HallOfFame replacement must be 'quality' or 'fifo', got {replacement!r}"
            raise ValueError(msg)
        self._capacity = capacity
        self._replacement: Literal["quality", "fifo"] = replacement
        # `deque` keeps insertion order cheaply (FIFO eviction is O(1)) and
        # supports indexed access for the quality-based eviction lookup.
        self._entries: deque[_HoFEntry] = deque()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def capacity(self) -> int:
        """Maximum number of entries the buffer can hold."""
        return self._capacity

    @property
    def replacement(self) -> Literal["quality", "fifo"]:
        """Configured eviction policy."""
        return self._replacement

    def __len__(self) -> int:
        """Return the current number of entries (<= capacity)."""
        return len(self._entries)

    def fitnesses(self) -> list[float]:
        """Snapshot of current fitnesses in insertion order.

        Useful for tests + the rebalance heuristic in
        `CoevolutionLoop`. The list is a copy; mutating it does not
        affect the HoF.
        """
        return [entry.fitness for entry in self._entries]

    def genomes(self) -> list[Genome]:
        """Snapshot of current genomes in insertion order."""
        return [entry.genome for entry in self._entries]

    # ------------------------------------------------------------------
    # Push / sample
    # ------------------------------------------------------------------

    def push(self, genome: Genome, fitness: float) -> bool:
        """Insert ``genome`` with its observed ``fitness`` into the buffer.

        Returns
        -------
        bool
            ``True`` if the entry was inserted (either because the buffer
            was below capacity or because eviction made room),
            ``False`` if the entry was rejected by the quality-based
            policy (new fitness <= all existing entries).
        """
        if len(self._entries) < self._capacity:
            self._entries.append(_HoFEntry(genome=genome, fitness=fitness))
            return True

        if self._replacement == "fifo":
            self._entries.popleft()
            self._entries.append(_HoFEntry(genome=genome, fitness=fitness))
            return True

        # Quality-based: find the lowest-fitness entry; replace iff the new
        # fitness is strictly greater. Tie at the lowest is rejected so the
        # buffer doesn't churn on equal-fitness pushes (preserves recency
        # of the original champion).
        lowest_idx = min(
            range(len(self._entries)),
            key=lambda i: self._entries[i].fitness,
        )
        if fitness > self._entries[lowest_idx].fitness:
            # `deque` doesn't support indexed deletion in O(1); convert to
            # list, swap, and rebuild the deque. HoF capacities in M5 are
            # 8 (per design.md D3), so the cost is trivial.
            entries = list(self._entries)
            entries[lowest_idx] = _HoFEntry(genome=genome, fitness=fitness)
            self._entries = deque(entries)
            return True
        return False

    def sample(
        self,
        rng: np.random.Generator,
        n: int,
        *,
        replace: bool = True,
    ) -> list[Genome]:
        """Sample ``n`` genomes from the HoF.

        Parameters
        ----------
        rng
            Seeded `numpy.random.Generator` for reproducibility.
        n
            Number of genomes to draw.
        replace
            When True (default), draws with replacement. When False,
            ``n`` must not exceed ``len(self)`` or `ValueError` is
            raised.

        Returns
        -------
        list[Genome]
            ``n`` genomes drawn from the buffer.

        Raises
        ------
        ValueError
            If ``n < 0``, or if ``replace=False`` and ``n > len(self)``,
            or if the buffer is empty.
        """
        if n < 0:
            msg = f"HallOfFame.sample n must be >= 0, got {n}"
            raise ValueError(msg)
        if not self._entries:
            msg = "HallOfFame.sample called on empty buffer"
            raise ValueError(msg)
        if not replace and n > len(self._entries):
            msg = (
                f"HallOfFame.sample with replace=False requires "
                f"n <= len(self) ({len(self._entries)}), got n={n}"
            )
            raise ValueError(msg)
        if n == 0:
            return []
        indices = rng.choice(len(self._entries), size=n, replace=replace)
        return [self._entries[int(i)].genome for i in indices]

    def mix_with_pop(
        self,
        rng: np.random.Generator,
        pop: Sequence[Genome],
        *,
        frac_hof: float = DEFAULT_FRAC_HOF,
    ) -> list[Genome]:
        """Sample a mixed opposition set: ``frac_hof`` from HoF, rest from ``pop``.

        Returns a list of length ``len(pop)`` containing approximately
        ``round(frac_hof * len(pop))`` HoF samples and the remainder from
        ``pop``. Both draws use replacement so the returned list size
        is stable regardless of `len(self)` and `len(pop)`.

        Empty-HoF fallback: when ``len(self) == 0``, all entries come
        from ``pop`` (the live signal is the only thing available).
        Empty-pop is rejected with `ValueError` because the caller is
        almost certainly misusing the API.

        Parameters
        ----------
        rng
            Seeded RNG. Two HoFs with identical contents and identical
            seeded RNGs SHALL produce identical mix sequences (per spec
            scenario "Reproducible Sampling Under Seeded RNG").
        pop
            Current-generation population on the side being evaluated
            against.
        frac_hof
            Fraction of the returned list drawn from the HoF. Default
            0.3 per design.md D3.

        Raises
        ------
        ValueError
            If ``frac_hof`` is outside ``[0.0, 1.0]`` or ``pop`` is empty.
        """
        if not 0.0 <= frac_hof <= 1.0:
            msg = f"frac_hof must be in [0.0, 1.0], got {frac_hof}"
            raise ValueError(msg)
        if not pop:
            msg = "HallOfFame.mix_with_pop requires a non-empty pop"
            raise ValueError(msg)

        sample_size = len(pop)
        if not self._entries:
            # Empty-HoF fallback: all entries come from pop (sampled with
            # replacement so the return shape is stable).
            indices = rng.choice(sample_size, size=sample_size, replace=True)
            return [pop[int(i)] for i in indices]

        n_hof = round(frac_hof * sample_size)
        # Clamp into [0, sample_size] — defends against floating-point
        # rounding edge cases at frac_hof=0 / 1.
        n_hof = max(0, min(sample_size, n_hof))
        n_pop = sample_size - n_hof

        # Draw HoF samples first, then pop samples — two separate
        # rng.choice calls so the RNG-state advancement is deterministic
        # and reproducible across runs at fixed seed.
        hof_picks: list[Genome] = []
        if n_hof > 0:
            hof_indices = rng.choice(len(self._entries), size=n_hof, replace=True)
            hof_picks = [self._entries[int(i)].genome for i in hof_indices]
        pop_picks: list[Genome] = []
        if n_pop > 0:
            pop_indices = rng.choice(len(pop), size=n_pop, replace=True)
            pop_picks = [pop[int(i)] for i in pop_indices]
        return hof_picks + pop_picks

    # ------------------------------------------------------------------
    # Checkpoint
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-compatible dict.

        Genome ``params`` round-trip via `list[float]` so the resulting
        dict has no numpy dependency. ``birth_metadata`` is preserved
        verbatim — callers MUST ensure their birth_metadata values are
        JSON-serialisable (e.g. encoder shape_maps using tuples need
        list-conversion before reaching this method; the M5 weight
        encoders' `shape_map` is already tuple-keyed lists which json
        accepts).
        """
        return {
            "capacity": self._capacity,
            "replacement": self._replacement,
            "entries": [
                {
                    "genome": {
                        "params": entry.genome.params.tolist(),
                        "genome_id": entry.genome.genome_id,
                        "parent_ids": list(entry.genome.parent_ids),
                        "generation": entry.genome.generation,
                        "birth_metadata": entry.genome.birth_metadata,
                    },
                    "fitness": entry.fitness,
                }
                for entry in self._entries
            ],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> HallOfFame:
        """Reconstruct from a dict produced by :meth:`to_dict`.

        Insertion order is preserved (equivalent to ``deque(entries)``
        from the serialised list). Capacity and replacement policy are
        restored verbatim.
        """
        hof = cls(
            capacity=int(data["capacity"]),
            replacement=data["replacement"],
        )
        for entry_dict in data["entries"]:
            genome_dict = entry_dict["genome"]
            genome = Genome(
                params=np.asarray(genome_dict["params"], dtype=np.float32),
                genome_id=genome_dict["genome_id"],
                parent_ids=list(genome_dict["parent_ids"]),
                generation=int(genome_dict["generation"]),
                birth_metadata=dict(genome_dict.get("birth_metadata", {})),
            )
            hof._entries.append(
                _HoFEntry(genome=genome, fitness=float(entry_dict["fitness"])),
            )
        return hof
