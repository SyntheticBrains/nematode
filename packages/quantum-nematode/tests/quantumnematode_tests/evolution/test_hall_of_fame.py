"""Tests for :mod:`quantumnematode.evolution.hall_of_fame` (task 4.4).

Covers spec scenarios from `evolution-framework/spec.md` "Hall-of-Fame Buffer":

- Quality-Based Eviction (default).
- FIFO Eviction Ablation.
- Mix-With-Pop Sampling (70/30 fraction).
- Reproducible Sampling Under Seeded RNG.
- Checkpoint Round-Trip.

Plus input-validation cases (capacity, replacement, n) so future refactors
can't quietly drop the guards.
"""

from __future__ import annotations

import numpy as np
import pytest
from quantumnematode.evolution.genome import Genome
from quantumnematode.evolution.hall_of_fame import HallOfFame


def _make_genome(params_value: float, *, genome_id: str = "g") -> Genome:
    """Construct a minimal `Genome` for HoF testing.

    Tests don't exercise the encoder round-trip — the HoF is brain-
    agnostic and just tracks `(genome, fitness)` pairs. A 1-D
    `params` vector with one float is enough to verify identity-by-
    inspection in checkpoint round-trip tests.
    """
    return Genome(
        params=np.array([params_value], dtype=np.float32),
        genome_id=genome_id,
        parent_ids=[],
        generation=0,
    )


# ---------------------------------------------------------------------------
# Construction + input validation
# ---------------------------------------------------------------------------


class TestConstruction:
    """Constructor surface + input-validation guards."""

    def test_default_replacement_is_quality(self) -> None:
        """Default `replacement` SHALL be `"quality"` per design.md D3."""
        hof = HallOfFame(capacity=8)
        assert hof.replacement == "quality"

    def test_capacity_below_one_raises(self) -> None:
        """`capacity < 1` SHALL raise `ValueError`."""
        with pytest.raises(ValueError, match="capacity"):
            HallOfFame(capacity=0)

    def test_invalid_replacement_raises(self) -> None:
        """Unknown replacement policy SHALL raise `ValueError`."""
        with pytest.raises(ValueError, match="replacement"):
            HallOfFame(capacity=8, replacement="oldest")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Quality-based eviction (default)
# ---------------------------------------------------------------------------


class TestQualityEviction:
    """Spec scenario "Quality-Based Eviction (Default)"."""

    def test_below_capacity_appends_unconditionally(self) -> None:
        """Pushes below capacity SHALL always succeed and grow the buffer."""
        hof = HallOfFame(capacity=3, replacement="quality")
        for i in range(3):
            assert hof.push(_make_genome(float(i)), fitness=float(i)) is True
        assert len(hof) == 3

    def test_at_capacity_higher_fitness_evicts_lowest(self) -> None:
        """At capacity, a strictly-greater fitness SHALL evict the lowest entry."""
        hof = HallOfFame(capacity=3, replacement="quality")
        hof.push(_make_genome(1.0, genome_id="a"), fitness=0.1)
        hof.push(_make_genome(2.0, genome_id="b"), fitness=0.5)
        hof.push(_make_genome(3.0, genome_id="c"), fitness=0.3)
        assert hof.fitnesses() == [0.1, 0.5, 0.3]

        # New fitness 0.4 > lowest existing (0.1) → "a" evicted.
        assert hof.push(_make_genome(4.0, genome_id="d"), fitness=0.4) is True
        ids = [g.genome_id for g in hof.genomes()]
        assert "a" not in ids
        assert "d" in ids
        assert sorted(hof.fitnesses()) == [0.3, 0.4, 0.5]

    def test_at_capacity_equal_or_lower_fitness_rejected(self) -> None:
        """At capacity, fitness ≤ lowest existing SHALL leave buffer unchanged."""
        hof = HallOfFame(capacity=3, replacement="quality")
        hof.push(_make_genome(1.0, genome_id="a"), fitness=0.1)
        hof.push(_make_genome(2.0, genome_id="b"), fitness=0.5)
        hof.push(_make_genome(3.0, genome_id="c"), fitness=0.3)

        # Equal to lowest (0.1): rejected — preserves the original
        # champion's recency on ties.
        assert hof.push(_make_genome(99.0, genome_id="reject"), fitness=0.1) is False
        # Below lowest: also rejected.
        assert hof.push(_make_genome(99.0, genome_id="reject"), fitness=0.0) is False

        ids = [g.genome_id for g in hof.genomes()]
        assert ids == ["a", "b", "c"]

    def test_quality_with_capacity_one(self) -> None:
        """Capacity-1 quality buffer SHALL hold only the strongest entry seen."""
        hof = HallOfFame(capacity=1, replacement="quality")
        hof.push(_make_genome(1.0, genome_id="weak"), fitness=0.1)
        hof.push(_make_genome(2.0, genome_id="strong"), fitness=0.9)
        hof.push(_make_genome(3.0, genome_id="weaker"), fitness=0.05)
        ids = [g.genome_id for g in hof.genomes()]
        assert ids == ["strong"]


# ---------------------------------------------------------------------------
# FIFO eviction ablation
# ---------------------------------------------------------------------------


class TestFIFOEviction:
    """Spec scenario "FIFO Eviction Ablation"."""

    def test_fifo_evicts_oldest_regardless_of_fitness(self) -> None:
        """At capacity, FIFO SHALL evict the oldest-pushed entry on every push."""
        hof = HallOfFame(capacity=3, replacement="fifo")
        # Push descending fitnesses — quality eviction would reject the
        # last two; FIFO must accept them and roll the buffer.
        hof.push(_make_genome(1.0, genome_id="a"), fitness=0.9)
        hof.push(_make_genome(2.0, genome_id="b"), fitness=0.5)
        hof.push(_make_genome(3.0, genome_id="c"), fitness=0.3)
        assert hof.push(_make_genome(4.0, genome_id="d"), fitness=0.05) is True
        assert hof.push(_make_genome(5.0, genome_id="e"), fitness=0.01) is True

        ids = [g.genome_id for g in hof.genomes()]
        # First two pushed were "a" and "b" — both should be evicted; "c"
        # is now the oldest survivor.
        assert ids == ["c", "d", "e"]


# ---------------------------------------------------------------------------
# Mix-with-pop sampling
# ---------------------------------------------------------------------------


class TestMixWithPop:
    """Spec scenarios "Mix-With-Pop Sampling" + "Reproducible Sampling"."""

    def test_mix_fraction_approximately_correct(self) -> None:
        """`mix_with_pop` SHALL return ~`frac_hof * len(pop)` HoF entries.

        With `frac_hof=0.3` and `len(pop)=10`, expect 3 HoF samples and
        7 pop samples. Uses sentinel genome IDs so HoF vs pop origin is
        identifiable in the result.
        """
        hof = HallOfFame(capacity=4, replacement="quality")
        for i in range(4):
            hof.push(_make_genome(float(i), genome_id=f"hof_{i}"), fitness=float(i))
        pop = [_make_genome(99.0, genome_id=f"pop_{i}") for i in range(10)]

        rng = np.random.default_rng(seed=42)
        out = hof.mix_with_pop(rng, pop, frac_hof=0.3)

        assert len(out) == 10
        n_from_hof = sum(1 for g in out if g.genome_id.startswith("hof_"))
        n_from_pop = sum(1 for g in out if g.genome_id.startswith("pop_"))
        assert n_from_hof == 3  # round(0.3 * 10)
        assert n_from_pop == 7

    def test_empty_hof_falls_back_to_all_pop(self) -> None:
        """When HoF is empty, `mix_with_pop` SHALL return all-from-pop."""
        hof = HallOfFame(capacity=4, replacement="quality")  # empty
        pop = [_make_genome(99.0, genome_id=f"pop_{i}") for i in range(5)]
        rng = np.random.default_rng(seed=42)
        out = hof.mix_with_pop(rng, pop, frac_hof=0.3)
        assert len(out) == 5
        assert all(g.genome_id.startswith("pop_") for g in out)

    def test_reproducible_sampling_under_seeded_rng(self) -> None:
        """Two HoFs with identical contents + identical seed SHALL produce identical mixes."""
        hof_a = HallOfFame(capacity=4, replacement="quality")
        hof_b = HallOfFame(capacity=4, replacement="quality")
        for i in range(4):
            g = _make_genome(float(i), genome_id=f"hof_{i}")
            hof_a.push(g, fitness=float(i))
            hof_b.push(g, fitness=float(i))
        pop = [_make_genome(99.0, genome_id=f"pop_{i}") for i in range(8)]

        rng_a = np.random.default_rng(seed=42)
        rng_b = np.random.default_rng(seed=42)
        out_a = hof_a.mix_with_pop(rng_a, pop, frac_hof=0.3)
        out_b = hof_b.mix_with_pop(rng_b, pop, frac_hof=0.3)
        assert [g.genome_id for g in out_a] == [g.genome_id for g in out_b]

    def test_invalid_frac_hof_raises(self) -> None:
        """`frac_hof` outside `[0.0, 1.0]` SHALL raise `ValueError`."""
        hof = HallOfFame(capacity=4)
        hof.push(_make_genome(1.0), fitness=0.5)
        pop = [_make_genome(2.0)]
        rng = np.random.default_rng(seed=0)
        with pytest.raises(ValueError, match="frac_hof"):
            hof.mix_with_pop(rng, pop, frac_hof=1.5)
        with pytest.raises(ValueError, match="frac_hof"):
            hof.mix_with_pop(rng, pop, frac_hof=-0.1)

    def test_empty_pop_raises(self) -> None:
        """Empty `pop` SHALL raise `ValueError` (caller misuse)."""
        hof = HallOfFame(capacity=4)
        hof.push(_make_genome(1.0), fitness=0.5)
        rng = np.random.default_rng(seed=0)
        with pytest.raises(ValueError, match="non-empty pop"):
            hof.mix_with_pop(rng, [], frac_hof=0.3)


# ---------------------------------------------------------------------------
# Sample
# ---------------------------------------------------------------------------


class TestSample:
    """Direct `sample` surface (used by future tests + ablation tooling)."""

    def test_sample_with_replacement(self) -> None:
        """Sampling with replacement SHALL allow drawing more entries than the buffer holds."""
        hof = HallOfFame(capacity=2)
        hof.push(_make_genome(1.0, genome_id="a"), fitness=0.5)
        hof.push(_make_genome(2.0, genome_id="b"), fitness=0.7)
        rng = np.random.default_rng(seed=42)
        out = hof.sample(rng, n=5, replace=True)
        assert len(out) == 5
        assert all(g.genome_id in {"a", "b"} for g in out)

    def test_sample_without_replacement_oversize_raises(self) -> None:
        """`replace=False` with `n > len(self)` SHALL raise."""
        hof = HallOfFame(capacity=2)
        hof.push(_make_genome(1.0), fitness=0.5)
        with pytest.raises(ValueError, match="replace=False"):
            hof.sample(np.random.default_rng(seed=0), n=2, replace=False)

    def test_sample_empty_buffer_raises(self) -> None:
        """Sampling from an empty buffer SHALL raise."""
        hof = HallOfFame(capacity=4)
        with pytest.raises(ValueError, match="empty"):
            hof.sample(np.random.default_rng(seed=0), n=1)


# ---------------------------------------------------------------------------
# Checkpoint round-trip
# ---------------------------------------------------------------------------


class TestCheckpointRoundTrip:
    """Spec scenario "Checkpoint Round-Trip"."""

    def test_to_dict_from_dict_preserves_state(self) -> None:
        """`to_dict` → `from_dict` SHALL preserve capacity, policy, and entry order."""
        hof = HallOfFame(capacity=3, replacement="fifo")
        hof.push(_make_genome(1.0, genome_id="a"), fitness=0.1)
        hof.push(_make_genome(2.0, genome_id="b"), fitness=0.5)
        hof.push(_make_genome(3.0, genome_id="c"), fitness=0.3)

        restored = HallOfFame.from_dict(hof.to_dict())

        assert restored.capacity == 3
        assert restored.replacement == "fifo"
        assert [g.genome_id for g in restored.genomes()] == ["a", "b", "c"]
        assert restored.fitnesses() == [0.1, 0.5, 0.3]
        for original, restored_g in zip(hof.genomes(), restored.genomes(), strict=True):
            np.testing.assert_array_equal(original.params, restored_g.params)

    def test_round_trip_preserves_sampling_under_seeded_rng(self) -> None:
        """Restored HoF SHALL produce identical `mix_with_pop` sequences under same RNG."""
        hof = HallOfFame(capacity=3, replacement="quality")
        for i in range(3):
            hof.push(_make_genome(float(i), genome_id=f"hof_{i}"), fitness=float(i))
        pop = [_make_genome(99.0, genome_id=f"pop_{i}") for i in range(5)]

        restored = HallOfFame.from_dict(hof.to_dict())

        rng_a = np.random.default_rng(seed=12345)
        rng_b = np.random.default_rng(seed=12345)
        out_a = hof.mix_with_pop(rng_a, pop, frac_hof=0.3)
        out_b = restored.mix_with_pop(rng_b, pop, frac_hof=0.3)
        assert [g.genome_id for g in out_a] == [g.genome_id for g in out_b]

    def test_round_trip_through_json(self) -> None:
        """`to_dict` output SHALL be JSON-serialisable and survive json round-trip."""
        import json

        hof = HallOfFame(capacity=2, replacement="quality")
        hof.push(_make_genome(1.0, genome_id="a"), fitness=0.5)
        hof.push(_make_genome(2.0, genome_id="b"), fitness=0.7)

        # json.dumps must accept the dict; json.loads must give a dict
        # with the same shape from_dict accepts.
        as_json = json.dumps(hof.to_dict())
        restored = HallOfFame.from_dict(json.loads(as_json))
        assert restored.fitnesses() == [0.5, 0.7]
