# 034 supporting — connectome rewired-null control forensics

Supporting analysis for [Logbook 034](../../034-connectome-structure-controls.md). Committed
artefacts only (analysis outputs); the 16 raw per-run training `.out` files are uncommitted
evaluation forensics under `tmp/evaluations/t7-connectome-controls/` (gitignored, per the
025/029/030/033 precedent).

## Files

- `controls.json` — the `connectome_structure_controls.py` output: per-arm plateau-tail C3 ranked
  success, the paired-seed delta (wild-type − rewired) with the 80% bootstrap CI + BH-FDR, and the
  verdict.
- `success-per-seed.csv` — `arch, seed, plateau_success_pct, mean_foods` (16 rows, 2 arms × 8 seeds).

## Rewiring validation (real Cook-2019 connectome)

3709 chemical + 1093 gap edges, 302 neurons. `rewire_degree_preserving(seed=1)` moved **91% of
chemical edges**, preserved every neuron's in/out degree (chemical) and degree (gap) exactly, ran in
0.1s, and emitted no low-acceptance warning — `swaps_per_edge = 10` mixes cleanly. The null is a
genuine scramble, not a near-copy. Degree preservation guarantees the per-post fan-in — hence the
`w_chem` init scale and the `g_gap` fan-in normalisation — is identical to wild-type; only *which*
neurons connect changes.

## Matched-initialisation design

Each run draws its rewiring from a **dedicated** `np.random.default_rng(rewire_seed = run seed)`,
independent of the weight-init RNG, so the topology's `w_chem` init draws land at the same RNG state
as the wild-type run for that seed. The wild-type and rewired arms are therefore compared under
identical PPO recipe, budget (6000ep), substrate, and initialisation stream — the pairing isolates
topology, nothing else.

## The smoke → panel reversal (why n=8 is the bar)

The single-seed smoke (seed 1) read wild-type 57.2% > rewired 46.5% (~11 pt) — superficially "specific
wiring". At n=8 the direction **reverses and vanishes**: wild-type 52.78 vs rewired 56.06, d=−3.28,
q=0.770. Seed 1 was a favourable draw for wild-type (its best-but-one seed); the rewired-null's seed-1
draw was among its weakest. Across 8 paired seeds the difference washes into noise — the same
connectome seed-variance the 029 ranking flagged (per-seed success spans 35–68% for wild-type alone).

## Verdict

**DEGREE-STATISTICS.** Wild-type 52.78% vs rewired-null 56.06% (n=8); paired delta d=−3.28,
CI[−8.56, +1.61] (spans zero), BH-FDR q=0.770. The wild-type connectome is statistically
indistinguishable from its degree-preserving rewirings — its 029 5th-of-6 standing is a property of
its connectivity *statistics*, not its specific wiring.

## Reproduce

```shell
# 2 arms × 8 seeds × 6000ep (parallel, OMP_NUM_THREADS=1), then the harness:
#   configs/scenarios/foraging_predator_thermal/connectomeppo_small_continuous2d_combined_klinotaxis.yml
#   (+ the _rewired_null.yml variant)
uv run python scripts/analysis/connectome_structure_controls.py \
    --manifest <run-dir>/_manifest.txt --out controls.json
```
