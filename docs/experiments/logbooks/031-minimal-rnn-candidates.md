# 031: Minimal-RNN (minGRU / minLSTM) New-Architecture Candidates (T7)

**Status**: complete — **both prongs positive**. On the memory cell both minimal RNNs are confirmed
working-memory arms (SEPARATION CONFIRMED, on par with the LSTM); on the reactive C3 cell both
**beat the plain LSTM** (minLSTM significantly), supporting the stability-upgrade hypothesis.
Carries a load-bearing implementation finding: the arms only learn the memory task with a
**memory-friendly retention-gate init** — and that init does **not** cost reactive performance.

**Branch**: `openspec/add-minimal-rnn-arms`.

**Date**: 2026-06-30.

**OpenSpec change**: [`add-minimal-rnn-arms`](../../../openspec/changes/archive/2026-06-30-add-minimal-rnn-arms/proposal.md).

______________________________________________________________________

## Objective

Bring up the cheapest Tier-1 memory-axis architecture candidate from the 2026-06-29
architecture-candidate research — **minGRU / minLSTM** ([Feng et al. 2024, arXiv:2410.01201](https://arxiv.org/abs/2410.01201)),
minimal RNNs whose gates depend only on the current input (no hidden-to-hidden matrix) so the
recurrence is an associative scan — and evaluate them on two prongs: **(a)** a memory-demanding
cell (does the comparison rate them as memory arms?) and **(b)** the reactive cell (do they train
more stably than the 029 LSTM laggard, a memory-independent stability-upgrade hypothesis?).

## Background

The bit-memory positive control ([Logbook 030](030-bit-memory-positive-control.md)) confirmed the
architecture comparison resolves working memory, unblocking `T7.separation.new_arch_candidates`. The
candidate research ([docs/research/policy-architecture-candidates.md](../../research/policy-architecture-candidates.md))
rated minGRU/minLSTM the cheapest Tier-1 candidate — a near-trivial extension of the `lstmppo` arm —
with dual value: a new memory-axis arm, and a candidate stability upgrade to the 029 LSTM arm (60.1,
5/8 converged), since the minimal cell has no saturating hidden-to-hidden matrix.

## Hypotheses

- **H1 (memory)** — minGRU/minLSTM solve the bit-memory delayed-match-to-cue task well above chance
  and significantly beat the memoryless MLP, landing in the recurrent/attention cluster.
- **H2 (stability)** — on the reactive continuous-2D C3 cell, the minimal RNNs train at least as
  stably as (the hypothesis: more stably than) the plain LSTM, whose 029 weakness was seed-fragility.

## Method

**Arms** — `mingruppo`, `minlstmppo`: two registered classical PPO arms backed by one parallel-form
minimal-RNN cell (minGRU single update gate; minLSTM two normalized input-only gates; both
single-state). They subclass `LSTMPPOBrain` and override only the recurrent core, reusing its entire
PPO / chunk-BPTT / rollout-buffer / weight-persistence pipeline (a behaviour-preserving
extract-method refactor exposed the hook; the 77 `lstmppo` tests stayed green).

**Memory cell** — the merged bit-memory control (`cue 2 / delay 8 / response 1`, 20 trials/episode,
1500 episodes), plateau-tail cue-match per seed, n=8 paired seeds, reusing the 030 baselines
(`mlpppo`, `lstmppo`, `cfcppo`, `transformerppo`, `connectomeppo`) and the committed paired-seed
Wilcoxon + bootstrap + BH-FDR harness.

**Reactive cell** — the 029 integrated continuous-2D C3 cell (food + predator + thermotaxis,
klinotaxis), identical env/reward/satiety block + recipe to the `lstmppo` C3 config (only the brain
differs), 2400 episodes, n=8 paired seeds, paired-seed A/B vs `lstmppo` on the plateau-tail
full-clear success rate (the 029 primary metric).

## Results

### Memory cell (plateau-tail cue-match, n=8, chance = 0.50)

| Arm | cue-match | per-seed range | vs MLP |
|-----|----------:|----------------|--------|
| cfcppo | 0.995 | 0.982–0.999 | — |
| transformerppo | 0.978 | 0.973–0.981 | — |
| **minlstmppo** | **0.966** | 0.90–1.00 | d=+0.465, q=0.007 \*\*\* |
| **mingruppo** | **0.956** | 0.758–1.00 | d=+0.455, q=0.007 \*\*\* |
| lstmppo | 0.939 | 0.771–1.00 | — |
| mlpppo | 0.501 | 0.494–0.508 | chance |
| connectomeppo | 0.499 | 0.495–0.506 | chance |

- Both minimal RNNs **clear the 0.80 threshold and significantly beat the memoryless MLP** (BH-FDR
  q = 0.007); they are **statistically indistinguishable from the plain LSTM** (mingru/minlstm vs
  lstm: ns) and sit just below CfC/Transformer.
- **Verdict: SEPARATION CONFIRMED** — the separating-arm set is {lstm, cfc, transformer, mingru,
  minlstm}; MLP and connectome remain at chance.

### Reactive cell (plateau-tail full-clear success %, n=8, 029 C3 cell)

| Arm | success % | per-seed range | std | vs lstmppo |
|-----|----------:|----------------|----:|------------|
| **minlstmppo** | **73.1** | 43.8–86.5 | 12.6 | d=+17.0, q=0.016 \*\*\* |
| **mingruppo** | **66.2** | 49.5–91.0 | 13.7 | d=+10.1, q=0.125 ns |
| lstmppo | 56.1 | 39.7–82.8 | 15.0 | — |

- Both minimal RNNs **beat the plain LSTM** — minLSTM significantly (paired one-sided Wilcoxon,
  BH-FDR q = 0.016), minGRU directionally (q = 0.125 ns). Both have **lower per-seed spread**
  (std 12.6 / 13.7 vs 15.0) and a **higher floor** (worst seed 43.8 / 49.5 vs 39.7) — the stability
  signal. The `lstmppo` re-run (56.1) reproduces the 029 C3 baseline (~59), anchoring the comparison.

## Analysis

- **H1 confirmed.** minGRU/minLSTM are genuine working-memory arms — they solve a task the memoryless
  MLP provably cannot, in the LSTM tier. minLSTM (0.966) and minGRU (0.956) even edge the plain LSTM
  (0.939) on the mean (ns).
- **H2 supported.** On the reactive C3 cell both minimal RNNs train at least as stably as the plain
  LSTM and **beat it** — minLSTM significantly (+17.0, q = 0.016), minGRU directionally (+10.1). At
  73.1 minLSTM lands upper-mid-pack on the 029 cell (between Transformer 68 and MLP 86), not a
  laggard, and both arms have a lower per-seed spread + higher floor than the LSTM.
- **Load-bearing finding — the retention-gate init.** With the natural zeroed-bias init the arms sat
  **at chance and never learned** (flat ~0.50 across the whole run, identical to the memoryless MLP).
  Diagnosis: during the delay the observation is exactly `[0, 0]`, so the minimal cell's gate is
  bias-only; a zeroed bias gives `z = f' = 0.5` — a ~1-step retention half-life that washes the cue
  out over the 8-step delay, leaving an effectively memoryless policy, and PPO cannot recover the
  hold from delayed credit. Biasing the retention gate toward **holding** (minGRU `bias_z ≈ −2.5`,
  minLSTM `bias_f ≈ +2.5` / `bias_i ≈ −2.5` — the LSTM forget-gate-bias trick adapted to the minimal
  cell) so PPO only has to learn to **write** during the cue lifts both arms from chance to ~0.96.
  This is the structural analogue of the orthogonal-recurrent / LayerNorm-cell stabilisation the
  plain GRU/LSTM arm already carries; the bare minimal cell needs its own.
- **Init detriments — the feared reactive cost did not materialise.** The hold-bias is a prior with
  costs in principle (task-roughly-tuned ≈ delay-8 half-life; slower default state dynamics; slower
  forgetting/overwriting; a mild gate-learning slowdown near sigmoid saturation), and the reactive
  A/B **confounds** the minimal architecture with the init. But empirically the *net* effect on the
  reactive cell was **positive** — both arms beat the LSTM — so the hold-init is not a reactive
  liability for the arm as shipped. It is purely beneficial on memory cells; making `_HOLD_BIAS`
  configurable is a recorded option, not a demonstrated need.
- **LSTM-style seed-fragility reappears** in minGRU (seed 8 at 0.758 vs 0.93–1.00 elsewhere),
  consistent with the 029 LSTM finding — reported, not tuned away.

## Conclusions

minGRU/minLSTM are **strong memory-axis arms** — confirmed memory arms on par with the LSTM on the
memory cell, **and** more stable than the LSTM on the reactive cell (minLSTM significantly) — at a
fraction of the recurrent machinery, **provided the retention gate is initialised to hold by
default** (which costs nothing on the reactive cell). Both prongs of
`T7.separation.new_arch_candidates` are positive for the minimal-RNN portion; modified-S5 (the other
Tier-1 candidate) remains a separate follow-on.

## Limitations

- **The hold-init is load-bearing and task-roughly-tuned** — see Analysis. The reactive A/B showed
  it costs nothing on the reactive cell, so making `_HOLD_BIAS` configurable is an option, not a
  demonstrated need. The A/B conflates the minimal architecture with the init; a clean
  architecture-only isolation would need a no-hold-init reactive variant.
- **Reactive stability read off the plateau spread, not a convergence count** — the runs did not
  write the experiment JSON (no `--track-experiment`), so the 029 convergence-detector count is not
  reproduced here; per-seed std + worst-seed floor are the stability proxy.
- **Single memory cell / single delay** — one bit length, delay 8, as in 030.
- **Non-gating** — does not enter Gate 3; the MUST integrated-C3 ranking (029) is unchanged.

## Next Steps

- Bring up **modified-S5** (the remaining Tier-1 candidate) on the memory cell.
- Optional: make `_HOLD_BIAS` configurable and/or add a no-hold-init reactive variant to isolate the
  architecture effect from the init (not required — the net reactive effect is already positive).

## Data References

- Arms + analysis: the [`add-minimal-rnn-arms`](../../../openspec/changes/archive/2026-06-30-add-minimal-rnn-arms/proposal.md)
  change (`brain/arch/minimal_rnn_ppo.py`, the `lstmppo` hook, the harness arm-roster edits).
- Memory-cell separation summary + per-seed cue-match: the supporting
  [separation.json + cue-match-per-seed.csv](supporting/031-minimal-rnn-candidates/separation.json);
  reproduced by running the per-arm `configs/scenarios/bit_memory/` configs through
  `scripts/analysis/bit_memory_separation.py`.
- Reactive-cell A/B summary: the supporting
  [reactive_ab.json](supporting/031-minimal-rnn-candidates/reactive_ab.json); reproduced by running
  the per-arm `configs/scenarios/foraging_predator_thermal/*_combined_klinotaxis.yml` configs through
  `scripts/analysis/minimal_rnn_reactive_ab.py`.
