# 030: Bit-Memory Working-Memory Positive Control (T7)

**Status**: complete — **SEPARATION CONFIRMED**. The architecture comparison resolves working
memory: the recurrent/attention arms (CfC / Transformer / LSTM) significantly beat the memoryless
MLP on an artificial delayed-match-to-cue task, while the connectome sits at chance alongside the
MLP. The gate fires → the memory-axis follow-ons (`T7.separation.new_arch_candidates`,
`T7.separation.ars_depletion`) are unblocked.

**Branch**: `openspec/add-bit-memory-positive-control`.

**Date**: 2026-06-30.

**OpenSpec change**: [`add-bit-memory-positive-control`](../../../openspec/changes/add-bit-memory-positive-control/proposal.md).

______________________________________________________________________

## Objective

Decide one yes/no question before investing in memory-capable architectures: **can the Phase-6
architecture comparison separate working memory at all?** Ship a deliberately-artificial
positive control — a task a memoryless policy provably cannot solve above chance — and check
whether the comparison detects the gap.

## Background

[Logbook 025](025-weight-search-architecture-ranking.md) (T4 grid) and
[Logbook 029](029-continuous-architecture-ranking.md) (T7 continuous) both found that on the
**reactive** cells the architecture comparison does not discriminate — the gradient is readable
locally with a one-step temporal derivative, so working memory is never exercised and the field
ties (or the MLP wins outright). The 2026-06-29 architecture-candidate research
([docs/research/policy-architecture-candidates.md](../../research/policy-architecture-candidates.md))
made the consequence concrete: the strongest new candidates (modified-S5, minGRU) are memory-axis
architectures whose advantage is *expected to be null on a reactive cell*. So before bringing them
up — or building the biologically-plausible area-restricted-search (ARS) cell — we need an
instrument that confirms the comparison can resolve memory in principle. A **null here would itself
be a strong finding**: it would mean the comparison cannot detect working memory, and the
memory-axis programme should not be pursued.

## Hypotheses

- **H1** — The memory arms (LSTM, CfC, Transformer) solve the task well above chance and
  significantly beat the memoryless MLP.
- **H2** — The connectome sits at chance alongside the MLP: its recurrence is *within-step*
  settling (the hidden state resets every forward call, `connectome_ppo.py`), so it carries no
  cross-step state and cannot retain the cue across the delay.

## Method

**Task** — a non-spatial **delayed-match-to-cue** probe. Each trial has three phases: a **cue**
phase (a binary cue `-1/+1` on a dedicated channel), a **delay** phase (cue withheld, channel
`0`), and a go-signalled **response** phase where the agent must act on the *remembered* cue
(the sign of its continuous turn). The cue is sampled uniformly per trial, so chance = 0.50.

**No external memory aids (the validity crux)** — the observation is exactly two channels, cue +
go-signal. STAM (the project's recency buffer) and all gradient sensing are withheld and a
load-time assertion enforces it; otherwise a memoryless policy could read the cue back off its
input. Retaining the cue across the delay therefore requires *internal recurrent state*.

**Arms / protocol** — the 5 T7 MUST arms (`mlpppo`, `lstmppo`, `cfcppo`, `transformerppo`,
`connectomeppo`) with their existing continuous heads; no new architecture. 20 trials/episode,
`cue 2 / delay 8 / response 1` (span 11, within the Transformer's window 16 so the cue is still
attendable). Primary metric: the plateau-tail (final-quarter) **cue-match rate** per seed, n=8
paired seeds, 1500 episodes, with the committed paired-seed Wilcoxon + 80% bootstrap CI + BH-FDR
layer (the 025/029 methodology). A learnability pre-check (LSTM on an easy delay-2 setting → 0.92)
de-risked the run before the full panel.

## Results

### Cue-match by arm (plateau-tail, n=8, chance = 0.50)

| Arm | cue-match | per-seed range | reads as |
|-----|----------:|----------------|----------|
| **cfcppo** | **0.995** | 0.982–0.999 | memory |
| **transformerppo** | **0.978** | 0.973–0.981 | memory |
| **lstmppo** | **0.939** | 0.771–1.000 | memory (2 fragile seeds) |
| mlpppo | 0.501 | 0.494–0.508 | chance |
| connectomeppo | 0.499 | 0.495–0.506 | chance |

### Pairwise (one-sided Wilcoxon, BH-FDR)

- Each memory arm beats the MLP by **d ≈ +0.44 to +0.49**, **q = 0.006** (\*\*\*).
- **mlpppo vs connectomeppo: d = +0.003, q = 0.174 (ns)** — statistically indistinguishable, both
  at chance.

**Verdict: SEPARATION CONFIRMED.** All three memory arms clear the pre-registered 0.80 threshold
and significantly beat the memoryless MLP; the MLP itself is at chance.

## Analysis

- **H1 confirmed.** The comparison resolves working memory: a recurrent/attention policy solves a
  task a memoryless one provably cannot, and the project's statistics layer detects it cleanly.
- **H2 confirmed.** The connectome is statistically indistinguishable from the MLP at chance —
  direct evidence that its biological recurrence is within-step settling, not cross-step working
  memory. (This sharpens how its 029 mid-pack rank should be read: it has no working-memory
  advantage to bring.)
- **LSTM seed-fragility** reappears (2 of 8 seeds at 0.77 / 0.84 vs six at 0.96–1.00; mean 0.939),
  consistent with the 029 finding. It is reported, not tuned away — and it still beats the MLP
  decisively (q = 0.006).
- **Why n≥8 (an n=4 false-null caught and avoided).** A preceding n=4 read showed the identical
  separation in the means (memory 0.92–1.00 vs MLP/connectome 0.50) but the harness returned NULL:
  with 4 paired seeds the one-sided Wilcoxon's *floor* p-value is 1/2⁴ = 0.0625, so q can never
  reach < 0.05 regardless of effect size. Scaling to n=8 (floor 1/2⁸ ≈ 0.004) made the overwhelming
  effect significant. The n=4 verdict was a statistical-floor artifact, not a real null.

## Conclusions

The Phase-6 comparison **can** separate working memory. The bit-memory positive control fires, so
the memory-axis programme is worth pursuing: `T7.separation.new_arch_candidates` (bring up
minGRU/minLSTM + modified-S5 on a memory-demanding cell) and `T7.separation.ars_depletion` (the
biologically-plausible twin) are **unblocked**. The connectome-at-chance result is a clean,
reusable datapoint about what the wild-type wiring does and does not compute.

## Limitations

- **Deliberately artificial / non-biological** — this is an instrument, not a behavioural claim.
  Its value is confirming the comparison's resolving power; the biologically-valid demonstration of
  the same separation is the ARS cell (`T7.separation.ars_depletion`).
- **Single trial structure** — one cue bit, one delay length (8), within the Transformer window. A
  longer-delay variant (span > window) that would separate unbounded memory (LSTM/CfC) from
  windowed attention (Transformer) is a recorded follow-on, not part of this gate.
- **Non-gating** — does not enter Gate 3; the MUST integrated-C3 ranking (029) is unchanged.
- **Tooling caveat** — the `run_simulation` session summary is foraging-centric and labels every
  bit-memory episode `Status: FAILED` / 0% success (it only counts goal/food terminations). The
  cue-match rate via the separation harness is the real metric; the summary's success% does not
  apply to this task.

## Next Steps

- Proceed with `T7.separation.new_arch_candidates`: bring up minGRU/minLSTM (cheapest; also a
  candidate stability upgrade to the 029 laggard LSTM arm) and modified-S5, evaluated on a
  memory-demanding cell with the existing recurrent arms as the yardstick.
- Build the ARS biological twin (`T7.separation.ars_depletion`) — source-dynamics-driven within-
  episode memory — and check the separation reproduces on a biologically-valid task.

## Data References

- Task + analysis: the [`add-bit-memory-positive-control`](../../../openspec/changes/add-bit-memory-positive-control/proposal.md)
  change (env phase machine, runner scoring, `scripts/analysis/bit_memory_separation.py`).
- Supporting analysis (committed): [supporting/030-bit-memory-positive-control/](supporting/030-bit-memory-positive-control/details.md)
  — per-seed cue-match CSV + the separation summary JSON.
- Raw per-run training outputs (40 `.out`) are uncommitted evaluation forensics under
  `tmp/evaluations/bit-memory-prep/` (gitignored, per the 025/029 precedent).
