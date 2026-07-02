# 033: Chemosensory Associative-Memory Probe — the Naturalistic Remember-and-Use Twin of the Bit-Memory Control (T7)

**Status**: completed — **separation**. On a naturalistic within-episode chemosensory
associative-learning task with probabilistic reversal (a working-memory **update** demand), all five
memory arms (Transformer / CfC / minGRU / minLSTM / LSTM) clear the update threshold and
significantly beat the memoryless MLP, which is pinned at chance. The reversal split further
separates **genuine updaters** (Transformer / CfC / LSTM — reversal ≈ non-reversal accuracy) from
**hold-biased** minimal-RNNs (minGRU / minLSTM — near-perfect retention, weaker overwrite). With the
bit-memory control ([030](030-bit-memory-positive-control.md)) and the ARS null
([032](032-ars-source-depletion.md)), this is the **second** naturalistic data point and it resolves
the ARS-null interpretation: the null was the environment's under-demand, **not** an inability of the
architectures to carry memory — given a genuine naturalistic memory demand, they separate cleanly.

**Branch**: `openspec/add-associative-memory-probe`.

**Date**: 2026-07-03.

**OpenSpec change**: `add-associative-memory-probe` (outcome channel + task machine + harness;
committed §1–§5, calibrated §6, green).

## Objective

The ARS-via-depletion null ([032](032-ars-source-depletion.md)) left one ambiguity: was the null
because the architectures **cannot** carry naturalistic within-episode memory, or because the ARS
*task* under-demanded it (a short-horizon demand the adaptive chemosensor already covers)? The
artificial bit-memory control ([030](030-bit-memory-positive-control.md)) showed the arms **can**
separate working memory on a deliberately non-biological delayed-match-to-cue. This probe is the
**naturalistic "remember-and-use" analogue** — a within-episode chemosensory associative-learning
task — asking whether that separation reproduces on a biologically-motivated associative demand
rather than an abstract bit, and (via a reversal twist) whether the arms genuinely **update** a held
association or merely **hold** it.

## Background

Real *C. elegans* does chemosensory associative learning — gustatory and olfactory cues become
appetitive or aversive after pairing with food or starvation — and re-learns when the contingency
flips. That is a genuine "remember the association, act on it later, revise it on new evidence"
demand, unlike ARS (driven by current sensation + slow neuromodulatory state). Logbook 025 / [029](029-continuous-architecture-ranking.md):
on the reactive foraging cell, architecture does not discriminate (a memoryless MLP ties or wins,
the gradient is locally readable). 030: an artificial delayed-match **separates** the memory arms
from the MLP. This experiment builds the naturalistic associative twin.

## Hypothesis

A **delayed-associative-match with probabilistic within-trial reversal** creates a working-memory
*update* demand: two cues are conditioned early in a trial (one rewarded), with probability
`reversal_prob` a reversal block re-presents them with flipped outcomes, then after a delay the agent
gives a binary readout (sign of its continuous turn) of the **current** rewarded cue. The observation
is exactly `[cue, outcome, go_signal]` — no STAM, no gradients — so only internal recurrent state can
carry (and overwrite) the association across the delay. The key property: at `reversal_prob = 0.5` a
**hold-only** policy (never updates) is at chance, like the memoryless MLP, so above-chance overall
accuracy requires genuine **update**. A memoryless MLP is pinned near chance; the recurrent /
attention arms should separate.

## Method

### Mechanism (committed §1–§5, byte-identical when off)

- **Outcome channel** — a dedicated `outcome` sensory module + `outcome_signal` on `BrainParams`,
  registered alongside the existing `cue` / `go_signal` channels (bit-memory carried cue + go only;
  the associative task adds the paired-outcome channel).
- **Task machine** (`env/associative_memory.py`, dedicated class mirroring `env/bit_memory.py`) — a
  per-trial phase machine: conditioning (both cues presented, one rewarded) → **probabilistic
  reversal** (flipped outcomes, fired with `reversal_prob`) → delay (channels zeroed) → response
  (go-signalled binary readout). Cue↔outcome pairing and the reversal are re-sampled per trial from
  the env RNG.
- **No-external-aid contract** — `assert_associative_observation_clean` fails loudly if the resolved
  observation carries anything beyond the task channels (no STAM, no gradient sensing), so the
  association must live in recurrent state.
- **Separation harness** (`scripts/analysis/associative_memory_separation.py`) — plateau-tail
  (final-quarter) response accuracy per arm + the **reversal / non-reversal split** (parsed from the
  printed per-episode line), the pairwise paired-seed Wilcoxon + 80% bootstrap CI + BH-FDR layer
  (reused verbatim from the arch ranking), and the verdict.
- Off by default (`associative_memory_task` unset) = byte-identical; the config validator pins
  `reward_correct = 1` / `penalty_wrong = 0` so accuracy = reward / responses.

### Calibration (§6, before the panel)

Learnability pre-check with the Transformer (the reliable detector). On an **easy** config (no
reversal, delay 4 — pure hold) response accuracy rose 0.51 → **0.93**; on the **default** config
(reversal_prob 0.5, delay 8 — the update demand) 0.51 → **0.99** with the reversal-split at **0.98**
— genuine update, not hold-only. The defaults (20 trials, `cond_steps_per_cue` 1, `reversal_prob`
0.5, delay 8 → worst-case span 13 < the Transformer window 16, per-arm entropy carried from the
`bit_memory` configs) were validated with **no retune**.

### Panel (continuous-2D substrate, plateau-tail response accuracy)

6 arms (MLP + LSTM / CfC / Transformer / minGRU / minLSTM) × **8 paired seeds** × 1500 episodes,
headless, parallelised (`OMP_NUM_THREADS=1`, `xargs -P8`). Chance = 0.50.

## Results

### Separation confirmed (8 seeds, 1500 episodes)

| arm | overall | reversal | non-reversal | read |
|-----|-----|-----|-----|-----|
| **transformerppo** | **0.989** | 0.984 | 0.995 | genuine update, near-ceiling, tightest seeds (0.983–0.994) |
| **cfcppo** | 0.914 | 0.920 | 0.908 | genuine update (rev ≈ non-rev); one weak seed (0.657) |
| **mingruppo** | 0.860 | **0.732** | 0.989 | **hold-biased** — near-perfect hold, weaker update (gap +0.26) |
| **minlstmppo** | 0.852 | **0.711** | 0.994 | **hold-biased** — near-perfect hold, weaker update (gap +0.28) |
| **lstmppo** | 0.811 | 0.787 | 0.835 | genuine update; most seed-variable (0.62–0.93) |
| **mlpppo** | 0.490 | 0.495 | 0.487 | at chance on **both** → the memoryless anchor |

Pairwise (paired-seed Wilcoxon, BH-FDR): **every memory arm beats the MLP at q = 0.007** \*\*\*. The
Transformer beats **every** other arm (q = 0.007) — the clear top. The other four
(CfC / minGRU / minLSTM / LSTM) are a statistical cluster (pairwise ns, q 0.15–0.37). The memoryless
MLP is within the chance band (0.490) → **VERDICT: SEPARATION**.

### The reversal split: update vs hold

At `reversal_prob 0.5` a hold-only policy is at chance overall (~half the trials flip the cue), so
overall accuracy above chance requires genuine **update**. The split shows two regimes among the
memory arms:

- **Genuine updaters** (reversal ≈ non-reversal): Transformer, CfC, LSTM — they overwrite the
  association on reversal evidence.
- **Hold-biased** (non-reversal ≫ reversal): minGRU (0.99 / 0.73), minLSTM (0.99 / 0.71) — near
  perfect at *retaining* the initial association, markedly weaker at *overwriting* it.

The hold-bias is a **converged plateau, not undertraining**: windowing reversal accuracy over the
1500 episodes, minGRU is flat at 0.73 across the final two eighths and minLSTM ~0.72, while CfC and
the Transformer climb to 0.92 / 0.98. The minimal-RNN arms *settle into* a hold-heavy solution rather
than passing through it toward full update.

## Analysis

**Separation reproduces on a naturalistic associative demand.** Unlike the reactive foraging cell
(025/029, MLP ties/wins) and unlike ARS (032, MLP best), a task that genuinely requires holding **and
updating** a cue↔outcome association across a delay cleanly separates the memory arms from the
memoryless MLP — with the same detector ordering as the artificial bit-memory control (030). The MLP
is at chance on *both* reversal and non-reversal trials: with no cue in the observation at readout it
can neither hold nor update — the structural memoryless anchor.

**The reversal twist adds a behavioural axis the bit-memory hold task could not.** The minimal-RNN
arms (minGRU / minLSTM) clear the threshold but are **hold-biased** — near-ceiling retention, weaker
overwrite. This plausibly reflects the memory-friendly **retention-gate init** they carry ([031](031-minimal-rnn-candidates.md)):
initialised to hold, they settle into a hold-heavy basin. That attribution is a hypothesis, not
causally tested here (a minGRU-without-retention-init ablation would confirm it). The genuine updaters
(Transformer / CfC / LSTM) overwrite on reversal evidence; the Transformer does so near-perfectly.

**This resolves the ARS-null interpretation.** The ARS null (032) could have meant either "the
architectures can't carry naturalistic within-episode memory" or "the ARS environment under-demanded
it." This probe settles it: given a genuine naturalistic memory-update demand, the memory arms
**do** separate — so the ARS null was the *environment's* under-demand (a short-horizon demand the
adaptive sensor covers), not an architectural incapacity. The two naturalistic probes together give
the "capability-yes / natural-foraging-demand-limited" conclusion its second, converging data point.

## Conclusions

- **Separation for the naturalistic associative-memory hypothesis.** All five memory arms clear the
  0.80 update threshold and beat the memoryless MLP at q = 0.007; the MLP is at chance.
- The **reversal split** distinguishes **genuine updaters** (Transformer / CfC / LSTM) from
  **hold-biased** minimal-RNNs (minGRU / minLSTM) — the latter a converged property, plausibly the
  031 retention-gate init.
- Together with 030 (artificial control separates) and 032 (ARS null, short-horizon), this **resolves
  the memory-separation question**: the comparison detects working memory when a task demands it; the
  ARS null was the environment, not the architectures.
- The **outcome channel + associative task machine are committed, tested, byte-identical when off** —
  a reusable capability for phase-7 naturalistic-memory work.

## Limitations

- **Single-shot within-episode pairing** compresses the real minutes-to-hours associative timescale →
  biologically-*inspired*, not faithful; the faithful slow-forming version (neuromodulator-gated
  plasticity across trials / episodes) is **phase-7** (the L4 deliverable).
- **The current arms already solve the probe** (Transformer 0.99, CfC 0.91) → limited separation
  headroom for further memory candidates (e.g. modified-S5) on this cell as-is; a harder demand
  (longer delay, capacity, or memory folded into foraging) is needed to distinguish them.
- **Recurrent-PPO seed instability** persists (LSTM overall 0.62–0.93; seed 2 reads below chance on
  reversal while holding non-reversal) — reported, not tuned away; consistent with 029/032.
- The **retention-gate-init → hold-bias** attribution is a hypothesis (untested here).
- Per-arm entropy was **carried from the `bit_memory` configs**, not the "matched 0.2" the tracker
  sketched; the separation is robust to this (all five arms clear the threshold), and the reliable
  detector reaches ceiling, so no retune was warranted.

## Next Steps

- [ ] **modified-S5** (`T7.separation.new_arch_candidates`, remaining Tier-1 gated candidate) — but
  paired with a **harder memory demand**, since the current arms already solve this probe; otherwise
  it would tie at ceiling (low information). Best deferred to phase-7 with a naturalistic-memory
  foraging task where it can distinguish itself.
- [ ] **NCP fidelity arm** (`T7.separation.ncp_fidelity_arm`, non-gated) — the tap-withdrawal
  worm-circuit wiring; a biological-fidelity / interpretability arm, not a leaderboard contender.
- [ ] **Retention-gate-init ablation** — minGRU/minLSTM without the retention init on this probe, to
  causally test whether the init drives the hold-bias.
- [ ] **Faithful slow-forming associative memory** (plasticity across trials / episodes with
  neuromodulatory dynamics) is **phase-7** — recorded in `docs/roadmap.md` § Known Gaps + Phase 7
  (ties to the L4 plasticity / neuromodulation deliverable and the naturalistic-memory-fidelity gap).

## Data References

- Mechanism + harness: the `add-associative-memory-probe` change — outcome channel
  (`packages/quantum-nematode/quantumnematode/brain/modules.py`, `brain/arch/_brain.py`), task
  machine (`packages/quantum-nematode/quantumnematode/env/associative_memory.py`), runner
  (`agent/runners.py`), config + no-aid contract (`utils/config_loader.py`), harness
  (`scripts/analysis/associative_memory_separation.py`), per-arm cells
  (`configs/scenarios/associative_memory/{arm}_small_associative_memory.yml`), and their tests.
- Supporting analysis (committed): [supporting/033-associative-memory-probe/](supporting/033-associative-memory-probe/details.md)
  — the `separation.json` summary + the per-seed accuracy/reversal/non-reversal CSV + calibration and
  per-seed forensics.
- Design sketch: [docs/research/associative-memory-probe.md](../../research/associative-memory-probe.md).
