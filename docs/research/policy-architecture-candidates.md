# Policy-Architecture Candidates for the High-Fidelity Continuous Substrate

**Purpose**: Survey neural-network architecture candidates — beyond the existing arm set —
for use as the RL **policy** on the continuous-2D klinotaxis substrate (T7+), with a project-fit
rating across four factors and a recommendation tied to the existing memory-separation gate.
**Status**: Research & Planning (informs T7-subsequent / Phase-7 arm selection; nothing built yet)
**Last Updated**: 2026-06-29

______________________________________________________________________

## Table of Contents

01. [Executive Summary](#executive-summary)
02. [How this survey was produced](#how-this-survey-was-produced)
03. [The decisive finding — the memory axis gates everything](#the-decisive-finding)
04. [Ranked shortlist — new architectures](#ranked-shortlist)
05. [Per-candidate detail](#per-candidate-detail)
06. [In-repo SHOULD/MAY arm review](#in-repo-shouldmay-arm-review)
07. [Exotic paradigms — unrated, not negative](#exotic-paradigms)
08. [Recommended sequence](#recommended-sequence)
09. [Sources](#sources)
10. [Open questions](#open-questions)

______________________________________________________________________

## Executive Summary

This is the first new-architecture scan since the T4-era survey. It asks: now that we run a
high-fidelity continuous-2D substrate (float kinematics, Euclidean fields, Fick gradients,
adaptive klinotaxis sensing, tanh-Gaussian continuous action, PPO), are there policy
architectures **not already in the repo** worth adding as T7+ arms?

**The load-bearing answer reframes the question.** The single most-verified finding across the
literature is that recurrence / state-space / world-model machinery only pays off under partial
observability, long horizons, or noise. On our *current reactive C3 cell* every strong new
candidate (S5, minGRU, Mamba) is **expected to tie or lose to MLP** — i.e. piloting them now
would reproduce the [Logbook 029](../experiments/logbooks/029-continuous-architecture-ranking.md)
null (MLP 89.0 ≫ the field). So the decision is **not** "which new arm to build" but "**build the
memory-separation gate first; the new arms are only worth bringing up if that gate fires.**" This
confirms and populates the already-scoped `T7.separation.bit_memory_control` →
`T7.separation.ars_depletion` structure rather than inventing new sequencing.

**Top picks — memory-gated** (conditional on the memory gate separating the arms):

- **minGRU / minLSTM** — cheapest new arm (a near-trivial extension of the existing `lstmppo`
  recurrent path); also a candidate *stability* upgrade to our laggard LSTM arm, independent of memory.
- **Modified S5 (structured SSM)** — the strongest memory-axis candidate; strictly memory-cell-dependent.

**Top pick — memory-independent** (NOT gated by the memory cell; must not be deferred by a null memory result):

- **NCP (worm-circuit wiring)** — the one candidate evaluable *without* the memory cell; highest
  biological fidelity, but heavy overlap with our existing CfC + connectome arms, so a fidelity/
  interpretability arm, not a leaderboard contender — schedule it independently of the gate.

Everything else (LTC, Mamba world-models, the exotic-unrated set, and every in-repo SHOULD/MAY
arm) is **skip or defer** for T7.

______________________________________________________________________

## How this survey was produced

A multi-agent deep-research harness (105 agents): 5 search angles → 23 primary sources fetched →
111 falsifiable claims extracted → 25 verified by 3-vote adversarial verification (need 2/3
refutes to kill). **21 confirmed, 4 killed.** All 21 confirmed claims rest on primary
peer-reviewed sources (NeurIPS / ICLR / AAAI / Nature MI / Nature Comput Sci); no blog/marketing
source was load-bearing. The project-fit layer (ratings against *our* task, *our* existing arms,
and the 029 result) is added on top of that verified landscape.

**Two over-strong claims were refuted (0-3) and must not be carried forward:** "MLP is the single
best architecture on continuous control" and "only Transformer-XL / GTrXL / Mamba-2 can solve the
hardest memory tasks (LSTM/GRU/MLP fail)." Also flagged: Mamba's long-term-*recall* superiority is
contested by a real critical literature (MQAR / associative-recall failures), so SSM memory ratings
describe *claimed*, not settled, capability.

______________________________________________________________________

## The decisive finding

**The memory axis is the decisive discriminator, and it is currently untestable.** Verified
repeatedly: recurrent/SSM/world-model advantage is *memory-conditioned*. RLBenchNet (PPO head-to-head
across MLP/LSTM/GRU/Mamba/Transformer-XL) finds MLP equals recurrent nets on fully-observable
continuous control and recurrent nets only pull ahead under partial observability; Ni et al. 2021
prove "for any MDP there exists an optimal memoryless policy." This is exactly the 029 result on
our reactive cell.

**Consequence for sequencing:** the memory-demanding task is the gate that makes *any* new
memory architecture worth its bring-up cost. The cheapest form of that gate already exists in the
tracker — `T7.separation.bit_memory_control`, an artificial hold-a-cue-across-a-delay positive
control — and it can run on the **existing** arms with no new code. New-arch bring-up should be
**conditional on that control firing.**

______________________________________________________________________

## Ranked shortlist

Factors are 1–5. **"Top MLP"** is split `reactive → memory` where cell-dependent (nothing is
expected to beat MLP on the reactive cell; the memory cell is where the contest lives).

| # | Candidate | Top MLP | Memory | Bio | Stab/eff | Bring-up | Needs memory cell | Verdict |
|---|-----------|:------:|:------:|:---:|:-------:|----------|:----------------:|---------|
| 1 | **minGRU / minLSTM** | 3→4 | 3 | 2 | **4** | Cheap (extends `lstmppo`) | Partial | **Worth-a-pilot — best value/cost** |
| 2 | **Modified S5** (SSM policy) | 3→4 | **5** | 2 | 4 | Moderate | **Yes** | **Worth-a-pilot — best memory candidate** |
| 3 | **NCP** (tap-withdrawal wiring) | 3 | 3 | **5** | 3 | Moderate | No | **Pilot only if distinct-from-CfC/connectome wanted** |
| 4 | **LTC** (liquid time-constant) | 2 | 3 | 5 | 2 | Moderate | No | **Low-priority — redundant with our CfC** |
| 5 | Mamba world-models (DRAMA/KalMamba/Dec-Mamba) | 2 | 4 | 1 | 2 | Expensive | Yes | **Skip — model-based, not a PPO policy** |
| 6 | BAAIWorm / MetaWorm (biophysical connectome) | — | — | 5 | 1 | Very expensive | n/a | **Skip as arm; keep as fidelity reference** |

______________________________________________________________________

## Per-candidate detail

### 1. minGRU / minLSTM — *cheapest new arm; a stability play even before memory*

Minimal RNN revival: drop the previous-hidden-state dependence from the gates → fully
parallel-scan, fewer parameters than GRU/LSTM. Matches Decision-Transformer/Mamba on D4RL
continuous control (minLSTM 78.1, minGRU 78.2 vs DT 76.4; [arXiv:2410.01201](https://arxiv.org/abs/2410.01201),
verified 3-0). **Project hook:** our LSTM arm was the 029 *stability laggard* (60.1, 5/8 converged);
minGRU's whole pitch is a leaner, faster, more-stable recurrent core — so a cheap A/B against
`lstmppo` could buy a stability/efficiency upgrade to the recurrent arm *independent of memory*.
Caveat: the D4RL evidence is offline behaviour-cloning, not online PPO, with large per-task variance.

### 2. Modified S5 — *the strongest memory-axis candidate*

An SSM with a modified parallel-scan that supports in-trajectory hidden-state reset, engineered
as a drop-in RNN-replacement RL policy. Solves POPGym Repeat-Previous-Hard that RNNs cannot
(0.91 vs GRU −0.46), >5× faster than RNNs, validated on low-dim DMControl meta-RL (obs ≤12,
action ≤2 — closest published match to our regime; [arXiv:2303.03982](https://arxiv.org/abs/2303.03982),
verified 3-0). **Caveat the verifiers flagged:** its memory wins are on *discrete-action* POPGym;
no source closes the discrete→continuous(tanh-Gaussian)-action gap. **Strictly memory-cell-dependent**
— no point before the gate fires.

### 3. NCP (Neural Circuit Policy / tap-withdrawal wiring) — *highest fidelity, heaviest overlap*

The canonical "C. elegans circuit as RL policy": a sparse sensory→inter→command→motor wiring on
continuous-time (LTC) neurons. The tap-withdrawal circuit matches DNN policies on classic control;
a 19-neuron NCP solves continuous lane-keeping ([arXiv:1803.08554](https://arxiv.org/abs/1803.08554),
[Nature MI 2020](https://www.nature.com/articles/s42256-020-00237-3), verified 3-0). **Project
reservation:** NCP sits squarely between our **CfC** (closed-form liquid net, 75.8 at 029) and our
**connectome** (Cook-2019 wired net, 52.2) — a *designed sparse worm-shaped wiring on liquid
neurons*. On the reactive cell it would likely land in that same 52–76 band and add interpretability,
not a win. It also natively exposes ~2 sensory inputs, so our multi-channel klinotaxis
(food/predator/temp × L/R) needs a wiring expansion or a learned input adapter (the lineage's
HalfCheetah follow-up used a linear adapter). Distinct enough to be its own arm only if the
single-cell-interpretability story is wanted. **The one candidate evaluable on either cell.**

### 4. LTC (liquid time-constant) — *redundant with CfC*

Time-continuous RNN of linear first-order systems with state-coupled time-constants, solved by an
ODE solver ([arXiv:2006.04439](https://arxiv.org/abs/2006.04439), verified 3-0). Our **CfC is
literally the closed-form approximation of LTC**, built by the same authors to remove LTC's
ODE-solver cost. Pilot only if an explicit ODE-exact-vs-closed-form comparison is ever wanted.

### 5. Mamba world-models (DRAMA / KalMamba / Decision-Mamba) — *not a PPO policy*

A family using Mamba/SSM as the *world-model* or as an offline return-conditioned sequence model,
with a separate conventional controller (actor-critic / SAC / supervised next-action) — a
different, much costlier project than a PPO-policy swap, and evaluated only on Atari100k / DMC
locomotion / D4RL, none classic-control-scale reactive ([arXiv:2410.08893](https://arxiv.org/abs/2410.08893),
[arXiv:2406.15131](https://arxiv.org/pdf/2406.15131), [arXiv:2406.00079](https://arxiv.org/pdf/2406.00079),
verified 3-0). The SSM-as-policy idea is already better served by S5 (#2).

### 6. BAAIWorm / MetaWorm — *fidelity reference, not an arm*

136-neuron Hodgkin-Huxley connectome model fit by gradient descent to calcium-imaging activity —
**not RL-trained** ([Nature Comput Sci 2024](https://www.nature.com/articles/s43588-024-00738-w),
verified 3-0). Converting it to a PPO policy means discarding its calcium-fit objective atop a stiff
biophysical simulator. But its chemotaxis pipeline (gradient→sensory current→motor→muscle, realistic
zigzag klinotaxis) is the **closest published analogue to our substrate physics** — high-value as a
**fidelity-validation target for `T7.validation`**, not a competing arm.

______________________________________________________________________

## In-repo SHOULD/MAY arm review

The survey also asked whether any *existing* non-MUST arm earns a T7 row. Grounded in the brain
registry (`BrainType` / `_registry.py`), the relevant arms all exist as registered brains:

| In-repo arm | Registry | Verdict for T7 |
|-------------|----------|----------------|
| Spiking (`spikingppo`, `qsnnppo`) | LIF + surrogate-grad PPO | **Low priority.** Reactive → noisier-MLP/connectome band; the only future story is spiking + e-prop/STDP temporal-credit on the memory cell, lower priority than S5/minGRU. |
| Reservoir (`qrc`, `qrh`, `crh`) | fixed recurrence + trained readout | **Low priority.** Memory-capable in principle, but reservoirs are typically beaten by *trained* recurrence; S5/minGRU dominate the same niche. |
| Quantum (`equivariantquantum`, `qvarcircuit`, hybrids) | PQC / hybrid | **Skip.** Settled negative at this scale (Logbook 025 controlled attribution). |

**Verdict: no in-repo SHOULD/MAY arm earns a T7 row on its own merits.** This matches the existing
`T7.prep.should_may_continuous` "non-gating, opportunistic" framing — and the research gives the
positive reason to *not* prioritise them: the reservoir/spiking arms' only conceivable edge is on
the memory cell, where the new candidates out-compete them.

______________________________________________________________________

## Exotic paradigms

The survey deliberately spanned exotic paradigms. Several returned **no surviving verified
evidence** on low-dim continuous control and are therefore **unrated here — an absence of positive
evidence, not a proven dead-end**: Kolmogorov-Arnold Networks (KAN), diffusion / flow-matching / consistency policies,
Dreamer-V3 / TD-MPC2 / IRIS as whole agents, predictive-coding / active-inference / free-energy nets,
NEAT / HyperNEAT / weight-agnostic nets, Hopfield / modern-associative-memory, dendritic-computation
nets, hypernetworks / fast-weights, and LRU / linear-attention specifically.

Two project notes:

- **NEAT / topology-evolution** is the least promising *for us*: our `feedforwardga` arm (evolved
  weights) came dead last at 15.0 (029); evolving topology on top is unlikely to beat gradient PPO
  on continuous control. (NEAT remains the formal L3/T8 evolutionary-search test, but as a
  *methodological* exercise, not a leaderboard contender.)
- **Active inference** is the most *conceptually* aligned (a worm minimising surprise about its
  gradient world) but it's a different objective, not a PPO swap — high-risk, defer to a dedicated
  exploration if ever motivated.

None of the unrated exotics jump ahead of minGRU / S5 / NCP on current evidence.

______________________________________________________________________

## Recommended sequence

The research **confirms** the existing gate-first structure and sharpens the ordering for maximum
compute efficiency. (Note: the earlier "normalize_advantages as a foundation" idea is dropped — the
recurrent-collapse it was meant to treat was the `fix-continuous-distance-reward-metric` reward bug,
since fixed; a PopArt value-normalization change was explored and dropped as not load-bearing. The
memory gate is the single real prerequisite.)

1. **`T7.separation.bit_memory_control` on the *existing* arms first** — the cheap, unambiguous
   artificial bit-memory positive control, no new code. **This is the gate.** It answers "can the
   comparison separate working memory at all?" A null here is itself a strong finding and stops the
   new-arch spend.
2. **If it separates** (recurrent / Transformer > MLP) → bring up the Tier-1 new arms to see if they
   beat the *existing* recurrent arms on memory: **minGRU** (cheapest; doubles as the LSTM-stability
   fix) and **modified-S5** (strongest memory candidate), with LSTM/GRU as the control yardstick.
   Optionally elevate `T7.separation.ars_depletion` (the biological twin) — also gated on separation.
   (**NCP is the exception — not gated on memory; brought up independently, see below.**)
3. **If it does not separate** → the null settles it; the new memory arms stay deferred (they would
   reproduce the reactive null on the biological cell too).

**NCP** is the only candidate not gated on memory — it can be brought up independently if the
single-cell-interpretability / worm-fidelity arm is wanted, but it is a fidelity arm, not a
leaderboard contender. Everything else is skip/defer.

______________________________________________________________________

## Sources

Primary, verified (3-0 unless noted):

- **S5-as-RL-policy** — Lu et al., *Structured State Space Models for In-Context RL*, NeurIPS 2023 — [arXiv:2303.03982](https://arxiv.org/abs/2303.03982)
- **minGRU/minLSTM** — Feng et al. (Mila/Borealis), *Were RNNs All We Needed?*, 2024 — [arXiv:2410.01201](https://arxiv.org/abs/2410.01201)
- **Neuronal Circuit Policies** — Lechner & Hasani, 2018 — [arXiv:1803.08554](https://arxiv.org/abs/1803.08554)
- **Neural Circuit Policies (NCP/wirings)** — Lechner et al., *Nature MI* 2020 — [s42256-020-00237-3](https://www.nature.com/articles/s42256-020-00237-3)
- **Liquid Time-Constant Networks** — Hasani et al., AAAI 2021 — [arXiv:2006.04439](https://arxiv.org/abs/2006.04439); CfC — Hasani et al., *Nature MI* 2022 — [s42256-022-00556-7](https://www.nature.com/articles/s42256-022-00556-7)
- **DRAMA** (Mamba world-model), ICLR 2025 — [arXiv:2410.08893](https://arxiv.org/abs/2410.08893); **KalMamba**, RLC 2024 — [arXiv:2406.15131](https://arxiv.org/pdf/2406.15131); **Decision Mamba** — [arXiv:2406.00079](https://arxiv.org/pdf/2406.00079)
- **BAAIWorm / MetaWorm** — *Nature Comput Sci* 2024 — [s43588-024-00738-w](https://www.nature.com/articles/s43588-024-00738-w)
- **RLBenchNet** (PPO architecture head-to-head) — 2025 — [arXiv:2505.15040](https://arxiv.org/abs/2505.15040); **Ni et al.**, *Recurrent Model-Free RL is a Strong POMDP Baseline*, 2021 — [arXiv:2110.05038](https://arxiv.org/abs/2110.05038)

______________________________________________________________________

## Open questions

1. Does modified-S5 retain its >5×-faster-than-RNN advantage with **continuous (tanh-Gaussian)
   action heads**? Its published memory wins are on discrete-action POPGym; no source closes the gap.
2. Does `bit_memory_control` (or `ars_depletion`) impose enough temporal-credit load to separate
   S5/minGRU/LSTM from MLP at all — i.e. is the memory horizon long enough that linear-state
   recurrence beats a one-step temporal derivative? Without a target horizon spec, the memory
   candidates cannot be sized.
3. For the NCP arm, what is the cleanest way to feed multi-channel klinotaxis inputs into a wiring
   that natively exposes ~2 sensory inputs — expand the connectome wiring, or a learned linear
   input/output adapter? This determines whether NCP is a faithful-circuit arm or just a sparse RNN
   with a worm-shaped core.
