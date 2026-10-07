# 080: A Leaky-Integrator Connectome Learns Both Block-V Cells, but Far Slower Than the Settling One (Phase 8b B.2a)

**Status**: completed — **`inferior` on both cells. B.2a's positive control fails; B.2b closes
*unreachable-with-reason*.**

B.2a gave every neuron a membrane potential that persists across an episode's steps, with gap junctions
as ohmic coupling and the sensors as a held current, and asked whether PPO learns block V's two cells
on it at least as well as on the memoryless settling substrate. It does not.

| cell | leaky / settling plateau | `auc_success`, leaky − settling | 80% CI | margin | seeds below | verdict |
|---|---|---|---|---|---|---|
| hard350 | 67.7% / 75.7% | **−0.139** | [−0.179, −0.099] | 0.0143 | 15 of 16 | **inferior** |
| thermal, target 35 | 20.0% / 64.4% | **−0.216** | [−0.250, −0.183] | 0.0239 | 15 of 16 | **inferior** |

Beside them, on episodes to competence: the leaky substrate is **967 episodes later** on hard350 \[−1206,
−729\] and **908 later** on thermal [−1039, −772], and only half its thermal seeds reach competence.

**It took a recalibration to learn at all.** At the input gain first registered, the leaky substrate
passed the sensory signal to the policy about 400 times more weakly than the settling map and learned
nothing at any time constant. With one input gain calibrated on untrained brains, it learns, but slower,
and the less it carries across steps the better it learns.

**Date**: 2026-10-07.

**OpenSpec change**: `add-across-step-state`, which adds `dynamics: leaky` to `connectome-ppo-brain`.

**Pre-registration**: [supporting/080-across-step-state/launch.md](supporting/080-across-step-state/launch.md),
committed before any scored run, amended twice before the panel with the maintainer's decisions: the
recalibration, and the re-size to 16 seeds.

## Objective

Every connectome result so far ran on a substrate with no state across steps. B.2 is the rung that adds
it, and B.2b — plastic gap junctions under PPO, MUST since the 8b re-plan — runs only if this substrate
first learns the cells at least as well as the one it replaces (D20 as amended).

## Method

### The substrate

`dynamics: leaky`. Within each step, four semi-implicit Euler sub-steps of
`τ dv/dt = −v − L v + Wᵀ tanh(v) + I`. The leak and the gap Laplacian `L` are stepped implicitly, so the
step is stable on the raw gap weights, whose row sums reach 232 against at most 6.2 of chemical fan-in.
PPO replays contiguous 16-step chunks from each step's stored starting potentials, with gradients
through time inside a chunk. Tests pin the settling path byte-identical, the contraction at every τ and
depth, and replay reproducing the rollout within 1e-5.

It differs from the settling substrate in four ways, so a verdict is about the substrate as built:

- the potentials persist across steps;
- the sensors are a held current, not the initial state;
- gap junctions act on potential differences, not as added drive;
- signal can cross the settling budget's hop limit by travelling across steps.

### The first pilot, and the recalibration

The τ pilot (0.2, 1 and 5 steps; seeds 1101–1104) left no τ eligible: the leaky wild type plateaued at
0–1.6% against 77.3% and 69.9% ([first-pilot.json](supporting/080-across-step-state/first-pilot.json)).
The cause was measured on untrained brains. The steady state's sensitivity of the policy mean to its
input was 1.4 × 10⁻³ at every τ, against 0.6 for settling: each hop has a gain below one, because the leak
pulls potentials toward zero, the weights are scaled for unit fan-in, and gap coupling shunts. The
design had checked stability and replay but not this gain.

The maintainer chose one recalibration, with a stopping rule. **`input_gain`**, a fixed multiplier on
the sensor current, was chosen without training as the gain whose sensitivity ratio to settling is
closest to 1 on the worse cell: **512** (0.84 hard350, 0.90 thermal;
[calibration.json](supporting/080-across-step-state/calibration.json)). The recurrent gain stayed at
1, below the critical gain of all 64 untrained wild types checked (lowest 1.09); raising it instead
makes activity self-sustained.

### The repeat pilot

At input gain 512 on fresh seeds 1105–1108, the pre-written rule chose **τ = 0.2 steps**, the only τ
readable on both cells ([pilot.json](supporting/080-across-step-state/pilot.json)).

| τ (steps) | hard350 leaky / settling plateau | thermal leaky / settling plateau |
|---|---|---|
| 0.2 | 41.9% / 77.8% | 26.9% / 68.2% |
| 1 | 28.2% / 77.8% | 0.2% / 68.2%, fails its floor |
| 5 | 0.1% / 77.8% | 0.0% / 68.2%, fails its floor |

Its deficit, −0.265 and −0.205 `auc_success`, was about 19 and 9 times the margins, so the maintainer
re-sized the panel from 128 to 16 seeds per cell before launch. At 16 seeds `inferior` needed a mean
below about −0.034 and −0.056, and `non_inferior` above about +0.006 and +0.008.

### The controls and the panel

- **Control 1, MLP-PPO, seeds 1201–1208**: every seed reaches competence on both cells, at 88.4–99.2%
  on hard350 and 47.5–89.7% on thermal ([mlp.json](supporting/080-across-step-state/mlp.json)).
- **Identity check**: 12 settling re-runs match the committed runs on every `Run:` line and the final
  `w_chem`, bit for bit, which licenses pairing against the committed settling runs
  ([identity.json](supporting/080-across-step-state/identity.json)).
- **Control 2**: the leaky wild type, learning and frozen, on seeds 641–656 (hard350) and 513–528
  (thermal), 64 runs, paired with the committed settling wild type. Gate preflight `readable` on both
  cells before launch.

## Results

| cell | gate | leaky plateau / floor | settling plateau / floor | verdict |
|---|---|---|---|---|
| hard350 | passes | 67.7% / 0.0% | 75.7% / 0.0% | **inferior** |
| thermal | passes | 20.0% / 0.0% | 64.4% / 0.0% | **inferior** |

Both intervals sit wholly below −δ, far past the 16-seed threshold. Achieved spread: sd 0.132 on hard350
and 0.104 on thermal, against the 0.062 and 0.10 proxies. The hard350 proxy erred low, as registered,
and does not move a verdict.

## Analysis

**On hard350 the leaky substrate nearly reaches the settling plateau but takes far longer to get there**
(67.7% against 75.7%, 967 episodes later). **On thermal it stays far below** (20.0% against 64.4%).

**Learning falls as τ rises**, on hard350 in both pilots and on thermal in the repeat pilot (the first left thermal at zero throughout). The substrate learns best when it carries least across
steps, and not at all when it carries most. The panel does not separate why. The across-step state may
be harder for PPO to credit, even with gradients through 16-step chunks. Or a longer τ may only slow the
response within a step. τ is confounded with both.

## What this establishes, and what it does not

- **Established**: on both block-V cells under PPO, the leaky substrate as built — leak, ohmic gap
  coupling, held sensor current, input gain 512, τ = 0.2 steps — learns worse than the settling substrate
  by more than either cell's registered margin. B.2a's positive control fails.
- **Established**: at unit input gain it attenuates the sensory signal about 400-fold and learns nothing.
  Its stability on the raw gap counts holds by construction.
- **Not established**: that memory across steps is what costs learning. The substrate differs from
  settling in four ways at once, and the τ trend is confounded.
- **Not established**: that no other calibration would pass. The registered budget was one
  recalibration; untried settings include per-neuron time constants, a recurrent gain near criticality,
  the sensors as initial state rather than current, and a longer training budget.

## Registered consequences

- **B.2b closes *unreachable-with-reason***, as D20 as amended requires when B.2a's positive control
  fails. The diagnosis above is its deliverable. Gap placement against gap strength stays open; it can
  be asked on the settling substrate, where a plastic gap matrix needs no leak.
- **B.2c** (bout durations) stays SHOULD behind C.0b's reversal, no longer tied to a dynamical substrate.
- **C.2c**, which needed B.2, loses its route through it.
- **8b's carried controls are complete**, and 8b moves to the body (C.0).

## Artefacts

- [launch.md](supporting/080-across-step-state/launch.md): the registration with both amendments.
- [first-pilot.json](supporting/080-across-step-state/first-pilot.json),
  [calibration.json](supporting/080-across-step-state/calibration.json),
  [pilot.json](supporting/080-across-step-state/pilot.json),
  [preflight.json](supporting/080-across-step-state/preflight.json): the two pilots, the calibration
  and the gate evidence.
- [identity.json](supporting/080-across-step-state/identity.json),
  [mlp.json](supporting/080-across-step-state/mlp.json): the reuse licence and Control 1.
- [control.json](supporting/080-across-step-state/control.json),
  [per-seed.csv](supporting/080-across-step-state/per-seed.csv): the panel.
- Reproduce: `scripts/analysis/across_step_calibration.py`; `scripts/analysis/across_step_control.py`
  (`--first-pilot`, `--pilot`, `--identity`, `--mlp`, and the panel). Campaign directories archived
  off-repo.
