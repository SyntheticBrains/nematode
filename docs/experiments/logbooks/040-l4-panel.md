# 040: The L4 2×2 Panel — Is the Wild-Type Wiring Legible to a Local Three-Factor Rule? (7a-i / Phase 7)

**Status**: completed — **`sanity_floor_fail`** under the pre-registered verdict map. On the
continuous integrated-C3 cell, with n = 8 paired seeds, the plastic wild-type connectome under the
minimal rate-based three-factor rule beats **none** of its registered contrasts: not its
degree-preserving rewired-null (T1 **+6.6**, CI[−5.5, +20.7], q = 0.50, 4/8 seeds), not its
frozen-weights floor (T2 **+9.6**, q = 0.50, 5/8), and not its unmodulated-Hebbian floor (T3
**−12.4**, CI[−34.2, +10.7], q = 0.73, 4/8). The matched-rule MLP yardstick sits at chance, so the
band test passes by construction and says nothing. What the panel *did* find is that outcomes on
this cell are **fixed points set by the random initial weights**, not learning trajectories: the
unmodulated Hebbian floor reaches **78%, 64% and 67%** on three seeds with no reward at all, and
reward modulation shifts the plastic arm's distribution upward on average — producing the two best
learning curves in the project (70.5% and 28.4%) — without doing so reliably. The one wiring
signal is descriptive and in the floors: wild-type Hebbian beats rewired Hebbian by **+16.5** points
(5/8 seeds) while the frozen floors tie (+0.4). The roadmap's "L4 plasticity fails to beat its
baselines" branch is realised, and it is citable.

**Branch**: `feat/l4-panel`, `feat/l4-panel-run` (panel), with the rule changes on their own
branches.

**Date**: 2026-09-06.

**OpenSpec changes**: `add-l4-panel` (registration, harness, arms, the panel; capability
`l4-plasticity-panel`), preceded by `add-l4-rule-scale`, `add-l4-modulator-centring`,
`add-l4-rule-robustness` and `add-l4-mlp-frozen-readout` (rule and arm fixes the pilots forced,
each pre-registered on its own; all archived).

## Objective

Resolve roadmap D2 under the D10 arm set: does the specific *C. elegans* wiring become
load-bearing under a biologically plausible three-factor rule — local, reward-modulated, no
gradients — where under PPO it was inert ([Logbook 034](034-connectome-structure-controls.md))?
The pre-registered tests: (i) plastic wild-type beats plastic rewired-null on paired seeds at
q < 0.05; (ii) plastic wild-type reaches the matched-rule MLP's band. Two sanity floors (frozen
weights, unmodulated Hebbian) and a learning-gain contrast complete a four-test BH-FDR family.

## Background

Phase 6 fixed two facts: under PPO the wild-type connectome ranks 5th of 6
([029](029-continuous-architecture-ranking.md)) and a degree-preserving rewired-null matches it
([034](034-connectome-structure-controls.md)), so under gradient learning the wiring is inert.
Phase 7's headline asks whether the animal's own rule family reads the wiring differently. The
substrate froze mode-off after the D7 gate ([038](038-state-dependent-std-gate.md)), so
Logbook 029 stays the descriptive frame and no PPO arm is a panel row.

The rule (A.3) is `Δw = η·δ·E − η·λ_w·w` on the chemical synapses, `δ = r − b` an EMA prediction
error, `E` a temporally causal eligibility trace `E ← λE + M∘(h_prev ⊗ h)`, applied once per
step under `no_grad`, with a frozen anatomical motor readout and frozen sensory gains. The floors
(A.4) run under the same rule with updates frozen, or with the modulator removed. The rewired
plastic arm (A.6) is one `wiring` key off the plastic arm; the MLP yardstick (A.5) runs the same
rule through a substrate-generic seam.

## Hypothesis

Pre-registered before any run ([details](supporting/040-l4-panel/details.md) links the archived
registration): wild-type plastic > rewired-null plastic (T1), > frozen (T2), > Hebbian (T3), and
wild-type learning gain > rewired learning gain (T4), corrected together; the band test on the
MLP; an ordered verdict map (`sanity_floor_fail`, `rewired_beats_wild_type`, `recovery`,
`structure_only`, `robustness`, `inconclusive`); every panel result a *performance* claim unless
bar (a) ensemble-invariance and bar (b) grounding are met.

## Method

Seven arms: wiring {wild-type, rewired-null} × rule {frozen, Hebbian, three-factor}, plus the
tanh MLP under the three-factor rule with a frozen readout. Seeds 1–8 paired across every arm
(`rewire_seed` derived from the run seed). Ranked metric: the committed plateau-tail (final
quarter) full-clear success; per-seed convergence from the level-agnostic detector; paired
one-sided Wilcoxon, 80% bootstrap CI, BH-FDR — the 029/034 layer, imported. Run through
`scripts/run_campaign.py` from a committed launch record; analysed by
`scripts/analysis/l4_panel.py`.

**Recipe and budget were pinned by pilots on disjoint seeds (101–102) by rules stated in
advance**, and the road to them is the method's most important part:

- **Pilot 1** (registered grid `{0.003, 0.01, 0.03}`): every arm at the floors; a fifth to 96% of
  synapses clamped on the bound; the MLP dead after one −10 kick. The grid was two orders of
  magnitude too hot, and a matched *rate* was not a matched *rule* (the MLP's per-weight trace is
  ~1000× smaller). → `add-l4-rule-scale`: `tanh(δ/σ)` modulator and per-tensor `E/ρ` trace
  normalisation, both running-RMS, default-off.
- **Probe 2**: the scaled rule made the connectome *worse*. `tanh` of a skewed prediction error is
  not zero-mean (a −10 death and a +2 food both compress to ±1; foods are more frequent), a
  reward-blind Hebbian drive. → `add-l4-modulator-centring`: `tanh(δ/σ) − c`.
- **Pilot 2** (grid `{3e-4, 1e-3, 3e-3}`): plastic arms near their floors, a fifth of synapses
  clamped, the MLP exploding or dying. Three of our own defaults: action noise frozen at std 1.0
  on every plastic arm; decay too weak to hold a coherent drive; unbounded ReLU units on the
  yardstick. → `add-l4-rule-robustness`: shared `initial_log_std`, homeostatic incoming-norm
  scaling (synaptic scaling; the decay is then inert), configurable MLP activation.
- **Probe 4**: with homeostasis and `initial_log_std = −1.0` the wild-type arm reached 64.7% at
  600 episodes against a 2% floor. **Pilot 3** pinned rate `1e-3` and budget 3000.
- **Probes 5–6**: the MLP yardstick destroys a 96% random policy within three episodes. With a
  plastic readout its output rows self-amplify into saturation (density 1e17); with the readout
  frozen (`add-l4-mlp-frozen-readout`) its hidden layers still collapse toward rank one (a dense
  layer fed a mostly-zero input has every unit's Hebbian update pointing the same way). Read as a
  property of local rules on dense stacks; the arm entered as registered.

The panel then ran once, 56 runs, 2 h on 16 workers, with one registered extension (wild-type
plastic seed 3, still climbing at 3000, re-run fresh at 4500: converged, 28.4%). No sensitivity
pass, since the verdict is not `robustness`.

## Results

### The registered family (paired seeds, BH-FDR α = 0.05)

| test | contrast | mean Δ | 80% CI | q | +seeds | result |
|---|---|---|---|---|---|---|
| T1 | wt_plastic − rn_plastic | +6.6 | −5.5 … +20.7 | 0.50 | 4/8 | fail |
| T2 | wt_plastic − wt_frozen | +9.6 | −3.3 … +22.9 | 0.50 | 5/8 | fail |
| T3 | wt_plastic − wt_hebbian | −12.4 | −34.2 … +10.7 | 0.73 | 4/8 | fail |
| T4 | wt gain − rn gain | +6.2 | −7.0 … +21.4 | 0.50 | 4/8 | fail |

Band (wt_plastic − mlp_plastic): +16.8, CI[+6.9, +28.0], PASS by construction. **Verdict:
`sanity_floor_fail`.** Ensemble invariance: T1 and T4 positive on 4/8 seeds; no dynamics claim.

### Per-arm plateau-tail full-clear success (%), seeds 1–8

| arm | mean | per seed | converged |
|---|---|---|---|
| wt_frozen | 8.2 | 36.9, 0.8, 7.7, 14.1, 1.1, 0.4, 2.8, 2.0 | 8/8 |
| wt_hebbian | 30.2 | 78.3, 1.9, 3.3, 64.4, 0.0, 0.0, 26.5, 67.3 | 8/8 |
| wt_plastic | 17.8 | 0.0, 70.5, 28.4, 1.9, 27.5, 9.2, 0.7, 4.5 | 8/8 |
| rn_frozen | 7.8 | 8.7, 4.8, 4.5, 2.9, 4.8, 15.2, 21.1, 0.4 | 8/8 |
| rn_hebbian | 13.7 | 12.1, 44.1, 0.0, 24.1, 5.1, 6.8, 12.0, 5.6 | 8/8 |
| rn_plastic | 11.2 | 13.6, 6.3, 7.1, 2.1, 2.0, 46.9, 0.3, 11.3 | 8/8 |
| mlp_plastic | 1.1 | 1.9, 0.0, 0.4, 0.0, 1.5, 0.4, 3.6, 0.7 | 8/8 |

Descriptive pairs (uncorrected): wt_hebbian − rn_hebbian **+16.5**, CI[+0.4, +31.8], 5/8;
wt_frozen − rn_frozen +0.4, CI[−5.7, +6.5], 4/8. Peak tracked action densities: every connectome
arm ≤ 20 (median 3–5); the MLP in the tens of thousands.

## Analysis

1. **Outcomes are fixed points, seeded by the initial weights.** Every connectome arm's plateau
   is reached within a few hundred episodes and depends more on the seed than on the rule. The
   frozen wild-type arm alone spans 0.4% to 36.9% across seeds on identical wiring. The Hebbian
   floor — no reward — settles within its first block at 78%, 64%, 67%, 27%, or at 0%. Under
   homeostasis the unmodulated rule converges to a fixed point of the wiring's correlation
   structure, and which fixed point depends on where it starts.
2. **Reward modulation shifts the distribution upward, not reliably.** The plastic arm produced
   two genuine learning curves (seed 2: 41% → 73% over 3000 episodes; seed 3: to 28%) and also
   took seed 1's 37% frozen policy to zero within a block. Averaged over eight seeds it is +9.6
   points over frozen and −12.4 under the Hebbian floor. At n = 8 that is a null on every test,
   and the estimate's sign on T3 says the modulator is not adding to what alignment finds.
3. **The wiring signal is in the floors.** Reward-free alignment finds better fixed points on the
   real wiring than on its degree-matched scramble (+16.5, 5/8) while the frozen floors tie. This
   is the "advantage attributable to the wiring's correlation structure under any Hebbian
   process" the design flagged as the weaker result, and it is the only wiring-specific signal in
   the panel. It is uncorrected and one panel; it is also cheap to test properly.
4. **The yardstick cannot hold a policy under this rule**, with any readout, because a local
   Hebbian update on a dense layer collapses the representation without lateral decorrelation —
   the failure that cerebellum-like circuits solve with anti-Hebbian depression and structured
   inhibition (Perks et al., *Nature* 2026-09-02). D10's "substrate-generic" rule is generic in
   code and not in effect; test (ii) is uninformative and the "recovery" verdict is unreachable
   in substance.
5. **Method findings that outlast the verdict.** A matched *rate* is not a matched *rule* across
   substrates (trace magnitudes differ by 10³); a bounded modulator must be re-centred or it
   becomes a Hebbian drive; a frozen action noise at std 1.0 caps every plastic arm; runaway
   onto the bound needs a homeostatic control, not a decay knob. Each was a default nobody
   examined until a run was long enough to show it, and each is now a pre-registered mechanism.

## Conclusions

- The minimal three-factor rule, as registered, does not make the wild-type wiring load-bearing
  on this cell: **`sanity_floor_fail`**, the roadmap's "fails to beat its baselines" branch. The
  claim type is performance; no dynamics claim is admissible.
- The panel's substantive finding is about **initialisation and fixed points**: on the wild-type
  wiring, reward-free Hebbian alignment from random weights yields a competent policy on three of
  eight seeds and a dead one on two, and the wild-type wiring's fixed points look better than the
  rewired-null's. The prior over policies that a wiring imposes on random initialisations is the
  quantity the next panel should measure directly.
- The matched-rule MLP yardstick is not a yardstick under a local rule; a dense feedforward stack
  needs decorrelation the rule does not provide.

## Limitations

- n = 8 with frozen floors spanning 36 points across seeds: the registered tests are
  underpowered for effects of ten points, and the null on T1/T2/T4 is inconclusive by power. The
  Hebbian floor's bimodality makes T3 a test with enormous variance.
- The substrate's weights are random draws on real edges; Cook 2019's synapse counts are unused.
  Initialisation is the dominant factor and was not a design axis.
- Every arm ran one recipe; the pilot seeds were favourable draws (frozen floors near 2%).
- The band test is vacuous against a collapsed yardstick; test (ii) contributed nothing.
- Descriptive pairs are uncorrected and one panel deep.

## Next Steps

In the ratified order: **panel 2** — the Hebbian wiring contrast (wt_hebbian vs rn_hebbian) as the
registered primary at n ≥ 16 with a frozen-floor *prior sweep* over many seeds and initialisation
(random vs synapse-count-scaled) as a factor; then **S.2** the imitation warm start, which also
supplies the good initial policy this panel lacked; then **7a-ii** with the third factor made
*structured* (pathway-specific instruction through the receptor atlas) rather than a global scalar.
A rule variant with anti-Hebbian/decorrelating terms, and a sparse random MLP arm, are the
yardstick's follow-ups.

## Data References

- Registration and design: `openspec/changes/archive/2026-09-06-add-l4-panel/`; capability
  `openspec/specs/l4-plasticity-panel/spec.md`.
- Everything the panel produced: [supporting/040-l4-panel/](supporting/040-l4-panel/details.md)
  — `panel.json`, `per-seed.csv`, `curves.csv`, `_manifest.txt`, `launch.md` (with the extension),
  pilots 1–3 and probes 2–6, `details.md`.
- Harness: `scripts/analysis/l4_panel.py`; pilot runner `scripts/campaigns/l4_panel_pilot.py`.
- Rule mechanisms: `learning_rules/three_factor.py` (scaling, centring, homeostasis);
  `brain/arch/_plasticity_config.py`; `brain/arch/_mlp_topology.py` (`plastic_layers`).
