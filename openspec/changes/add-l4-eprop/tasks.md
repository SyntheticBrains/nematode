# Tasks

## 1. The mechanism

- [ ] 1.1 Add `"eprop"` to `EligibilityMode` in
  [`_plasticity_config.py`](../../../packages/quantum-nematode/quantumnematode/brain/arch/_plasticity_config.py),
  with a `LearningSignalRouting` Literal (`"symmetric" | "random_motor" | "random" | "scalar"`) and a
  `plasticity_learning_signal` field defaulting to `"random"`. Document at the seam what the truncation
  drops and that symmetric feedback reaches only the readout pool.
- [ ] 1.2 Validators, as model validators **and** as an explicit guard the brain re-runs — `model_copy`
  skips the former and that is how the campaign runner derives arms, the gap R.1c's review found:
  - `eprop` requires `plasticity_node_noise == 0.0`;
  - `eprop` rejects any `plasticity_perturbation_set` other than the default, there being no
    perturbation to restrict;
  - `plasticity_learning_signal` set **explicitly** is rejected by every eligibility mode but `eprop`,
    so its default never trips the guard on an existing config;
  - `eprop` is rejected on a **discrete-action** config, whose score function this change does not
    implement;
  - `eprop` is rejected by `mlpppo`, which has no fold-in call site here — a refusal at load rather
    than a raise mid-episode;
  - a PPO config may declare none of it.
- [ ] 1.3 The eligibility itself, per substrate. Connectome: accumulate
  `eps_ij = Σ_s psi_j^(s+1) · h_i^(s)` across the settling loop under `no_grad`, into a transient
  buffer registered in `_TRANSIENT_BUFFERS`. MLP: the single-pass form `psi_j · h_i` per plastic layer.
  Both leave the forward pass **bitwise unchanged** — assert it.
- [ ] 1.4 The fold-in seam: `apply_learning_signal(score)` on the plastic-topology Protocol, called once
  per environment step immediately after the action is sampled, computing `L` from the routing and doing
  `E ← decay·E + M ∘ (eps · L)`. A no-op for every eligibility mode but `eprop`, so the existing
  substrates pay nothing.
- [ ] 1.5 The score function is taken **pre-squash** — `_continuous_action_step`'s `pre_tanh`, not the
  bounded action — since `∂ log π / ∂ mu` is `(u − mu)/sigma²` and the squash correction does not depend
  on `mu`. A test pins it against autograd on the committed head.
- [ ] 1.6 `symmetric` routing derives `B` from the readout **through the mean-pool**:
  `readout[k, class(j)] / |class(j)|` on the 39 pooled units and zero elsewhere. A test asserts the 263
  non-pool units receive exactly zero, since that is the arm's defining property and not a bug.
- [ ] 1.7 `random_motor` is `random`'s projection masked to the same 39 units, drawn from the **same**
  generator draw so the two arms differ by the mask alone at one seed.
- [ ] 1.8 **A forward whose signal was never folded in must fail**, not credit a stale `eps`. The
  failure is silent and reads as a rule that learns slowly, so it is a raised error and a test.
- [ ] 1.9 `B` is drawn once per run from the run seed and **persisted with the checkpoint**: a reloaded
  policy that draws a new projection is learning against a different feedback path than the one it was
  trained with.
- [ ] 1.10 Tests: the derivative is the activation's (numeric check against autograd on a small net);
  the trace matches a closed-form `eps` on a two-unit two-step settling case; `scalar` gives `L = 1`
  exactly; the forward is unchanged with the mode on; each guard in 1.2 fires under `model_copy` as well
  as construction.

## 2. Stage 1 — the positive control, which is a stop clause

- [ ] 2.1 Three arms in
  [`l4_rule_positive_control.py`](../../../scripts/analysis/l4_rule_positive_control.py) at the existing
  pins and over the existing rate grid: `eprop_symmetric`, `eprop_random`, `eprop_scalar`. Any rate
  passing counts as a pass, as for the committed arms.
- [ ] 2.2 `eprop_symmetric` **must pass** — on a one-step scalar-action task it is the exact REINFORCE
  gradient of the plastic layer, so a failure is an implementation fault and the control is **VOID**.
- [ ] 2.3 `eprop_scalar` **must not pass** — a rule with no per-unit signal solving a cue-to-target
  association means the task is not discriminating, and the control is **VOID**.
- [ ] 2.4 `eprop_random` is reported and required of nothing; its result bounds what stage 2 could show
  and is stated in the record either way.
- [ ] 2.5 A cross-check that the existing `three_factor`, `hebbian` and `analytic` arms are **unchanged
  to the committed values** — the harness gained arms, and if the old ones moved, something shared did.
- [ ] 2.6 **No stage-2 run is launched until 2.2 and 2.3 both hold.** Record which.

## 3. The arms

- [ ] 3.1 Five configs from the committed `hard350` cell: `eprop_{symmetric,random_motor,random,scalar}`
  and one shared `eprop_frozen`, at `initial_log_std: -1.0`, `plasticity_node_noise: 0.0`, the committed
  anatomical readout, no perturbation set, and the pinned rule settings written out rather than
  inherited — `plasticity_normalise_modulator`, `plasticity_normalise_trace` and
  `plasticity_homeostasis` on, `plasticity_rate: 0.001`, `trace_decay: 0.9`, `forward_pass_depth: 4`.
- [ ] 3.2 Exact-key test: the four learning configs differ from each other in
  `plasticity_learning_signal` **alone**, and from the frozen one in `freeze_updates` alone.
- [ ] 3.3 State in the launch record why R.1c's frozen floor is **not** reused: it ran at
  `plasticity_node_noise` 0.1, the perturbation enters the forward pass whether or not updates are
  frozen, so its floor is a noisier policy than this one.
- [ ] 3.4 A **pilot on disjoint seeds 101–104** before any registered seed: the arms run, the
  eligibility is non-zero, `symmetric` writes nothing outside the pool on real runs, and the frozen
  floor sits where a no-learning policy sits on this cell.
- [ ] 3.5 **A rate check on the same disjoint seeds**: the committed `0.001` and one decade either side,
  learning arm only. The normalised trace is what makes the rate transferable, and this is what stops a
  `does_not_learn` reading being a rate artefact — the discipline R.1c's σ calibration established after
  a carried-over value cost 31.5% of the arm's level. **The committed rate stands unless it is visibly
  off**, and any change is recorded with its reason before the campaign.

## 4. Harness

- [ ] 4.1 `scripts/analysis/l4_eprop.py`, reusing R.1c/R.1d's statistics layer and drift reader. Per-arm
  contrast against the shared frozen control, paired one-sided, BH-FDR across the **four** learning
  arms, 80% bootstrap CIs.
- [ ] 4.2 Both registered minima against each arm: the absolute 1.0-food floor, and 10% of the gap to
  PPO's **matched 18.945**, with the larger binding.
- [ ] 4.3 **Beats-its-floor and reaches-competence reported separately**, as R.1c and R.1d do, since
  only the second bears on R.1b.
- [ ] 4.4 Credited-synapse drift per arm, read through the connectome's `state["topology"]` layout — the
  reader R.1c had to repair. The registered comparison is R.1c's **1.37–1.38×** and R.1d's
  **1.38–1.42×**: a third structural axis holding the same number is the finding, and a different number
  is a bigger one.
- [ ] 4.5 **The spatial prediction**: where the weight change lands by hop distance from the 39-unit
  motor pool, using the committed
  [`_readout_hop_distances`](../../../packages/quantum-nematode/quantumnematode/brain/arch/connectome_ppo.py#L917)
  walk. e-prop drops the multi-hop terms, so a learning arm's change should concentrate near the pool,
  and `random` — the only broad arm with a per-unit signal — is where the prediction is testable.
- [ ] 4.6 **The two matched contrasts reported explicitly**, since the verdict does not turn on them but
  the interpretation does: `symmetric − random_motor` isolates the signal's direction at matched
  breadth, and `random − random_motor` isolates breadth at a matched signal source.
- [ ] 4.7 The four registered readings — `does_not_learn`, `learns_below_competence`, `learns_the_cell`,
  `void` — with the consequence of each in the harness, not in prose.
- [ ] 4.8 Tests for 4.1–4.7, including a fixture in each reading, one where `scalar` matches `random`
  (which makes the result about the trace, not the signal), and one where `symmetric` and
  `random_motor` match (which makes it about the pool, not the direction).

## 5. The stop clauses

- [ ] 5.1 Stage 1 passes before stage 2 launches (2.6).
- [ ] 5.2 **059's three-active-week implementation bound.** The case for this programme over running 7b
  under PPO is that it is cheaper; if the implementation passes three active weeks, that is itself a
  stopping condition. Record the date work started.
- [ ] 5.3 Re-score R.1c's committed `motor` learning and frozen logs through this harness and confirm
  **3.751** and **3.150**. A mismatch means the harnesses disagree and nothing here is comparable.
- [ ] 5.4 `launch.md` committed before anything runs.

## 6. Campaign

- [ ] 6.1 80 runs: four learning arms and one frozen floor × seeds 1–16 at 3000 episodes, with
  `--track-experiment` so drift and the hop-distance reading have weights to read.
- [ ] 6.2 Per-seed CSV, the per-arm table and the verdict under `supporting/063-l4-eprop/`.

## 7. The record

- [ ] 7.1 Logbook 063: the per-arm table, the `scalar` ablation's position and both matched contrasts
  stated in **every** verdict branch, the drift column against R.1c's and R.1d's, the hop-distance
  reading, and the verdict against 059's three registered outcomes.
- [ ] 7.2 The experiments index row and `CHANGELOG.md`.
- [ ] 7.3 The tracker's R.2 entry, and R.1b's remaining gate restated to whatever this leaves it.
- [ ] 7.4 State plainly what this may **not** be cited as, per the design's last section — including
  that the fourth cell of the routing 2×2, true directions reaching all 302 units, is one the mechanism
  forbids rather than one that was skipped.
- [ ] 7.5 If the reading is `does_not_learn`, 059's first outcome fires: record that the programme stops,
  that 7b proceeds under PPO, and that the plausibility claim is given up — as a decision with its
  arithmetic, not as an omission.
