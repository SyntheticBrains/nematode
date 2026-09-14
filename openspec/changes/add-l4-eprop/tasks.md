# Tasks

## 1. The mechanism

- [x] 1.1 Add `"eprop"` to `EligibilityMode` in
  [`_plasticity_config.py`](../../../packages/quantum-nematode/quantumnematode/brain/arch/_plasticity_config.py),
  with a `LearningSignalRouting` Literal (`"symmetric" | "random_motor" | "random" | "scalar"`) and a
  `plasticity_learning_signal` field defaulting to `"random"`. Document at the seam what the truncation
  drops and that symmetric feedback reaches only the readout pool.
- [x] 1.2 Validators, as model validators **and** as an explicit guard the brain re-runs — `model_copy`
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
- [x] 1.3 The eligibility itself, per substrate. Connectome: accumulate
  `eps_ij = Σ_s psi_j^(s+1) · h_i^(s)` across the settling loop under `no_grad`, into a transient
  buffer registered in `_TRANSIENT_BUFFERS`. MLP: the single-pass form `psi_j · h_i` per plastic layer.
  Both leave the forward pass **bitwise unchanged** — assert it.
- [x] 1.4 The fold-in seam: `apply_learning_signal(score)` on the plastic-topology Protocol, called once
  per environment step immediately after the action is sampled, computing `L` from the routing and doing
  `E ← decay·E + M ∘ (eps · L)`. A no-op for every eligibility mode but `eprop`, so the existing
  substrates pay nothing.
- [x] 1.5 The score function is taken **pre-squash** — `_continuous_action_step`'s `pre_tanh`, not the
  bounded action — since `∂ log π / ∂ mu` is `(u − mu)/sigma²` and the squash correction does not depend
  on `mu`. A test pins it against autograd on the committed head.
- [x] 1.6 `symmetric` routing derives `B` from the readout **through the mean-pool**, and derives it
  **live** rather than caching it — the anatomical contrast overwrites the orthogonal draw after the
  topology is built, and a loaded checkpoint can substitute the readout again, so a cached transpose
  would describe a readout the arm never ran with. Found by the pooling-divisor test:
  `readout[k, class(j)] / |class(j)|` on the 39 pooled units and zero elsewhere. A test asserts the 263
  non-pool units receive exactly zero, since that is the arm's defining property and not a bug.
- [x] 1.7 `random_motor` is `random`'s projection masked to the same 39 units, drawn from the **same**
  generator draw so the two arms differ by the mask alone at one seed.
- [x] 1.8 **A forward whose signal was never folded in must fail**, not credit a stale `eps`. The
  failure is silent and reads as a rule that learns slowly, so it is a raised error and a test.
- [x] 1.9 `B` is drawn once per run from the run seed and **persisted with the checkpoint**: a reloaded
  policy that draws a new projection is learning against a different feedback path than the one it was
  trained with.
- [x] 1.10 Tests: the derivative is the activation's (numeric check against autograd on a small net);
  the trace matches a closed-form `eps` on a two-unit two-step settling case; `scalar` gives `L = 1`
  exactly; the forward is unchanged with the mode on; each guard in 1.2 fires under `model_copy` as well
  as construction.

## 2. Stage 1 — the positive control, which is a stop clause

- [x] 2.1 Three arms in
  [`l4_rule_positive_control.py`](../../../scripts/analysis/l4_rule_positive_control.py) at the existing
  pins and over the existing rate grid: `eprop_symmetric`, `eprop_random`, `eprop_scalar`. Any rate
  passing counts as a pass, as for the committed arms.
- [x] 2.2 `eprop_symmetric` **must pass** — on a one-step scalar-action task it is the exact REINFORCE
  gradient of the plastic layer, so a failure is an implementation fault and the control is **VOID**.
- [x] 2.3 `eprop_scalar` **must not pass** — a rule with no per-unit signal solving a cue-to-target
  association means the task is not discriminating, and the control is **VOID**.
- [x] 2.4 `eprop_random` is reported and required of nothing; its result bounds what stage 2 could show
  and is stated in the record either way.
- [x] 2.5 A cross-check that the existing `three_factor`, `hebbian` and `analytic` arms are **unchanged
  to the committed values** — the harness gained arms, and if the old ones moved, something shared did.
- [ ] 2.6 **No stage-2 run is launched until 2.2 and 2.3 both hold.** Record which.

## 3. The arms

- [x] 3.1 Seven configs from the committed `hard350` cell: `eprop_{symmetric,random_motor,random,scalar}`,
  `eprop_plastic_readout` and `eprop_readout_only` (task 8)
  and one shared `eprop_frozen`, at `initial_log_std: -1.0`, `plasticity_node_noise: 0.0`, the committed
  anatomical readout, no perturbation set, and the pinned rule settings written out rather than
  inherited — `plasticity_normalise_modulator`, `plasticity_normalise_trace` and
  `plasticity_homeostasis` on, `plasticity_rate: 0.001`, `trace_decay: 0.9`, `forward_pass_depth: 4`.
- [x] 3.2 Exact-key test: the four frozen-readout learning configs differ from each other in
  `plasticity_learning_signal` **alone**, `eprop_plastic_readout` differs from `eprop_random` in
  `plasticity_plastic_readout` alone, and the frozen config differs from `eprop_random` in
  `freeze_updates` alone.
- [x] 3.3 State in the launch record why R.1c's frozen floor is **not** reused: it ran at
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

- [x] 4.1 `scripts/analysis/l4_eprop.py`, reusing R.1c/R.1d's statistics layer and drift reader. Per-arm
  contrast against the shared frozen control, paired one-sided, BH-FDR across the **six** learning
  arms, 80% bootstrap CIs.
- [x] 4.2 Both registered minima against each arm: the absolute 1.0-food floor, and 10% of the gap to
  PPO's **matched 18.945**, with the larger binding.
- [x] 4.3 **Beats-its-floor and reaches-competence reported separately**, as R.1c and R.1d do, since
  only the second bears on R.1b.
- [x] 4.4 Credited-synapse drift per arm, read through the connectome's `state["topology"]` layout — the
  reader R.1c had to repair. The registered comparison is R.1c's **1.37–1.38×** and R.1d's
  **1.38–1.42×**: a third structural axis holding the same number is the finding, and a different number
  is a bigger one.
- [x] 4.5 **The spatial prediction**: where the weight change lands by hop distance from the 39-unit
  motor pool, using the committed
  [`_readout_hop_distances`](../../../packages/quantum-nematode/quantumnematode/brain/arch/connectome_ppo.py#L917)
  walk. e-prop drops the multi-hop terms, so a learning arm's change should concentrate near the pool,
  and `random` — the only broad arm with a per-unit signal — is where the prediction is testable.
- [x] 4.6 **The four matched contrasts reported explicitly**, since the verdict does not turn on them
  but the interpretation does: `symmetric − random_motor` isolates the signal's direction at matched
  breadth, `random − random_motor` isolates breadth at a matched signal source, and
  `plastic_readout − random` isolates the readout at a matched signal and matched breadth — the one
  stage 1 predicts will be the largest — and `plastic_readout − readout_only` isolates the
  substrate's own plasticity at a matched readout.
- [x] 4.7 The four registered readings — `does_not_learn`, `learns_below_competence`, `learns_the_cell`,
  `void` — with the consequence of each in the harness, not in prose.
- [x] 4.8 Tests for 4.1–4.7, including a fixture in each reading, one where `scalar` matches `random`
  (which makes the result about the trace, not the signal), and one where `symmetric` and
  `random_motor` match (which makes it about the pool, not the direction).

## 5. The stop clauses

- [ ] 5.1 Stage 1 passes before stage 2 launches (2.6).
- [x] 5.2 **059's three-active-week implementation bound.** The case for this programme over running 7b
  under PPO is that it is cheaper; if the implementation passes three active weeks, that is itself a
  stopping condition. Record the date work started.
- [x] 5.3 Confirm every comparator against the record that measured it, rather than re-scoring R.1c's
  logs through a harness whose labels do not match them: R.1c's committed `motor` arm gives **3.751**
  learning, **3.150** frozen and a **+0.601** shift, and R.1d's protocol gives PPO's matched **18.945**.
  Read from their JSON by test. This caught a transcription error before the campaign: **4.361** was
  carried in four places as the `motor` arm's level when it is the σ-calibration figure at σ 0.1, a
  different run.
- [x] 5.4 `launch.md` committed before anything runs.

## 6. Campaign

- [ ] 6.1 112 runs: six learning arms and one frozen floor × seeds 1–16 at 3000 episodes, with
  `--track-experiment` so drift and the hop-distance reading have weights to read.
- [ ] 6.2 Per-seed CSV, the per-arm table and the verdict under `supporting/063-l4-eprop/`.

## 7. The record

- [ ] 7.1 Logbook 063: the per-arm table, the `scalar` ablation's position and all four matched contrasts
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

## 8. The plastic-readout arm, added after stage 1 and before any arm ran

Stage 1 measured that a **frozen** readout is what stops a broadcast projection working: feedback
alignment needs the forward path to the output to come into alignment with the feedback matrix, and a
frozen readout cannot. Same task, same rule, same projection, 20,000 trials — frozen, the broadcast
arm's best is **−0.7031** against a cue-blind floor of −0.6909; plastic, it reaches the optimum of
**−0.1353 exactly** on every seed. So `random`, the arm the plausibility claim rests on, is
structurally unable to work in the four registered arms' configuration.

**This change's original exclusion of a plastic readout was not supported by the evidence it cited.**
Logbook 040 measured the collapse under the **Hebbian** rule, where a plastic output layer's
post-synaptic factor is its own *output* and its rows self-amplify. Under e-prop the factor is its own
*error*, and there is no such loop.

- [x] 8.1 `plasticity_plastic_readout`, rejected outside `eprop` with 040's mechanism as the reason,
  and rejected together with `freeze_updates` — that combination would be reported as a
  plastic-readout floor, and there is no such arm.
- [x] 8.2 The readout as a **second plastic tensor** on the seam: its own trace, an all-true mask, the
  fan-in axis its `[action, class]` layout implies, and the action mean as its post-synaptic activity.
- [x] 8.3 Its eligibility, which is **exact**: `E[k, c] ← decay·E[k, c] + score_k · pooled_c`. The
  readout's post-synaptic units are the action dimensions, so its learning signal is the identity and
  no path is dropped. Pinned against autograd end to end, and against the same identity on the MLP.
- [x] 8.4 **Excluded from the homeostatic rescale**, through a per-tensor flag the topology supplies
  and the rule honours, so no other substrate changes. The rescale returns each unit's incoming norm
  to construction and the readout's scale is part of what this arm asks about — R.1d measured +4.51
  foods from that scale alone. The rule's weight bound still applies at 3.0 per entry, which allows a
  readout norm of 8.49 against the 7.820 R.1d's PPO harvest reached.
- [x] 8.5 The harness reports **where the readout ended up** for this arm — norm and cosine to the
  anatomical default — against R.1d's two measured readouts (anatomical 1.414; PPO 7.820 at cosine
  −0.178), so "e-prop rediscovers something like PPO's decoding" and "it finds something else" are
  separable.
- [x] 8.6 Tests for 8.1–8.5, including that the four frozen-readout arms expose **one** plastic tensor
  and this one exposes two.

## 9. The readout-only control, added with the plastic-readout arm

The pilot on disjoint seeds 101–104 is why this exists rather than being a precaution:
`plastic_readout` reached **15.550 foods of 20 and 49.13% full clear** against a shared frozen floor
of 1.829, while `random` sat at **3.730** — inside R.1c's 2.4–4.4 band, as stage 1 predicted. That is
the first arm in this programme to reach competence on the connectome, and it is exactly the result
that must not ship with its alternative explanation untested: a plastic readout is an 8-parameter
linear map over four pooled motor-class means.

- [x] 9.1 `plasticity_plastic_tensors: readout_only`, which withholds `w_chem` from the seam, rejected
  without a plastic readout — it would otherwise leave nothing plastic and read as a frozen control
  wearing a learning arm's name.
- [x] 9.2 The topology decides what it exposes in **one** place, and every aligned list derives from
  it. Five parallel branches over the same two switches is how a trace comes to be paired with
  another tensor's mask; a test asserts the lists stay the same length for every arm.
- [x] 9.3 The config differs from `eprop_plastic_readout` in `plasticity_plastic_tensors` alone.
- [x] 9.4 The harness reports `plastic_readout − readout_only` as a matched contrast and carries a
  per-arm `chemical` column, so the table says which arms could write the substrate at all.
- [ ] 9.5 Its credited drift reads ~0, which is the check that the withholding actually happened.
- [x] 9.6 The pilot's verdict is **withheld**, not printed: at four seeds the exact paired test cannot
  reach the significance gate — its smallest reachable p is 2⁻⁴ = 0.0625, which BH across the arms
  pushes above it — so every arm reads `no_improvement` whatever it did. The harness printed
  `does_not_learn — the programme stops` on that evidence before this was fixed.
