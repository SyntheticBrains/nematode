# Tasks: The calibration-and-robustness surface

Phase 8 task **A.2**. The plan is authoritative in `docs/roadmap.md` § Phase 8 **D16** as amended
2026-09-20; the design decisions taken before anything ran are in this change's `design.md`.

**Registration discipline**: tasks 1–6 land and the protocol is registered **before any panel run**.
Task 7's pilot confirms the levels run, the gates fire and the cost estimate holds; the minimum
effect and the power come from A.1's committed hard350 per-seed spread, not from four pilot seeds.
Task 8 registers all of it — including the rule that picks the primary metric and the branch a sign
move takes — and only then does the PPO campaign launch. **No panel seed is touched before task 8**,
and the pilot uses its own band so it cannot contaminate one.

**Sequencing**: the PPO half (tasks 9–11) reads out before the reading half (tasks 12–14) launches.
It is what B.1 and B.2 cite, it is the only half that can force an A.1 re-read, and it is the
cheaper campaign on which to find a defect in the shared generator or driver.

- [ ] 1. **The config generator** — `scripts/campaigns/generate_operating_point_configs.py`, in the
  mould of `scripts/campaigns/l4_panel_pilot.py:41-54` but across five pins instead of one. It loads
  a committed parent, writes exactly one key under `brain.config`, and emits into
  `configs/scenarios/foraging/` using the established suffix vocabulary: `_wide` (already meaning
  `readout_width: per_neuron` in 18 committed configs), `_d2`/`_d3`/`_d6`,
  `_lsm15`/`_lsm10`/`_lsm05`/`_ls00`/`_lsp05`, `_r1e4`/`_r1e2`, `_td05`/`_td099`. Each file carries
  a header naming the pin under test, its parent, and the single-key delta, per the house
  convention. Guards: never emit `forward_pass_depth: 0` (refused only at construction, not load);
  never pair a non-zero `initial_log_std` with `continuous_std_mode: state_dependent` (refused at
  load); never emit two configs sharing a stem, since the campaign runner would hash-disambiguate
  the log names and the manifest builder keys on the bare stem.

- [ ] 2. **Configs** — roughly 57 generated, committed YAMLs, plus fifteen committed arms serving
  their levels unchanged.
  **PPO half, 32 arms**: 8 points (centre + `_wide` + three depths + three log-stds) × 2 wirings ×
  {learning, frozen}. The four committed block-V hard350 arms are the centre, so 28 are new. Nothing
  committed is a PPO `per_neuron` arm — all 18 `readout_width: per_neuron` configs are `three_factor`
  — so that level is entirely new.
  **Reading half, 40 arms**: 12 learning points × 2 wirings = 24 learning arms, plus frozen floors at
  the 8 **construction**-pin points only × 2 = 16. `plasticity_rate` and `trace_decay` cannot move a
  frozen arm, so the centre's floor is theirs. Eleven committed `eprop` arms cover the centre,
  `_wide` and `_r1e4` levels, so 29 are new.
  The generator skips a config that already exists rather than overwriting it.

- [ ] 3. **Config tests** — each file in the panel, **generated or committed**, loads through the
  real loader and differs from its centre parent in **exactly** the one key it names. Covering the
  committed arms matters: eleven of them were authored for earlier rungs, and one that turns out to
  carry a second delta is a config to regenerate rather than a fact to work around. Also: no two
  stems collide, and stating a pin's default explicitly is byte-identical to leaving the key absent,
  which is what makes the committed centre the operating point rather than a near miss of it.

- [ ] 4. **The pin-reach test** — new
  `packages/quantum-nematode/tests/quantumnematode_tests/brain/arch/test_connectome_pin_reach.py`,
  discharging this change's added requirement and Decision B.05. It asserts, per pin and per
  learner, that the level changes what the configured learner computes or updates — and asserts the
  negative cases that motivated the requirement: `plasticity_rate` and `trace_decay` leave a PPO arm
  unchanged, so they are confined to the reading half, and they leave a **frozen** arm unchanged
  under either learner, which is what licenses one floor serving all their levels. It also asserts the shared generator's
  **post-construction state is equal across pin levels**, per protocol principle 6's 2026-09-21
  clause: the comment at `connectome_ppo.py:888-893` claims the readout draw happens at the pooled
  shape whatever the width, and A.1 proved that exactly this kind of claim must be tested rather
  than read.

- [ ] 5. **The analysis driver** — `scripts/analysis/operating_point_surface.py`, in the
  `init_sharing_control.py` mould: an explicit `ARM_BY_STEM` built by a loop over pins × levels
  (never a regex — the suffix order is not free), `build_manifest`, `require_complete`, per-level
  calls to the **unmodified** `wiring_premise.efficiency_contrast`, the censoring-driven metric
  choice, the interaction against the campaign's own centre, and the surface report. CSV written
  with `lineterminator="\n"`.

- [ ] 6. **Driver tests** — mirroring `test_init_sharing_control.py`, including the
  `git diff --quiet origin/main` assertion that `wiring_premise.py` and
  `connectome_structure_efficiency.py` are untouched, the stem-mapping cross-product, seed
  freshness against the burnt bands (1–96, 101–108, 129–160), and the completeness gate refusing a
  partial panel.

- [ ] 7. **Pilot** — seeds 109–112, a subset of levels including both extremes of
  `forward_pass_depth`, configured the way the campaign will be. It confirms the levels run, the
  learning gates fire, and the cost estimate holds. It does **not** estimate the spread; four seeds
  cannot.

- [ ] 8. **Registration, before any panel seed** —
  `docs/experiments/logbooks/supporting/071-operating-point-surface/launch.md`, in 070's shape:
  the question; the design; the metric rule fixed before the censoring rates are known; the
  **sensitivity per level, from A.1's committed hard350 per-seed CSV** (prior committed data, frozen
  2026-09-21) as the interaction requirement demands; the minimum effect as a fraction of the centre
  wiring gap, **in both directions**; the registered branch a sign move takes, including the A.1
  re-read; the statement that the crossing pass is a second registration; the A.0 retention line;
  and the cost.

- [ ] 9. **PPO campaign** — 512 runs, seeds 161–176, through `scripts/run_campaign.py` at 16
  workers into a gitignored `campaigns/` directory, with detailed export off per the runner's own
  warning.

- [ ] 10. **Read the PPO surface** — against the registered branches. Report the interaction at each
  level with its sensitivity, the wiring gap's sign, and every gate. A level whose gate fails is
  reported as a gate failure, which is itself a sensitivity result, not worked around.

- [ ] 11. **The A.1 branch** — if the PPO half moves the sign at any level, register the A.1 re-read
  at that point as the registered outcome it is, and add the condition as a dated note at every site
  that cites block V or Logbook 070. If it does not, record that plainly with the panel's sensitivity
  beside it.

- [ ] 12. **Reading-learner campaign** — 640 runs, seeds 177–192, after the PPO half has read out.

- [ ] 13. **Read the reading-learner surface** — separately from the PPO one. `initial_log_std` is
  reported as two different quantities across the halves, not pooled: under PPO it is a trained
  parameter's start point, under the rule it is fixed exploration noise for the whole run.

- [ ] 14. **Logbook 071** — `docs/experiments/logbooks/071-operating-point-surface.md` plus its
  `supporting/` directory, the index row in `docs/experiments/README.md`, and the roadmap's A.2
  status rows and exit-criterion checkbox. The surface is reported as a surface: A.2 declares no
  winner, and which point a later rung runs at is that rung's calibration decision citing this one.

- [ ] 15. **Tracker** — tick A.2 in `openspec/changes/phase8-tracking/tasks.md` with its shipment
  status, and record the dated correction to the "shared by both learners" wording: the three pins
  are shared across the learners of `ConnectomePPOBrain`, but `readout_width` and
  `forward_pass_depth` are not declared on `MLPPPOBrainConfig` and are dropped there with a warning
  rather than an error, so a later rung must not read this surface onto an MLP arm.
