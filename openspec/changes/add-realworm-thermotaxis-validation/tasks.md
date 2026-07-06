# Tasks — real-worm behavioural-thermotaxis validation

> Extends the archived `realworm-behavioural-validation` capability (Logbook 035) with a second
> modality. The metric / grading / harness compute path is reused unchanged; the new surface is the
> setpoint-drive capture + the thermal reference + the harness modality selector.

## 1. Setpoint-drive behavioural capture (modality)

- [ ] 1.1 Add `SensingConfig.capture_behaviour_modality: food | thermotaxis` (default `food`,
  byte-identical). Add a `_behaviour_capture_fields` helper on the agent returning
  `(drive, drive-derivative, toward-drive direction, gradient strength)`: `food` returns today's food
  values (byte-identical); `thermotaxis` returns the setpoint drive `−|T − Tc|`, its one-step
  derivative, the toward-comfort direction (live `get_temperature` / `get_temperature_gradient`, flip
  when `T > Tc`), and the thermal-gradient strength. Wire it into the capture append.
- [ ] 1.2 Byte-identical test: `food` modality (default) is unchanged (existing capture tests pass).
- [ ] 1.3 Capture test: `thermotaxis` modality on a thermotaxis-enabled continuous env records a
  non-positive setpoint drive and a live non-zero thermal gradient.

## 2. Thermotaxis reference + modality-aware loading + harness flag

- [ ] 2.1 `data/thermotaxis/behavioural_bias_signatures.json` (same four statistic keys, all
  **sign-only** — thermal magnitudes are not literature-comparable): klinokinesis on the thermal-error
  derivative + weathervane toward comfort (Ryu & Samuel 2002; Clark 2007 / Luo 2014). Add a hardcoded
  thermotaxis fallback + `load_bias_signatures(modality=...)`; an explicit missing path still raises.
- [ ] 2.2 `--modality food|thermotaxis` on the harness → grades against the modality reference set +
  records the modality in the summary JSON; metrics / CI / floor / figures unchanged.
- [ ] 2.3 Tests: the thermotaxis reference set loads (four sign-only statistics, thermotaxis citation
  distinct from chemotaxis); the fallback mirrors the JSON.

## 3. Thermotaxis-dominant continuous cell

- [ ] 3.1 A `Tc`-seeking / isotherm-tracking continuous config (the thermal analogue of the food-only
  klinotaxis cell): linear thermal gradient with the spawn region off-setpoint, reward rewarding
  comfort, thermotaxis sensing `klinotaxis`, foraging pressure minimised, `capture_behaviour: true` +
  `capture_behaviour_modality: thermotaxis`. Difficulty (gradient strength / comfort band / reward
  weights) is a first draft — calibrated at the smoke.

## 4. Calibration / smoke (before the panel)

- [ ] 4.1 Single-seed smoke: train the MLP thermotaxis forager with capture on; calibrate `θ_sharp`;
  confirm the pipeline produces both curves under `--modality thermotaxis` and record which strategy
  the worm shows.
- [ ] 4.2 **PAUSE for user review of the smoke** (which strategies appear + the calibrated cell)
  before the full panel.

## 5. Evaluation + verdict

- [ ] 5.1 Panel: the trained MLP thermotaxis arm (+ connectome companion), n ≥ 8, post-convergence,
  headless, parallelised.
- [ ] 5.2 Run the harness (`--modality thermotaxis`); record both bias curves, statistics + CIs, and
  the per-strategy verdicts vs Luo et al. 2014 (klinokinesis + klinotaxis). Check robustness (θ_sharp,
  tail-window, floor fraction, CI level) as in 035.
- [ ] 5.3 **Conditional specificity control (iff the panel shows a positive thermal weathervane).**
  A `thermotaxis_mode: derivative` arm (temporal thermal sensing, no head-sweep) — the direct analogue
  of 035's food-derivative control; expect klinokinesis to persist and the weathervane to collapse if
  it is sensor-driven. Skip (note as a limitation) if the panel shows no weathervane.
- [ ] 5.4 **PAUSE for user review of the evaluation + verdict before writing the logbook.**

## 6. Logbook + tracker

- [ ] 6.1 Write Logbook 036 (objective / method / results / analysis / limitations, incl. the
  sign-only thermal-reference + homeostatic-setpoint caveats) + committed supporting artefacts (no
  `tmp/` references); feeds the T9a synthesis's biological-validation section alongside 035.
- [ ] 6.2 Add the logbook row to `docs/experiments/README.md`.
- [ ] 6.3 Tick `T7.validation.thermotaxis` in `openspec/changes/phase6-tracking/tasks.md` with the
  verdict (and correct the earlier "no new OpenSpec change" note to point at this change).
- [ ] 6.4 Document the `capture_behaviour_modality` flag + the `--modality` harness option in
  `AGENTS.md`.

## 7. Pre-merge gates

- [ ] 7.1 Targeted `pre-commit` during iteration; full `pre-commit run -a` before push.
- [ ] 7.2 `openspec validate add-realworm-thermotaxis-validation --strict`.
- [ ] 7.3 Full `uv run pytest -m "not nightly"` green (byte-identical `food`-modality invariant holds).
- [ ] 7.4 Archive the change in-PR (`openspec archive add-realworm-thermotaxis-validation -y`).
