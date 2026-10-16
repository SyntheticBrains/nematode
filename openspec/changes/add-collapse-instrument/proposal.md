## Why

Every arm in [Logbook 092](../../../docs/experiments/logbooks/092-thermal-body.md) collapsed on some
seeds after episode ~4,000 — 2 of 32 wild-type, 6 gap-only and 5 chemical-only learners peaked and then
fell below half their peak — and every 6,000-episode panel inherits it. The literature places this
failure at exactly this network scale and horizon as an optimiser-level property (Moalla et al. 2024;
Elsayed, Lan, Lyle & Mahmood 2024; Dohare et al. 2024), but nothing here can say which mechanism is
at work: the PPO updates report only mean losses, entropy and gradient norm, nothing per-update reaches
disk, and a run saves `final.pt` alone. D25 (roadmap v4.5 § Phase 9) makes collapse a registered
statistic and forbids any further 6,000-episode body panel before M.1 reads out; this change is M.1(a),
the instrument and the diagnosis. The remedies (M.1(b)) and the confirmatory panel (M.1(c)) are their
own change, registered on what this one finds.

## What Changes

- **Per-update telemetry**, opt-in and byte-identical when off (`training-telemetry`): with
  `--update-telemetry`, each PPO update on the MLP and connectome brains appends one JSON line to
  `exports/<session>/telemetry/updates.jsonl` — the episode and update index; policy entropy and
  `log_std`; approximate KL (`(r−1) − log r`) and clip fraction over the final epoch; value loss and
  explained variance; pre-clip gradient norm, the fraction of minibatches clipped, and actor and critic
  gradient norms separately; the raw advantages' mean, std and maximum magnitude before normalisation;
  the pre-squash action mean's magnitude and the fraction of steps with `|u| > 2`; layer-wise weight
  norms; the saturated-unit fraction (`|h| > 0.99`) and the effective rank (`srank`, δ = 0.01) of the
  features the heads read (the connectome's settled rates, the MLP's last hidden layer); and on the
  connectome the settling residual (the change in the rates between the last two settling iterations).
  Every quantity is computed from tensors the update already holds, under `no_grad`, with no draw from
  any generator.
- **Periodic checkpoints**, opt-in: `--checkpoint-every N` writes
  `exports/<session>/weights/ep<NNNNN>.pt` every `N` episodes with the existing weight format.
- **The collapse statistic** (`architecture-comparison-protocol`): `scripts/analysis/collapse.py`
  implements D25's per-seed definition, incidence with Wilson intervals, censored time-to-collapse, and
  a sensitivity analysis over the definition's three numbers, with tests; it is the module every later
  long panel cites.
- **The diagnosis** (M.1(a)), registered in a launch record before any log is read: M.9's 16 paired
  runs re-run with telemetry and checkpoints — the 8 collapsed wild-type and gap-only learners and, on
  the same 8 seeds, the other wiring's intact learner — after an identity check that the re-runs
  reproduce the committed per-seed plateaus and that telemetry on and off give identical runs; plus
  MLP-PPO's 8 retry seeds (9721–9728) at 6,000 episodes, to measure the MLP's own incidence.
  Event-aligned stacks around each onset against the paired intact run; the cause set and each cause's
  discriminating pattern fixed in the launch record. Readout: Logbook 096.

## Capabilities

**Added**: `training-telemetry` — the per-update telemetry and periodic checkpoints.
**Modified**: `architecture-comparison-protocol` — the collapse statistic as a registered reading
(ADDED requirement).

## Impact

- Code: `learning_rules/ppo.py` and `brain/arch/mlpppo.py` (the telemetry hook), a small shared
  telemetry module under `brain/arch/`, `scripts/run_simulation.py` (two flags), tests.
- Scripts: `scripts/analysis/collapse.py`, `scripts/analysis/collapse_diagnosis.py`.
- Docs: `docs/experiments/logbooks/supporting/096-collapse-diagnosis/`, Logbook 096, the tracker
  (M.1(a)), CHANGELOG, `docs/usage.md` for the flags.
- Machine time: 24 runs at 6,000 episodes through the body, ≈ 2 h at 16 workers; the identity
  check's re-runs are inside it.

## Breaking Changes

None.

## Backward Compatibility

Both flags default off; with them off a run writes nothing new and is byte-identical. With telemetry on
the run's episodes, rewards and final weights are identical to the run with it off (tested).
