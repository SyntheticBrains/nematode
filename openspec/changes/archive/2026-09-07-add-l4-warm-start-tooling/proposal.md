# Warm-start tooling: connectome weight persistence, rollout recording, behavioural cloning

## Why

Three panels (Logbooks 040–042) established that on the C3 cell the outcome of any local
learning rule is decided by the random initial weights: most random initialisations are dead, a
fifth are competent with no learning, and the rule's fixed points vary by seed more than by
wiring. The imitation warm start (roadmap D13, tracker S.2) is the item that removes that
confound — clone a competent policy into the connectome on every seed, then ask what the rule
does from there — and it also answers D13's older question of whether the connectome can *hold*
a good policy at all, or whether its fifth-of-six rank under PPO (Logbook 029) was a
representability limit rather than a learnability one.

None of that can run today, for three concrete reasons found in recon:

- **The connectome brain cannot load weights.** It implements neither half of the
  `WeightPersistence` protocol (issue #308); the entry point's `--load-weights` and the
  config-level `weights_path` raise on it. A warm-started arm is a config pointing at a `.pt`
  file, so this is the gate.
- **Nothing records what a brain saw.** Tracking exports carry the sampled action and a state
  label; the `BrainParams` the brain read are not written anywhere. Cloning needs observation
  and action pairs from the teacher.
- **The action mean is not reported.** Continuous brains expose only the sampled, squashed
  action. The clone should target the teacher's policy, not its exploration noise, and no brain
  offers `top_only` on the continuous path.

Ratified with Chris 2026-09-07: S.2 splits into this tooling change and a separate
pre-registered panel; the clone trains two parameter sets (the plastic set the rule can inherit,
and the full PPO set for representability); the panel fine-tunes both; one teacher — the best of
eight MLP-PPO seeds — supplies every student's rollouts.

## What Changes

- **Connectome `WeightPersistence`**: `get_weight_components` / `load_weight_components` on
  `ConnectomePPOBrain` — a `topology` component (every parameter and wiring buffer), `value` and
  `optimizer` when the PPO rule is live, and `training_state`. Loading refuses a wiring or
  std-mode mismatch before touching anything, resets the rollout buffer, and under the plastic
  rule resets the rule's running state. Closes #308. Byte-identical when unused.
- **`ActionData.continuous_mean`**: the squashed action mean beside the sampled action, set by
  the MLP-PPO and connectome brains on the continuous path; `None` elsewhere.
- **Rollout recording**: `--record-rollouts PATH` on the simulation entry point writes one JSON
  line per step — episode, step, the `BrainParams` the brain read, the sampled action and the
  action mean. Off by default; the runners make no call when no recorder is attached.
- **A behavioural-cloning trainer** (`scripts/campaigns/l4_behavioural_clone.py`): builds a
  student from a config at a seed, preprocesses recorded observations through the student's own
  feature path, regresses its squashed action mean onto the teacher's by mean squared error in
  the action space over a chosen parameter set — `plastic` (chemical weights only, behind the
  config's readout) or `full` (everything PPO trains) — reports train and held-out loss, and
  saves the result through `save_weights`.
- Tests for each, including a self-cloning end-to-end test (a frozen connectome as its own
  teacher) and the frozen-reference byte-identity tests unchanged.

Out of scope: the teacher's training, the panel's arms, seeds, family and verdict (the next
change); any change to the rule or the substrate.

## Capabilities

**Modified**: `weight-persistence` (connectome), `continuous-action-policy` (reported mean),
`cli-interface` (the recorder flag).
**New**: `behavioural-cloning` (the recorder and the trainer).

## Impact

- Edited: `brain/arch/connectome_ppo.py` (persistence, the mean), `brain/arch/mlpppo.py` (the
  mean), `brain/actions.py` (`ActionData`), the agent runners (the recorder hook),
  `scripts/run_simulation.py` (the flag); new `scripts/campaigns/l4_behavioural_clone.py` and a
  recorder module; tests; `docs/architectures.md`, `docs/usage.md`, `CHANGELOG.md`.
- Default runs are byte-identical: no recorder, no load, no change to any sampled action.
