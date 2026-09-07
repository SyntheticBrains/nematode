# Design: warm-start tooling

## Context

The connectome brain's policy is a function of the current observation alone: every forward
pass starts from a zero hidden state and settles through a fixed number of within-step hops, and
the eligibility trace is the only thing that remembers the previous step. Offline behavioural
cloning on independent observation-action pairs is therefore valid. The brain already has a
batched forward (used by the PPO update's re-evaluation) and a preprocessing path from
`BrainParams` to feature arrays, so a trainer can reuse both rather than reimplement the sensory
projection. The MLP-PPO brain implements `WeightPersistence` with `policy`, `value`, `optimizer`
and `training_state` components and a std-mode check; the connectome mirrors that shape.

## Goals / Non-Goals

**Goals**

- Make a warm-started connectome arm expressible as a config plus a `.pt` file.
- Record exactly what a teacher saw and meant to do, once, in a form any student can consume.
- Clone into either of the two parameter sets the panels use, deterministically per seed.
- Change nothing about any run that does not ask for these.

**Non-Goals**

- Training the teacher, choosing rollouts, the arms, the family (the next change).
- DAgger or any on-policy correction: the teacher's own visitation distribution is the dataset.
- Persistence for the plastic rule's running state across sessions (baseline, scales, traces
  reset on load; a warm start is a fresh start from better weights).

## Decisions

### D1. Connectome weight persistence

`get_weight_components` returns:

| component | contents | when |
|---|---|---|
| `topology` | the topology module's full `state_dict`: `w_chem`, every sensory gain, `readout`, `log_std` or the std head, plus the wiring buffers `m_chem` and `g_gap` | always |
| `value` | the PPO critic's `state_dict` | PPO rule live |
| `optimizer` | the PPO optimiser's `state_dict` | PPO rule live |
| `training_state` | episode count, `continuous_std_mode`, `learning_rule`, `wiring`, `weight_init`, `connectome_source` | always |

`load_weight_components` validates before mutating: the std mode through the existing
`raise_on_std_mode_mismatch`; the wiring by comparing the saved `m_chem` and `g_gap` buffers to
the receiving brain's (a warm start onto a different wiring is a silent error the mechanism must
refuse; the rewired arm loads a clone made on its own wiring). It then loads `topology`, loads
`value` and `optimizer` only if present and the PPO rule is live (a plastic-rule brain ignores
them; a PPO brain given a file without them keeps its fresh critic and optimiser, which is the
warm-start case), resets the rollout buffer, and under the plastic rule resets the rule's
running state. `learning_rule` and `weight_init` in `training_state` are recorded, not
enforced: loading a full-set clone made under the PPO config into a plastic-rule brain is the
intended path, and the readout it carries replaces the anatomical one — a deliberate,
config-visible choice the panel registers per arm. A round trip is bit-identical.

### D2. The reported action mean

`ActionData` gains `continuous_mean: tuple[float, float] | None`, the tanh-squashed Gaussian mean
rescaled to the action bounds exactly as the sample is — the action the policy would take with
its noise removed. Set by the MLP-PPO and connectome brains on the continuous path in the same
call that builds the sampled action; every other brain leaves it `None`. Nothing reads it in a
default run.

### D3. Rollout recording

A `RolloutRecorder` (new module beside the brain package's actions) is attached to the agent when
`--record-rollouts PATH` is given. The runners call `recorder.record(episode, step, params, action_data)` immediately after `run_brain` returns, and make no call when no recorder is
attached — the default path is untouched. Each call appends one JSON line: `episode`, `step`,
`params` (the `BrainParams` dump with `None` fields and the `action` field dropped), `action`
(the sampled continuous action), `action_mean`, `probability`. The file is flushed per episode
and closed at session end. Recording is orthogonal to tracking and to learning; the teacher is
recorded with `freeze_updates: true` so its policy is stationary across the recording.

### D4. The cloning trainer

`scripts/campaigns/l4_behavioural_clone.py --config C --seed S --rollouts R --parameter-set {plastic,full} --out W [--epochs --lr --batch-size --holdout]`:

1. Build the student from config `C` at seed `S` through the same factory the entry point uses,
   so its initial weights are exactly the arm's at that seed.
2. Read `R`; reconstruct each `BrainParams`; run the student's own preprocessing to the feature
   arrays it feeds its batched forward; stack them.
3. Forward in batches; map the pooled motor activity through the student's readout to the
   Gaussian mean; squash and rescale exactly as sampling does; loss = mean squared error against
   the teacher's `action_mean` in the action space (bounded, scale-consistent; a target the
   teacher saturated is matched at the bound rather than chased to infinity).
4. Optimise with Adam over the chosen set. `plastic`: `w_chem` alone, with the mask applied to
   its gradient so the update never leaves the wiring, behind whatever readout the config built
   (anatomical under the plastic configs). `full`: `w_chem`, every gain, the readout and the
   noise parameter — what PPO trains.
5. Hold out a seeded fraction of episodes; report initial, final train and held-out loss, and
   the parameter set's norm change; refuse to save if the final loss is not below the initial.
6. Save through `save_weights` with the `topology` and `training_state` components; a
   `clone.json` beside it records the arguments, the rollout file's hash, the losses and the
   student's seed.

Deterministic given the seed (the holdout split and batch order draw from a generator seeded by
`S`). The trainer never runs the environment; a clone's behavioural quality is the panel's
question.

### D5. Tests

- Persistence: round trip bit-identical on the real C3 configs under PPO and under the plastic
  rule; wiring mismatch refused (wild-type file into a rewired brain, and the reverse) before
  any mutation; std-mode mismatch refused; `value`/`optimizer` present only under PPO and
  ignored by a plastic brain; buffer and rule state reset on load; the frozen-reference and
  wiring-arm byte-identity tests unchanged.
- The mean: on both brains `continuous_mean` equals the squashed, rescaled mean recomputed from
  the same forward; `None` on a discrete brain.
- Recorder: a short headless run with the flag writes one line per step with the fields above
  and the same `BrainParams` values the brain's history saw; without the flag no file and no
  call.
- Trainer: self-cloning — record a frozen connectome's rollouts on a small foraging config, clone
  a fresh student of the same wiring at another seed; held-out loss falls by an order of
  magnitude; under `plastic` the gains, readout and noise are bit-identical before and after;
  under `full` they change; the saved file loads into a brain built from the same config and
  reproduces the student's mean on a recorded observation.

## Risks / Trade-offs

- **A clone's loss is not its behaviour.** The trainer reports only fit; a low-loss clone can
  still forage badly if the teacher's visited states do not cover the student's. That is the
  panel's measurement, and the reason the panel keeps frozen clone arms as its representability
  floors.
- **Loading a full-set clone into a plastic brain replaces the anatomical readout.** Intended and
  registered per arm; the mechanism records `learning_rule` in `training_state` so the record
  shows which readout an arm ran with.
- **The plastic set may not reach the teacher.** Chemical weights behind a fixed readout and
  random gains may not represent the MLP's policy; that gap is the representability result, not
  a defect.

## Open Questions

None.
