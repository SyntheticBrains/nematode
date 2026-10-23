## Context

The per-step runner (`agent/runners.py`, `StandardEpisodeRunner.run`) advances one animal at a time.
On the frozen 1-s primitives worm, a step does six things:

1. starts any assay phase (`advance_assay`);
2. senses (`_create_brain_params` → `_compute_temporal_data`):
   - the lawn odour at the worm and at its two head-sweep points;
   - the STAM buffer's record and derivative;
   - the adaptive fold-change readout;
   - food contact, satiety, ingestion, locomotor state and proprioception;
3. runs the brain;
4. moves the worm by the primitives;
5. eats from the lawn and restores satiety;
6. decays satiety.

Everything else in the step is dead when learning is off:

- `get_state`'s gradient, which feeds only quantum brains;
- the reward, which feeds metrics only;
- the PPO buffer, frozen;
- predators, temperature and oxygen, all disabled.

What the code survey found (2026-10-24):

- **The brains already batch.** The connectome has `forward_with_hidden_batched`, used by PPO replay.
  The MLP's actor is an `nn.Sequential`.

- **One batching bug.** The connectome's `modulator_release` takes `.mean()` over the whole tensor, so
  on a `(B, N)` batch it averages across animals.

- **Two random streams per animal:**

  - the environment's numpy generator: bout starts, kinds, durations, angles, placements and headings;
  - torch's global generator: one 5-value normal per step, for the policy.

  `set_global_seed(run_seed)` and `env.rng = default_rng(run_seed)` fix both per episode
  (`scripts/analysis/lawn_states.py`, `capture`).

- **No vectorised environment exists.** Parallelism is by process only.

The maintainer's decisions of 2026-10-24:

- frozen evaluation only;
- layered parity;
- sampled noise from per-animal streams;
- a target of at least 20× per core.

## Goals / Non-Goals

**Goals**

- N animals of one config per call, with no per-animal Python on the step path except random draws and
  the rare bout start.
- Per animal, the per-step runner's behaviour on the same seed: exact over short horizons, statistically
  indistinguishable over full assays.
- Captures that every B readout and the classifier read unchanged.
- A measured cost that sizes W.

**Non-goals**

- **Training.** C.3 trains by reward on the per-step runner.

- **Configs outside the declared surface:**

  - random lawn placement, whose geometry differs per animal;
  - point food;
  - free locomotion;
  - the kinematic body;
  - leaky connectome dynamics;
  - plasticity;
  - any other sensing mode.

  Each is refused, and each can be added later behind the same parity test.

- **GPUs and compiled kernels.** A numba or torch rewrite of sensing waits until the pilot shows it is
  needed.

## Decisions

### Decision A: Arrays per animal, one clock for all

`BatchedEvaluator(config, controller, seeds, *, capture=True, mean_action=False)` builds N animals for
the N seeds. All animals share the config and the step clock, so they share the assay's phase schedule
and its lawn geometry. A centred lawn's cells are the same for every animal, and only densities differ.
The state is:

| State | Shape | Notes |
|---|---|---|
| position, heading | (N, 2), (N,) | float64, as the per-step runner's floats |
| bout kind, remaining time, speed, turn rate, ventral side | (N,) each | `PrimitiveState` as arrays |
| lawn density | (N, cells) | one row per animal; refreshed when a phase turns lawns on |
| satiety, step intake | (N,), (N,) | |
| STAM: concentrations, position deltas, action index | (N, B), (N, B, 2), (N, B) | ring buffers of the STAM size B, filled as the per-step runner's deque |
| adaptive baseline, previous sensing position | (N,), (N, 2) | |
| connectome modulators | (N, 2) float32 | |
| previous action present | bool | step 0 has no action (proprioception, STAM's action index) |

The per-step runner's order is kept exactly within a step. That order includes the details that change
values:

- the odour is tanh'd;
- head-sweep points are clamped to `grid_size − 1` and then into the circle;
- food contact is read before the move;
- `step_intake` is zeroed by `track_step` before eating;
- satiety restores before it decays.

### Decision B: Controllers

A `BatchedController` maps the step's batched senses to raw actions `(N, 5)` in `[-1, 1]`. It has
`reset(N)` and `act(senses, noise)`. There are four adapters:

- **`BatchedMLP(brain)`**: features in `extract_classical_features`' sorted module order (16 dims on the
  E.2/E.3 MLP), the actor's mean, and the brain's `log_std`.
- **`BatchedConnectome(brain)`**: the 6-dim state (food features and the modulators), and
  `forward_with_hidden_batched` with the modulators' held current. The concentrations update per
  animal through a batch-safe `modulator_release` (`.mean(-1)`). The single-sample path keeps its
  output, and a test pins it.
- **`BatchedConstantHazards`** and **`BatchedPirouetteHazards`**: vectorised forms of E.2's
  controllers, with identical outputs per animal.

Brains are run with no gradient, frozen. A brain built with traces, plasticity, leaky dynamics, node
noise or e-prop is refused.

### Decision C: Random streams per animal

- **The environment's stream.** Animal *i*'s environment generator is `np.random.default_rng(s_i)`.
  Its draws come in the per-step runner's order: placements and headings at phase starts, the ventral
  side on the first step, then each step's start test while idle, and the kind, duration and angle at a
  start.
  - The start test is a Python loop over idle animals, one `random()` each.
  - A bout start calls E.2's own `_start_bout` with that animal's generator, so durations and angles
    are bit-identical.
- **The policy noise.** Animal *i*'s noise comes from `torch.Generator().manual_seed(s_i)`, five
  `randn` values a step. Then `pre_tanh = mean + std · ε` and `action = tanh(pre_tanh)`, mapped to the
  bounds.
  - Torch's `Normal(mean, std).sample()` under a global seed equals `randn` on a generator with that
    seed, checked 2026-10-24.
  - Layer 1 of the parity test pins the arithmetic to 1e-6.
- **Mean-action mode** skips the noise.
- **Cost.** At N = 256 the per-animal draws were measured at 0.41 µs (torch) and 0.11 µs (numpy) per
  animal-step, well inside the 25-µs target.

The per-step reference for animal *i* is one episode of the same config and controller, with `env.seed = s_i` and `set_global_seed(s_i)` before it runs. That is how `lawn_states.capture` seeds an
evaluation episode.

### Decision D: Layered parity

1. **State level.** For 8 seeds × 4 controllers × the 4 assays, the reference runs 50 steps and logs:

   - each step's feature vector;
   - each step's action mean;
   - the connectome's modulators;
   - the STAM derivative.

   The evaluator, run on the same seeds, must match every logged value within 1e-6.

2. **Trajectory level.** On the same runs, extended to 300 steps, positions, headings, bout kinds,
   satiety and lawn densities must be equal within 1e-9.

   - Batched matrix products can round differently at the last bit. One flipped start test would then
     send a trajectory elsewhere, so exactness is required over a short horizon only.
   - A divergence before step 300 fails the test and is reported with its first step and field.

3. **Statistic level** (slow tier). Full Scheer and Flavell assays on seeds 5001–5032, for the MLP and
   `PirouetteHazards`. Per animal, the paired difference must have a 95% bootstrap interval containing
   zero, with mean |difference| below 0.02, in each of:

   - on-lawn fraction;
   - roaming fraction, from the classifier;
   - reorientation-bout rate per minute;
   - final satiety;
   - total intake.

**Must-fail.** Each layer is re-run against an evaluator with one deliberate fault: the derivative scale
left at the written 50 instead of the derived 250. It must fail layer 1. A second fault, the head-sweep
sides swapped, must fail layers 1 and 2. The tests can therefore fail.

### Decision E: Capture

With `capture=True` the evaluator returns, per animal, a list of `BehaviourStep`. It has the per-step
runner's fields, plus `primitive` and `phase`, and the protocol record. `lawn_states.episode_windows`
and the classifier read it unchanged. Capture is arrays during the run, converted once at the end.

### Decision F: The cost pilot

`scripts/analysis/batched_cost_pilot.py` times, on seeds 5001–5032 repeated to reach N:

- each controller (MLP, connectome, the two hazard controllers);
- on each assay;
- at N = 32, 128 and 256;
- against the per-step runner on 8 of those seeds.

It writes `docs/experiments/logbooks/supporting/104-batched-evaluator/cost_pilot.json` with:

- µs per animal-step, and the speedup per core;
- the wall time of a 200-animal evaluation per assay;
- W's projected budget, as a formula of population, generations, arms and draws per arm, evaluated at
  the tracker's C.2 and W defaults.

Logbook 104 records the pilot and the parity results. Missing the 20× target is a result for W's
sizing, not a merge blocker.

## Risks / Trade-offs

- **Silent drift from the per-step runner.** A later change to sensing in `agent.py` would not reach
  the evaluator. The layer-1 and layer-2 parity tests run in the default tier, on every PR. Any drift
  fails them.
- **Python loops at bout starts.** These are rare: about 0.03 starts per animal-second at the untrained
  policy's rates, fewer when trained.
- **Memory under capture.** 256 animals × 5,400 steps × about 12 fields is about 130 MB. Capture can be
  turned off for fitting, where only summary statistics are needed. B adds summary-only readouts there.
- **The connectome's batched forward differs in last bits.** That is documented in its docstring, and
  is why layer 2's horizon is short.

## Migration

None. This is a new package. The only change to existing code is `modulator_release`'s batch axis, and
its single-sample result is pinned unchanged.
