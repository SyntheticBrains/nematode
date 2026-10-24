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
  - state-dependent action noise;
  - a reserve that ends the episode (`starvation_ends_episode: true`): all animals share one clock,
    so none may stop early;
  - any other sensing mode.

  Each is refused, and each can be added later behind the same parity test.

- **GPUs and compiled kernels.** A numba or torch rewrite of sensing waits until the pilot shows it is
  needed.

## Decisions

### Decision A: Arrays per animal, one clock for all

`BatchedEvaluator(config, controller, seeds, *, mean_action=False)` builds N animals for
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

**Tolerances come from float32.** Batched and single-sample forward passes differ at the last bits of
float32. On 2026-10-24 the measured differences were:

- MLP: 2 × 10⁻⁷;
- a 302 × 302 settling product: 1.5 × 10⁻⁶;
- one row at N = 1 against the same row at N = 256: 5 × 10⁻⁸.

Exact equality is therefore not the test.

1. **State level.** For 4 seeds × 4 controllers × the 4 assays, the reference runs 50 steps and logs:

   - each step's feature vector;
   - each step's action mean;
   - the connectome's modulators;
   - the STAM derivative.

   The evaluator, run on the same seeds, must match every logged value within 1e-5.

2. **Trajectory level.** On the same runs, extended to 300 steps:

   - discrete events must match exactly: every bout's kind and start step;
   - positions and headings must match within 1e-3 mm and 1e-3 rad over the first 150 steps;
   - satiety and lawn densities must match within 1e-6 over the same 150 steps.

   Float32 rounding grows around the closed sensory loop. On Scheer's lawn edge, where the odour
   gradient is steepest, the MLP's difference grew about tenfold every 40 steps (measured
   2026-10-24): 10⁻⁷ at step 0, 10⁻⁵ by step 180, 10⁻³ by step 228. Its bouts all still matched at
   step 300. Continuous values are therefore held over 150 steps, where the worst case measured was
   1.6 × 10⁻⁵, and bouts over all 300. A fault moves either by orders of magnitude more. A flipped start
   test is improbable: the per-step chance is about 10⁻⁶ of the start probability. If one does flip,
   it is reported, not tolerated.

   - Batched matrix products can round differently at the last bit. One flipped start test would then
     send a trajectory elsewhere, so exactness is required over a short horizon only.
   - A divergence before step 300 fails the test and is reported with its first step and field.

3. **Statistic level** (slow tier). Full Scheer and Flavell assays on seeds 5001–5032, for the MLP and
   `PirouetteHazards`. These are four tests, one per assay × controller, at about 75 s each: one
   5-minute test would sit alone at the end of a CI shard. Per animal, the paired difference must have
   a 95% bootstrap interval containing zero, in each statistic below. A statistic equal within 1e-9
   on every animal passes outright.

   This is a check for bias, not identity. Over a full hour the closed loop decorrelates some pairs:
   the MLP's bout rate on Scheer differed by about ±0.15/min between paired animals, a mean of 0.036,
   measured 2026-10-24. Thirty-two animals cannot hold a mean difference to 0.02. The statistics:

   - on-lawn fraction;
   - roaming fraction, from the classifier;
   - reorientation-bout rate per minute, read from the primitive labels: this is a diagnostic check,
     and graded readouts never use labels (D36);
   - final satiety;
   - total intake.

**Must-fail.** Each layer is re-run against an evaluator with one deliberate fault: the derivative scale
left at the written 50 instead of the derived 250. It must fail layer 1. A second fault, the head-sweep
sides swapped, must fail layers 1 and 2. The tests can therefore fail.

### Decision E: What the evaluator returns

Every run returns `BatchedTrajectories`: compact arrays over animals and steps.

- **Fields:** x, y, heading, concentration, the derivative feature, satiety, intake, on-lawn, primitive
  code and phase code, plus each animal's final state and the protocol record.
- **Types:** float32 for the continuous fields, int8 for the codes. That is about 45 MB for 256 animals
  over 5,400 steps.
- **`to_behaviour_steps(i)`** builds animal *i*'s `BehaviourStep` list on request, with the per-step
  runner's fields, so `lawn_states.episode_windows`, the classifier and B's readouts read it unchanged.

Fitting reads the arrays directly.

### Decision F: The cost pilot

`scripts/analysis/batched_cost_pilot.py` times, on seeds 5001–5032 repeated to reach N:

- each controller (MLP, connectome, the two hazard controllers);
- on each assay;
- at N = 32, 128 and 256;
- against the per-step runner on 8 of those seeds.

It pins torch to one thread, and OMP, MKL and OpenBLAS to one thread each, as the campaign runner does,
so its figures are per core. It also reports a figure at the machine's default thread count beside
them.

The pilot is an engineering measurement, not a scored registration, so it needs no launch record. Its
config, seeds, thread settings and code version are written into the record.

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
- **Memory.** The compact arrays are about 45 MB at 256 animals × 5,400 steps. `BehaviourStep` lists
  are built only on request.
- **The connectome's batched forward differs in last bits.** That is documented in its docstring, and
  is why layer 2's horizon is short.

## Migration

None. This is a new package. The only change to existing code is `modulator_release`'s batch axis, and
its single-sample result is pinned unchanged.
