## Why

Phase 10's fits and its wiring reading evaluate frozen controllers on many simulated animals:

- C.2 fits the connectome by sep-CMA, at 200 or more animals per evaluation;
- W repeats that fit for every null arm and every rewiring draw;
- B grades every controller and every negative on 256 animals per assay.

Today one animal runs through the per-step runner at about 0.5 ms a step. That was measured on
2026-10-24: 2.65 s for a 5,400-step Flavell episode on the MLP, 3.7 s on the connectome, both under a
profiler. So one evaluation of 200 animals on a 90-minute assay takes about 9 minutes on one core, and
a fit of thousands of evaluations is out of reach.

The time goes to per-animal Python, not to the brains:

- each step evaluates the lawn's odour six times;
- it builds a `BrainParams` dataclass;
- it steps the STAM buffer through list comprehensions;
- the brain's forward pass is about a third of the connectome's step.

This is tracker item T.1: many animals per call, no per-step Python per animal, carrying E.2's
primitives and E.3's assays, with a parity test against the per-step runner and a cost pilot on
seeds 5001–5032 that sizes W.

## What Changes

The maintainer's decisions of 2026-10-24 are marked (decided).

- **`BatchedEvaluator`** (`quantumnematode/evaluation/`) runs N animals of one config in lockstep. It
  holds every per-animal state as arrays:

  - position, heading and primitive bout;
  - lawn density, satiety and intake;
  - STAM buffer, adaptive baseline and previous position;
  - the connectome's modulator concentrations.

  Each step senses, acts, moves, eats and decays satiety for all N at once, in the per-step runner's
  order.

- **Frozen evaluation only** (decided). It runs fixed controllers:

  - the fitted connectome and MLP, through their batched forward passes;
  - the non-learning hazard controllers, vectorised.

  Reward training keeps the per-step runner.

- **A declared surface.** It covers the 1-s point worm under locomotor primitives, on the lawn food
  model with centred lawns or an assay protocol, with the klinotaxis, STAM and fold-change sensing the
  E.2 and E.3 configs use. Anything else is refused at construction, naming the setting.

- **Per-animal random streams, policy noise sampled** (decided). Animal *i* with seed `s_i` gets:

  - a numpy generator `default_rng(s_i)` for its environment;
  - a torch generator seeded `s_i` for its five policy-noise values a step.

  These are the streams the per-step runner gives an episode with that run seed. Results are therefore
  reproducible per seed and independent of batch size. Measured cost: about 0.5 µs per animal-step at
  N = 256. A mean-action mode is an option.

- **Layered parity** (decided):

  1. given the same state, features and brain outputs match the per-step runner within 1e-6;
  2. with matched streams, trajectories match exactly over the first 300 steps;
  3. over full assays, graded statistics match the per-step runner's on the same seeds, within their
     paired bootstrap intervals.

  A deliberately broken evaluator must fail each layer.

- **A speed target of at least 20× per core** (decided): at most 25 µs per animal-step at N = 256.

- **A cost pilot** on seeds 5001–5032. It measures µs per animal-step at N = 32, 128 and 256 for each
  controller on each assay, and the projected cost of one 200-animal evaluation and of W's fits. It is
  recorded in Logbook 104. If the target is missed, the pilot records it; it does not block the merge.

- **Captures** in the same shape as the per-step runner's `BehaviourStep` series, so the classifier and
  every B readout read batched animals unchanged.

## Capabilities

### New Capabilities

- `batched-evaluation`: the evaluator, its declared surface, per-animal streams, parity and capture.

## Impact

- **Code**:
  - new `quantumnematode/evaluation/` (`batched.py`, `controllers.py`, `__init__.py`);
  - a batch-safe modulator release in `brain/arch/connectome_ppo.py` (`modulator_release` averages over
    the whole batch on a `(B, N)` input today);
  - `scripts/analysis/batched_cost_pilot.py`.
- **Default behaviour**: none. The per-step runner, the brains' single-sample paths and every config
  are unchanged.
- **Cost**: about 8–12 engineering days, the tracker's estimate. The pilot runs in about an hour.
- **Seeds**: parity and pilot on 5001–5032, the band reserved for them.
- **Downstream**:
  - B grades on it;
  - F.0 and C.2 fit with it;
  - W's draws per arm are sized from the pilot.
