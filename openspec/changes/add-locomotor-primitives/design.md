## Context

E.1 gave the point worm a 1-second step (`step_seconds`). The worm still moves by a free `(speed, turn)`
each step: `env/continuous_2d.py` `_kinematic_move` turns, then translates, and backs up along an
unchanged heading when `allow_reversal` is on. Nothing in the repository holds a reversal, omega or
delta turn's duration, speed or angle.

The brains' continuous actions are a tanh-squashed Gaussian in `brain/arch/_policy.py`. The body's
`action_space: body_drive` is the precedent for a second action space. It touched five places:

- the `action_space` literal and its validator in `brain/arch/dtypes.py`;
- the dimension and bounds in `_policy.py`;
- each brain's head size;
- the config loader's brain–environment cross-validator;
- the runner's dispatch in `agent/runners.py`.

What the worm senses is also limited. It has no sense of the lawn under it: lawns reach the brain only
as odour and satiety. Its proprioception is grid-only: on the continuous worm, `_proprioception_core`
reports a constant 1 and a stale cardinal direction.

The maintainer's decisions of 2026-10-23:

- a hazard interface;
- kinematics from published values;
- a food-contact sense;
- pause as a run at zero speed.

## Goals / Non-Goals

**Goals**

- One primitive layer every controller drives through the same five numbers.
- Kinematics from data, vendored with provenance and pinned by a test.
- Byte-identical defaults: `locomotion: free` draws nothing new from the RNG.
- A positive control that needs no learning, with a must-fail arm.

**Non-goals**

- **Re-extracting kinematics from tracks.** The published values stand. B.1 checks them against
  Scheer's on-food flags on held-out cohorts, and a mismatch there is a dated amendment.
- **The trajectory reorientation detector.** That is B.2. The positive control here measures each bout
  from its own trajectory, which is enough to check that executed kinematics match their source.
- **The connectome's mapping of `food_contact` and `locomotor_state` onto sensory neurons.** That is C.2's
  encoder. The connectome brain runs the primitive action space through its existing learnable readout,
  and does not read the two new modules.
- **Whether PPO learns through the primitives.** That is C.3's registration, with its own pilot.
- **Assay protocols** (E.3) and the batched evaluator (T.1).

## Decisions

### Decision A: The action space and the hazard interface

Under `action_space: primitives` a brain emits five numbers in `[-1, 1]` each second (tanh-squashed, as
every continuous action is):

| # | Output | Mapping |
|---|---|---|
| 0 | run speed | `v = v_max · (x + 1) / 2`, in [0, v_max] |
| 1 | run curvature | `ω = ω_max · x`, in rad/s |
| 2–4 | start hazard for reversal, omega, delta | `λ = exp(ln λ_min + (x + 1)/2 · (ln λ_max − ln λ_min))`, per second |

- **Speed.** The speed's lower bound is 0, not −1: backing is the reversal primitive's.
- **Log-scaled hazards.** The rates span 0.05 to 3 per minute across conditions, so the hazards are
  log-scaled, with λ_min = 10⁻⁴/s and λ_max = 1/s. A uniform action covers that span evenly. The
  untrained policy's mean is the geometric middle, 0.01/s, about 0.6/min, inside the observed range.

**Drawing a start** (competing hazards). This happens on each step while no bout is running:

- with total `Λ = Σλ_k` and step `Δt`, a bout starts with probability `1 − exp(−Λ Δt)`;
- if one starts, its kind is drawn with probability `λ_k / Λ`;
- both draws come from the environment's RNG.

A constant-hazard controller is therefore a Poisson process, exactly B.4's matched negative.

**Rejected:** a categorical choice sampled by the brain. It is correct reinforcement learning, but it
needs a hybrid action buffer, and a separate adapter for each non-learning arm.

### Decision B: The primitive layer

`Continuous2DParams.locomotion: Literal["free", "primitives"] = "free"`, from `continuous.locomotion`.
Under `primitives` the runner calls `env.move_agent_primitives(action)` in place of
`move_agent_normalized`. The layer, `env/primitives.py`, holds per-agent bout state: kind, remaining
time and per-second heading rate.

**A run** lasts one step. The worm turns by `ω Δt`, then moves `v Δt` along the new heading, through
the same clamped move as today. Pause is a run with `v ≈ 0`.

**A reversal bout**:

- **duration** `D ~ Gamma`, mean 2.69 s, SD 1.75 s;
- **speed**: backs at `v_rev = 0.14 mm/s` along an unchanged heading, the point worm's backing as
  today;
- **steps**: `D` is split into `⌈D/Δt⌉` steps, the last one partial, so distance is `v_rev · D`
  exactly;
- **heading**: unchanged (Zhao 2003: the bearing barely changes while backing). The shallow post-reversal
  turns are the run curvature that follows.

**An omega or delta bout**:

- **duration** `D ~ Gamma`;
- **angle**: `θ` is drawn from a truncated normal;
- **sense**: toward the episode's ventral side for an omega, its dorsal side for a delta;
- **heading**: rotated at `θ/D` per second, with no translation (the coiled worm's centroid barely
  moves).

**The ventral side** is `+1` or `−1`, drawn once per episode from the environment's RNG. The point
worm lies on one side, as the animal does on agar.

**During a bout** the brain is still run each step, so its state and modulators keep their time course
and the record has an action every step. Its outputs move nothing until the bout ends. A start can be
drawn on the step the bout ends.

**Walls** clamp every displacement as today.

**Kinematics** (`data/locomotion/primitives.json`; V = verified in text, F = read from a figure, about
±10%):

| Parameter | Value | Source |
|---|---|---|
| `v_max` | 0.2 mm/s (= the crawl speed E.1's step derives from) | Zhao 2003 Fig 3A 0.184 mm/s start (V); Ben Arous 2009 ~0.17 mm/s without food (F) |
| `ω_max` | 50°/s | Pierce-Shimomura 1999, the sharp-turn threshold that defines a run (V) |
| reversal speed | 0.14 mm/s | Zhao 2003 Fig 5A (V) |
| reversal duration | Gamma, mean 2.69 s, SD 1.75 s | Zhao 2003 Fig 5B: 2.3 ± 1.6 s without omega (n = 249), 3.4 ± 1.8 s with (n = 135), pooled (V) |
| omega angle | Normal(155°, 25°) truncated to [90°, 200°], ventral | Broekmans 2016 Fig 4C, foraging (F) |
| omega duration | Gamma, mean 7 s, SD 3 s | Broekmans 2016 Fig 4–suppl 1, escape omegas; foraging not reported (V) |
| delta angle | Normal(90°, 30°) truncated to [30°, 170°], dorsal | Broekmans 2016 Fig 4C (F) |
| delta duration | Gamma, mean 6 s, SD 2 s | Broekmans 2016 Fig 4–suppl 1 (V) |
| hazard range | 10⁻⁴ to 1 per second | covers Gray 2005 Fig 1E: 0.05–3 per minute across conditions (F) |

Each entry carries its source, figure, condition and whether it is verified or read from a figure.
Every source above is off food, and the file says so. B.1's check is the on-food test.

### Decision C: The two senses

**`food_contact`** (`ModuleName.FOOD_CONTACT`, one feature):

- strength = the density of the lawn cell under the worm, 0 off every lawn;
- quality is not sensed: the mechanosensory bacteria sense reads presence, not nutrition (Sawin 2000);
- carried on `BrainParams.food_contact`, set by the agent only under `food_model: lawns`;
- a brain that does not list the module is unchanged.

**`locomotor_state`** (`ModuleName.LOCOMOTOR_STATE`, four features):

- a one-hot of the running primitive: run, reversal, omega, delta;
- available only under `locomotion: primitives`, and refused elsewhere at load;
- it lets a controller sequence a turn after a reversal (the pirouette) without the sequence being
  built into the layer. It is the worm's proprioception of its own motor state.

The connectome brain reads neither module. Its sensory mapping is fixed by `sensing_mode`, and how new
senses enter it is C.2's encoder question. The MLP reads both.

### Decision D: Cross-validation and scope

At load:

- **brain and environment must agree.** `action_space: primitives` and `locomotion: primitives` come
  together or not at all, the body-drive pattern. Only `connectomeppo` and `mlpppo` implement it.
- **`primitives` requires:**
  - `action_mode: continuous`;
  - `signed_speed: false` and `allow_reversal: false`, since backing is the reversal primitive;
  - `continuous_std_mode: state_independent`, because the state-dependent head is sized for two
    outputs;
  - the point body;
  - an environment step of at most 1 s, since bouts last 1–15 s and need second resolution. On D.1's
    5-s cell this means `step_seconds: 1.0`.
- **`locomotor_state`** requires `locomotion: primitives`.
- **`food_contact`** requires `food_model: lawns`.

`utils/step_duration.py` marks the new fields as not step-coupled in its field walk. The kinematics
are in seconds, and `max_step_mm` and `max_turn_rad` are unread under primitives.

### Decision E: Capture

`BehaviourStep.primitive: str | None` records `run`, `reversal`, `omega` or `delta` under primitives,
and is absent otherwise (the optional-field pattern). It is for diagnosis. B's readouts never read it
(D36). A test in B.2 will assert that.

### Decision F: The non-learning controllers

`quantumnematode/env/hazard_controllers.py` holds two controllers. Each maps a step's sensed state to
the five raw outputs in `[-1, 1]`, so it drives the layer through the same mapping as a brain.

- **`ConstantHazards(v, rates)`**: a fixed run speed, zero curvature and fixed rates. This is the
  Poisson-timed negative.
- **`PirouetteHazards(v, λ_0, β, rates)`**: the Pierce-Shimomura klinokinesis rule.
  - The reversal and turn hazards are `λ_0 · exp(−β · dC/dt)`, read from the same STAM derivative the
    brains see. They rise as the concentration falls.
  - Runs are straight at `v`.

They run through a small driver loop. It reads the agent's sensing and calls
`env.move_agent_primitives`, with no brain.

## Positive control (protocol principle 4)

The new component is the primitive layer. It needs no learning to be checked.

1. **Kinematics.** Force each primitive with a constant hazard of λ_max for that kind only. Over 2,000
   bouts per kind:

   - each bout's duration, displacement and heading change are measured from the trajectory alone;
   - their distributions match the vendored ones (two-sample KS against draws from the source
     distributions, p > 0.01);
   - omega and delta turn to opposite sides within an episode;
   - the labels are not read.

2. **Expressiveness, with a must-fail arm.**

   - **Setup.** On one lawn in a 20 mm arena, worms start 6 mm off the lawn edge, on seeds 5994–5999.
   - **Arm.** `PirouetteHazards` must reach the lawn within 20 minutes more often than `ConstantHazards`
     at the same mean hazard and speed, measured on the pirouette arm's own runs. The difference must
     be bootstrap-significant at 95% over 6 × 32 episodes.
   - **Must-fail.** `ConstantHazards` alone must reach the lawn no more often than with its hazards set
     to zero. Timing without sensing is not chemotaxis.

   If the pirouette arm fails, the layer cannot express klinokinesis, and E.2 does not close.

3. **Byte identity.** A `locomotion: free` run is identical to before the option, RNG draws included.

## Risks / Trade-offs

- **Off-food kinematics on food.** Every source is off food, and on-food reversals may be shorter and
  slower. B.1's check reads Scheer's on-food flags on cohorts held out of the benchmark. If they
  disagree beyond the sources' spread, the kinematics are amended, dated, before any fit.
- **Figure-read angles (±10%).** B grades turn-angle distributions only through reorientations
  detected from trajectories, against real tracks read by the same detector. A systematic angle error
  shows as every controller failing that statistic, which is diagnosable.
- **The brain runs during bouts.** PPO stores actions that moved nothing, which adds credit noise.
  That is C.3's concern. The alternative, skipping the brain during bouts, would stop its clock, and
  the modulators' time course with it.
- **Log-scaled hazards bias the untrained policy** to about 0.6 reorientations of each kind per
  minute. That is within the real range, and it is the same for every arm.
- **Omega translation set to zero.** The coiled centroid moves little, but not zero. A trajectory
  detector reading displacement at turns may need B.2's tolerance.

## Migration

None. Every committed config keeps `locomotion: free` and `action_space: speed_turn`.
