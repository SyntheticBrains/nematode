## Context

E.1 gave the point worm a 1-second step, and E.2 gave it locomotor primitives. Every lawn run is still
D.1's cell:

- `LawnField.place` scatters `count` lawns at random in a square arena (`env/lawns.py`);
- the worm starts at the arena centre, on a lawn when `start_on_lawn` centres the first lawn there;
- the episode ends at `max_steps` or by starvation (`agent/runners.py`, `_handle_starvation_check`).

Satiety decays linearly at `satiety_decay_rate` per step. On the D.1 cell at 1 s that is 0.16/s from
300, so a worm off food dies in about 31 minutes. Walls clamp each axis of a square arena
(`Continuous2DEnvironment._turn_then_move`).

The maintainer's decisions of 2026-10-24:

- satiety is one slow reserve;
- each assay uses its reference's arena;
- the food history is simulated;
- lawn density is uniform.

## Goals / Non-Goals

**Goals**

- Each reference's assay reproduced as a protocol: arena, lawn, start, food history, duration and
  analysis window.
- Captures carry each step's phase, and the file carries the protocol's timing, so B can bin from the
  reference's time zero.
- A satiety that behaves like an energy reserve over hours, and never ends an assay.
- Byte-identical defaults.

**Non-goals**

- **The readouts.** Reorientation rates, roaming fractions and leaving rates belong to B (B.2–B.3),
  registered on these protocols.
- **The lawn edge's density, multiple worms per plate, and lawn growth during an assay.** Each is a
  later amendment if B needs it.
- **The time zero of López-Cruz's bins.** The protocol's time zero is the removal; B checks the
  recording offset against the deposited data.
- **A metabolic cost of locomotion.** The assay configs set it to zero. No published rate is in hand,
  and D.1's value was chosen to make roaming starve the worm within an hour.

## Decisions

### Decision A: The protocol is a list of phases

`EnvironmentConfig.assay: AssayConfig | None = None`:

```yaml
assay:
  name: food_removal_lopezcruz
  phases:
    - {name: fed,   duration_s: 2700, food: lawns, place: lawn_centre}
    - {name: assay, duration_s: 2700, food: none,  place: arena_centre}
  assay_phase: assay          # the phase the reference records
  analysis_from_s: 0          # within it; Scheer's is 1200, Calhoun's 120
```

**Food.**

- `lawns`: the configured lawns, freshly placed whenever a phase turns them on.
- `none`: a bare plate, with no lawn field, no odour and no intake.

**Placement** at a phase's start:

| `place` | Effect |
|---|---|
| `keep` | the worm stays where it is |
| `lawn_centre` | the worm goes to the first lawn's centre |
| `off_lawn` | uniform over bare agar, at least `start_clearance_mm` from every lawn edge and 1 mm from the wall |
| `arena_centre` | the worm goes to the arena centre, as a transfer to a fresh plate |

Every placement but `keep` draws a uniform heading.

**State across phases.** The worm's satiety, its controller (brain state, modulators, STAM buffer) and
its primitive state carry across phases unchanged, as the animal's do across a transfer.

**Timing.** The phases' durations sum to the episode's length. The config's `max_steps` times its
**reference step** must equal that sum, checked at load. The reference step is
`step_worm_seconds(max_step_mm)`, the step its per-step values are written for. That is 5 s on the
written config and the configured step on the derived one, so the same check holds before and after
E.1's derivation. On López-Cruz's config, 1,080 written steps × 5 s and 5,400 derived steps × 1 s
both make 5,400 s.

The runner's step loop starts phase `i` at the first step whose start time is `Σ_{j<i} duration_j`.
A phase begins **before that step's sensing**, so the step's observation and capture already see the
new food and the moved worm.

`lawn_centre` and `off_lawn` require the phase's `food: lawns`.

**The transfer's odour step.** At a removal the odour falls to zero, while the STAM buffer still holds
the lawn's concentrations. For one step the worm senses a large negative change. Real worms sense
their removal too, and López-Cruz drops the first 2 minutes as affected by the transfer, so the step
is kept and the buffer carries over. B's bins start after it.

Rejected: a fixed set of named protocols in code. The phases are four lines of config each, and B may
need a variant (a shorter fed phase, a second transfer) without new code.

### Decision B: Arenas and lawn placement

- **`continuous.arena_shape: Literal["square", "circle"] = "square"`.** A circle has diameter
  `world_size_mm` and its centre at the square's centre. A move ending outside it is projected
  radially onto the wall: the worm slides along the wall, as it does along a square one. Sensing
  positions are clamped the same way.
- **`lawns.placement: Literal["random", "centre"] = "random"`.**
  - `centre` puts a single lawn at the arena centre, and is refused with `count` above 1.
  - `random` is refused on a circular arena: the square placement could put a lawn outside the
    circle.
- **Assays require** `food_model: lawns`, the continuous substrate, one worm, `start_on_lawn: false`,
  and no `manyworlds_mode`. The protocol's `place` decides the start instead. The phase clock lives in
  the standard episode runner only.
- **Head-sweep sample points** are clamped into the circle as well as the square.

### Decision C: Satiety as a slow reserve

`SatietyConfig.starvation_ends_episode: bool = True`. False keeps satiety clamped at zero, and the
episode runs to its end. Assays require false.

The assay configs set the reserve to empty over 12 h of basal decay:

- `satiety_decay_rate` = 300 / (43,200 s / 5 s) = 0.0347 per reference step, 0.00694/s at 1 s;
- `initial_satiety` = 300;
- `movement_cost_per_mm` = 0.

Off food the reserve falls linearly: 87.5% after 90 minutes, half at 6 h, zero at 12 h.

**Lawn depletion over an assay.** At D.1's intake (2% of a cell per 5-s step) a stationary worm halves
its 1-mm cell in about 170 s. That is shorter than the real dwelling bout (τ = 482 s), so depletion,
not state, would end dwelling. Real OP50 lawns are not visibly grazed by one worm in an hour. D.1 chose
its rate so that worms would leave its patch cell.

The assay configs therefore set:

- a per-cell half-life of about 2 h under a stationary worm: a per-second intake of 1 × 10⁻⁴, written
  as `intake_fraction: 0.0005` for the 5-s reference step;
- `satiety_per_intake: 3.0`, so a fresh cell still restores about 0.09/s, more than ten times the
  decay.

Depletion stays in the model at this slower rate. This was the maintainer's decision of 2026-10-24.
The half-life is chosen without a direct measurement, grounded only in lawns surviving an hour. Reward
per second of eating falls with it, and C.3 sets its own reward. The
references say:

- 30 min off food changes the response to refeeding;
- more than 4 h changes pumping;
- 12 h is the fasted state.

A fitted controller reads the reserve through `internal_state`. The 10-minute local-search memory is
its own to carry.

### Decision D: Capture and the record

- **`BehaviourStep.phase: str | None`** records each step's phase name under an assay, and is absent
  otherwise.
- **`behaviour_capture.json`** gains a top-level `"assay"` under an assay, with the protocol's name,
  each phase's name, start step and duration, the assay phase, and the analysis window in steps. A
  default capture is unchanged.
- **Satiety and on-lawn status** are recorded on bare-plate phases too: satiety as usual, `on_lawn: false`. Today a missing lawn field drops both.

### Decision E: The configs

`scripts/campaigns/generate_assay_configs.py` writes each protocol on E.2's 1-s primitives worm, for
the MLP with food contact and locomotor state, and for the connectome with modulators:

| Protocol | Arena | Lawn | Phases | Analysis |
|---|---|---|---|---|
| `scheer_lawn` | circle, 10 mm | 1.5 mm radius, centred | `assay` 3600 s, `off_lawn` | from 1200 s |
| `food_removal_lopezcruz` | circle, 100 mm | 25 mm radius, centred | `fed` 2700 s at `lawn_centre`; `assay` 2700 s, `none`, `arena_centre` | from 0 |
| `food_removal_calhoun` | square, 25.4 mm | 11 mm radius, centred | `fed` 2700 s at `lawn_centre`; `assay` 1800 s, `none`, `arena_centre` | from 120 s |
| `uniform_lawn_flavell` | circle, 35 mm | 16 mm radius, centred | `assay` 5400 s at `lawn_centre` | from 0 |

- **Scheer's start** keeps `start_clearance_mm` at 0.5. The well is 10 mm across and the lawn 3 mm,
  so bare agar is a ring 3.5 mm wide.
- **López-Cruz's pre-assay lawn** covered its 5 × 5 cm barrier region. A 25 mm-radius disc stands for
  it, at 1,963 cells.
- **Calhoun's worms fed overnight.** The 45-min fed phase approximates that, and is stated in the
  config's header.
- **Flavell's plate was 600 cm².** The 35 mm plate is his exploration assay's, so the lawn's cell
  count stays near 800.

Every other setting is the D.1 lawn cell's: intake, quality, the odour field and the rewards. The
satiety and movement cost are set per Decision C.

## Checks (protocol principle 4)

The new component is the protocol layer. Its checks need no learning:

1. **Phases.**
   - At a removal, the lawn field is gone: the odour and intake are zero everywhere.
   - The worm is moved, and its heading is redrawn.
   - Its satiety, brain and primitive state are the ones it had a step before.
   - The phase labels change at the configured step.
2. **Placement.**
   - `off_lawn` starts lie on bare agar, at least the clearance off the lawn and inside the well.
   - `lawn_centre` and `arena_centre` land where named.
3. **The circular wall.** No position leaves the circle over 10,000 random moves, and a worm driven
   outward slides along the wall.
4. **The reserve.**
   - On the satiety manager directly: off food, satiety is linear, 87.5% at 5,400 s and 0 at
     43,200 s (±1 step). A worm moved to a fresh cell each step stays at its maximum.
   - Through the runner: a reserve set near empty reaches zero and the episode runs to `max_steps`.
   - On an assay lawn, a stationary worm's cell keeps at least 40% of its density after 90 minutes.
5. **Timing.**
   - A config whose `max_steps` and phase sum disagree is refused.
   - The capture's phase starts and analysis window match the protocol.
6. **Byte identity.** Every existing config runs as before. Free locomotion and D.1's cells are
   untouched.
7. **Smokes.** Each of the eight generated configs runs one episode on seeds 5991–5993, captures, and
   classifies the Scheer and Flavell captures.

## Risks / Trade-offs

- **Arena size against cost.** López-Cruz's 100 mm arena makes the integer grid 100 × 100, and its fed
  lawn 1,963 cells. The odour sum costs O(cells) per step, about 3× the D.1 cell's. T.1 measures it.
- **Approximated food histories.** Calhoun's history is approximated, and López-Cruz's thin lawn is
  modelled as density 1. Both are stated. B's positive controls on real tracks are unaffected, and any
  mismatch shows as every controller failing the same statistic.
- **No locomotion cost.** A reward-trained controller has no energetic reason to slow down. That is
  C.3's to set; fitted controllers do not read reward.
- **The depletion rate is a choice.** It is grounded in lawns outlasting an assay, not in a measured
  grazing rate. If B's dwelling-duration or leaving-with-depletion statistics fail every controller,
  this is the first parameter to revisit.
- **Satiety's linear decay.** The reserve falls linearly, not with the animal's dynamics. It is a slow
  state the controller can read, not a model of metabolism.

## Migration

None. Every existing config has no `assay`, a square arena, random placement and starvation on.
