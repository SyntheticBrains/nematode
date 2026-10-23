## Why

Phase 10's benchmark grades simulated worms against real-worm data on standard on-lawn assays (D34).
A grade means something only if the simulated worm runs the assay its reference ran: the same arena,
lawn, start, food history, duration and analysis window. Today every lawn run is D.1's cell:

- four random lawns in a 20 mm square;
- the worm starting on a lawn;
- one hour;
- a satiety that empties in about 31 minutes off food at the 1-s step, and ends the episode when it
  does.

None of the references runs that assay. The protocol sweep of 2026-10-24 established what each one does:

| Reference | Arena | Lawn | Start | Food history | Duration |
|---|---|---|---|---|---|
| Scheer & Bargmann 2023 | 10 mm round well | one 3 mm spot | on bare agar; the worm finds the lawn | fed | 60 min, the last 40 analysed |
| López-Cruz et al. 2019 | about 80 cm² round barrier | none | picked off food | 45 min on a thin uniform lawn | 45 min after removal |
| Calhoun et al. 2014 | 25.4 mm square | none | picked off food | overnight on a patch | 30 min, the first 2 dropped |
| Flavell et al. 2013 | large uniformly seeded plate | uniform | on food | continuously fed | 90 min |

The sweep also corrected three premises:

1. **Scheer's worms are not placed on the lawn.** They start on bare agar and find it, so the
   tracker's "off-food start" is part of Scheer's protocol.
2. **López-Cruz 2019 reports no exponential fit.** Its deposited data (CC BY 4.0) give
   γ = 0.115 ± 0.011/min only when fitted from the 4–6 min bin. The curve rises before it decays, so
   the fit's start and the time zero of removal are B's to register.
3. **Hunger and the food memory run on different clocks.** Local search fades within about 10
   minutes. Pumping and quiescence change over hours off food, with 12 h the standard fasted state. One
   satiety that empties in 30 minutes fits neither.

This is tracker item E.3.

## What Changes

The maintainer's decisions of 2026-10-24 are marked (decided).

- **An assay protocol** (`environment.assay`): a list of phases. Each phase gives a name, a duration in
  worm-seconds, its food (a lawn layout, or a bare plate), and where the worm is put at its start:

  - kept where it is;
  - on the lawn's centre;
  - at a uniformly drawn point on bare agar;
  - at the arena centre, as a transfer.

  Every relocation draws a new heading. The worm's satiety and its controller's state carry across
  phases, as the animal's do across a transfer. One phase is marked as the assay phase, with an
  analysis window inside it.

- **Per-reference arenas** (decided): a circular arena option (`continuous.arena_shape: circle`), whose
  wall clamps the worm radially. A lawn can also be placed at the arena centre (`lawns.placement: centre`).

- **The food history is simulated** (decided). López-Cruz's 45 minutes on a uniform lawn is a first
  phase, then a transfer to the bare plate. Local search is then a memory the controller has to carry.

- **Satiety as one slow energy reserve** (decided).

  - It empties over 12 h off food, the standard fasted state.
  - Assays turn off death by starvation (`satiety.starvation_ends_episode: false`).
  - The roughly 10-minute food memory is the controller's own; no sense hands it a timer.
  - In assay configs locomotion has no metabolic cost. No published rate is in hand.

- **Uniform lawn density** (decided). Scheer's denser lawn edge is not modelled; B revisits it if its
  edge statistics fail every controller.

- **Capture records the phase.** Each step gets its phase. The file gets the protocol's phases, start
  steps and analysis window, only when an assay is configured.

- **Generated assay configs** on the 1-s primitives worm, for the MLP and the connectome:

  - `scheer_lawn`: the 10 mm well, a 3 mm lawn, 60 min, analysed from 20 min;
  - `food_removal_lopezcruz`: a 100 mm circle, 45 min fed, then 45 min bare;
  - `food_removal_calhoun`: a 25.4 mm square, 45 min fed, then 30 min bare, analysed from 2 min.
    Calhoun's worms fed overnight; the 45-min fed phase is the stated approximation;
  - `uniform_lawn_flavell`: a 35 mm circle uniformly covered, 90 min.

## Capabilities

### New Capabilities

- `assay-protocols`: phases, placements, the circular arena, the slow reserve and the assay configs.

### Modified Capabilities

- `realworm-behavioural-validation`: capture records the phase and the protocol.

## Impact

- **Code**:
  - `utils/config_loader.py`: `AssayConfig`, the arena shape, lawn placement, the satiety flag and
    validation;
  - `env/continuous_2d.py` and `env/env.py`: the circular wall, phase transitions and placement;
  - `env/lawns.py`: centred placement;
  - `agent/runners.py`: the phase clock;
  - `agent/satiety.py`: optional starvation termination;
  - `agent/agent.py` and `report/`: capture of the phase and protocol;
  - `utils/step_duration.py`: the new fields classified;
  - `scripts/campaigns/generate_assay_configs.py`.
- **Default behaviour**: none. With no `assay` block, a square arena, random placement and the
  starvation flag at its default, runs are byte-identical.
- **Cost**: about 3–5 engineering days. No campaign.
- **Seeds**: smokes on 5991–5993.
- **Downstream**:
  - B registers each assay's readout on these protocols;
  - T.1 batches them;
  - C.3's reward-trained comparator decides its own reward on them.
