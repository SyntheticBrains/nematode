## Overview

C.0 is the body's prerequisites: a step constant, reversal, a frozen substrate and D19 recorded. The
maintainer chose four settings before this change was written:

- Emmons 2024 as 8b's substrate;
- D19 as a decision record now, with the generator built in C.1c;
- symmetric reversal;
- validation by code tests plus a short gate-only pilot.

## Decisions

### Decision A: The step constant (C.0a)

`env/worm_time.py` holds `CRAWL_SPEED_MM_PER_S = 0.2` and `UNDULATION_PERIOD_S = 1.6`, the roadmap's
kinematic figures, with Chung & Kim 2025/2026 as the modelled reference and C.3 as the place they are
validated. It also holds `step_worm_seconds(max_step_mm) = max_step_mm / CRAWL_SPEED_MM_PER_S`. A
full-speed step covers `max_step_mm`, so its duration in worm time is that distance at the crawl speed.
At block V's `max_step_mm: 1.0`, about one body length, it is **5.0 worm-seconds**, or 3.1 undulation
periods; D22 rounds this to three periods integrated inside the step.

The constant inherits the crawl speed's uncertainty: 0.15–0.3 mm/s gives 3.3–6.7 s. It is recorded, not
used: C.1c's generator consumes it, and a test pins it on both block-V cells' configs.

### Decision B: Signed speed (C.0b)

**Environment.** `continuous.allow_reversal: bool = False`.

- When true, `_kinematic_move` clamps speed to `[-max_step_mm, max_step_mm]` and moves the worm along its
  heading by the signed speed.
- The turn still applies first, as today. The heading is the head's direction and does not flip on
  reversal, so backing up keeps the head where it was.
- `move_agent_normalized` accepts `speed_norm` in `[-1, 1]` when reversal is on.

**Scope.** Reversal is single-agent: continuous motion is driven only by the single-agent runner, and both of the environment's motion entry points pass through the one clamp.

**Sensing under reversal follows from the existing code.**

- The lateral head-sweep sample is taken across the heading, so it stays attached to the head.
- The rate-of-change feature reads the actual displacement, so backing down a gradient reads as a fall.
- Contact zones are taken against the heading, so a predator behind a reversing worm is still posterior.

Tests pin all three.

**Brains.** The shared continuous-brain config gains `signed_speed: bool = False`, refused under discrete
actions. `_policy.continuous_action_bounds(signed_speed)` returns `([-1, -1], [1, 1])` or today's
`([0, -1], [1, 1])`. The five continuous brains — connectome, MLP, LSTM, CfC and transformer PPO — read
their bounds from it instead of hard-coding them.

Under signed speed the tanh-squashed policy's centre moves from speed 0.5 to 0. An untrained policy
therefore starts near stationary on average, not at half speed. That is a real change in initial
behaviour, and the pilot measures whether learning survives it.

**Agreement.** Simulation-config validation refuses a continuous brain whose `signed_speed` differs from
the environment's `allow_reversal`, in either direction:

- a signed brain in a clamping environment would stop wherever it meant to reverse;
- a reversing environment with an unsigned brain is reversal nobody can use.

**Behaviour capture.** When reversal is on, each captured step records `speed_signed`, the step's signed
displacement along the heading in mm. The field is absent when reversal is off, so existing capture files
are byte-identical. It is what bout statistics need (B.2c, C.3).

**The anatomical readout.** It maps forward minus backward motor-class drive to speed. Under signed speed
that contrast reaches reversal rather than stopping at zero. Only the plastic-rule arm uses it; it is
noted, not changed.

### Decision C: Emmons 2024 as 8b's substrate

`connectome_source: Literal["cook_2019_hermaphrodite", "emmons_2024_hermaphrodite"]`, with Cook the
default. The brain loads the chosen source; the wild type a measured prior is defined on is the same
source.

Emmons 2024 has the same 302 neurons, the same 3,709 chemical synapses and the same 956 neuromuscular
synapses. Four gap pairs differ: ALML–BDUL and ALMR–BDUR are new, and BDUL–PLML and BDUR–PLMR are
stronger (23 to 37 sections). A test pins that the built chemical mask is identical and that the gap
buffer differs only in those entries.

**From C.1, 8b's configs use Emmons 2024.** 8a's results stay on Cook 2019, and so does the optional M.8
detour, which reads an 8a question. The per-connection sign table stays Cook-keyed; no brain reads it, and
it is identical on Emmons (tested since H.2).

### Decision D: D19 recorded as decided (C.0d)

The body carries the rhythm; the brain sets segmental drive. The generator has four parts:

- **Phase** comes from a head relaxation switch, after Ji et al. 2021 (*eLife*, fitted to phase-response
  data; placed in SMDD by Yeon et al. 2018). Head bending reverses when curvature crosses a threshold,
  and relaxes between switches.
- **Forward propagation** is a front-to-back relay, after Wen et al. 2012. Each segment's bending follows
  the curvature of the body in front of it, so the wave travels tailward.
- **Backward propagation** is the mirrored tail-to-head relay. Its source is a hypothesis (A-type motor
  neurons, Gao et al. 2018), and Wormlight found the backward mode the hard part.
- **The interface.** The brain's drive through the neuromuscular map sets each segment's dorsal–ventral
  amplitude and bias, which gives speed and steering. The sign of its net forward-versus-backward drive
  selects the wave's direction, which is C.0b's reversal carried into the body. The generator integrates
  about three undulation periods inside each step (Decision A) and exposes curvature and phase, the
  interface that replaced C.0c (D22).

**Parameter sources.** The period (~1.6 s) and the crawl wavelength come from the kinematic references
C.3 adopts (Fang-Yen et al. 2010). The switch threshold and relay delay are fitted in Ji et al. 2021 and
taken from there. Any parameter left free is calibrated **once on C.1d's MLP positive control and frozen
across arms**, with a registered sensitivity check, as D18 requires of muscle gains.

**What this licenses.** A wiring contrast through this generator is a contrast of segmental drive and
steering, not of rhythm generation (D19 as amended). This change records the decision in D19 and the
tracker; C.1c builds and tests the generator.

### Decision E: Validation, a gate-only pilot

On hard350 with reversal on (`allow_reversal` and `signed_speed`), 3,000 episodes, **seeds 1301–1308**,
disjoint from every band and earlier pilot:

| learner | config parent | substrate |
|---|---|---|
| MLP-PPO, learning and frozen | width-64 hard350 MLP-PPO | none |
| connectome PPO, settling wild type, learning and frozen | block V's hard350 arm | Emmons 2024 |

**The gate.** Each learner's plateau beats its frozen floor: the 80% interval of the paired difference
lies above zero, the same test the panels use.

- **Both pass:** C.0b validates, and the substrate freezes.
- **A learner fails:** the pilot is the diagnosis. Its plateau, floor and how often it reverses go to the
  maintainer before C.1 registers.

**Reported beside, never read:** each learner's plateau against its committed reversal-off runs on
the same cell. *(Amended before the pilot ran: the fraction of steps with negative speed was to be
reported too, but it needs behaviour capture on every step of 3,000-episode runs, about 150 MB a run.
Reversal use is read where bout statistics are, through the body at C.1d with C.3's instruments.)*

The connectome comparison changes two things at once, reversal and substrate, so it is description only.

32 runs. The cost is estimated from the hard350 runs these arms share: about 17 minutes per connectome
learning run and 11 per frozen run at 16 workers, and less for MLP-PPO. That comes to under an hour.

## Risks

- **Reversal may be used as a second forward gear**, moving backward toward food with the head turned
  away. The worm's sensing is head-based, so this is self-limiting; it is read through the body at C.1d.
- **A centred initial policy may learn slower.** The pilot reads only whether each learner still learns;
  any slowdown is reported, and C.1 registers its own floors and baselines anyway.
