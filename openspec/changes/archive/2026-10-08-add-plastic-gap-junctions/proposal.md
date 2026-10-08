## Why

Logbooks 075 and 078 found that most of block V's lead over the degree-preserving null came from
that null's rewired gap junctions: about half on hard350 and about 84% on the thermal cell at target 35.
Neither panel could say why, because a degree-preserving rewiring moves two things at once:

- which neurons are coupled (**placement**);
- how strongly, since EM counts travel with their edges and each neuron's total gap strength changes
  (**strength**).

B.2b was to answer this with plastic gap junctions. It closed *unreachable-with-reason* when the leaky
substrate it was registered on failed its positive control (Logbook 080). The question does not need
that substrate. The maintainer chose to run it as the optional M.8, alongside C.1's build, on the
settling substrate and the Cook 2019 connectome, since it reads an 8a question.

## What Changes

- **A gap-only null.** `wiring: rewired_gap_junctions_only` holds the chemical graph at the wild
  type's and rewires only the gap junctions by the degree-preserving swap, counts travelling with their
  edges. Against the wild type it differs in gap placement and per-neuron gap strength, and nothing
  else.
- **Plastic gap junctions.** `plastic_gaps: true` gives every existing gap pair a learnable positive
  multiplier on its strength, starting at 1 and symmetric by construction, which PPO learns with the
  rest of the brain. No pair can be created. Off by default and byte-identical when off. It is refused
  under leaky dynamics, which precomputes its operator from fixed gaps.
- **A panel on the thermal cell at target 35.** The wild type and the gap-only null, each learning
  with fixed gaps, learning with plastic gaps, and frozen, on seeds 513–576. The wild type's
  fixed-gap runs are reused from Logbook 078, licensed by an identity check. Three readings,
  corrected together:
  - the lead with fixed gaps, as the base;
  - the lead with plastic gaps;
  - their interaction.
    They read whether the wild type's gap advantage survives once each wiring can tune its own
    strengths.
- **The tracker** records the maintainer's decisions: M.8 runs alongside C.1, and M.7 lands during C.1,
  before C.3.

## Capabilities

**Modified**: `connectome-ppo-brain`, with two added requirements:

- a gap-only rewired null;
- plastic gap junctions under PPO.

## Impact

- `connectome/rewiring.py` (`rewire_chemical`); `brain/arch/connectome_ppo.py` (the wiring value and
  `plastic_gaps`)
- `configs/scenarios/thermal_foraging/`: four new arms through a generator
- `scripts/analysis/plastic_gaps.py`: new
- Tests: the null's guarantees, plasticity's symmetry, positivity and gradient, the refusal under
  leaky, byte-identical off, the configs, the readings
- After the readout: Logbook 082, tracker M.8

## Breaking Changes

None. Both new options default to today's behaviour.
