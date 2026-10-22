## Why

The phase's STOP condition was met on 2026-10-20. On log-compressed gap counts, the wild type shows no
wiring effect over its gap-only null on the point worm or through the body:

- [Logbook 099](../../../docs/experiments/logbooks/099-logcount-contrast.md): +0.003 [−0.009, +0.016];
- [Logbook 100](../../../docs/experiments/logbooks/100-body-logcount.md): −0.022 [−0.045, +0.005].

The registered consequence is that the phase after 9 inherits a modelling question: *how should
electrical synapses enter a learned connectome model?* On 2026-10-20 the maintainer chose to **open that
question inside Phase 9**, before the STOP statement ships.

Both corrections read so far shrink the weights: unit weights in G.0, and log counts in 099 and 100.
Neither fixes what caused the artefact. The model divides each pair's weight by `√(d_i d_j)`, where
`d` is the binary gap degree. That normaliser counts partners, not synapses, so a high-count pair
between two low-degree neurons couples far above 1. ALA–CAN reaches about 90 under counts.

Two principled models remain unread:

- **Strength-normalised coupling** divides each pair by `√(s_i s_j)`, where `s` is each neuron's total
  gap weight. Every coupling is then at most 1, and the gap matrix's eigenvalues lie in [−1, 1]. Each
  neuron's partners keep the relative weights the counts give them. It is the bound the code's own
  comment once wrongly claimed the degree normaliser provided.
- **Ohmic coupling at ×0.03** is the most physical model: current is proportional to the potential
  difference. It was built and calibrated in G.0-A, and it learned more slowly than log counts. Its
  wiring gap was barred from reading there ([Logbook 097](../../../docs/experiments/logbooks/097-gap-substrate.md)).

The maintainer chose these two, and the point worm first. A model goes through the body only if it reads
`placement` there.

## What Changes

- **Brain option** (`connectome-ppo-brain`) `gap_normalisation`:

  - `degree`, the default, is byte-identical to today;
  - `strength` divides each gap weight by `√(s_i s_j)`, with `s_i` the sum of neuron `i`'s gap weights
    after `gap_weight_mode`.

  It is computed on each wiring's own matrix, as the degree normaliser is. It is refused with gap
  junctions off and with `plastic_gaps`.

  Tests:

  - every coupling is at most 1;
  - the spectral radius is 1;
  - `unit` weights under `strength` equal `unit` under `degree`;
  - the default is byte-identical;
  - each refusal.

- **The operating-point probe** (097's instrument, before any run): the strength model's untrained
  action spread and saturated fraction against the gap-off brain. It must keep at least half the gap-off
  spread, as the ohmic scale had to. If it does not, the step is a dated decision, never a calibration
  inside the pilot.

- **Configs**: the strength model's wild type and gap-only null, learning and frozen, on the thermal cell
  at target 35 on Emmons with count weights. The ohmic model reuses G.0-A's four `*_g0a_ohmic_*_t35`
  configs unchanged.

- **Gate evidence**:

  - for ohmic, G.0-A's pilot arms on 9871–9886 (wild type 61.1%, null 54.1%, floors 0%);
  - for strength, a pilot on 9905–9912, configured as the panel;
  - smokes on 9913–9914.

- **The panel**: both models' contrasts on seeds **4201–4264**, each wiring learning and frozen, 3,000
  episodes, **512 runs**.

- **Two readings**, `gap(strength)` and `gap(ohmic)` on `auc_success`, wild type − null, paired by
  seed. They are BH-FDR corrected together as a family of two, and each is classified by `mc.classify`
  at **m = 0.0577**. The verdict map, fixed before any run, names what the pair of outcomes does to the
  STOP statement.

- **D24**: matched statistics and a shortcut audit for both nulls, committed before launch.

- **Harness** `scripts/analysis/gap_model_question.py`, with subcommands `plan`, `probe`, `preflight`,
  `table` and `score`.

## Impact

- **Cost**:
  - engineering: about ½–1 day for the option, probe and harness;
  - the strength pilot: about 1.4 hours (32 runs);
  - the panel: about 7.5 hours (512 runs, at the point worm's measured 23 minutes per learning run and
    4.6 per frozen run);
  - the body, for a model that reads `placement`: about 34 hours, in its own change.
- **Seeds**: new bands 4201–4264, 9905–9912 and 9913–9914, added to the seed-band table.
- **Downstream**:
  - a `placement` model qualifies the STOP statement ("the effect depends on how gap junctions are
    modelled"), and goes to its own body registration;
  - two `no_wiring_effect` readings leave the STOP statement standing, backed by four corrections;
  - the 9a synthesis follows either way.
