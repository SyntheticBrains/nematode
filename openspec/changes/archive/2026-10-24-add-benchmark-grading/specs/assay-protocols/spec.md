## MODIFIED Requirements

### Requirement: Generated assay configs

The repository SHALL carry generated assay configs on the 1-second primitives worm, for the MLP and the
connectome:

- `scheer_lawn`: a 10 mm well, a 3 mm centred lawn, 60 min from an off-lawn start, analysed from 20 min;
- `uniform_lawn_scheer`: a 10 mm well whose lawn covers it (radius 5 mm, no wall clearance), 60 min from
  the centre on the lawn, analysed from 20 min;
- `food_removal_lopezcruz`: a 100 mm circle, 45 min on a 25 mm-radius lawn, then 45 min bare from the
  centre;
- `food_removal_calhoun`: a 25.4 mm square, 45 min fed, then 30 min bare, analysed from 2 min;
- `uniform_lawn_flavell`: a 35 mm circle, a 16 mm-radius lawn, 90 min.

#### Scenario: The assay configs run

- **WHEN** each assay config runs one episode on a smoke seed with capture on
- **THEN** it SHALL complete with its phases in order
- **AND** its capture SHALL classify where the protocol has food

#### Scenario: The uniform well stays on food

- **WHEN** `uniform_lawn_scheer` runs one episode on a smoke seed
- **THEN** every captured step SHALL be on the lawn, including at the well's rim, in the per-step
  runner and the batched evaluator
