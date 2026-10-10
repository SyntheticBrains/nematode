## Why

Phase 8b's B.3 + D.1 asks whether a learner's foraging on patchy food takes the roaming and dwelling
states real worms show, and, in B.3, whether serotonin and PDF on the connectome gate them as they do
in the worm (Flavell et al. 2013). This change is **D.1**, the first of the two: the lawns, the internal
state the brain can read, the roaming/dwelling instrument, and a positive control. B.3's modulator field
starts only if that control passes, since a new readout needs its own positive control before it reads
a substrate (the phase protocol).

Nothing of this exists yet. Food on the point worm is points with a capture radius; satiety reaches
`BrainParams.satiety` but no sensory module reads it; there is no roaming/dwelling classifier and no
reference data. Logbook 032 found that per-point depletion alone creates only a short-horizon demand.

**D.1 runs on the point worm, not the kinematic body.** Roaming and dwelling differ in speed and in
reorientation. C.3 (Logbook 086) found the body's speed pinned between 0.10 and 0.12 body lengths/s in
every run, trained or untrained, since its wave floor keeps it crawling, and its sharp turns are steering
pivots with no omega posture. The point worm sets its speed, reversal included, and turns freely.

## What Changes

- **Lawns with edges** on the continuous point-worm environment: `foraging.food_model: lawns` places
  disc lawns, each with a density grid about a body length per cell, a quality, and an odour field
  summed over its remaining cells. `points` stays the default, byte-identical.
- **Continuous intake**: each step, the cell under the worm loses a fraction of its density; the worm
  gains reward and satiety in proportion to what it ate, scaled by the lawn's quality. Intake does not
  depend on speed, so dwelling is not rewarded by construction.
- **An internal-state sensory module**, `internal_state`, giving the brain its satiety.
- **Behaviour capture** gains the worm's satiety, intake and whether it is on a lawn.
- **A roaming/dwelling instrument**: Flavell et al. 2013's speed and angular-speed line and Scheer &
  Bargmann 2023's two-state model. The line is calibrated at the point worm's 5-second step against
  the authors' labels on 1,586 real wild-type animals (Dryad, CC0), and applied unchanged to simulated
  worms on lawns. Ji et al. 2021's patch-foraging, food-density and mutant readings give the
  reference directions.
- **A positive control**: MLP-PPO with the internal-state module on a patchy-lawn cell, against its
  untrained policy and against MLP-PPO without the module, registered before it runs.

## Capabilities

**New**:

- `patchy-lawns`: the lawn food model, its density grid, odour field, intake and quality.

**Modified**:

- `brain-architecture`: the internal-state sensory module.
- `realworm-behavioural-validation`: the roaming/dwelling instrument and behaviour capture's new fields.

## Impact

- `env/env.py`, `env/continuous_2d.py`, a new `env/lawns.py`: the lawn model.
- `agent/food_handler.py`, `agent/runners.py`: intake, reward and satiety per step.
- `brain/modules.py`: `internal_state`.
- `report/dtypes.py`, `agent/agent.py`: behaviour capture's fields.
- `validation/roaming_dwelling.py`: new.
- `utils/config_loader.py`: the lawn configuration.
- `data/roaming_dwelling/`: Ji et al. 2021's tracks, with provenance.
- Configs, analysis script, tests, Logbook 087, tracker D.1, roadmap.
