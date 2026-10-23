## Why

Phase 10 grades simulated worms on the statistics that define the worm's own strategies:

- reorientation rates and their decay in local search;
- pirouettes against dC/dt;
- roaming and dwelling;
- reversals at the lawn edge.

The point worm cannot produce any of these. Its action is a free `(speed, turn)` each step, so a
"reversal" is whatever the policy makes of a negative speed, and an omega turn is one large heading
change. Neither has the duration, speed or angle of the animal's.

D36 fixes the remedy. The worm gets locomotor primitives:

- **run**, **reversal bout**, **omega/delta reorientation** and **pause**;
- kinematics fixed from data, and timing chosen by the controller;
- the same primitives for every arm and every negative;
- reorientations read from trajectories, never from primitive labels.

This is tracker item E.2. It builds on E.1's 1-second worm (#471).

The plan review's literature sweep corrected two premises (2026-10-23):

1. **Delta turns are not shallow turns.** In Broekmans et al. 2016 they are the deepest coils: about
   90° to the dorsal side, against about 150° to the ventral side for omegas. The shallow turns are the
   small heading changes after short reversals.
2. **Both named sources are off food.** Broekmans 2016 recorded worms off food in a copper ring;
   Gray 2005 deposited no data. The only open on-food kinematics are Scheer & Bargmann 2023, the
   benchmark's own source.

## What Changes

The maintainer's decisions of 2026-10-23 are marked (decided).

- **A new action space, `action_space: primitives`** (connectome and MLP-PPO). It is five continuous
  outputs, read every second:

  - run speed and run curvature;
  - a start hazard for each of reversal, omega and delta.

  It is a **hazard interface** (decided): the primitive layer, not the brain, draws each start from
  the hazards, so every controller drives the primitives the same way. That covers the mechanistic
  baseline's stochastic switch, the fitted connectome, PPO, and the Poisson negative's constant
  hazards. PPO's buffer is unchanged.

- **The primitive layer** (`continuous.locomotion: primitives` in the environment):

  - **Run.** The controller sets speed in [0, 0.2 mm/s] and curvature bounded by the sharp-turn
    threshold.
  - **Reversal.** It backs at a fixed speed for a duration drawn from data.
  - **Omega and delta.** Each turns the heading by an angle drawn from data, over a drawn duration,
    toward the episode's ventral or dorsal side.
  - During a bout the controller is still read each second, but its outputs move nothing.
  - **Pause is a run at zero speed** (decided). No pause duration is invented.

- **Kinematics from published values** (decided), vendored at `data/locomotion/primitives.json` with
  provenance and pinned by a test. The sources are Zhao 2003, Gray 2005, Broekmans 2016, Pierce-Shimomura
  1999 and Ben Arous 2009. B.1 checks them against Scheer's on-food flags on held-out cohorts.

- **A food-contact sense** (decided). This is the lawn edge question E.2 was to settle.

  - A new sensory module, `food_contact`: the lawn density under the worm, 0 off a lawn.
  - It stands for the mechanosensory bacteria sense behind basal slowing (Sawin 2000).
  - With it, edge reversals (Scheer's head pokes) can arise.

- **A locomotor-state sense**, `locomotor_state`: which primitive is running. With it a controller can
  sequence a reorientation after a reversal, the pirouette structure, without the sequence being
  built in.

- **Capture** records each step's primitive, for diagnosis only.

- **Two non-learning controllers**, `ConstantHazards` (the Poisson negative) and `PirouetteHazards` (a
  reorientation hazard rising as concentration falls). They are E.2's positive control and B.4's
  matched negative.

- **Configs**: 1-s primitive arms of the connectome lawn worm and the MLP lawn worm (with food contact
  and locomotor state), each smoked.

## Capabilities

### New Capabilities

- `locomotor-primitives`: the primitive layer, its kinematics, the hazard interface, the two senses
  and the non-learning controllers.

### Modified Capabilities

- `continuous-action-policy`: the `primitives` action space.
- `realworm-behavioural-validation`: capture records the primitive under primitives.

## Impact

- **Code**:
  - `env/continuous_2d.py` and a new `env/primitives.py`;
  - `brain/arch/_policy.py`, `dtypes.py`, `mlpppo.py` and `connectome_ppo.py`;
  - `brain/modules.py`, adding the two modules;
  - `agent/agent.py` and `agent/runners.py`;
  - `utils/config_loader.py` (cross-validators) and `utils/step_duration.py`;
  - `report/dtypes.py` and `report/behaviour_export.py`.
- **Data**: `data/locomotion/primitives.json` with `PROVENANCE.md`. No new download.
- **Default behaviour**: none. `locomotion: free` and `action_space: speed_turn` stay the defaults and
  draw nothing from the RNG.
- **Cost**: about 4–6 engineering days, plus smokes. No campaign.
- **Seeds**: smokes on 5991–5999. The positive control's simulated episodes are test fixtures on
  seeds 5994–5999.
- **Downstream**:
  - T.1 batches the primitive layer;
  - B.2's detector reads trajectories these primitives produce;
  - B.4's Poisson negative is `ConstantHazards`;
  - C.1–C.3 drive the hazard interface;
  - the connectome's mapping of the two new senses onto neurons is C.2's encoder question.
