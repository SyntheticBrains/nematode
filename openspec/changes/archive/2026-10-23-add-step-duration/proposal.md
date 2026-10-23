## Why

Phase 10 grades simulated worms against real-worm data on standard lawn assays (roadmap v5.0, D33). At
the point worm's 5 worm-seconds per step those assays cannot be read:

- a reversal bout or an omega turn lasts a few seconds, no longer than one step, so it can only be one
  step's change of heading;
- a pirouette is a cluster of such events;
- local search is scored as a reorientation rate per minute, which needs the reorientations
  themselves.

D35 fixes the point worm's step at **1 worm-second**, from **one config value** from which every
step-coupled constant derives, with 5 s staying the default. Today there is no such value. The step's
duration is *inferred* from `max_step_mm` divided by the crawl speed (`env/worm_time.py`). The brain and
the classifier each carry a copy of their own (`modulator_step_seconds`, `roaming_dwelling.STEP_SECONDS`).
Dozens of constants measured in steps sit across env, agent and brain configs, unlinked to any of the
three.

This is tracker item E.1. The locomotor primitives (E.2) build on it as the next change, by the
maintainer's decision of 2026-10-23.

## What Changes

- **One value.** A new top-level config key, `step_seconds: float | None = None`. Unset, nothing is
  derived and every run is byte-identical to today's, by construction.

- **Values authored at the reference step, then derived** (the maintainer's choice, 2026-10-23). A
  config's per-step values stay as they are written. They are read as given for the *reference step*:
  the step the config's `max_step_mm` implies, 5 s at 1 mm. When `step_seconds` is set, the loader
  converts every step-coupled constant from the reference step to the configured one. Each kind of
  constant has one rule:

  - **amounts per step** scale linearly: distance, turn, satiety decay, regrowth, per-step reward terms
    and the modulator step;
  - **fractions per step** compound: lawn intake and the adaptive sensor's α;
  - **per-step discounts** compound: PPO γ and GAE λ;
  - **lengths in steps** multiply: the STAM buffer, `max_steps`, `membrane_tau_steps` and the stuck
    threshold;
  - **the derivative feature** is expressed per reference step.

  The derived values are logged at load and printed to stdout, so a campaign's per-run log keeps them;
  the derivation is idempotent.

- **Refused at load** (the maintainer's choice, 2026-10-23; D41's scope). A non-default step is
  refused, with the reason, when anything it touches has no declared derivation:

  - the grid substrate;
  - point food;
  - thermal, oxygen, predators, pheromones and social feeding;
  - multi-agent runs;
  - the memory tasks;
  - the kinematic body (Phase 11);
  - within-lifetime plasticity;
  - every brain other than `connectomeppo` and `mlpppo`.

- **The environment carries the configured step.** The duration is read from `step_seconds` when set,
  and from the crawl-speed relation when unset, which is unchanged.

- **Captures record their step.** `behaviour_capture.json` carries `step_seconds` when it differs
  from 5 s. The roaming/dwelling classifier reads the step from the capture or config, and downsamples a
  shorter step's track to 5 s before measuring. That keeps the HMM's calibration, which is the
  roadmap's registered fallback for "the 1-s step breaks calibrations". The klinokinesis/weathervane
  harness refuses a non-5-s capture: its statistics are per transition, and B.2 repairs them.

- **Two 1-s lawn configs**, generated: the connectome with modulators and the MLP reading internal
  state, on D.1's lawn cell. They are smoked on seeds 5991–5999.

- **Documentation**: the STAM module's "1–2 seconds per step" corrected; the configuration docs; the
  tracker updated (H.2 closed on #470's merge; E.1's dated amendment for the body step).

## Capabilities

### New Capabilities

- `step-duration`: the configured step duration, the derivation rules, their refusals and their record.

### Modified Capabilities

- `continuous-2d-environment`: the step's duration is the configured value when set, and the crawl-speed
  relation otherwise.
- `realworm-behavioural-validation`: captures record their step; the classifier measures every track at
  5-second steps; the bias-curve harness refuses non-5-s captures.

## Impact

- **Code**:
  - `utils/config_loader.py`, holding the key, the derivation and the refusals;
  - `env/continuous_2d.py` and `env/worm_time.py`;
  - `report/behaviour_export.py`;
  - `validation/roaming_dwelling.py`;
  - `agent/stam.py` (docstring);
  - the consumers of `rd.STEP_SECONDS` (`scripts/analysis/lawn_states.py`, `patch_leaving_reference.py`);
  - `scripts/analysis/behavioural_chemotaxis_validation.py`;
  - `scripts/campaigns/generate_lawn_configs.py`.
- **Behaviour at the default**: none. Every existing config leaves `step_seconds` unset.
- **Cost**: about 2–3 engineering days, plus a smoke run. No campaign.
- **Seeds**: smokes on 5991–5999 (the tracker's smoke band).
- **Downstream**: E.2's primitives are specified in seconds against this value. T.1's batched evaluator
  inherits the derivation, and C.3's registration chooses the training-budget settings this change
  leaves alone.
