## Why

C.0 is what must land, validate and freeze before C.1 registers a body (roadmap § Phase 8, item 8). Four
things are missing.

- **What a step is in worm time.** No constant exists in code. The roadmap's figure, about five
  worm-seconds (D22), is derived only in prose.
- **Reversal.** The continuous environment clamps speed to `[0, max_step_mm]`, and all five continuous
  brains hard-code speed bounds of `[0, 1]`. Backward motion is impossible, so VA/DA and VB/DB cannot
  mean different things through a body, escape cannot be reversal-plus-turn, and there are no bouts to
  validate against.
- **The substrate 8b freezes on.** The vendored Cook 2019 workbook states no licence. Emmons 2024 (*PLoS
  Biol* 22:e3002939) republishes the same matrices under CC BY 4.0 with the lab's corrections, and has
  been loadable since PR #404, but the brain accepts only Cook.
- **D19, who generates the rhythm, is not yet recorded as decided.**

## What Changes

- **C.0a: the step constant.** `env/worm_time.py` records the crawl speed, the undulation period and
  `step_worm_seconds(max_step_mm)`. At block V's `max_step_mm: 1.0` this gives 5 worm-seconds, about
  three undulation periods. Nothing consumes it until C.1c.
- **C.0b: signed speed, off by default and byte-identical when off.**
  - `continuous.allow_reversal` lets the environment move a worm backward along its heading, up to
    `max_step_mm`.
  - A shared `signed_speed` brain-config key widens the speed bound to `[-1, 1]` for every continuous
    brain, through one policy helper.
  - Simulation-config validation refuses a brain and an environment that disagree.
  - Behaviour capture records the signed speed when reversal is on.
- **Emmons 2024 as a connectome source.** `connectome_source: emmons_2024_hermaphrodite`; Cook 2019
  stays the default, so every 8a result reproduces. 8b freezes on Emmons 2024 from C.1. The optional M.8
  detour stays on Cook 2019, since it reads an 8a question.
- **C.0d: D19 recorded as decided.** The generator's form is the head relaxation switch (Ji et al. 2021)
  with front-to-back propagation (Wen et al. 2012) and a mirrored backward relay. The record also sets its
  parameter sources and its calibration protocol. It is built in C.1c, where the body it drives exists.
- **Validation.** A gate-only pilot on hard350 with reversal on:
  - MLP-PPO and the settling connectome wild type on Emmons 2024, each learning and frozen;
  - 8 disjoint seeds each;
  - the question is whether each learner still beats its frozen floor.

## Capabilities

**Modified**:

- `continuous-2d-environment`: signed speed, and the step constant.
- `continuous-action-policy`: signed speed bounds that agree with the environment.
- `connectome-ppo-brain`: the Emmons 2024 source.

## Impact

- `env/continuous_2d.py`, `env/worm_time.py` (new), `utils/config_loader.py` (the reversal flag and the
  agreement check), `brain/arch/dtypes.py` (`signed_speed`), `brain/arch/_policy.py` (the bounds
  helper).
- The five continuous brains: `connectome_ppo`, `mlpppo`, `lstmppo`, `cfc_ppo` and `transformer_ppo`.
- `agent/agent.py` and `report/dtypes.py` (signed speed in behaviour capture).
- `brain/arch/connectome_ppo.py` (`connectome_source`).
- Pilot configs and `scripts/analysis/body_prerequisites.py`.
- Tests: byte-identical when off, backward motion, sensing and contact zones under reversal, the bounds
  and the agreement check, the Emmons brain differing from Cook in exactly the four gap pairs, and the
  step constant.
- After the readout: Logbook 081; the tracker's C.0a, C.0b and C.0d; roadmap D19, C.0 and the substrate
  freeze.

## Breaking Changes

None. Every new key defaults to today's behaviour.
