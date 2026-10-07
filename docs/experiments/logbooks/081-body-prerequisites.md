# 081: The Body's Prerequisites Are in Place: Signed Speed Validates, and 8b Freezes on Emmons 2024 (Phase 8b C.0)

**Status**: completed — **validated.** With reversal on, MLP-PPO and the connectome's settling wild type
on Emmons 2024 both learn hard350 well above their frozen floors.

| learner, hard350, reversal on | plateau | floor | plateau − floor, 80% CI | gate |
|---|---|---|---|---|
| MLP-PPO, width 64 | 97.1% | 0.0% | [96.2, 97.9] | **passes** |
| connectome PPO, settling wild type, Emmons 2024 | 80.2% | 0.0% | [78.3, 82.0] | **passes** |

C.0 lands, validates and freezes before C.1 registers:

- **C.0a**: a full-speed step is **5.0 worm-seconds**.
- **C.0b**: signed speed, validated above.
- **C.0c**: already absorbed into C.1c.
- **C.0d**: D19 recorded as decided.
- **The substrate**: **Emmons 2024** with reversal on.

**Date**: 2026-10-07.

**OpenSpec change**: `add-body-prerequisites`.

## Objective

C.1 puts a body under the brain. Before it registers, the platform needs four things:

- what a step is in worm time;
- reversal, so forward and backward motor classes can mean different things;
- a frozen substrate;
- D19, who generates the rhythm, decided.

Signed speed changes every continuous brain's action space and the untrained policy's starting
behaviour. A new component on a validated platform needs its own positive control, so each learner
must still learn the cell with reversal on.

## Method

### What landed

- **C.0a, the step constant.** `env/worm_time.py`: a full-speed step of `max_step_mm` lasts
  `max_step_mm / 0.2 mm/s`. At block V's 1.0 mm that is 5.0 worm-seconds, 3.1 undulation periods of
  1.6 s. The crawl speed and period are the roadmap's figures, with Chung & Kim as the modelled
  reference and C.3 as where they are validated. A crawl speed of 0.15–0.3 mm/s would make the step
  3.3–6.7 s.
- **C.0b, signed speed.**
  - `continuous.allow_reversal` moves a worm backward along its unchanged heading, up to `max_step_mm`.
  - `signed_speed` widens every continuous brain's speed bound to `[-1, 1]` through one shared helper.
  - Loading refuses a brain and an environment that disagree.
  - Behaviour capture records the signed displacement only under reversal, so existing capture files are
    unchanged.
  - Both are off by default and byte-identical when off. The full suite passes with both off.
- **The substrate.** `connectome_source: emmons_2024_hermaphrodite`, the same reconstruction under CC
  BY 4.0 with the lab's corrections. A test pins that it differs from Cook 2019 only around the four
  gap pairs Emmons adds or strengthens.
- **C.0d.** D19 is recorded as decided in the roadmap and the tracker.
  - **Phase** comes from Ji et al. 2021's head relaxation switch.
  - **Forward propagation** is Wen et al. 2012's front-to-back relay; **backward** is the mirrored
    relay.
  - **The brain** sets segmental drive. The sign of its net forward-versus-backward drive selects the
    wave's direction.
  - **Free parameters** are calibrated once on C.1d's MLP and frozen.
  - C.1c builds it.

### The validation

On hard350 with reversal and signed speed on, 3,000 episodes, seeds 1301–1308:

- MLP-PPO (width 64), learning and frozen;
- the connectome's settling wild type on Emmons 2024, learning and frozen.

That is 32 runs, in 27 minutes. **The gate:** each learner's plateau beats its frozen floor, paired
by seed, with the 80% interval of the difference above zero.

## Results

Both learners pass, on every seed (connectome 73.5–84.8%, MLP 93.2–99.2%).

Reported beside, not read: both plateaus sit at or above the committed reversal-off runs on the same
cell. Those are 88.4–99.2% for MLP-PPO ([Logbook 080](080-across-step-state.md)'s control) and
75.7–77.8% for the connectome on Cook 2019 (Logbooks 079 and 080). The connectome comparison changes
reversal and substrate together, so it says only that neither cost learning visibly.

## What this establishes, and what it does not

- **Established**: with reversal on, both learners still learn hard350 well above their floors, so
  signed speed is a usable action space for C.1. Its starting policy is centred on zero speed rather
  than half speed.
- **Established**: 8b's substrate is frozen on Emmons 2024 with reversal on. 8a's results and the
  optional M.8 detour stay on Cook 2019.
- **Not established**: how the worms use reversal. Counting backward steps needs per-step capture,
  about 150 MB a run at this length, so reversal use is read through the body at C.1d, where C.3's bout
  instruments apply.
- **Not established**: whether reversal changes block V's wiring contrast. No null ran here, and C.1e
  registers its own floors and baselines in a new reference frame.

## Biological fidelity of these choices, and where each is revisited

Recorded at the branch review. Figures marked *recalled* are from memory of the literature and are to
be checked against their sources where the named later step uses them.

- **Step of 5 worm-seconds.** It rests on a 0.2 mm/s crawl and a 1.6 s undulation. That crawl speed is
  typical of forward crawling off food; worms on food move slower. Reported crawling periods span
  roughly 1.5–3 s across assays (*recalled*), so 1.6 s is the fast end, and "three periods per step"
  may be nearer two. **Adequate now**, since nothing consumes the constant yet. **Revisited at C.1c**,
  which fixes the period and wavelength from Fang-Yen et al. 2010, and at C.3's kinematic validation.
- **Symmetric reversal.** Backward crawling speed is broadly comparable to forward, so the bound is
  reasonable. Real reversals, though, are brief bouts of a few head swings, often followed by an omega
  turn (the pirouette; Pierce-Shimomura et al. 1999). Nothing here limits how long a reversal lasts, so a
  learner could back up for many steps, which a worm rarely does. **Adequate now; revisited at C.1d**,
  where reversal use is first read with C.3's bout instruments. B.2c's bout-duration target is the
  registered check. A cap or a cost is the remedy if reversal turns out to be used as a second gear.
- **Heading and sensing under reversal.** The head stays the head, the head-sweep sample stays at the
  head, and the rate-of-change feature follows the actual displacement. That matches a worm backing up
  tail-first while its head sensors read what the movement brings. **Faithful.**
- **Sharp reorientation is still missing.** Turning is capped at 0.5 rad per step and there is no omega
  turn, so the reversal-then-turn pirouette cannot be fully expressed on the point worm. **Known and
  planned**: in C.1 turning emerges from body curvature, and C.3 grades omega-turn geometry.
- **The anatomical readout.** VB/DB drive forward and VA/DA drive backward. Signed speed makes the
  backward classes meaningful for the first time. **An improvement in fidelity.**
- **Emmons 2024.** It is the same single-animal reconstruction with the lab's corrections, so fidelity is
  unchanged or better. Two notes:
  - **Mechanosensory pairs.** Its four changed gap pairs couple the touch neurons ALM and PLM to BDU. The
    predator-contact projection injects into ALM and PLM, so a predator cell on Emmons is not directly
    comparable to one on Cook. The food-only hard350 cell is unaffected.
  - **Individual variability.** One animal's wiring is the standing limitation. M.2's wild-type-vs-wild-
    type control (Cook against Witvliet dataset 8) is where it would be measured.
- **D19, the rhythm from outside the brain.** In the worm, the motor circuit generates the rhythm, with
  proprioceptive coupling. An outside generator is a deliberate simplification, so the claim is scoped
  to segmental drive and steering. The route that would test rhythm generation by the wiring, C.2c, ran
  through B.2 and has none since B.2 closed (Logbook 080). **Acceptable now; the open question is
  recorded in the tracker**, and would need its own registration later.

## Artefacts

- [validation.json](supporting/081-body-prerequisites/validation.json): each learner's per-seed
  plateaus, floors and gate.
- Reproduce: `scripts/campaigns/generate_body_prerequisite_configs.py`;
  `scripts/analysis/body_prerequisites.py --logs <campaign>/logs --out validation.json`. The campaign
  directory is archived off-repo.
