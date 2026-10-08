# C.1d — the MLP positive control through the kinematic body: registration and launch

**Registered 2026-10-08, before any scored run.** Change: `add-body-positive-control`. The design's
decisions A–E, B′ and B″ are this registration's detail; this record fixes the arms, gates, verdict map,
sizing, cost and launch.

## The question

**Can a learner that is known to learn hard350 learn it through the body?** MLP-PPO learns hard350 with
reversal on as a point worm: 97.1% plateau against a 0% floor ([Logbook 081](../../081-body-prerequisites.md)).
If it cannot learn the same cell through the 25-number body drive and the kinematic body, nothing read
later from a connectome through the body is interpretable. This is the body's positive control, and
every C.1e wiring contrast waits on it.

## The arms

MLP-PPO, width 64, through the kinematic body at the calibrated steering gain (2, now the body's
default), hard350, 3,000 episodes, learning and frozen. **Seeds 1505–1512 (8).** The configs are C.0's
MLP reversal pair with the body switched on, generated and re-checked through the real loader:

- `mlpppo_small_continuous2d_fick_adaptive_klinotaxis_hard350_ppo_w64_body.yml`
- `mlpppo_small_continuous2d_fick_adaptive_klinotaxis_hard350_ppo_w64_body_frozen.yml`

## The gates and the verdict

**Metric**: plateau success, the final-quarter success rate of each run.

- **Floor gate**: the learning arm beats its frozen floor, paired by seed, with the 80% bootstrap
  interval of plateau minus floor above zero.
- **Competence**: every seed's plateau is at least 30%.

| floor gate | competence | verdict |
|---|---|---|
| passes | every seed ≥ 30% | **passes**: C.1e may register |
| passes | any seed < 30% | **fallback** |
| fails | | **fails**: C.1d's deliverable is the diagnosis |
| fewer than 8 paired seeds | | **incomplete** |

**The fallback**, fixed now. Its gate-only pilot runs the learning arm at 500, 700 and 1,000 steps on
**seeds 1513–1516** (`..._body_steps{500,700,1000}.yml`). The shortest length at which all four seeds
reach 30% is chosen, and the control re-runs there, learning and frozen, on **seeds 1517–1524**. If no
length qualifies, competence is recorded as unmet and C.1e registers with that condition stated.

**Sequencing.** The fallback pilot launches together with the control. It is read only if the control
reads `fallback`; otherwise it is reported beside the control as description. The re-run waits for the
control's reading.

## The kinematics (descriptive)

Each learning run's final weights are evaluated frozen for 10 held-out episodes, with posture capture.
Frequency, wavelength, speed and reversal fraction are read with design E's rules (undulating steps only
for frequency and wavelength; 1 mm wall exclusion) against the adopted bands: frequency 0.2–0.45 Hz,
wavelength 0.5–0.8 body lengths, speed 0.12–0.3 body lengths per second. A control outside a band is
recorded as a kinematic condition on every later body result, not as a foraging failure.

**Half-step check**: the same weights at 40 sub-steps; each instrument's mean over the 8 runs must agree
within 10% (reversal fraction: 10% or 0.01, whichever is larger). A failure is a property of the body's
integration, recorded as such.

**The worm's reversal rate** is cited as an order of magnitude only: several a minute off food, falling
over the first ~15 minutes (Zhao et al. 2003; Gray, Hill & Bargmann 2005), so about 0.08–0.4 per
5-second step. It grades nothing.

## Before launch

**Three calibration pilots**, all on seeds 1501–1504, four gains, learning and frozen, 32 runs each. The
body changed between them, each time on a finding the previous pilot made, and each change is recorded
in the design before the next pilot ran:

| pilot | body | chosen gain | what it found |
|---|---|---|---|
| [uncapped](pilot-uncapped.json) | as C.1c left it | 1 | seeds locked into one gait; seed 1502 crawled backward on every step |
| [capped](pilot-capped.json) | reversals last one step (B′) | 1 | the MLP silenced 2.7–5 of 12 segments per step, the head most |
| [floored](pilot.json) | wave amplitude ≥ 0.25 of peak (B″) | **2** | below |

**The floored pilot** ([pilot.json](pilot.json), [pilot-kinematics.json](pilot-kinematics.json)):

| gain | mean plateau | seeds (1501–1504) | floor |
|---|---|---|---|
| 0.5 | 5.2% | 4.9, 8.5, 5.5, 1.7 | 0% |
| 1 | 15.4% | 11.7, 17.5, 3.5, 28.8 | 0% |
| **2** | **37.7%** | 50.0, 34.9, 32.8, 32.9 | 0% |
| 4 | 39.2% | 46.3, 44.4, 26.3, 40.0 | 0% |

The rule (highest mean plateau, ties within 5 points to the smaller gain) chose **2**; gain 4 is +1.6,
a tie. The neighbours, reported for D18's sensitivity check: gain 1 is −22.3 points, gain 4 +1.6.
Silencing no longer available, the policy steers through the dorsal–ventral bias, which needs more gain.

At gain 2 the instruments read frequency 0.300 Hz, wavelength 0.647 body lengths, speed 0.125 body
lengths per second (seed 1503 at 0.118, just under the band), reversal fraction 0.005–0.027, with 97–99%
of wall-clear steps undulating.

**Gate preflight, read from the pilot at the registered point.** The floor gate is clear: every floor is
0% and every learning plateau at gain 2 is above 30. **Competence is near the bar**: three of four seeds
sit within 5 points of 30% (32.8, 32.9, 34.9). Eight fresh seeds may well put one under it, which is why
the fallback pilot launches with the control. The gate preflight script reads four-arm wiring panels and
does not apply to a two-arm control; this paragraph is its reading.

**Sizing.** The floor gate's effect at gain 2 is about 38 points against a 0% floor with a seed spread
near 8 points, so 8 paired seeds clear an 80% interval with a wide margin. Competence is a per-seed rule
and is not sized.

**Cost**, from the floored pilot's run times (40–44 minutes per 350-step run at 16 workers):

| part | runs | minutes per run |
|---|---|---|
| control, learning and frozen | 16 | about 41 |
| fallback pilot, 500 steps | 4 | about 59 |
| fallback pilot, 700 steps | 4 | about 82 |
| fallback pilot, 1,000 steps | 4 | about 117 |

Run as two campaigns side by side, 8 and 10 workers on 18 cores, with the longest fallback runs first,
the wall time is **about 2 hours**.

**Readiness.** The gates were read at the registered gain on this cell. The cost comes from a pilot
configured as the campaign. The pilot seeds (1501–1504) are disjoint from the control's (1505–1512) and
the fallback's (1513–1524). Every verdict branch is named.

## Biological conditions recorded

- Reversals are short only: one step, 1.5 head swings. Long reversals before an omega turn cannot occur;
  revisited at C.3, where pirouettes are graded.
- A segment's wave is damped to a quarter of the peak at most, never silenced, after Wen et al. 2012.
- The steering gain has no direct measurement; it is this control's calibration, frozen for every arm.

## Launch

From a worktree at this commit, so the main checkout can change without touching the running code:

```bash
cd ../nematode-c1d
B=configs/scenarios/foraging/mlpppo_small_continuous2d_fick_adaptive_klinotaxis_hard350_ppo_w64_body
O="-- --theme headless --track-experiment --no-detailed-export --no-file-log"
uv run python scripts/run_campaign.py --config ${B}.yml --config ${B}_frozen.yml \
  --seeds 1505-1512 --runs 3000 --workers 8 --output-dir ../nematode/campaigns/c1d-control $O &
uv run python scripts/run_campaign.py --config ${B}_steps1000.yml --config ${B}_steps700.yml \
  --config ${B}_steps500.yml --seeds 1513-1516 --runs 3000 --workers 10 \
  --output-dir ../nematode/campaigns/c1d-fallback-pilot $O &
wait

# Then, from the main checkout: copy the worktree's records in, and score.
rsync -a --ignore-existing ../nematode-c1d/experiments/ experiments/
rsync -a --ignore-existing ../nematode-c1d/exports/ exports/
uv run python scripts/analysis/body_control.py control --logs campaigns/c1d-control/logs --out control.json
uv run python scripts/analysis/body_control.py fallback --logs campaigns/c1d-fallback-pilot/logs --out fallback-pilot.json
uv run python scripts/analysis/body_control.py kinematics --stage control --logs campaigns/c1d-control/logs --out kinematics.json
```

## Retention (A.0)

Committed: this record, the three pilots' readings and kinematics, and the control's `control.json`,
`fallback-pilot.json` and `kinematics.json`. Archived off-repo: the pilot, control and fallback campaigns'
raw logs and the worktree's session records.
