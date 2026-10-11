# B.3 — serotonin and PDF on the connectome: registration and launch

**Registered 2026-10-11, before any panel run.** Change: `add-b3-modulators`. Both pilots' readings
and the preflight are committed beside this record.

## The question

**When serotonin or PDF is removed from a trained connectome, does it move the way the mutant worms
move?** The connectome is sign-grounded and carries both modulators.

- **Serotonin.** *tph-1* worms, which make no serotonin, roam more (Flavell 2013).
- **PDF.** *pdfr-1* worms, which lack PDF's receptor, dwell more (Flavell 2013).

The design's Decisions A–H give the circuit, the field, and why the reading and the substrate changed
after the first pilot.

## The cell and the substrate

**The cell** is stage 0's: D.1's lawn cell with entropy 0.005, Emmons 2024, settling to depth 4, and
reversal. It has 4 lawns of 2.5 mm, with the worm starting on one. Episodes are 720 steps, 3,000 per
run, with no state-shaping reward.

**The substrate:**

- **Signs grounded and held.** Synapse signs come from the transmitter atlas (`synapse_signs: atlas`)
  and are enforced under PPO (`enforce_synapse_signs: true`). The atlas grounds 3,176 of 3,709
  chemical synapses.
- **The modulator field, as Decisions A–E describe it:**
  - serotonin, released by NSM, inhibits AIY, ASI and RIF through MOD-1;
  - PDF, released by AVB, PVP and SIAV, excites AIY, RIM and RIA through PDFR-1 and inhibits NSM;
  - NSM is driven by the last move's bite, as a fraction of a full bite;
  - the concentrations decay with τ = 60 s;
  - the four pathway strengths are learnable, starting at 1.

**The arms**, each a config stem `connectomeppo_small_continuous2d_fick_adaptive_klinotaxis_lawns_` +:

| arm | stem suffix | role |
|---|---|---|
| intact | `modulated_atlas` | the learner |
| floor | `modulated_atlas_frozen` | its untrained policy |
| raised *tph-1* | `modulated_atlas_tph1` | trained without serotonin; beside |
| raised *pdfr-1* | `modulated_atlas_pdfr1` | trained without PDFR-1; beside |
| satiety link | `modulated_atlas_satiety` | NSM's drive × (1 + hunger); beside |

**The acute knockouts** are evaluations, not trained arms. Each takes the intact arm's trained
weights and evaluates them with the knockout set:

- `modulated_atlas+tph1` holds serotonin at zero;
- `modulated_atlas+pdfr1` removes PDFR-1's current and PDF's inhibition of NSM.

Each runs the intact arm's 30 held-out evaluation episodes, on the same lawns and seeds.

**Seeds 2001–2032 (32), 160 runs.** Seeds 2001–2019 are stage 0's, so the two pair beside the
readings. Every run is evaluated frozen for **30 held-out episodes** with D.1's instrument.

## Gates, in order

1. **Intake.** The intact arm's intake plateau beats its floor's, paired by seed, with the 80%
   interval above zero.
2. **Dwelling.** The intact arm's on-lawn dwelling share against its floor's, paired by seed and
   classified at stage 0's minimum of 0.278. The gate **passes on significant dwelling of any size**:
   `move_wt`, or `below` with a positive mean. The knockouts need a dwelling state to shift, not
   stage 0's effect size. This was decided with the user after the sign-grounded pilot put the panel
   near stage 0's bar.

If either gate fails, the panel is **unreadable** and no knockout is read.

## The readings

**Two, one per mutant.** Each is the acute knockout's **mean on-lawn speed** (mm/s) minus the intact
network's, paired by seed. Each difference is multiplied by the mutant's published direction, so a
reproduced mutant reads positive:

| knockout | published direction | oriented reading |
|---|---|---|
| *tph-1* | more roaming, so faster | knockout − intact |
| *pdfr-1* | more dwelling, so slower | intact − knockout |

**The minimum is 0.009 mm/s,** a third of the 0.028 mm/s between real worms' calibrated dwelling
(0.013) and roaming (0.041) speeds. Those speeds are the states' typical ones: the exponentiated
means of the HMM's log-speed emissions, less its 0.001 mm/s floor. The reading is an arithmetic mean
of window speeds. The minimum is set from the biology, not from the pilots' effects.

**Each reading** is classified by `mc.classify` at that minimum, with the two q values (two-sided
Wilcoxon) **Holm-adjusted** across the pair:

| state | verdict | what it licenses |
|---|---|---|
| `move_wt` | **mutant direction** | removing the modulator moves the trained network's on-lawn speed the mutant's way by at least 0.009 mm/s |
| `move_null` | **opposite direction** | it moves speed against the mutant by at least 0.009 mm/s |
| `below` | **below the minimum** | a significant shift smaller than 0.009 mm/s, in the direction of its sign |
| `no_move` | **no effect at the minimum** | the interval sits inside ±0.009 mm/s and includes zero |
| `unresolved` | **unresolved at this sensitivity** | none; the achieved interval is reported as the bound |

**Reported beside, never read as verdicts:**

- each acute knockout on the dwelling share;
- the raised *tph-1* and *pdfr-1* arms and the satiety link against the intact arm, on both measures;
- the learned pathway strengths;
- the concentrations in each state;
- how many seeds take both states;
- the intact arm against stage 0's connectome on the dwelling share, over seeds 2001–2019 (stage 0's
  signs were random, so this compares substrates as well as modulators);
- each reading's achieved spread and MDE, beside the minimum, never used to re-read it.

## The pilots

**First pilot, random signs** (seeds 9401–9404; [pilot-states.json](pilot-states.json)):

- Intake passed, +2.43 [+2.09, +2.77].
- Two seeds dwelt in every window; two roamed with brief dwells.
- Acute *tph-1* slowed both roaming seeds, against the mutant. Acute *pdfr-1* slowed one, with it.

Under random signs, what a receptor cell does downstream is arbitrary per seed, so the substrate
changed (Decision H). On the classified share, each knockout could move only half the seeds, so the
reading moved to speed (Decision G).

**Sign-grounded pilot** (seeds 9411–9418; [pilot-atlas-states.json](pilot-atlas-states.json)), which
sizes this panel. Seeds 9415–9418 were added after the spec review, so that the dwelling gate could
resolve, which it cannot at four seeds.

- **Intake** passes: +2.66 [+2.50, +2.81].
- **Dwelling shares over the floor,** per seed: +0.23, +0.99, −0.00, −0.01, +0.95, +0.82, +0.98,
  +0.00, a mean of **+0.495** [+0.30, +0.71]. Stage 0's panel read +0.514.
  - Five of the 8 seeds dwell. Two of them switch states with many bouts: 9411 (16 roaming, 35
    dwelling) and 9416 (11 and 6).
  - On 9411, both knockouts abolish dwelling.
- **Acute *tph-1*, oriented speed,** per seed: +0.037, −0.002, +0.011, −0.003, +0.001, −0.007,
  +0.002, −0.036 mm/s, a mean of **+0.000** (sd 0.020).
- **Acute *pdfr-1*,** per seed: −0.046, −0.008, +0.064, +0.000, −0.019, −0.015, +0.014, −0.001 mm/s,
  a mean of **−0.001** (sd 0.032).
- **Learned strengths:**
  - PDF's inhibition of NSM fell below 1 on 7 of 8 seeds (0.13–0.74);
  - NSM's ingestion gain rose above 1 on 6 of 8 (1.3–2.4).

## Sizing

At 32 seeds, with the eight-seed pilot's spreads, the MDE is `2.487 × sd / √32`:

- ***tph-1*: 0.0088 mm/s,** just below the 0.009 minimum. 31 seeds would reach it, so the panel is
  sized for this reading. The pilot's mean is +0.000, so a `no_move` reading, no effect at the
  minimum, is a live outcome and an informative one.
- ***pdfr-1*: 0.014 mm/s.** This is above the minimum, so this reading may come out
  **unresolved**. Reaching the minimum would need about 77 seeds. It is registered at this size
  anyway, decided with the user: the panel's size is set by *tph-1* and the arms beside.
- **The dwelling gate:** the pilot's mean is +0.495 with sd 0.45, which gives an MDE of 0.20 at 32
  seeds. Its 80% interval already excludes zero at 8 seeds.

## Before launch

- **Gate preflight** ([preflight.json](preflight.json)): the registered gates and readings evaluated
  through `lawn_states.read_knockouts` on the eight committed sign-grounded pilot seeds.
  - Intake passes.
  - Dwelling is `unresolved` at n = 8: +0.495 [+0.30, +0.71] with q 0.078. The interval excludes
    zero. The two-sided Wilcoxon misses 0.05 because three seeds sit just below their floors.
  - At 32 seeds, with five in eight dwelling, the gate is expected to pass. If the panel dwells as
    rarely as the first four pilot seeds did, the panel is unreadable.
- **Cost:** the two sign-grounded pilot batches each ran 20 runs in 35 to 38 minutes on 16 workers,
  so 160 runs take about **5 hours**. Evaluating 56 runs took about a minute, so the panel's 224 take
  about 5.
- **Seeds:** the panel's (2001–2032) are disjoint from the pilots' (9401–9404, 9411–9418).
- **Spec review:** `/nematode-review-spec add-b3-modulators`, after which the proposal was brought in
  line with the design, the minimum's basis was stated, the glutamate-sign caveat added (Decision H),
  and the pilot extended to eight seeds.

## Launch

From a worktree at this commit:

```bash
git -C ../nematode-d1 checkout --detach <this commit>
cd ../nematode-d1
L=configs/scenarios/foraging/connectomeppo_small_continuous2d_fick_adaptive_klinotaxis_lawns_modulated_atlas
uv run python scripts/run_campaign.py --config ${L}.yml --config ${L}_frozen.yml \
  --config ${L}_tph1.yml --config ${L}_pdfr1.yml --config ${L}_satiety.yml --seeds 2001-2032 \
  --runs 3000 --workers 16 --output-dir ../nematode/campaigns/b3-panel \
  -- --theme headless --track-experiment --no-detailed-export --no-file-log

# Then, from the main checkout:
rsync -a --ignore-existing ../nematode-d1/experiments/ experiments/
rsync -a --ignore-existing ../nematode-d1/exports/ exports/
uv run python scripts/analysis/lawn_states.py --logs campaigns/b3-panel/logs \
  --arms modulated_atlas modulated_atlas_frozen modulated_atlas+tph1 modulated_atlas+pdfr1 \
    modulated_atlas_tph1 modulated_atlas_pdfr1 modulated_atlas_satiety \
  --learner-arm modulated_atlas --floor-arm modulated_atlas_frozen --minimum 0.278 \
  --knockout-minimum 0.009 --knockout-measure speed --seeds $(seq 2001 2032) --episodes 30 \
  --compare-stage0 docs/experiments/logbooks/supporting/089-connectome-lawns/panel.json \
  --out panel.json --csv per-run.csv
```

## Retention (A.0)

**Committed:** this record, `preflight.json`, both pilots' readings, and the panel's `panel.json` and
`per-run.csv`.

**Archived off-repo:** the campaign directories and the worktree's session records.
