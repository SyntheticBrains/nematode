# V.4 — fresh rewirings: the registered protocol

Registered in `openspec/changes/add-wiring-fresh-rewiring`, reviewed and committed **before** any arm
runs. The instrument check below ran first, on committed data, and spent no registered seed.

## The question

After [L.0](../../064-l4-frozen-features.md), block V's learning-speed advantage is the **only
surviving wiring result in this project**:

| reading | what it found |
|---|---|
| endpoint under gradient learning ([034](../034-connectome-structure-controls/)) | inert — no endpoint advantage |
| under a local rule that **writes** the wiring ([R.1c](../../061-l4-reduced-perturbation.md), [R.2](../../063-l4-eprop.md)) | actively harmful — every substrate-writing arm below the frozen control |
| as **fixed features** ([L.0](../../064-l4-frozen-features.md)) | indistinguishable from a degree-matched shuffle |
| **learning speed under PPO** ([V.1](../../057-wiring-premise-contrast.md), [V.3](../../058-wiring-premise-difficulty.md)) | **+35.4%** and **+23.5%** off time-to-competence |

That last row has one open caveat, and this change exists to close it. `rewire_seed` is unset in every
wiring config, so each seed's rewired graph derives from its run seed — and **V.3's rewirings at seeds
1–32 are a subset of V.1's at 1–64**. The two positives do not corroborate each other on independent
nulls; they *share* them. A rewiring that happened to be unlucky at seed 7 was unlucky at seed 7 in
both records.

> *Does the advantage hold on rewired graphs no panel has used?*

Seeds **65–96** give 32 graphs fresh to both panels, and a test asserts the disjointness.

## What "fresh" does and does not buy

`rewire_seed` stays unset, deliberately — it is the coupling V.1 and V.3 ran under, and a replication
that removed it would not be replicating them. So at a fresh seed the **rewiring, the task draw and the
network initialisation all move together**. That is enough to answer whether the committed figure
survives graphs it has never seen. It is *not* enough to attribute the effect to the rewiring alone:
that needs `rewire_seed` pinned across seeds, which is a different experiment, and one neither V.1 nor
V.3 registered. The stricter question stays open and unregistered, and the record says so in both
directions of the outcome.

## The arms — no new configs

All eight already exist and are the ones V.1 and V.3 ran. A test asserts each is byte-unchanged since
its panel ran: the thermal four against **`431a4689`**, the hard350 four against **`48ba778c`**.

| cell | arm | config | wiring | learner |
|---|---|---|---|---|
| thermal | `wt_ppo` | `..._thermal_klinotaxis_t20` | wild type | PPO |
| thermal | `rn_ppo` | `..._thermal_klinotaxis_rewired_null_t20` | degree-preserving null | PPO |
| thermal | `wt_frozen` | `..._thermal_klinotaxis_frozen_t20` | wild type | nothing learns |
| thermal | `rn_frozen` | `..._thermal_klinotaxis_rewired_null_frozen_t20` | null | nothing learns |
| hard_food | `wt_ppo` | `..._hard350` | wild type | PPO |
| hard_food | `rn_ppo` | `..._hard350_rewired_null` | null | PPO |
| hard_food | `wt_frozen` | `..._hard350_frozen` | wild type | nothing learns |
| hard_food | `rn_frozen` | `..._hard350_rewired_null_frozen` | null | nothing learns |

Each rewired config differs from its wild-type partner in the **`wiring` key alone**, and a test
asserts `rewire_seed` is unset in all four.

**256 runs**: 2 cells × 4 arms × 32 seeds at 3000 episodes, with `--track-experiment`.

## The instrument does not change

This is the property the whole reading rests on. A replication varies the evidence and holds the
reading fixed; if the harness were edited here, *"the instrument changed"* would compete with *"the
effect is not there"*, and after the fact those are not separable.

So `scripts/analysis/wiring_premise.py` and `scripts/analysis/connectome_structure_efficiency.py` are
used **unmodified**, and a test asserts both are byte-identical to `main`. The new driver
`scripts/analysis/wiring_fresh_rewiring.py` is a **manifest builder and a branch reporter, and nothing
else** — it does not re-declare the 20% minimum, the verdicts, the crossing floor or the arm mapping,
all four of which the committed harness already owns. A test asserts it does not so much as define
those names.

### The instrument check (ran before anything else, on committed data)

V.3's committed panel re-scored through `wiring_premise.py` at seeds 1–32:

| | committed record ([058](../../058-wiring-premise-difficulty.md)) | re-scored today |
|---|---|---|
| `episodes_to_30pct_success` | wild **892** vs null **1165** | wild **892.03** vs null **1165.34** |
| time-to-competence gain | **+23.5%** | **+23.5%** (0.2345) |
| wild-better seeds | 21/32, q = 0.029 | 21/32, q = 0.029 |
| verdict | `specific_wiring_efficiency` | `specific_wiring_efficiency` |

The whole `efficiency` block compares equal field for field.

The same check on V.1's pooled 64 seeds, for the other cell this panel replicates:

| | committed record ([057](../../057-wiring-premise-contrast.md)) | re-scored today |
|---|---|---|
| `episodes_to_30pct_success` | +217.1, q = 0.001 | wild **396.36** vs null **613.47**, +217.11, q = 0.001 |
| time-to-competence gain | **+35.4%** | **+35.4%** |
| wild-better seeds | 43–48/64 across metrics | 44/64 on the primary, 43–48/64 across the four |
| verdict | `specific_wiring_efficiency` | `specific_wiring_efficiency` |

Its crossing rates are 98% (wild) and 100% (null), both above the 80% floor, so neither comparator was
censored. The instrument still reproduces both records it is replicating, so the panels are comparable.

## The branches, registered in advance

The harness's verdict names are what the record reports, with V.1's prose branches **mapped onto them**
rather than run alongside — a parallel vocabulary is how two records come to disagree about one run.

| harness verdict | prose branch | consequence |
|---|---|---|
| `specific_wiring` | **replicates** | the caveat closes; block V's positive is independent in rewiring |
| `below_min_effect` | **same direction, below the minimum** | a real but smaller effect, shrinkage named; licenses the follow-up, not the claim |
| `degree_statistics` | **does not replicate** | the first positive is **withdrawn on the record rather than defended** |

Three further harness verdicts are registered **as themselves**, because each is live rather than
hypothetical, and **none is evidence against the original result**:

- `saturated` — the cell cannot answer on this axis. This is what the klinotaxis cell returned in
  V.1's *own* pilot: both wirings at 100% full clear and a contrast of exactly zero.
- `no_learning` — a learning gate failed, so the contrast is uninterpretable. The gates are read
  before the contrast, as the harness has always done.
- `insufficient_seeds` — too few paired seeds survived to score the panel.

Separately, a contrast the harness flags **materially censored** below its 80% crossing floor — the case
[L.0](../../064-l4-frozen-features.md) met on `hard350` with five non-crossing seeds of 32 — is likewise
**not** evidence against the original. The censoring flag is computed through the harness's own
`crossing_rate()` and `CROSSING_FLOOR`, not re-derived.

### A split stays a split

Read **per cell, never pooled**. Block V's claim is that the effect generalises from a foraging cell
under thermal pressure to a foraging cell hard enough to discriminate — so one cell replicating and the
other not is evidence about the **scope** of that generalisation. If that happens the pooled reading is
**withheld**, and the split is never resolved toward whichever cell happens to support the original.

## The comparators, carried with their own variability

| cell | committed figure | seeds | per-panel spread |
|---|---|---|---|
| thermal | **+35.4%** | 64 | **+46.4%, +32.6%, +31.8%** — 15 points across panels of 16, 16 and 32 |
| hard_food | **+23.5%** | 32 | single panel; three of four efficiency metrics significant |

V.1's own estimate shrank across its panels, which is why a near-miss on the thermal cell is read
against that spread rather than as a clean failure.

## Power

32 paired seeds. A one-sided sign test needs **22/32** to clear q = 0.05. Against the comparator's
win-rate band (21/32 to 26/32, its observed rates across the efficiency metrics):

| win rate | power |
|---|---|
| 65.6% (21/32) | 43.4% |
| 73.4% (midpoint) | **79.2%** |
| 81.3% (26/32) | 97.3% |

**These are sign-test planning figures, not the registered procedure's power.** The registered
procedure is a paired rank test under BH-FDR across four metrics, which differs from a sign test in
both directions — more sensitive to effect size, penalised by the correction. The figures are here to
say what this panel can and cannot see, not to license a post-hoc reading of a null.

## The honest prior

I expect it to replicate on both cells, and I expect the **thermal cell is the likelier of the two to
come in under the bar** — not because the effect is absent there but because V.1's own per-panel
estimates spread 15 points around a mean 15 points above the minimum, so a fresh panel of 32 landing
below +20% is well inside that instrument's demonstrated variability. `hard_food` sits closer to the
bar in level (+23.5% against +20%) but was measured once, at 32 seeds, on a cell calibrated one step
wide — there its risk is **censoring**, not shrinkage.

Registered before the data: a thermal `below_min_effect` beside a hard_food `specific_wiring` is the
single most likely non-clean outcome, and it is a **split**, reported as one.

## If it does not replicate on either cell

[V.1](../../057-wiring-premise-contrast.md), [V.3](../../058-wiring-premise-difficulty.md), the 7a
shipment record ([059](../../059-7a-shipment.md)) and the roadmap each rest on the figure, and **each is
corrected in the same PR**. The first positive is withdrawn on the record rather than defended, and the
phase's citable results are restated — which after L.0 would leave the systematic negative as the only
one. That consequence is stated here, before the data, so that the cost of the outcome cannot argue
against reporting it.

______________________________________________________________________

## Reproduce

```bash
T=configs/scenarios/thermal_foraging/connectomeppo_small_continuous2d_thermal_klinotaxis
H=configs/scenarios/foraging/connectomeppo_small_continuous2d_fick_adaptive_klinotaxis_hard350

# 0. the instrument check — V.3's committed panel, re-scored through the harness that made it
uv run python scripts/analysis/wiring_premise.py \
  --manifest campaigns/wiring-premise-hard/_manifest.txt --out /tmp/v3_recheck.json

# 1. pilot on DISJOINT seeds 101-104 — 32 runs. No verdict is read at this seed count.
uv run python scripts/run_campaign.py \
  --config ${T}_t20.yml --config ${T}_rewired_null_t20.yml \
  --config ${T}_frozen_t20.yml --config ${T}_rewired_null_frozen_t20.yml \
  --config ${H}.yml --config ${H}_rewired_null.yml \
  --config ${H}_frozen.yml --config ${H}_rewired_null_frozen.yml \
  --seeds 101-104 --runs 3000 --output-dir campaigns/wiring-fresh-rewiring-pilot \
  -- --theme headless --track-experiment

# 2. the registered panel — 256 runs on seeds 65-96, fresh to both prior panels
uv run python scripts/run_campaign.py \
  --config ${T}_t20.yml --config ${T}_rewired_null_t20.yml \
  --config ${T}_frozen_t20.yml --config ${T}_rewired_null_frozen_t20.yml \
  --config ${H}.yml --config ${H}_rewired_null.yml \
  --config ${H}_frozen.yml --config ${H}_rewired_null_frozen.yml \
  --seeds 65-96 --runs 3000 --output-dir campaigns/wiring-fresh-rewiring \
  -- --theme headless --track-experiment

# 3. score — the committed harness reads it; this driver builds the manifest and reports the branch
uv run python scripts/analysis/wiring_fresh_rewiring.py \
  --campaign campaigns/wiring-fresh-rewiring \
  --out docs/experiments/logbooks/supporting/065-wiring-fresh-rewiring/fresh_rewiring.json
```
