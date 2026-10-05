# A.3 — the boundary-preserving null and the registered hop predictor: registration and launch

**Registered 2026-10-06, before any scored run and before the predictor is computed.** Change:
`add-boundary-null`.

## The question

A degree-preserving rewiring can route an injected sensor straight onto a readout motor neuron. The wild
type has no motor neuron one hop from a food sensor; the current null has about nine, the chemical-only
null about eight ([Logbook 071](../../071-operating-point-surface.md),
[074](../../074-null-strength-control.md)). A.2 found block V's advantage depth-critical, with the null
winning at a settling depth of two, and tied that to these shortcuts after the fact. Park
(arXiv:2609.39248) found nulls that keep the sensory-motor boundary erase a connectome's apparent
difference in an embodied fly. Two registered readings follow.

## Part 1 — the hop predictor (committed data, no training)

**Data.** The 48 committed hard350 PPO seeds at block V's point under the current null: A.2's centre
(seeds 161–176, the centre gap recovered from each level row as gap minus interaction, checked equal
across rows) and A.6's `full` level (seeds 305–336). Pooled; both are the same cell, learner, point and
null.

**Statistic.** Spearman's rho between each seed's **current-null one-hop count** — readout motor neurons
one hop from a food sensor over the propagating graph of chemical synapses and gap junctions, the
quantity Logbook 071 measured — and the same seed's **wild-type minus null `auc_success` gap**.

- **Predicted direction: negative.** More shortcuts, a faster null, a smaller wild-type lead.
- **Minimum: |rho| ≥ 0.3.**
- **Test:** two-sided permutation p (20,000 permutations, seeded), 80% bootstrap interval.

| verdict | condition |
|---|---|
| **predicts** | p < 0.05, rho ≤ −0.3 |
| **opposite** | p < 0.05, rho ≥ +0.3 |
| **below_minimum** | p < 0.05, absolute rho below 0.3 |
| **no_prediction** | p ≥ 0.05, interval inside (−0.3, +0.3) |
| **unresolved** | anything else |

**Spread and power.** The one-hop count runs 4–14 across these seeds (sd 2.4), looked at before
registration without the gaps. At n = 48 a true rho of −0.3 is detected about 55% of the time; the test
is registered at that power, not sized up to it. Reach within two hops saturates (37–39 of 39) and is
not used.

## Part 2 — the boundary-null panel

**The null.** `wiring: rewired_boundary_held`: the chemical-only null (gap junctions and autapses held)
with every chemical edge out of a boundary sensory neuron or into a boundary motor neuron held at the
wild type's. Sensory side: the 17 neurons any projection injects into (ASE, AWC, AWA, AFD, ASH, ASI, ALM,
AVM, PLM). Motor side: the 39 the readout pools (VB, DB, VA, DA). 544 edges held, 3,165 interior edges
rewired. Its one-hop count and reach within two hops equal the wild type's (0 and 26).

| property | chemical-only null | boundary-preserving null |
|---|---|---|
| chemical in- and out-degree | preserved | preserved |
| edges out of injected sensors, into readout motors | rewired | **the wild type's** |
| interior chemical edges | rewired | rewired |
| gap junctions, autapses | the wild type's | the wild type's |
| one- and two-hop sensor-to-motor routes | manufactured | **the wild type's** |

**The panel.** Wild type, chemical-only null and boundary null, each learning and frozen, on hard350
under PPO at block V's committed point (edge-order draw, pooled readout, depth 4). **Seeds 641–768
(128)**, fresh, 3,000 episodes, 768 runs. Two levels, `chemical` and `boundary`, share the wild-type runs.

**Two readings, corrected together** (BH-FDR per metric; B.1c's `classify`):

```text
interaction = gap(boundary) − gap(chemical)     paired by seed
lead        = gap(boundary)
```

**Primary `auc_success`**, episodes beside. **Minimum 0.0143**: 2/3 of A.6's committed lead over the
chemical-only null at this point (+0.0215). A.6's own minimum (0.041) exceeds the whole base.

| state | `interaction` | `lead` |
|---|---|---|
| `move_null` | **boundary** — holding the boundary shrinks the lead by at least the minimum | **null_leads** |
| `move_wt` | **shortcuts_helped_null** — the lead grows: the shortcuts had helped the null | **lead_remains** |
| `no_move` | **interior** if the lead over the boundary null excludes zero above, else **no_gap_to_attribute** | **no_lead** |
| `below` | **below_minimum** | **lead_below_minimum** |
| `unresolved` | **unresolved** | **unresolved** |

**Sensitivity** from A.6's committed PPO spread (interaction sd 0.061, chemical-gap sd 0.064): at 128
seeds the detectable effects are 0.0135 and 0.0140, at the minimum. Achieved spread reported beside,
never used to re-read.

**Gates.** Each learning arm beats its frozen floor; the two learning arms do not both reach 90%.

## Before launch

- **Gate preflight** ([preflight.json](preflight.json)): the boundary null learning and frozen on seeds
  305–308 (A.6's, disjoint from this band), read with A.6's wild-type and chemical-only runs. Both levels
  readable: chemical 76.4% / 73.4% on 32 seeds, boundary 67.6% / 76.4% on 4, floors at or near 0.
- **Dry run**: `boundary_null.score` ran end to end on those 4 seeds; no state or verdict was printed.
- **Cost**: A.6's own 16-way times on these configs, 16.4 min learning and 10.5 min frozen, give
  **about 10.8 hours**. The pilot ran 8-way and faster, so it is not the estimate.

## Launch

```bash
uv run python scripts/analysis/hop_predictor.py --out docs/experiments/logbooks/supporting/079-boundary-null/hop-predictor.json

uv run python scripts/campaigns/gate_preflight.py --panel boundary_null \
  --logs campaigns/a6-ppo/logs --logs campaigns/a3-boundary-pilot/logs
cfgs=(); for s in $(uv run python -c "import sys;sys.path.insert(0,'scripts/analysis');import boundary_null as b;print(' '.join(b.LEVELS_BY_STEM))"); do
  cfgs+=(--config configs/scenarios/foraging/$s.yml); done
uv run python scripts/run_campaign.py "${cfgs[@]}" --seeds 641-768 --runs 3000 --workers 16 \
  --output-dir campaigns/a3-boundary \
  -- --theme headless --track-experiment --no-detailed-export --no-file-log

uv run python scripts/analysis/boundary_null.py --campaign campaigns/a3-boundary \
  --out-dir build/a3 --out build/a3/control.json --csv build/a3/per-seed.csv
```

## Retention (A.0)

Committed: this record, the preflight, the predictor JSON, the panel's analysis JSON and per-seed CSV.
Archived off-repo: the pilot's and the campaign's raw logs.
