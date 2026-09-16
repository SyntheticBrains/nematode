# Does the wiring advantage survive rewirings that have never been used? (V.4)

## Why

After [L.0](../../../docs/experiments/logbooks/064-l4-frozen-features.md), block V's learning-speed
advantage is **the only surviving wiring result in this project**. Every other regime has come back
negative: the wiring is endpoint-inert under gradient learning
([034](../../../docs/experiments/logbooks/034-connectome-structure-controls.md)), actively harmful
under local rules that write it ([R.2](../../../docs/experiments/logbooks/063-l4-eprop.md)), and
indistinguishable from a degree-matched shuffle as fixed features (L.0). What stands is
[V.1](../../../docs/experiments/logbooks/057-wiring-premise-contrast.md)'s **+35.4%** and
[V.3](../../../docs/experiments/logbooks/058-wiring-premise-difficulty.md)'s **+23.5%** off
time-to-competence under PPO — one of the two results 7a shipped on, and the result Phase 7's synthesis
will be written around.

It has one open caveat, registered by 058 and never run. **Every wiring result so far draws its rewired
graphs from run seeds 1–64, and V.3 reuses 1–32 exactly** — so V.3's 32 rewirings are a *subset* of
V.1's 64. The two positives are not independent of each other: they share their null graphs. V.2 bounds
the concern without removing it, having regenerated all 64 rewirings and scored them on four graph
properties fixed before looking, finding **nothing** that predicts learning time.

So the question is narrow and it is the last one standing:

> *Does the advantage hold on rewirings that have never been used?*

## What changes

- **Both panels re-run on seeds 65–96** — fresh in rewiring, task and initialisation together, since
  `rewire_seed` is unset and each seed's rewiring derives from its run seed.

  | panel | cell | config family | prior seeds | here |
  |---|---|---|---|---|
  | V.1 | thermal, `target_foods_to_collect: 20` | `..._thermal_klinotaxis_t20` + 3 arms | 1–64 | **65–96** |
  | V.3 | hard food-only, `max_steps: 350` | `..._hard350` + 3 arms | 1–32 | **65–96** |

  Four arms each — `wt_ppo`, `rn_ppo`, `wt_frozen`, `rn_frozen` — **256 runs**. **No new configs**: all
  eight already exist and are the ones V.1 and V.3 ran.

- **Scored by the committed harness, unchanged, and it needs no help.** `wiring_premise.py`'s
  registered family already carries `thermal` (V5–V8) and `hard_food` (V9–V12) with exactly these arm
  names; it **imports and drives** `connectome_structure_efficiency` itself; and it carries the
  registered minimum, the peak-axis gates, the efficiency primary and the censoring floor. **Neither
  is modified and no sibling module is written** — that is the point of a replication. A sibling would reintroduce the
  risk that a difference in *reading*, rather than a difference in the world, explains a failure to
  replicate.

- **The verdict vocabulary is the committed harness's**, with V.1's prose branches mapped onto it
  rather than run in parallel. `wiring_premise.py` already assigns a per-cell verdict in order, with
  the gates read before the contrast, and already owns the registered **≥ 20%** minimum as
  `MIN_EFFICIENCY_GAIN`:

  | harness verdict | V.1's prose branch | what follows |
  |---|---|---|
  | `specific_wiring` | **replicates** | the caveat closes; block V's positive is independent in rewiring and the synthesis rests on it |
  | `below_min_effect` | **same direction, below the minimum** | a real but smaller effect, with the shrinkage named. **It licenses the follow-up, not the claim** |
  | `degree_statistics` | **does not replicate** | **the registered panel is reported as not holding, and the first positive is withdrawn on the record rather than defended** |
  | `saturated` | — | the cell cannot answer on this axis, as the klinotaxis cell could not. Not a replication failure |
  | `no_learning` | — | a gate failed: the arms did not learn, so the contrast is uninterpretable |
  | `insufficient_seeds` | — | fewer than the harness's minimum paired seeds survived; the panel is not scored |

  The last three are **registered here because they are live**, not for completeness: the harness also
  flags a contrast as **materially censored** below its `CROSSING_FLOOR` of 80% of seeds crossing the
  threshold, and L.0 has just met asymmetric censoring on `hard350`. A censored or saturated fresh
  panel is **not** evidence against the original result, and saying so in advance is what stops it
  being read as one.

- **What happens if the two cells disagree is fixed in advance**, because with one caveat and two
  panels it would otherwise be decided under the result: each cell is read on its own against the same
  three branches, and a **split** — one cell replicating and the other not — is reported as a split,
  with the pooled reading withheld. Block V's own claim is that the effect generalises from "a foraging
  cell under thermal pressure" to "a foraging cell hard enough to discriminate", so one cell holding
  and the other not is evidence about the *scope* of that generalisation and is recorded as such, never
  resolved toward whichever cell is more convenient.

Out of scope: any change to the two committed harnesses; the stricter experiment that holds
initialisations fixed while varying rewirings alone, which needs `rewire_seed` set explicitly and is a
different question from the one 058 registered; and L.1, which follows this.

## Capabilities

**Modified**: `plasticity-evaluation` — a result whose controls are drawn from the same seeds as its
runs states that coupling, and a replication of it uses the original instrument unmodified.

## Impact

- New: `scripts/analysis/wiring_fresh_rewiring.py` — a **manifest builder and branch reporter, and
  nothing else**. Recon shortened this: `wiring_premise.py` already drives
  `connectome_structure_efficiency` itself, already owns the ≥ 20% minimum, already emits per-cell
  verdicts with the gates read first, and already guards the censoring case. Re-implementing any of
  that here would be the duplication this change exists to avoid. Plus records under
  `supporting/065-wiring-fresh-rewiring/` and Logbook 065.
- Edited: the experiments index, `CHANGELOG.md`, the tracker (V.4), and — if it does not replicate —
  the roadmap, V.1, V.3 and the 7a shipment record, each of which rests on the figure.
- Compute: **256 runs** — two cells × four arms × 32 seeds at ~600–1800 s — about **10h** at the
  measured parallelism, plus a pilot on disjoint seeds.
