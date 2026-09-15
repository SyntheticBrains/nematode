# Tasks

## 1. The arms

- [x] 1.1 **No new configs.** All eight already exist and are the ones V.1 and V.3 ran:
  `..._thermal_klinotaxis_t20{,_rewired_null,_frozen,_rewired_null_frozen}.yml` and
  `..._fick_adaptive_klinotaxis_hard350{,_rewired_null,_frozen,_rewired_null_frozen}.yml`. Confirm by
  test that each is byte-unchanged since its panel ran — the thermal four against **`431a4689`**
  (`feat/wiring-premise-contrast`, V.1) and the hard350 four against **`48ba778c`**
  (`feat/wiring-premise-difficulty`, V.3). A replication that silently edited an arm would be measuring
  something else.
- [x] 1.2 Confirm `rewire_seed` is unset in all four rewired configs, so each seed's rewiring derives
  from its run seed and the arms pair — the coupling this change exists to work around, not remove.
- [x] 1.3 Record which seeds each prior panel used, from the committed records rather than from
  memory: V.1 seeds **1–64** on the thermal cell, V.3 seeds **1–32** on hard350, so **V.3's rewirings
  are a subset of V.1's** and the two positives share their nulls. A test asserts 65–96 is disjoint
  from both.

## 2. Harness

- [x] 2.1 `scripts/analysis/wiring_fresh_rewiring.py`, a **manifest builder and branch reporter, and
  nothing else**: build the `<cell> <arm> <seed> <out>` manifest and call the committed
  `wiring_premise.py`, which drives `connectome_structure_efficiency` itself. **Do not re-implement**
  the ≥ 20% minimum (`MIN_EFFICIENCY_GAIN`), the per-cell verdicts (`verdict()`), the censoring guard
  (`CROSSING_FLOOR`) or the efficiency-arm mapping (`EFFICIENCY_ARMS`) — the harness owns all four, and
  an earlier draft of this task would have duplicated them.
- [x] 2.2 **Both committed harnesses are READ-ONLY, and a test asserts it.** A replication varies the
  evidence and holds the reading fixed; a modified instrument would let "the instrument changed"
  compete with "the effect is not there", and those are not separable after the fact.
- [x] 2.3 **Report the harness's verdict names, with V.1's prose branches mapped onto them** —
  `specific_wiring` → replicates, `below_min_effect` → same direction below the minimum,
  `degree_statistics` → does not replicate — the last carrying that the first positive is **withdrawn
  on the record rather than defended**. A parallel vocabulary is how two records come to disagree about
  the same run, so the mapping is stated and the harness's names are what the record reports.
- [x] 2.3a **The harness's other three verdicts are registered readings too**, because they are live:
  `saturated` (the cell cannot answer on this axis — what the klinotaxis cell returned in V.1's pilot,
  and **not** a replication failure), `no_learning` (a gate failed, the contrast is uninterpretable),
  and `insufficient_seeds`. Separately, a contrast the harness flags **materially censored** below its
  80% crossing floor — the case L.0 met on `hard350` — is likewise **not** evidence against the
  original result. Each is reported as itself rather than collapsed into "does not replicate".
- [x] 2.4 **Per cell, never pooled across cells.** A split is reported as a split with the pooled
  reading withheld, as evidence about the scope of block V's generalisation.
- [x] 2.5 The committed comparators carried as fields: V.1's **+35.4%** pooled over 64 seeds with its
  per-panel spread (+46.4%, +32.6%, +31.8%), and V.3's **+23.5%** over 32 — so a near-miss on the
  thermal cell is read against V.1's own per-panel variability rather than as a clean failure.
- [x] 2.6 The power arithmetic as a field, as L.0 registered it: 32 pairs, 22/32 needed, **79.2%**
  against the comparator's win-rate midpoint, labelled **sign-test planning figures and not the
  registered procedure's power**.
- [x] 2.7 Tests for 2.1–2.6, including a fixture in each branch and one split across cells.

## 3. The stop clauses

- [ ] 3.1 A **pilot on disjoint seeds** before any registered seed: both cells run, the rewired arms
  differ from their wild-type partners, and the frozen floors sit where a no-learning policy sits.
  Seeds 101–104, as every pilot in this programme has used — and **disjoint from 65–96**.
- [x] 3.2 Re-score V.3's committed panel through **`wiring_premise.py`** — the harness that produced
  the committed figure — and confirm **+23.5%** on `episodes_to_30pct_success` at seeds 1–32. A mismatch
  means the instrument no longer reproduces the record it is replicating, and nothing here is
  comparable. This checks the *instrument*, so it must not be routed through the new driver's own
  reading.
- [x] 3.3 `launch.md` committed before anything runs, carrying the branches, the comparators, the power
  arithmetic and the honest prior.

## 4. Campaign

- [ ] 4.1 256 runs: two cells × four arms × seeds **65–96** at 3000 episodes, with `--track-experiment`.
- [ ] 4.2 Per-seed CSV, the per-cell tables and the branch under `supporting/065-wiring-fresh-rewiring/`.

## 5. The record

- [ ] 5.1 Logbook 065: both cells' tables, the gates and priors in **every** branch, the branch per
  cell, and the coupling stated — that fresh seeds vary rewiring *and* initialisation together, and
  what that does and does not separate.
- [ ] 5.2 The experiments index row and `CHANGELOG.md`.
- [ ] 5.3 The tracker's V.4 entry.
- [ ] 5.4 **If it does not replicate on either cell**: V.1, V.3, the 7a shipment record
  ([059](../../../docs/experiments/logbooks/059-7a-shipment.md)) and the roadmap each rest on the
  figure, and each is corrected in the same PR. The first positive is withdrawn on the record rather
  than defended, and the phase's citable results are restated — which after L.0 would leave the
  systematic negative as the only one.
- [ ] 5.5 **If it replicates**: the caveat closes, and the record states that block V's positive is now
  independent in rewiring — while remaining explicit that rewiring and initialisation still vary
  together, so the stricter question is open and unregistered.
