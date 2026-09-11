# Tasks

## 1. The statistics module

- [x] 1.1 `scripts/analysis/l4_mixture_statistic.py`: the family's three members over paired
  per-seed values — **F** (competent-fraction discordance, exact binomial on discordant pairs at
  the committed `COMPETENT_THRESHOLD`), **L** (each arm's mean over its own competent seeds, seeded bootstrap CI on the difference, one-sided), **W** (the existing `paired_seed_wilcoxon_bootstrap`) — every member one-sided at α 0.05, BH-FDR across them. F generalises `l4_panel3.discordance` from its wt/rn naming rather than re-deriving it; the threshold is imported from `l4_panel2`, not restated. `l4_panel.py` is not edited (it would be an import cycle).
- [x] 1.2 The outcome map as the registered direction table over (F, L) ∈ {+, −, 0}², every cell named, `mixed_response` on opposed significant contrasts or an explicit split, returning the branch and what it licenses.
- [x] 1.3 Tests: a level-only panel is not `no_effect`; a frequency-only panel is not `level_only` **and L is not significant on it**; opposed significant contrasts are `mixed_response`, not `degrades`;
  a seed competent in one arm only enters that arm's level; a panel where either arm has no competent seed reports L undefined rather than null; every cell of the direction table is reached by a constructed panel; `mixed_response` fires on a split and **not** on a panel that merely misses
  significance; the map's order is the registered one.

## 2. The graded metric

- [x] 2.1 Read plateau-tail mean foods from the committed per-seed tables (`foods`, already in every CSV) and run L and W on it as a parallel, separately corrected reading, over the competence the primary metric defines — no foods threshold is chosen.
- [x] 2.2 Test: an arm at its full-clear floor with a graded difference is reported by the graded
  family and not lost.

## 3. The re-read

- [x] 3.1 Read each committed table (040, 041, 042, 044, 045, 046, 047, 050, 052) from its
  committed per-seed CSV, not from a campaign directory.
- [x] 3.2 Assays (045, 050, 052) are contrasted against each seed's own committed comparator; 044
  is a prior sweep. Each is reported in its own protocol's terms, with no pooling across protocols.
- [x] 3.3 Every committed verdict is carried beside its re-read, unchanged, with the record stating
  which is the verdict. A disagreement is reported, not resolved.
- [x] 3.4 Test: the re-read of a committed table reproduces that table's committed W result, and the re-read of 042 reproduces its committed R2 discordance, so the re-read is reading the same numbers the record was scored on and F is the test that was registered.

## 4. Records and close-out

- [x] 4.1 Records under `supporting/053-l4-mixture-statistic/`: `family.json`, `re-read.csv`,
  `details.md` — including the n = 8 power limit, stated rather than discovered.
- [x] 4.2 `docs/architectures.md` or `docs/experiments/README.md`: the convention that a panel on a
  bimodal outcome is read with both components and a graded metric.
- [x] 4.3 `CHANGELOG.md`; tracker (I.2) and roadmap.
- [x] 4.4 State what is now unblocked: I.4's ladder re-read takes this as input, and any further
  panel or assay is registered with this family.
