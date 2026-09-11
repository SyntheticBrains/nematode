# Tasks

## 1. The statistics module

- [ ] 1.1 `scripts/analysis/l4_mixture_statistic.py`: the family's three members over paired
  per-seed values — **F** (competent-fraction discordance, exact binomial on discordant pairs at
  the committed `COMPETENT_THRESHOLD`), **L** (mean among seeds competent in either arm, paired,
  one-sided), **W** (the existing `paired_seed_wilcoxon_bootstrap`) — with BH-FDR across them.
  Imports the threshold from `l4_panel2` rather than restating it.
- [ ] 1.2 The ordered outcome map, `mixed_response` included, returning the first matching branch
  and what it licenses.
- [ ] 1.3 Tests: a level-only panel is not `no_effect`; a frequency-only panel is not `level_only`;
  a pair with exactly one competent arm enters L; an all-incompetent panel reports L undefined
  rather than null; `mixed_response` fires on a split and **not** on a panel that merely misses
  significance; the map's order is the registered one.

## 2. The graded metric

- [ ] 2.1 Read plateau-tail mean foods from the committed per-seed tables (`foods`, already in
  every CSV) and run the same family on it as a parallel, separately corrected reading.
- [ ] 2.2 Test: an arm at its full-clear floor with a graded difference is reported by the graded
  family and not lost.

## 3. The re-read

- [ ] 3.1 Read each committed table (040, 041, 042, 044, 045, 046, 047, 050, 052) from its
  committed per-seed CSV, not from a campaign directory.
- [ ] 3.2 Assays (045, 050, 052) are contrasted against each seed's own committed comparator; 044
  is a prior sweep. Each is reported in its own protocol's terms, with no pooling across protocols.
- [ ] 3.3 Every committed verdict is carried beside its re-read, unchanged, with the record stating
  which is the verdict. A disagreement is reported, not resolved.
- [ ] 3.4 Test: the re-read of a committed table reproduces that table's committed W result, so the
  re-read is reading the same numbers the record was scored on.

## 4. Records and close-out

- [ ] 4.1 Records under `supporting/053-l4-mixture-statistic/`: `family.json`, `re-read.csv`,
  `details.md` — including the n = 8 power limit, stated rather than discovered.
- [ ] 4.2 `docs/architectures.md` or `docs/experiments/README.md`: the convention that a panel on a
  bimodal outcome is read with both components and a graded metric.
- [ ] 4.3 `CHANGELOG.md`; tracker (I.2) and roadmap.
- [ ] 4.4 State what is now unblocked: I.4's ladder re-read takes this as input, and any further
  panel or assay is registered with this family.
