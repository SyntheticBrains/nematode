# Launch record: C.4, the fidelity table, and C.1 (v3)'s predicted mutants

**Part A registered 2026-11-27**, change `add-fidelity-table`. It is committed before `grade` runs, and fixes the one new reading, its rules and seeds, and the table's rows. A verdict registered here is never re-read under a different rule; anything else is reported beside it as description.

## Part A: the rules

### The authority

- **The C row of the roadmap's Phase 10 tracker:** C.4 is the fidelity table, with C.2 against C.1 as a secondary contrast. The tracker's C.4 task asks for the controllers and the negatives on held-out assays and the mutant set, under the per-controller encoding table registered before any fit.
  - **A stated deviation:** the negatives have no mutant encodings, so they enter on the wild type alone, as B v4 read them.
- **The maintainer's decisions:**
  - **2026-11-25 ([Logbook 133](../../133-c2-fit.md)):** C.2 enters as a failing in-sample row, and C.3 does not enter.
  - **2026-11-27 ([Logbook 136](../../136-c1-v4-fit.md)):** C.1 enters through its v3 answer.
  - **2026-11-27, for this change:** C.1's held-out cell stays "not read", and C.2 against C.1 is description, with no verdict on which is closer.
- **The encoding table** (`supporting/113-c1-fit/encodings.md`, registered 2026-10-26, bound). It fixes each controller's encodings, what each mutant is graded on, and the `unidentified` rule.

### The one new reading: C.1 (v3)'s predicted mutants

C.1 (v3) was fitted to the wild type alone ([Logbook 121](../../121-c1-v3-fit.md)), so its tph-1 and tax-4 rows are predictions. They have never been graded.

- **The restarts:** 0, 4 and 6 of `supporting/121-c1-v3-fit/fit.json` (bound). These are the three whose answers pass v3's wild type in at least 3 of 5 registered populations, recorded in the tracker on 2026-11-03. Restart 6 is the chosen answer.
- **The body:** `benchmark_grade.ARMS["mlp"]`, the pooled on-food reversal C.1 (v3) was fitted and graded on (six configs, bound).
- **The populations:** each restart on Logbook 121's registered populations, animals 525,000,000 + 200·g + i, g < 5, i < 200.
- **The simulation, in two passes:**
  - first the wild type alone on its 15 registered populations, for the reproduction check below, with no mutant simulated;
  - only if it reproduces, every population again: the wild type on every v4 assay, then tph-1 and tax-4 on Scheer's small lawn on the same animals.
  - The second pass's registered wild type must equal the first's exactly, or no prediction is read.
- **The encodings** (the table's, with launch record 121's extension):
  - tph-1 sets `a_intake` = `b_intake` = 0.
  - tax-4 sets `kappa_weathervane` = `kappa_off_food` = `beta_reversal` = `beta_omega` = 0.
  - pdfr-1 is not encodable.
- **The grading:** each mutant's differences from its own wild type, on v4's mutant bands at n = 200 (`bands_v4.json`, bound; v2's, carried byte-identical), at |z| ≤ 2.24 in the resolved direction.
  - tph-1 is graded on its roaming fraction, direction more. Its leaving is reported beside.
  - tax-4 is graded on its roaming fraction, direction less, and on its leaving by magnitude.
- **A restart's verdict:** a population passes a mutant when all of the mutant's graded statistics pass in that same population. A restart passes the mutant when at least 3 of its 5 registered populations pass it.
- **The prediction:** the `unidentified` rule (`c1_fit.mutant_verdict`) across restarts 0, 4 and 6:
  - `unidentified` if they disagree;
  - otherwise their shared verdict.
  - C.1 (v3)'s v4 wild-type failure (0 of 5, [Logbook 124](../../124-benchmark-v4.md)) is stated beside it.

### The reproduction check, before any prediction is read

The wild type must reproduce its committed grades on the same animals:

- for every restart and population, the v3 grade's D within 10⁻⁹ of its committed `population_D`, and its v3 wild-type pass equal to its committed `population_passes`;
- for restart 6, the v4 D within 10⁻⁹ of B v4's read of it and its v4 pass equal (`supporting/124-benchmark-v4/read.json`, bound);
- each of the 15 registered populations present exactly once, and B v4's read made at the same base.

If any value differs, no mutant is simulated. The grade record holds the wild type, the mismatches and no prediction, `table` refuses, and the reading returns to the maintainer.

### Beside the reading (description)

- **More populations:** restart 6 on 10 more populations, animals 555,000,000 + 200·g + i, g < 10, for its pass rates over 15, as C.2's.
- **tph-1's leaving,** its difference in exits per on-lawn hour, reported against its band centre and B.4's single held-out reading of the same difference, both from `supporting/110-benchmark-controls/controls.json` (bound; [Logbook 110](../../110-benchmark-controls.md)). Neither is read again.

### The table

`table` writes `fidelity_table.json` and `fidelity_table.md` from these committed records, all bound, and the grade record. It copies each recorded value, and computes only medians and counts over a record's populations.

- **C.1 (v3):**

  - its body and fitted families, and its recovery control (`121-c1-v3-fit/recovery.json`);
  - its wild type on v3 (3 of 5) and on v4 (0 of 5, by family, with D and D_fit), from Logbook 124's read;
  - its predicted mutants, from the grade record, with pdfr-1 not encodable;
  - its held-out cell: **not read**.

- **C.2:** its registered answer, from `133-c2-fit/fit.json`:

  - its body and fitted families, and F.0b-ii's failed gate;
  - its wild type on its first 5 registered populations, by family, with D and D_fit, and its pass rates over 15;
  - its fitted mutants;
  - its held-out reading, as read in its fit.

- **C.3:** not run in Phase 10.

- **The negatives:** B v4's eight, each as read in its controls.

- **The real-worm controls:** N2's unread dates (in two halves) and PD1074's held-out dates, both on activity, as B v4 read them, with W's minimum.

- **C.1 (v4)'s pilot answer:** from `136-c1-v4-fit/pilot.json` (bound), as description on discarded seeds.

- **C.2 against C.1:** description. The two rows are set side by side, with the conditions stated:

  - C.2's recovery control failed;
  - the bodies differ;
  - C.2's mutants are fitted and C.1's predicted;
  - the populations are not paired.

  No verdict is read on which controller is closer.

### What C.4 receives (the verdict map)

| C.1's prediction | Its C.4 row |
|---|---|
| `pass` | a predicted pass, read with C.1 (v3)'s v4 wild-type failure stated |
| `fail` | a predicted fail, with the same statement |
| `unidentified` | `unidentified`: the restarts that pass v3's wild type disagree |
| no prediction (reproduction failed) | `table` refuses, so no table is written; the logbook reports the mismatch, and C.1's mutant cells stay empty until the maintainer decides |

Each mutant is read separately. pdfr-1 is "not encodable" for C.1 whatever the reading. C.2's rows are its registered verdict, unchanged.

### Seeds

| Block | Animals | Use |
|---|---|---|
| Registered populations | 525,000,000 + 200·g + i, g < 5 | Logbook 121's, reused, so the mutants share their wild type's animals |
| More populations | 555,000,000 + 200·g + i, g < 10 | description |
| Smoke | 99,870,000 + 200·g + i, g < 2 | discarded |

Every block is disjoint from the fit, selection and recovery blocks of C.1 (v3), from C.1's v4 refit and its probes (641 M and 647–700 M), and from C.2 (700 M and above). A test enumerates them.

### The guards

- `grade` refuses unless this Part A is committed with every bound record unchanged. It also refuses if the package sources, the dependency lock and project files, or the analysis scripts it imports (`fidelity_table.CODE_PATHS`) differ from Part A's commit, or hold an untracked file. Part A's commit is the last commit that changed Part A's own text.
- `grade` and `table` each write their record once.
- `table` also refuses a grade record without predictions, and writes nothing if either of its files exists.
- Before writing, `table` checks every count it can recompute against its record (C.2's verdict, the negatives' failures, C.1 (v3)'s v4 passes), and refuses if any disagrees.

### The bound records

| Record | SHA-256 (first 16) |
|---|---|
| `supporting/121-c1-v3-fit/fit.json` | `8e66efed6b64f3c5` |
| `supporting/121-c1-v3-fit/recovery.json` | `abe621a90612a3cf` |
| `supporting/133-c2-fit/fit.json` | `e6cd923b599a5f2f` |
| `supporting/124-benchmark-v4/read.json` | `bece13586ac0e909` |
| `supporting/110-benchmark-controls/controls.json` | `8e0b0299e76a2ae9` |
| `supporting/136-c1-v4-fit/pilot.json` | `1436b1dced91e9fe` |
| `supporting/113-c1-fit/encodings.md` | `d3aa1af865dc74fa` |
| `data/benchmark/bands_v4.json` | `355177a62e3fed4d` |
| `data/benchmark/bands_v3.json` | `8af38429a6929141` |
| `data/benchmark/classifier.json` | `e4a059de3d5b826d` |
| `data/locomotion/primitives.json` | `8356a09a1078d887` |
| the six `mlpppo_…_internal_s1_primitives_assay_*.yml` configs | listed in `grade.json` |

### Disclosures

Everything seen before this part was written:

- **The committed records the table reads,** among them C.2's verdict, C.1 (v3)'s v4 failure and B v4's negatives.
- **Logbook 136's separation probe** (647 M, discarded) ran restart 6 on the speed-following arm with `r_dwell_on_food` at 0.0224 per s. That is a different body and one more parameter. tph-1 passed in 0 of 10 there (Δ −0.19) and tax-4 in 0 of 10 (Δ exits +12.6). Logbook 136's tax-4 probe (641 M) ran the same point. Both make a failing prediction likely.
- **This change's first smoke** (99,870,000, discarded) ran restart 6 on two populations on the registered body, and printed that both mutants failed in both. The revised smoke prints only whether the wild type repeats identically.
- **A reproduction preflight** ran restarts 0, 4 and 6 on their 15 registered populations, the wild type alone with no mutant simulated. All 15 reproduce their committed v3 D and passes, and restart 6 its v4 D (0 mismatches). The script is `probes/reproduction_preflight.py.txt` and its output `probes/reproduction_preflight.txt`. It read only values already committed.
- **An independent review** of the change, the script, its tests and this record, before it was committed. Its findings are fixed here: the restart rule is stated unambiguously; the wild type runs alone first; the checks are stricter; and the tests cover the simulation path, the git guards and every copied cell.
- **Not seen:** any mutant on the registered populations, or the 10 more populations.

### Cost

15 wild-type evaluations, then 25 with the mutants, each about 40 s: a few minutes on 16 workers. The first smoke ran in 53 s.
