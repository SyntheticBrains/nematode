# Launch record: C.1 refitted on benchmark v3, with its 25-parameter model

**Part A registered 2026-11-02**, change `fit-c1-v3`. It is committed before the recovery's preflight, the recovery and every fit. It fixes decisions 1–7 of the change's design with every threshold and seed. Part B, added after the recovery and before any fit, records the recovery's result; it changes no rule. A verdict registered here is never re-read under a different rule. Anything else is reported beside it as description.

## Part A: the rules

### The model (decisions 1–2)

C.1 (v3) is `BatchedMechanistic` with C.1's 20 parameters on F.0a's box (`npe_recovery.PRIOR`), plus five more, each uniform on its range:

| Parameter | What it does | Range | From |
|---|---|---|---|
| `v_off_food` | the run speed wherever food contact is 0, whatever the state | 0.005–0.20 mm/s | the state speeds' prior |
| `sigma_off_food` | the curvature noise wherever food contact is 0 | 5–60 °/s | the curvature priors' |
| `speed_log_sd_dwell`, `speed_log_sd_roam` | each state's run speed × exp(sd·z), z redrawn every 10 s, so the state's speed is the median | 0–1.5 | v2's reachability box |
| `kappa_off_food` | the weathervane gain wherever food contact is 0 | −50 to 50 | the weathervane gain's prior |

- **No range comes from search data** (D38).
- **The mechanisms were chosen by the probes below,** on all five families, including local search and chemotaxis. So C.1's held-out test is its mutant predictions, not its wild-type fit.
- **Without the new parameters the model is unchanged:** all 55 identity jobs are byte-identical (`identity.json`).

### The mutant encodings (decision 3)

They extend `113-c1-fit/encodings.md`, which stays byte-identical and bound:

| Mutant | C.1 (v3) |
|---|---|
| tph-1 | a_intake = b_intake = 0, as registered |
| tax-4 | κ_w = **κ_off** = β_r = β_o = 0: the registered encoding, with the off-food gain removed too, since it is chemosensory steering |
| pdfr-1 | not encodable, as registered |

The off-food speed, the off-food curvature and the two spreads are kinematic, not chemosensory, and every mutant keeps them. The `unidentified` rule applies as registered. `c1_v3_fit.mutant_parameters` implements this table.

### The reduced recovery (decision 4)

**The truths** are the two 25-parameter probe answers that reach the wild type:

- t = 0, `probes/basin_warm.json`;
- t = 1, `probes/basin_boxA.json`.

Both records are bound.

**`recovery-preflight`**, run once. Every D in it is the capped v3 D on v3's scales and family map (`predictive_distance_v2(capped=True)`). It reads, for each truth:

- **the observation:** the truth simulated at `540_000_000 + 200·(100·t) + i`;
- **the floor:** the median D from the observation to re-simulations 1–10, at `540_000_000 + 200·(100·t + q) + i`;
- **the oracle:** the same over re-simulations 11–20;
- **the prior D:** the median D from the observation to 200 prior draws of the 25-parameter box. The draws are `default_rng(5781)`, simulated at `541_000_000 + 200·j + i`;
- **the box centre's D:** over 10 re-simulations at `542_000_000 + 200·q + i`;
- **gain** = (prior D − D) / (prior D − floor).

It passes only if all three hold:

- each truth's prior D exceeds its floor by at least 1.0;
- the oracle's median gain is at least **0.85**;
- the box centre's median gain is at most **0.25**.

Otherwise no recovery fit runs, and the change returns to the maintainer.

**`recovery`**, only once the preflight has passed:

- **The fits:** each truth fitted twice (fit = 2t + j).
  - sep-CMA from the box centre, σ₀ 0.2, population 16, bounds [0, 1], CMA seeds 5782 + fit;
  - ranked by the definedness-first score against the truth's observation, on v3's scales;
  - fit animals at `530_000_000 + 200·k + i`, with k = 10,000·fit + 16·(n − 1) + m;
  - the doubly guarded plateau: the plateau rule, σ ≤ σ₀, and no median of 1,000 or more in the last 40 generations. Otherwise the cap of 330.
- **The score:**
  - per truth, the restart with the lower final objective (the mean of its last 20 medians) is chosen;
  - its answer is re-simulated 10 times, at k = 10,000·fit + 8,000 + q;
  - its gain is read on the capped D;
  - the other restart's gain, the stops and the parameter errors are reported. The errors are read after mapping the answer's dwelling and roaming parameters onto the truth's in whichever order is closer.
- **The gate:** **the median best-restart gain over the two truths (their mean) must be at least 0.75.** Otherwise no registered fit runs until the maintainer decides.

### The registered fit (decision 5)

- **The objective:** the ranking score against v3's band centres at n = 200, for 200 fresh animals per evaluation, over v3's four assays through its measurement models, without presence.
- **sep-CMA:**
  - **12 restarts** r = 0–11 from the box centre;
  - σ₀ 0.2, population 16, bounds [0, 1], CMA seeds 5769 + r;
  - the answer is `xfavorite`, clipped to the box.
- **The stop:** the doubly guarded plateau, or **the cap of 330**. That is ⌈1.5 × 218⌉, rounded up, from the longest 25-parameter probe plateau.
- **Selection:**
  - every plateaued restart is graded with presence on **5** selection populations, `526_000_000 + 200·(5r + s) + i`;
  - the restart passing most is chosen, ties going to the lower median D, then the lower r;
  - a capped restart cannot be chosen. If none plateaus, no verdict is read, and the run returns to the maintainer.
- **The verdict:**
  - every restart is graded with presence on 5 registered populations, `525_000_000 + 200·g + i`;
  - a population passes when every family passes: each statistic at |z| ≤ 3.0381, and local search and chemotaxis present;
  - **the chosen restart passes the wild type when at least 3 of its 5 populations pass.**
- **The verdict map:**
  - **pass:** C.1 (v3) enters C.4;
  - **fail:** the failing statistics are the diagnosis, and the roadmap's C.1 risk row ("the primitives or the benchmark are wrong; fixed before C.2, or the phase ships B and the diagnosis") goes to the maintainer.
- **Reported, not gated:**
  - every restart's stop, final score, selection and registered grades, median D over 10 re-simulations (`500_000_000 + 200·(10,000·r + 8,000 + q) + i`), parameters and reversal share against Roberts' 17.3%;
  - the restart spread.
- **The odds:**
  - a faithful controller passes 3 of 5 with probability 0.977 (Logbook 120);
  - from the probes, crudely, about 1 box-centre restart in 3 finds the passing basin;
  - the one that did passed 3 of 5, and the warm start passed 4 of 5;
  - so the verdict is expected to pass with a probability of roughly 70–90%.

### Seeds (decision 6)

| Seeds | Use |
|---|---|
| 500,000,000 + 200·k + i, k = 10,000·r + 16·(n − 1) + m; answers at 10,000·r + 8,000 + q | the fit (to 523,601,999) |
| 525,000,000 + 200·g + i, g < 5 | the registered populations |
| 526,000,000 + 200·(5r + s) + i, s < 5 | the selection populations |
| 530,000,000 + 200·k + i, k = 10,000·fit + …, answers at 10,000·fit + 8,000 + q | the recovery's fits |
| 540,000,000 + 200·(100·t + q) + i, q ≤ 20 | the truths' observations and re-simulations |
| 541,000,000 + 200·j + i, j < 200; draws `default_rng(5781)` | the prior draws |
| 542,000,000 + 200·q + i, q < 10 | the box centre's re-simulations |
| CMA 5769–5780; 5782–5785 | the fit's restarts; the recovery's fits |
| 85,000,000 | the recovery gate's preflight, discarded |
| 86,000,000–86,970,999; CMA 5990 | the smoke, discarded |
| 87,000,000–98,970,999; CMA 5760–5768 | the probes and the sensitivity tests, discarded |

Every registered block is disjoint from every other, which the tests check, and from every earlier scheme. Each restart's block spans 2,000,000 animals, so the blocks start at least that far apart.

### Disclosures

Seen before this record was committed, all on discarded seeds (`probes/`, with each script as run):

- **The probes,** each one sep-CMA restart against v3's ranking score, with C.1b's guarded stop, its answer graded with presence on 5 populations:

  | Probe | Model | Start | Stop | Passing | D | What fails |
  |---|---|---|---|---|---|---|
  | `readiness_probe` (98 M, CMA 5760) | 20 parameters | centre | 86 | 0 | 3.42–3.56 | dwelling speeds z +7, dwelling bouts, weathervane |
  | `probe_warm` (97 M, 5761) | 20 | C.1b's answer, σ 0.1 | 103 | 0 | 2.49–2.70 | the three new statistics (reverses 47%) |
  | `probe_wide` (96 M, 5762) | 20, 5 bounds widened | centre | 163 | 0 | 3.09–3.54 | dwelling speeds, lawn bouts |
  | `probe_structural` (95 M, 5763) | 22 | centre | 136 | 0 | 2.50–2.74 | Chen's onset rate, flips, weathervane |
  | `probe_spread` (93 M, 5764) | 24 | centre | 146 | 0 | 1.28–1.46 | weathervane z +4 |
  | `probe_full25` (90 M, 5765) | 25 | centre | 133 | 0 | 2.50–2.81 | kinematics, lawn bouts |
  | `basin_warm` (89 M, 5766) | 25 | the 24's answer, σ 0.1 | 126 | **4** | 0.85–1.04 | one roaming-bout length |
  | `basin_boxA` (88 M, 5767) | 25 | centre | 218 | **3** | 1.08–1.33 | a roaming-bout length, in 2 |
  | `basin_boxB` (87 M, 5768) | 25 | centre | 172 | 0 | 1.44–1.53 | the two roaming speeds |

- **The sensitivity tests** (3 paired populations each):

  - `search_sensitivity` (94 M): the search term barely matters;
  - `kappa_sensitivity` (92 M): the on-food weathervane gain keeps the worm on the lawn;
  - `kappa_off_sensitivity` (91 M): an off-food gain at a quarter of it fixes the weathervane, and 1 of 3 populations passes.

- **The recovery gate's preflight** (`recovery_gate_preflight.json`, 85 M) read the quantities `recovery-preflight` will re-read on fresh seeds:

  | | Truth t = 0 | Truth t = 1 |
  |---|---|---|
  | floor | 0.39 | 0.44 |
  | prior D | 7.08 | 7.10 |
  | oracle gain | 0.995 | 0.991 |
  | box-centre gain | 0.018 | 0.017 |

  Scored as recovery fits, the probes' answers gain:

  - 0.93–0.94 for the other passing answer;
  - 0.85–0.94 for the near basin (box B);
  - 0.665–0.666 for the far basin.

- **Chen's protocol:** naive worms were "directly removed from food and washed for three times before testing" (Chen et al. 2025, Methods), as the simulated assay assumes.

- **The smoke** (86 M, CMA 5990) ran every stage's path.

- **The maintainer's decisions at approval:** 12 restarts and 5 selection populations, taken before this record and seeing no registered reading.

- **Not seen:** the recovery's registered simulations (530–542 M), every registered fit, selection and registered population.

### Cost

At about 33 s per generation on 16 workers (the probes'):

| Stage | Estimate |
|---|---|
| `recovery-preflight` | about 252 simulations, about 10 min |
| `recovery` | 4 fits, about 6–8 h (the probes plateaued at 126–218) |
| `run` | 12 restarts, about 18 h typical and 36 h at most |
| `select`, `grade` | up to 120 simulations, about 5 min |

### The guards

- `recovery-preflight` and `recovery` refuse unless this record is committed and unchanged, with `113-c1-fit/encodings.md` and the two truths' records byte-identical to their versions in that commit.
- `recovery-preflight` runs once, and `recovery` refuses until the preflight has passed.
- `run`, `select`, `grade` and `read` also refuse unless this record carries Part B, with `recovery_preflight.json` and `recovery.json` bound and the recovery passing.
