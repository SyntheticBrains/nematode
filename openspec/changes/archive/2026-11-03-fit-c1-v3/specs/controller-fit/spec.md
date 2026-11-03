## ADDED Requirements

### Requirement: C.1 is refitted on benchmark v3 with its 25-parameter model

C.1 (v3) SHALL be C.1's 20 parameters on F.0a's box plus `v_off_food`, `sigma_off_food`, `speed_log_sd_dwell`, `speed_log_sd_roam` and `kappa_off_food`, on ranges set without search data. It SHALL be fitted to the real wild type on benchmark v3 by sep-CMA from the box centre, with 12 restarts, the definedness-first ranking score against v3's bands at n = 200, the doubly guarded plateau and a cap of 330. Selection on 5 separate populations per plateaued restart SHALL choose the restart, and the verdict SHALL be at least 3 of 5 registered populations passing with presence at |z| ≤ 3.0381.

#### Scenario: A capped restart is not chosen

- **WHEN** a restart reaches generation 330 without the guarded plateau
- **THEN** it SHALL NOT be chosen, though its registered grades SHALL be reported

#### Scenario: Seed blocks do not overlap

- **WHEN** the fit's, the answers', the selection's, the registered populations' and the recovery's animal seeds are enumerated
- **THEN** no animal seed SHALL appear in two of them

### Requirement: A reduced recovery gates the v3 fit

Before any registered v3 fit, sep-CMA SHALL recover C.1 (v3) at two known truths, the two passing probe answers. Each truth SHALL be fitted twice from the box centre on its observation, and the better restart by final objective SHALL be scored by its gain fraction, (prior D − D) / (prior D − floor). The truths' oracle SHALL reach a median gain of at least 0.85 and the box centre at most 0.25 before any recovery fit runs. The fit SHALL run only if the median gain over the two truths is at least 0.75.

#### Scenario: The recovery falls short

- **WHEN** the two truths' median gain is 0.70
- **THEN** no registered fit SHALL run until the maintainer decides

### Requirement: The probes are the v3 fit's pilot

The discarded-seed probes recorded in launch record 121 SHALL set the v3 fit's cap and stand as its reachability evidence, and no further pilot SHALL run. Every probe SHALL be disclosed in Part A with its seeds and outcome.

#### Scenario: The cap comes from the probes

- **WHEN** the cap is registered
- **THEN** it SHALL be at least 1.5 × the longest plateau generation among the 25-parameter probes

### Requirement: The 25-parameter model's mutant encodings

C.1 (v3)'s mutant encodings SHALL be C.1's registered encodings, except that tax-4 SHALL also set `kappa_off_food` to 0. The four kinematic parameters SHALL be kept by every mutant. The extension SHALL be registered in Part A, before any fit, and the original table SHALL stay byte-identical.

#### Scenario: tax-4 removes both weathervane gains

- **WHEN** C.1 (v3)'s tax-4 encoding is built
- **THEN** both `kappa_weathervane` and `kappa_off_food` SHALL be 0, and the off-food speed, curvature and spreads SHALL be unchanged
