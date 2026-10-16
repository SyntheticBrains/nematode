## ADDED Requirements

### Requirement: Collapse incidence is reported beside every long learning curve

Every registered panel whose runs exceed 3,000 episodes SHALL report collapse incidence per arm under
one per-seed definition — a trailing-200-episode peak of the success rate; onset at the first episode
where the 100-episode mean falls below half of that peak and stays below for at least 300 episodes;
censored at the horizon — with Wilson intervals, Kaplan–Meier time-to-collapse censored at the horizon,
and the primary metric computed both to episode 3,000 and to the horizon. A collapsed seed SHALL stay
in the primary reading. Best-so-far and best-block statistics SHALL NOT be used as comparison metrics.
The definition SHALL be implemented once, in `scripts/analysis/collapse.py`, and cited by every such
panel.

#### Scenario: A long panel reports without the statistic

- **WHEN** a logbook reports a panel of more than 3,000 episodes per run
- **THEN** it SHALL carry each arm's collapse incidence with its interval, time-to-collapse, and the
  primary metric to 3,000 episodes beside the full horizon

#### Scenario: A collapsed seed is dropped

- **WHEN** a reading excludes seeds because they collapsed
- **THEN** the exclusion SHALL be restated as a described subset beside the registered reading, which
  keeps every seed
