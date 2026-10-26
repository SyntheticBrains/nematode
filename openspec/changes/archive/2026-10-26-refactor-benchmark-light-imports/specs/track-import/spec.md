## MODIFIED Requirements

### Requirement: Grading imported tracks does not run the simulator

Grading imported tracks SHALL run no simulation, read no simulator config, and call no simulator code. Importing the grader and the track importer SHALL load no `torch` and no brain, agent or environment module, so the benchmark can be used without the optional PyTorch extra.

#### Scenario: Grading with the simulator disabled

- **WHEN** the batched evaluator and the simulation function are replaced by functions that raise, and a small CSV is graded
- **THEN** the grade SHALL complete

#### Scenario: Importing the grader in a fresh interpreter

- **WHEN** a fresh interpreter imports `quantumnematode.benchmark.grading` and `quantumnematode.benchmark.track_import`
- **THEN** no `torch` module, and no `quantumnematode.brain`, `quantumnematode.agent` or `quantumnematode.env` module, SHALL be loaded
