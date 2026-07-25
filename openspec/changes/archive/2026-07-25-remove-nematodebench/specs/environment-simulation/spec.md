# environment-simulation Specification

## MODIFIED Requirements

### Requirement: Rendering Symbol Verification

The system SHALL use consistent, documented symbols for predators across all rendering themes to maintain visual clarity and documentation accuracy.

The `Benchmark Category Name Verification` scenario is removed — it existed to prevent doc-code drift between the documented category strings and `benchmark/categorization.py`, which is deleted. The two rendering-symbol scenarios are unchanged and remain live.

#### Scenario: Emoji Theme Predator Symbol

- **GIVEN** rendering with theme mode "emoji"
- **WHEN** a predator is rendered
- **THEN** the predator SHALL be displayed as spider emoji: 🕷️
- **AND** this SHALL match the documented spec exactly
- **AND** the symbol SHALL be visually distinct from food (🍎) and agent (🪱)

#### Scenario: ASCII Theme Predator Symbol

- **GIVEN** rendering with theme mode "ascii"
- **WHEN** a predator is rendered
- **THEN** the predator SHALL be displayed as hash symbol: #
- **AND** this SHALL match the documented spec exactly
- **AND** the symbol SHALL be visually distinct from food and agent ASCII symbols

## REMOVED Requirements

### Requirement: Predator-Enabled Benchmark Categories

**Reason**: Specified the `predator_{small,medium,large}/{quantum,classical}` category strings derived by `benchmark/categorization.py`, deleted with the NematodeBench leaderboard system. The grid-size thresholds it referenced (small ≤ 20×20, medium ≤ 50×50, large > 50×50) existed only to bucket submissions; they are not used by the simulation itself. The requirement was also duplicated in `benchmark-management`, where it is removed under this same change.

**Migration**: None. Predator configuration and per-experiment predator metrics are captured by `experiment-tracking` § Predator Experiment Metadata Capture, which is unaffected.
