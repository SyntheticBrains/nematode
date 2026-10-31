## ADDED Requirements

### Requirement: The measurement model is faster with identical output

`measure` SHALL return, for every step, the same field values as before. Positions SHALL gain the same noise, and lost seconds SHALL hold NaN positions. Each noise splice SHALL make the same random draws and choose the same continuation, the first minimum in row-major order over its (candidate, offset) grid.

#### Scenario: A measured track is the same

- **WHEN** a track long enough to splice, with lost seconds, is measured by the new and the reference code from the same seed
- **THEN** every field of every step SHALL be equal

#### Scenario: Stored simulations re-simulate exactly

- **WHEN** stored v2 simulations are re-run after the change
- **THEN** every graded statistic and D SHALL equal the stored record exactly
