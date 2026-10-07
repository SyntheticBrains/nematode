## ADDED Requirements

### Requirement: Emmons 2024 as a connectome source

The brain SHALL accept `connectome_source: emmons_2024_hermaphrodite` beside the default
`cook_2019_hermaphrodite`, and SHALL build every wiring, null and measured prior on the chosen source.
The default SHALL be byte-identical to the brain before this option existed.

#### Scenario: Emmons differs from Cook in four gap pairs only

- **WHEN** wild-type brains are built on both sources at one seed
- **THEN** their chemical masks and chemical weights SHALL be identical
- **AND** their gap-junction buffers SHALL differ only in the entries of the four gap pairs Emmons 2024
  adds or strengthens
