## ADDED Requirements

### Requirement: A wiring effect traced to a model artefact is re-read on the corrected model before mechanism work builds on it

When a registered reading attributes a wiring contrast to an artefact of how the model represents a
connection type, mechanism work on that contrast SHALL NOT register until the contrast has been re-read
on the corrected model. The re-reading SHALL be its own registration. It SHALL use its own seeds,
disjoint from the reading that found the artefact and from the pilot that chose the correction. It SHALL
have a minimum and a verdict map fixed before any run, and its gates SHALL be evaluated before launch.
The selection that chose the corrected model SHALL NOT read the contrast.

#### Scenario: Mechanism work registers on the corrected model without the re-reading

- **WHEN** a mechanism panel registers on a corrected gap model
- **AND** no registered contrast between the wild type and the null has been read on that model
- **THEN** the registration SHALL be refused until the re-reading has read

#### Scenario: The re-reading reuses the selection pilot's seeds

- **WHEN** the re-reading's seeds overlap the seeds of the pilot that chose the corrected model
- **THEN** the registration SHALL be refused

#### Scenario: The re-reading finds no effect on one cell

- **WHEN** the re-reading reads no wiring effect on one cell
- **AND** the phase registers a STOP condition over more than one cell
- **THEN** the STOP condition SHALL stay open until every cell it names has read on the corrected model
