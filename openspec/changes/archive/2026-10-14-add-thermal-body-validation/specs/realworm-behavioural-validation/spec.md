## MODIFIED Requirements

### Requirement: Behaviour capture in the evaluation harness

The evaluation harness SHALL, when asked, record each step's behaviour as the simulation's behaviour
capture does, so that the bias-curve harness can read evaluation episodes. It SHALL take the capture
modality, food or thermotaxis, as the simulation's sensing configuration does, and record that
modality's drive.

#### Scenario: An evaluation writes a behaviour capture the bias-curve harness reads

- **WHEN** a run is evaluated with behaviour capture
- **THEN** it SHALL write a capture the bias-curve harness loads, one series per episode

#### Scenario: A thermal evaluation records the thermal drive

- **WHEN** a run on a thermotaxis cell is evaluated with behaviour capture and the thermotaxis
  modality
- **THEN** each step's captured drive SHALL be the setpoint error `−|T − Tc|` at the sensed position,
  not the food concentration
