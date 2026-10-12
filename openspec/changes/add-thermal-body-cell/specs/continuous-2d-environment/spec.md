## ADDED Requirements

### Requirement: Thermotaxis through the kinematic body

Under the kinematic body model, the continuous environment SHALL sample temperature, and the agent
its temperature derivative and lateral comparison, at the body's head, the same position its food
sensing reads.

#### Scenario: Temperature is read at the head

- **WHEN** a kinematic body with thermotaxis enabled has moved
- **THEN** the temperature the agent senses SHALL equal the environment's temperature at the body's
  head
