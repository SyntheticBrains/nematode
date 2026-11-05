## ADDED Requirements

### Requirement: An optional on-food dwelling reversal rate

`BatchedMechanistic` SHALL accept `r_dwell_on_food`, used only when given: a dwelling animal's reversal rate wherever food contact is above 0, in place of `r_dwell`, before the klinokinesis, edge and search terms. Off food, and while roaming, the state's rate SHALL apply. Without it, every action SHALL be byte-identical to the model's without the option.

#### Scenario: Off food nothing changes

- **WHEN** an animal is off food
- **THEN** its actions SHALL be those of the model without `r_dwell_on_food`

#### Scenario: On food only dwelling reversals change

- **WHEN** an animal is on food
- **THEN** its run speed, curvature, omega and delta hazards SHALL be unchanged, and its reversal hazard SHALL scale from `r_dwell` to `r_dwell_on_food` while dwelling and stay unchanged while roaming
