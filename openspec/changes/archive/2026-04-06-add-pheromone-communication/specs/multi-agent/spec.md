## MODIFIED Requirements

### Requirement: Pheromone Emission in Multi-Agent Step Loop

The MultiAgentSimulation orchestrator SHALL emit pheromones during the step loop based on agent events.

#### Scenario: Food-Marking Pheromone on Consumption

- **GIVEN** an agent consumes food at position (10, 10) at step 50
- **WHEN** the food competition phase resolves
- **THEN** a FOOD_MARKING pheromone source SHALL be added at (10, 10) with emission_step=50
- **AND** the emitter_id SHALL match the consuming agent's agent_id

#### Scenario: Alarm Pheromone on Predator Damage

- **GIVEN** an agent at position (15, 20) takes predator damage at step 30
- **WHEN** `apply_predator_damage_for(agent_id)` returns damage > 0
- **THEN** an ALARM pheromone source SHALL be added at (15, 20) with emission_step=30
- **AND** the emitter_id SHALL match the damaged agent's agent_id

#### Scenario: Pheromone Field Update Per Step

- **GIVEN** an active multi-agent episode
- **WHEN** each step completes
- **THEN** `env.update_pheromone_fields(current_step)` SHALL be called once
- **AND** expired sources SHALL be pruned

#### Scenario: Pheromones Disabled

- **GIVEN** a multi-agent config without pheromones enabled
- **WHEN** the step loop runs
- **THEN** no emission or field update calls SHALL be made

### Requirement: Pheromone Sensing in Multi-Agent Perception

The orchestrator SHALL include pheromone data in agent BrainParams during the perception phase.

#### Scenario: Per-Agent Pheromone Sensing

- **GIVEN** a multi-agent episode with pheromones enabled
- **WHEN** BrainParams is constructed for each agent
- **THEN** pheromone concentration/gradient SHALL be computed at that agent's position
- **AND** different agents at different positions SHALL receive different pheromone readings
