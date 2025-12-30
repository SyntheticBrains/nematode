# Spec Delta: Episode Rendering

## ADDED Requirements

### Requirement: Episode renderer encapsulates visualization logic

The system SHALL provide an `EpisodeRenderer` class that handles all rendering logic including frame display, screen clearing, and render timing decisions.

#### Scenario: Render frame during episode

**GIVEN** an EpisodeRenderer with an environment
**WHEN** render_frame(env, step=5, max_steps=100, text="Episode 1") is called
**THEN** the renderer SHALL display the environment grid with the provided text
**AND** SHALL include step information in the visualization

#### Scenario: Show only last frame mode

**GIVEN** an EpisodeRenderer with show_last_frame_only=True
**WHEN** render_if_needed() is called on step 50 of 100
**THEN** the renderer SHALL NOT display any output
**WHEN** render_if_needed() is called on step 100 of 100 (last step)
**THEN** the renderer SHALL display the final frame

#### Scenario: Show all frames mode

**GIVEN** an EpisodeRenderer with show_last_frame_only=False
**WHEN** render_if_needed() is called on any step
**THEN** the renderer SHALL display the frame for that step

### Requirement: Episode renderer handles screen clearing

The EpisodeRenderer SHALL manage screen clearing logic to prevent visual clutter during rendering.

#### Scenario: Clear screen between frames

**GIVEN** an EpisodeRenderer in show all frames mode
**WHEN** rendering consecutive frames
**THEN** the renderer SHALL clear the screen before each frame
**AND** SHALL use appropriate terminal control sequences

#### Scenario: No screen clearing in last frame only mode

**GIVEN** an EpisodeRenderer in show_last_frame_only mode
**WHEN** rendering the final frame
**THEN** the renderer SHALL display the frame without clearing previous terminal output

### Requirement: Episode renderer supports headless mode

The EpisodeRenderer SHALL support disabling all rendering for headless execution (e.g., during testing or batch runs).

#### Scenario: Headless rendering mode

**GIVEN** an EpisodeRenderer with enabled=False
**WHEN** render_if_needed() is called
**THEN** no output SHALL be produced
**AND** no terminal control sequences SHALL be executed
