"""The per-step behavioural record, in a module with no package imports.

The benchmark reads it, so it must load without the brain, the agent or the environment.
"""

from __future__ import annotations

from dataclasses import dataclass


# A plain slotted dataclass (not a pydantic ``BaseModel`` like ``SimulationResult``) is deliberate:
# this is a lightweight per-step record appended in the capture loop, and it round-trips through
# ``dataclasses.asdict`` / ``BehaviourStep(**d)`` for the behaviour-capture JSON — per-step pydantic
# validation would add cost with no benefit for an internally-produced value.
@dataclass(slots=True)
class BehaviourStep:
    """One per-step continuous behavioural record for real-worm chemotaxis validation.

    Captured opt-in (``SensingConfig.capture_behaviour``) at the agent step where the continuous
    state and the sensing are computed together. The two klinotaxis bias curves are computed as a
    pure function of a ``list[BehaviourStep]``: turn-rate vs ``dc_dt`` (klinokinesis) and
    curving-rate vs the bearing ``wrap(grad_dir - heading_rad)`` (klinotaxis weathervane).

    Attributes
    ----------
    step : int
        Episode step index.
    x, y : float
        Continuous position (mm).
    heading_rad : float
        Continuous heading (radians, world frame).
    concentration : float
        Local food concentration at the sensing position.
    dc_dt : float
        Temporal derivative of the food concentration (STAM).
    grad_dir : float
        Local food-gradient direction (radians, world frame), captured live (the food field mutates
        during a foraging run, so it cannot be recomputed post-hoc).
    grad_strength : float
        Local food-gradient magnitude (weak-gradient steps can be down-weighted / excluded).
    speed_signed : float | None
        Recorded only when the environment allows reversal: the displacement since the previous
        captured step projected on the heading, in mm, negative when the worm backed up; zero at
        an episode's first captured step, which follows no move. Also recorded under locomotor
        primitives, whose reversal bouts back the worm up. ``None`` otherwise, and then omitted
        from the export, so captures without reversal are unchanged.
    satiety, intake, on_lawn : float | None, float | None, bool | None
        Recorded only under the lawn food model: satiety, the quality-weighted food eaten on the
        move into this step, and whether the worm is inside a lawn. ``None`` otherwise, and then
        omitted from the export.
    primitive : str | None
        Recorded only under locomotor primitives: the primitive that moved the worm into this step
        (``run``, ``reversal``, ``omega`` or ``delta``; ``run`` at an episode's first step). For
        diagnosis only: behaviour is graded from trajectories, never from this label. ``None``
        otherwise, and then omitted from the export.
    phase : str | None
        Recorded only under an assay protocol: the name of the phase this step ran in. ``None``
        otherwise, and then omitted from the export.
    """

    step: int
    x: float
    y: float
    heading_rad: float
    concentration: float
    dc_dt: float
    grad_dir: float
    grad_strength: float
    speed_signed: float | None = None
    satiety: float | None = None
    intake: float | None = None
    on_lawn: bool | None = None
    primitive: str | None = None
    phase: str | None = None
