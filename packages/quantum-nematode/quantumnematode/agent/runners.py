"""Episode execution strategies for the quantum nematode agent."""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np

from quantumnematode.brain.arch import ClassicalBrain
from quantumnematode.dtypes import (  # noqa: TC001 - used at runtime
    AgentPath,
    FoodHistory,
    GridPosition,
)
from quantumnematode.env import Direction
from quantumnematode.env.oxygen import OxygenZone
from quantumnematode.env.temperature import TemperatureZone
from quantumnematode.logging_config import logger
from quantumnematode.report.dtypes import TerminationReason

if TYPE_CHECKING:
    from quantumnematode.agent import QuantumNematodeAgent, RewardConfig
    from quantumnematode.brain.actions import ActionData
    from quantumnematode.env.associative_memory import AssociativeMemoryTask
    from quantumnematode.env.bit_memory import BitMemoryTask

# Chance accuracy for the binary bit-memory cue-match (a per-episode success flag is set
# when the episode's cue-match rate beats this; the analysis harness uses the logged rate).
_BIT_MEMORY_CHANCE = 0.5
# Chance accuracy for the associative-memory binary readout (same 50% two-alternative baseline).
_ASSOCIATIVE_MEMORY_CHANCE = 0.5


@dataclass
class EpisodeData:
    """Data collected during a single simulation episode.

    Attributes
    ----------
    steps : int
        The number of steps taken in the episode.
    rewards : float
        The total reward accumulated during the episode.
    foods_collected : int
        The number of food items collected during the episode.
    distance_efficiencies : list[float]
        The distance efficiencies recorded during the episode.
    satiety_history : list[float]
        The satiety levels at each step (for dynamic foraging environments).
    health_history : list[float]
        The health (HP) levels at each step (when health system is enabled).
    temperature_history : list[float]
        The temperature at each step (when thermotaxis is enabled).
    predator_encounters : int
        Number of times agent entered predator detection radius.
    successful_evasions : int
        Number of times agent exited predator detection radius without dying.
    in_danger : bool
        Whether agent is currently within predator detection radius.
    """

    steps: int
    rewards: float
    foods_collected: int
    distance_efficiencies: list[float]
    satiety_history: list[float]
    health_history: list[float]
    temperature_history: list[float]
    oxygen_history: list[float]
    predator_encounters: int = 0
    successful_evasions: int = 0
    in_danger: bool = False


@dataclass
class EpisodeResult:
    """Result of processing a single simulation episode.

    Attributes
    ----------
    agent_path : AgentPath
        The path taken by the agent in the episode.
    termination_reason : TerminationReason
        The reason for episode termination, if applicable.
    food_history : FoodHistory | None
        Food positions at each step (DynamicForagingEnvironment only).
    """

    agent_path: AgentPath
    termination_reason: TerminationReason
    food_history: FoodHistory | None = None


class EpisodeRunner(Protocol):
    """Protocol for episode execution strategies.

    Episode runners encapsulate different modes of executing simulation episodes
    (e.g., standard single-trajectory, many-worlds branching). They access agent
    components and helper methods to execute the episode logic.
    """

    def run(
        self,
        agent: QuantumNematodeAgent,
        reward_config: RewardConfig,
        max_steps: int,
        **kwargs: dict[str, Any],
    ) -> EpisodeResult:
        """Execute an episode using this runner's strategy.

        Parameters
        ----------
        agent : QuantumNematodeAgent
            The agent instance containing brain, environment, and components.
        reward_config : RewardConfig
            Configuration for the reward system.
        max_steps : int
            Maximum number of steps for the episode.
        **kwargs : Any
            Additional runner-specific parameters.

        Returns
        -------
        StepResult
            The result of the episode execution, including path and termination reason.
        """
        ...


class StandardEpisodeRunner(EpisodeRunner):
    """Runs a standard episode using step-by-step execution.

    This runner orchestrates the main episode loop by directly accessing agent
    components and helper methods. The runner delegates to the agent's internal
    components (FoodConsumptionHandler, SatietyManager) and helper methods for
    episode execution.

    Notes
    -----
    The implementation directly accesses agent private members (_food_handler,
    _satiety_manager, helper methods) as part of the episode execution architecture.
    This provides a clean separation between episode orchestration (runner) and
    agent state management.
    """

    def __init__(self) -> None:
        """Initialize the standard episode runner."""
        # One-time guard: warn if a continuous-2D env runs with a discrete brain
        # (continuous-action heads not yet implemented) so the fallback isn't silent.
        self._warned_discrete_on_continuous = False

    def _terminate_episode(  # noqa: PLR0913
        self,
        agent: QuantumNematodeAgent,
        params: Any,  # noqa: ANN401
        reward: float,
        *,
        success: bool,
        termination_reason: TerminationReason,
        learn: bool = True,
        update_memory: bool = True,
        food_history: FoodHistory | None = ...,  # type: ignore[assignment]
    ) -> EpisodeResult:
        """Handle episode termination with consistent cleanup.

        Parameters
        ----------
        agent : QuantumNematodeAgent
            The agent instance.
        params : Any
            Brain parameters from the current step.
        reward : float
            Current accumulated reward.
        success : bool
            Whether the episode completed successfully.
        termination_reason : TerminationReason
            The reason for episode termination.
        learn : bool
            Whether to call brain.learn (default True).
        update_memory : bool
            Whether to call brain.update_memory (default True).
        food_history : FoodHistory | None
            Override for food_history in the result. Defaults to agent.food_history.

        Returns
        -------
        EpisodeResult
            The result of the episode execution.
        """
        if learn and isinstance(agent.brain, ClassicalBrain):
            agent.brain.learn(params=params, reward=reward, episode_done=True)

        if update_memory:
            agent.brain.update_memory(reward)

        agent.brain.post_process_episode(episode_success=success)
        # Predator lifecycle hook — once per episode at termination.
        # Mirrors MultiAgentSimulation.run_episode. Predators don't have
        # an agent-style success notion (their fitness signal is the
        # per-predator counters); pass None.
        for predator in agent.env.predators:
            predator.brain.post_process_episode(episode_success=None)
        agent._metrics_tracker.track_episode_completion(
            success=success,
            steps=agent._episode_tracker.steps,
            reward=agent._episode_tracker.rewards,
            foods_collected=agent._episode_tracker.foods_collected,
            distance_efficiencies=agent._episode_tracker.distance_efficiencies,
            predator_encounters=agent._episode_tracker.predator_encounters,
            successful_evasions=agent._episode_tracker.successful_evasions,
            termination_reason=termination_reason,
        )
        # Log temporal sensing diagnostics (once per episode, if STAM active)
        if agent._stam is not None and len(agent._stam) > 0:
            stam_state = agent._stam.get_memory_state()
            # Log per-channel weighted means and derivatives dynamically
            means_parts = []
            deriv_parts = []
            for idx, ch_def in enumerate(agent._active_channels):
                means_parts.append(f"{ch_def.name}={stam_state[idx]:.3f}")
                deriv = agent._stam.compute_temporal_derivative(idx)
                deriv_parts.append(f"{ch_def.name}={deriv:.4f}")
            logger.info(
                f"Temporal sensing summary: "
                f"STAM entries={len(agent._stam)}, "
                f"channels={agent._stam.num_channels}, "
                f"weighted_means=[{', '.join(means_parts)}], "
                f"derivatives=[{', '.join(deriv_parts)}], "
                f"action_entropy={stam_state[-1]:.3f}",
            )

        resolved_food_history = agent.food_history if food_history is ... else food_history
        return EpisodeResult(
            agent_path=agent.path,
            termination_reason=termination_reason,
            food_history=resolved_food_history,
        )

    def _apply_brave_foraging_bonuses(
        self,
        agent: QuantumNematodeAgent,
        reward: float,
    ) -> tuple[str, float]:
        """Apply brave foraging bonuses for collecting food in danger zones.

        Returns
        -------
        tuple[str, float]
            (brave_msg for logging, updated reward)
        """
        brave_msg = ""

        # Temperature brave foraging bonus (discomfort zones)
        if agent.env.thermotaxis.enabled:
            zone = agent.env.get_temperature_zone()
            if zone in (TemperatureZone.DISCOMFORT_COLD, TemperatureZone.DISCOMFORT_HOT):
                brave_bonus = agent.env.thermotaxis.reward_discomfort_food
                if brave_bonus > 0:
                    reward += brave_bonus
                    agent._episode_tracker.track_reward(brave_bonus)
                    brave_msg = f" [Brave foraging bonus: +{brave_bonus}]"
                    logger.debug(
                        f"Brave foraging bonus: +{brave_bonus} ({zone.value} zone)",
                    )

        # Oxygen brave foraging bonus (danger zones)
        if agent.env.aerotaxis.enabled:
            o2_zone = agent.env.get_oxygen_zone()
            if o2_zone in (OxygenZone.DANGER_HYPOXIA, OxygenZone.DANGER_HYPEROXIA):
                o2_bonus = agent.env.aerotaxis.reward_discomfort_food
                if o2_bonus > 0:
                    reward += o2_bonus
                    agent._episode_tracker.track_reward(o2_bonus)
                    brave_msg += f" [O2 brave bonus: +{o2_bonus}]"

        return brave_msg, reward

    def _handle_food_collection(
        self,
        agent: QuantumNematodeAgent,
        reward_config: RewardConfig,
        params: Any,  # noqa: ANN401
        reward: float,
    ) -> tuple[EpisodeResult | None, float]:
        """Handle food collection after agent moves.

        Returns
        -------
        tuple[EpisodeResult | None, float]
            A termination result if victory was achieved (or None), and the
            updated reward value.
        """
        if not agent.env.reached_goal():
            return None, reward

        food_result = agent._food_handler.check_and_consume_food()
        if not food_result.food_consumed:
            return None, reward

        agent._episode_tracker.track_food_collection(
            distance_efficiency=food_result.distance_efficiency,
        )

        # Log food collection with distance efficiency and health restoration
        dist_eff_msg = ""
        if food_result.distance_efficiency is not None:
            dist_eff = food_result.distance_efficiency
            dist_eff_msg = f" (Distance efficiency: {dist_eff:.2f})"

        health_msg = ""
        if food_result.health_restored > 0:
            health_msg = (
                f", HP +{food_result.health_restored:.1f} "
                f"to {agent.env.agent_hp:.1f}/{agent.env.health.max_hp:.1f}"
            )
            # Apply healing reward (learning signal for recovering health)
            healing_reward = reward_config.reward_health_gain
            reward += healing_reward
            agent._episode_tracker.track_reward(healing_reward)

        # Apply brave foraging bonuses
        brave_msg, reward = self._apply_brave_foraging_bonuses(agent, reward)

        logger.info(
            f"Food #{agent._episode_tracker.foods_collected} collected! "
            f"Satiety restored by {food_result.satiety_restored:.1f} to "
            f"{agent.current_satiety:.1f}/{agent.max_satiety}{health_msg}{dist_eff_msg}{brave_msg}",
        )

        # Check for victory condition (collected target number of foods)
        if agent.env.has_collected_target_foods(agent._episode_tracker.foods_collected):
            logger.info(
                "Successfully completed episode: collected target of "
                f"{agent.env.foraging.target_foods_to_collect} food!",
            )
            return self._terminate_episode(
                agent,
                params,
                reward,
                success=True,
                termination_reason=TerminationReason.COMPLETED_ALL_FOOD,
                update_memory=False,
            ), reward

        return None, reward

    def _check_predator_contact(
        self,
        agent: QuantumNematodeAgent,
        reward_config: RewardConfig,
        params: Any,  # noqa: ANN401
        reward: float,
        *,
        after_movement: bool = False,
    ) -> tuple[EpisodeResult | None, float]:
        """Check for predator contact and apply damage.

        Parameters
        ----------
        after_movement : bool
            Whether this check happens after predators moved (affects log message).

        Returns
        -------
        tuple[EpisodeResult | None, float]
            A termination result if the agent died (or None), and the
            updated reward value.
        """
        if not agent.env.is_agent_in_damage_radius():
            return None, reward

        damage = agent.env.apply_predator_damage()
        log_prefix = "Predator stepped on agent!" if after_movement else "Predator contact!"
        logger.info(
            f"{log_prefix} Took {damage:.1f} damage. "
            f"HP: {agent.env.agent_hp:.1f}/{agent.env.health.max_hp:.1f}",
        )

        # Apply damage penalty (learning signal for taking damage)
        damage_penalty = -reward_config.penalty_health_damage
        reward += damage_penalty
        agent._episode_tracker.track_reward(damage_penalty)

        # Check if health depleted
        if not agent.env.is_health_depleted():
            return None, reward

        # Track final health (0 HP) before returning
        agent._episode_tracker.track_health(agent.env.agent_hp)

        logger.warning(
            "Failed to complete episode: health depleted from predator damage!",
        )
        # Apply death penalty
        penalty = -reward_config.penalty_predator_death
        reward += penalty
        agent._episode_tracker.track_reward(penalty)

        return self._terminate_episode(
            agent,
            params,
            reward,
            success=False,
            termination_reason=TerminationReason.HEALTH_DEPLETED,
        ), reward

    def _handle_predator_phase(
        self,
        agent: QuantumNematodeAgent,
        reward_config: RewardConfig,
        params: Any,  # noqa: ANN401
        reward: float,
        step_index: int = 0,
    ) -> tuple[EpisodeResult | None, float]:
        """Handle predator checks before and after predator movement.

        Parameters
        ----------
        step_index : int
            Episode-level step counter at the start of this phase. Forwarded
            into `env.update_predators` -> `PredatorBrainParams.step_index`
            so time-aware predator brains receive the live counter.

        Returns
        -------
        tuple[EpisodeResult | None, float]
            A termination result if the agent died (or None), and the
            updated reward value.
        """
        # Track danger status at start of step (before any movement)
        was_in_danger_at_step_start = agent._episode_tracker.in_danger

        # Check for predator collision/damage BEFORE predators move
        result, reward = self._check_predator_contact(
            agent,
            reward_config,
            params,
            reward,
        )
        if result is not None:
            return result, reward

        # Move predators after agent moves
        agent.env.update_predators(step_index=step_index)

        # Check danger status at end of step (after both agent and predators moved)
        is_in_danger_at_step_end = agent.env.is_agent_in_danger()

        # Track state transitions across the entire step
        if not was_in_danger_at_step_start and is_in_danger_at_step_end:
            # Entered danger zone during this step - increment encounters
            agent._episode_tracker.predator_encounters += 1
        elif was_in_danger_at_step_start and not is_in_danger_at_step_end:
            # Exited danger zone during this step - successful evasion
            agent._episode_tracker.successful_evasions += 1

        agent._episode_tracker.in_danger = is_in_danger_at_step_end

        # Check for predator collision/damage AFTER predators move
        # (predator may step onto agent's position)
        result, reward = self._check_predator_contact(
            agent,
            reward_config,
            params,
            reward,
            after_movement=True,
        )
        return result, reward

    def _handle_temperature_effects(
        self,
        agent: QuantumNematodeAgent,
        reward_config: RewardConfig,
        params: Any,  # noqa: ANN401
        reward: float,
    ) -> tuple[EpisodeResult | None, float]:
        """Apply temperature zone effects (rewards/penalties and HP damage).

        Returns
        -------
        tuple[EpisodeResult | None, float]
            A termination result if health was depleted (or None), and the
            updated reward value.
        """
        if not agent.env.thermotaxis.enabled:
            return None, reward

        # Track temperature at agent position
        current_temp = agent.env.get_temperature()
        if current_temp is not None:
            agent._episode_tracker.track_temperature(current_temp)

        temp_reward, temp_damage = agent.env.apply_temperature_effects()
        if temp_reward != 0.0:
            reward += temp_reward
            agent._episode_tracker.track_reward(temp_reward)

        # Apply HP damage penalty for temperature damage (immediate learning signal)
        if temp_damage > 0:
            damage_penalty = -reward_config.penalty_health_damage
            reward += damage_penalty
            agent._episode_tracker.track_reward(damage_penalty)
            logger.debug(
                f"Temperature HP damage penalty applied: {damage_penalty} "
                f"(took {temp_damage:.1f} damage)",
            )

            # Track health after temperature damage
            agent._episode_tracker.track_health(agent.env.agent_hp)

        # Check if temperature damage depleted health
        if temp_damage > 0 and agent.env.is_health_depleted():
            logger.warning(
                "Failed to complete episode: health depleted from temperature damage!",
            )
            # Apply death penalty (reuse predator death penalty)
            penalty = -reward_config.penalty_predator_death
            reward += penalty
            agent._episode_tracker.track_reward(penalty)

            return self._terminate_episode(
                agent,
                params,
                reward,
                success=False,
                termination_reason=TerminationReason.HEALTH_DEPLETED,
            ), reward

        return None, reward

    def _handle_oxygen_effects(
        self,
        agent: QuantumNematodeAgent,
        reward_config: RewardConfig,
        params: Any,  # noqa: ANN401
        reward: float,
    ) -> tuple[EpisodeResult | None, float]:
        """Apply oxygen zone effects (rewards/penalties, HP damage).

        Returns
        -------
        tuple[EpisodeResult | None, float]
            Episode result (if terminated by oxygen damage) and
            updated reward value.
        """
        if not agent.env.aerotaxis.enabled:
            return None, reward

        # Track oxygen at agent position
        current_o2 = agent.env.get_oxygen()
        if current_o2 is not None:
            agent._episode_tracker.track_oxygen(current_o2)

        o2_reward, o2_damage = agent.env.apply_oxygen_effects()
        if o2_reward != 0.0:
            reward += o2_reward
            agent._episode_tracker.track_reward(o2_reward)

        # Apply HP damage penalty for oxygen damage (immediate learning signal)
        if o2_damage > 0:
            damage_penalty = -reward_config.penalty_health_damage
            reward += damage_penalty
            agent._episode_tracker.track_reward(damage_penalty)
            logger.debug(
                f"Oxygen HP damage penalty applied: {damage_penalty} (took {o2_damage:.1f} damage)",
            )

            # Track health after oxygen damage
            agent._episode_tracker.track_health(agent.env.agent_hp)

        # Check if oxygen damage depleted health
        if o2_damage > 0 and agent.env.is_health_depleted():
            logger.warning(
                "Failed to complete episode: health depleted from oxygen damage!",
            )
            penalty = -reward_config.penalty_predator_death
            reward += penalty
            agent._episode_tracker.track_reward(penalty)

            return self._terminate_episode(
                agent,
                params,
                reward,
                success=False,
                termination_reason=TerminationReason.HEALTH_DEPLETED,
            ), reward

        return None, reward

    def _handle_starvation_check(
        self,
        agent: QuantumNematodeAgent,
        reward_config: RewardConfig,
        params: Any,  # noqa: ANN401
        reward: float,
    ) -> tuple[EpisodeResult | None, float]:
        """Check for starvation and terminate if starved.

        Returns
        -------
        tuple[EpisodeResult | None, float]
            A termination result if the agent starved (or None), and the
            updated reward value.
        """
        if not agent._satiety_manager.is_starved():
            return None, reward

        logger.warning("Failed to complete episode: agent starved!")
        # Apply starvation penalty to both brain reward and episode tracker
        penalty = -reward_config.penalty_starvation
        reward += penalty
        agent._episode_tracker.track_reward(penalty)

        return self._terminate_episode(
            agent,
            params,
            reward,
            success=False,
            termination_reason=TerminationReason.STARVED,
        ), reward

    @staticmethod
    def _bit_memory_turn(action: ActionData) -> float:
        """Return the binary-response source — the continuous turn or a discrete L/R vote.

        ``sign`` of the returned value is compared to the cue: for the continuous arms this
        is the normalized turn component, for a discrete arm a LEFT/RIGHT choice.

        Parameters
        ----------
        action : ActionData
            The top action emitted by the brain this step.

        Returns
        -------
        float
            The signed response source: the normalized turn for a continuous action, or
            -1.0 / +1.0 for a discrete LEFT / RIGHT (0.0 for any other discrete action).
        """
        if action.continuous is not None:
            return float(action.continuous[1])  # (speed, turn) -> turn
        from quantumnematode.brain.actions import Action

        if action.action == Action.LEFT:
            return -1.0
        if action.action == Action.RIGHT:
            return 1.0
        return 0.0

    def _terminate_bit_memory(  # noqa: PLR0913
        self,
        agent: QuantumNematodeAgent,
        params: Any,  # noqa: ANN401
        reward: float,
        bm: BitMemoryTask,
        *,
        learn: bool = True,
        update_memory: bool = True,
    ) -> EpisodeResult:
        """Log the per-episode cue-match rate (the analysis metric) and terminate.

        Parameters
        ----------
        agent : QuantumNematodeAgent
            The agent instance.
        params : Any
            Brain parameters from the final step.
        reward : float
            The reward delivered on the terminating step.
        bm : BitMemoryTask
            The bit-memory phase machine, read for the cue-match rate + response count.
        learn : bool
            Whether to call brain.learn at termination (default True).
        update_memory : bool
            Whether to call brain.update_memory at termination (default True).

        Returns
        -------
        EpisodeResult
            The episode result, tagged with the bit-memory termination reason.
        """
        rate = bm.cue_match_success_rate
        logger.info("BitMemory: cue_match=%.4f responses=%d", rate, bm.num_responses)
        return self._terminate_episode(
            agent,
            params,
            reward,
            success=rate > _BIT_MEMORY_CHANCE,
            termination_reason=TerminationReason.BIT_MEMORY_COMPLETED,
            learn=learn,
            update_memory=update_memory,
            food_history=None,
        )

    def _run_bit_memory_step(
        self,
        agent: QuantumNematodeAgent,
        max_steps: int,
        prev_action: ActionData | None,
        *,
        render_text: str | None,
        show_last_frame_only: bool,
    ) -> tuple[EpisodeResult | None, ActionData | None]:
        """Run one bit-memory step: cue/go observation -> action -> response scoring.

        Bypasses all foraging/predator/thermal dynamics (movement is inert). The reward is
        the *previous* response's score (``run_brain`` treats it as the previous-step
        reward, mirroring the foraging timing); the current action is scored against the cue
        when this is a response step. The episode ends once every trial completes.

        Parameters
        ----------
        agent : QuantumNematodeAgent
            The agent instance (drives the brain + owns the bit-memory env).
        max_steps : int
            The episode step budget. Completion is normally driven by the trial counter;
            this is only a mis-size guard for a config whose budget is too small.
        prev_action : ActionData | None
            The previous step's top action, forwarded into the brain params.
        render_text : str | None
            Text to display during rendering.
        show_last_frame_only : bool
            Whether to render only the last frame.

        Returns
        -------
        tuple[EpisodeResult | None, ActionData | None]
            The episode result once the task terminates (else None), and this step's top
            action (carried forward as the next step's ``prev_action``).
        """
        bm = agent.env.bit_memory
        if bm is None:  # defensive: the caller guards this
            return None, prev_action

        reward = bm.take_reward()
        agent._episode_tracker.track_reward(reward)
        input_data = agent._prepare_input_data(0.0)
        params = agent._create_brain_params(action=prev_action)
        action = agent.brain.run_brain(
            params=params,
            reward=reward,
            input_data=input_data,
            top_only=True,
            top_randomize=True,
        )
        if len(action) != 1:
            error_msg = f"Invalid action length: {len(action)}. Expected 1."
            raise ValueError(error_msg)
        top_action = action[0]
        agent._episode_tracker.track_step()

        # All trials complete: the final response's reward was just delivered above
        # (take_reward -> run_brain). Finalise + terminate.
        if bm.done:
            return self._terminate_bit_memory(agent, params, reward, bm), top_action

        # Score this step's action against the cue (a no-op outside the response phase).
        bm.record_response(self._bit_memory_turn(top_action))

        episode_done = agent._episode_tracker.steps >= max_steps
        if isinstance(agent.brain, ClassicalBrain):
            agent.brain.learn(params=params, reward=reward, episode_done=episode_done)
        agent.brain.update_memory(reward)

        # Movement is inert — record the (unchanged) position + foods so the agent-level
        # len(path) == len(food_history) invariant the foraging loop maintains still holds.
        agent.path.append((agent.env.agent_pos[0], agent.env.agent_pos[1]))
        agent.food_history.append([(round(fx), round(fy)) for fx, fy in agent.env.foods])
        agent._render_step(max_steps, render_text, show_last_frame_only=show_last_frame_only)

        if episode_done:  # budget exhausted before all trials (mis-sized config)
            return (
                self._terminate_bit_memory(
                    agent,
                    params,
                    reward,
                    bm,
                    learn=False,
                    update_memory=False,
                ),
                top_action,
            )

        bm.advance()
        return None, top_action

    def _terminate_associative_memory(  # noqa: PLR0913
        self,
        agent: QuantumNematodeAgent,
        params: Any,  # noqa: ANN401
        reward: float,
        am: AssociativeMemoryTask,
        *,
        learn: bool = True,
        update_memory: bool = True,
    ) -> EpisodeResult:
        """Log the per-episode response accuracy (overall + reversal split) and terminate.

        The reversal / non-reversal split makes the working-memory *update* demand directly
        readable (a hold-only policy is at chance on the reversal fraction).
        """
        acc = am.response_accuracy
        # Print (not just log) so the reversal / non-reversal split reaches the run's ``.out`` for
        # the separation harness — the overall accuracy is also derivable from the episode reward,
        # but the split (which arms actually *update* the association) is not.
        print(  # noqa: T201
            f"AssocMemory: accuracy={acc:.4f} reversal={am.reversal_accuracy:.4f} "
            f"non_reversal={am.non_reversal_accuracy:.4f} responses={am.num_responses}",
        )
        return self._terminate_episode(
            agent,
            params,
            reward,
            success=acc > _ASSOCIATIVE_MEMORY_CHANCE,
            termination_reason=TerminationReason.ASSOCIATIVE_MEMORY_COMPLETED,
            learn=learn,
            update_memory=update_memory,
            food_history=None,
        )

    def _run_associative_memory_step(
        self,
        agent: QuantumNematodeAgent,
        max_steps: int,
        prev_action: ActionData | None,
        *,
        render_text: str | None,
        show_last_frame_only: bool,
    ) -> tuple[EpisodeResult | None, ActionData | None]:
        """Run one associative-memory step (mirrors ``_run_bit_memory_step``).

        cue/outcome/go observation -> action -> response scoring against the *current*
        (post-reversal) rewarded cue. Bypasses all foraging/predator/thermal dynamics (movement
        inert). The reward is the previous response's score; the episode ends once every trial
        completes.
        """
        am = agent.env.associative_memory
        if am is None:  # defensive: the caller guards this
            return None, prev_action

        reward = am.take_reward()
        agent._episode_tracker.track_reward(reward)
        input_data = agent._prepare_input_data(0.0)
        params = agent._create_brain_params(action=prev_action)
        action = agent.brain.run_brain(
            params=params,
            reward=reward,
            input_data=input_data,
            top_only=True,
            top_randomize=True,
        )
        if len(action) != 1:
            error_msg = f"Invalid action length: {len(action)}. Expected 1."
            raise ValueError(error_msg)
        top_action = action[0]
        agent._episode_tracker.track_step()

        if am.done:
            return self._terminate_associative_memory(agent, params, reward, am), top_action

        # Score this step's action against the current rewarded cue (no-op outside the response
        # phase). The binary-response reader is arch-agnostic — reused from the bit-memory path.
        am.record_response(self._bit_memory_turn(top_action))

        episode_done = agent._episode_tracker.steps >= max_steps
        if isinstance(agent.brain, ClassicalBrain):
            agent.brain.learn(params=params, reward=reward, episode_done=episode_done)
        agent.brain.update_memory(reward)

        # Movement is inert — record the (unchanged) position + foods so the agent-level
        # len(path) == len(food_history) invariant the foraging loop maintains still holds.
        agent.path.append((agent.env.agent_pos[0], agent.env.agent_pos[1]))
        agent.food_history.append([(round(fx), round(fy)) for fx, fy in agent.env.foods])
        agent._render_step(max_steps, render_text, show_last_frame_only=show_last_frame_only)

        if episode_done:  # budget exhausted before all trials (mis-sized config)
            return (
                self._terminate_associative_memory(
                    agent,
                    params,
                    reward,
                    am,
                    learn=False,
                    update_memory=False,
                ),
                top_action,
            )

        am.advance()
        return None, top_action

    def run(  # noqa: C901, PLR0911, PLR0912, PLR0915
        self,
        agent: QuantumNematodeAgent,
        reward_config: RewardConfig,
        max_steps: int,
        **kwargs: Any,  # noqa: ANN401
    ) -> EpisodeResult:
        """Run a standard episode.

        Complete implementation that matches agent.run_episode() logic exactly.

        Note: This runner intentionally accesses private agent members (_food_handler,
        _satiety_manager, _prepare_input_data, _create_brain_params, _render_step)
        as part of the episode execution architecture.

        Parameters
        ----------
        agent : QuantumNematodeAgent
            The agent instance.
        reward_config : RewardConfig
            Configuration for the reward system.
        max_steps : int
            Maximum number of steps for the episode.
        **kwargs : Any
            Additional keyword arguments:
            - render_text : str | None - Text to display during rendering
            - show_last_frame_only : bool - Whether to show only the last frame

        Returns
        -------
        StepResult
            The result of the episode execution, including path and termination reason.
        """
        render_text: str | None = kwargs.get("render_text")
        show_last_frame_only: bool = kwargs.get("show_last_frame_only", False)

        # Initialize the agent's direction
        agent.env.current_direction = Direction.UP

        # Prepare brain for new episode (e.g., save parameters for potential rollback)
        agent.brain.prepare_episode()
        # Lifecycle hook for stateful predator brains (no-op for the
        # heuristic; learnable predator brains reset hidden state here).
        # Mirrors MultiAgentSimulation.run_episode which fires the same
        # hook so single-agent and multi-agent paths stay symmetric.
        for predator in agent.env.predators:
            predator.brain.prepare_episode()

        # Reset STAM buffer for new episode (no cross-episode memory)
        if agent._stam is not None:
            agent._stam.reset()
        # Reset the bit-memory phase machine for the new episode (no cross-episode cue).
        if agent.env.bit_memory is not None:
            agent.env.bit_memory.reset()
        # Reset the associative-memory phase machine too (no cross-episode association).
        if agent.env.associative_memory is not None:
            agent.env.associative_memory.reset()
        # Reset the adaptive chemosensory background too (same per-episode boundary
        # as STAM — otherwise the background leaks across reused episodes).
        if agent._adaptive_food is not None:
            agent._adaptive_food.reset()
        agent._previous_position = None
        agent._last_heading = Direction.UP

        # Reset food handler tracking for new episode
        agent._food_handler.reset()

        reward = 0.0
        top_action = None
        stuck_position_count = 0
        previous_position = None

        for step_index in range(max_steps):
            logger.debug("--- New Step ---")

            # Bit-memory positive control: a non-spatial delayed-match-to-cue task. When
            # enabled, drive its phase machine instead of the foraging dynamics — movement
            # is inert and every foraging/predator/thermal handler below is bypassed.
            if agent.env.bit_memory is not None:
                result, top_action = self._run_bit_memory_step(
                    agent,
                    max_steps,
                    top_action,
                    render_text=render_text,
                    show_last_frame_only=show_last_frame_only,
                )
                if result is not None:
                    return result
                continue

            # Associative-memory probe: the same non-spatial pattern (conditioning/reversal/
            # delay/response). When enabled, drive its phase machine instead of foraging.
            if agent.env.associative_memory is not None:
                result, top_action = self._run_associative_memory_step(
                    agent,
                    max_steps,
                    top_action,
                    render_text=render_text,
                    show_last_frame_only=show_last_frame_only,
                )
                if result is not None:
                    return result
                continue

            gradient_strength, _gradient_direction = agent.env.get_state(agent.path[-1])

            # Track if agent stays in same position
            current_position = tuple(agent.env.agent_pos)
            if current_position == previous_position:
                stuck_position_count += 1
            else:
                stuck_position_count = 0
            previous_position = current_position

            # Calculate reward
            reward = agent.calculate_reward(
                reward_config,
                agent.env,
                agent.path,
                max_steps=max_steps,
                stuck_position_count=stuck_position_count,
                can_eat=agent.can_eat,
            )
            agent._episode_tracker.track_reward(reward)

            # Prepare input_data and brain parameters
            input_data = agent._prepare_input_data(gradient_strength)
            params = agent._create_brain_params(
                action=top_action,
            )
            action = agent.brain.run_brain(
                params=params,
                reward=reward,
                input_data=input_data,
                top_only=True,
                top_randomize=True,
            )

            # Only one action is supported
            if len(action) != 1:
                error_msg = f"Invalid action length: {len(action)}. Expected 1."
                logger.error(error_msg)
                raise ValueError(error_msg)

            top_action = action[0]

            # Dispatch on the env type (env-driven, not brain-coupled): the
            # continuous-2D env consumes the (speed, turn) vector; the grid env
            # consumes the discrete Action. A continuous-action brain emits
            # `continuous`; when it doesn't, the continuous env's _apply_movement
            # runs a coherent discrete move (it re-syncs the float position), so
            # the fallback is well-defined.
            from quantumnematode.env.continuous_2d import Continuous2DEnvironment

            if isinstance(agent.env, Continuous2DEnvironment):
                if top_action.continuous is not None:
                    # Continuous-action brains emit a normalized (speed, turn); the
                    # env rescales to physical units.
                    agent.env.move_agent_normalized(
                        *top_action.continuous,
                        agent_id=agent.agent_id,
                    )
                else:
                    # Continuous env, discrete brain: continuous-action heads are not
                    # active yet. Use a (coherent) discrete move and warn once.
                    if not self._warned_discrete_on_continuous:
                        logger.warning(
                            "Continuous-2D environment received a discrete action "
                            "(no (speed, turn) vector). Continuous-action heads are not "
                            "active for this brain; using discrete movement on the "
                            "continuous substrate.",
                        )
                        self._warned_discrete_on_continuous = True
                    agent.env.move_agent_for(agent.agent_id, top_action.action)
            else:
                agent.env.move_agent_for(agent.agent_id, top_action.action)

            # Track step (will add satiety later if dynamic environment)
            agent._episode_tracker.track_step()

            # Food collection (must happen immediately after agent moves)
            result, reward = self._handle_food_collection(agent, reward_config, params, reward)
            if result is not None:
                return result

            # Predator checks (before and after predator movement)
            result, reward = self._handle_predator_phase(
                agent,
                reward_config,
                params,
                reward,
                step_index=step_index,
            )
            if result is not None:
                return result

            # Satiety decay (after predator movement)
            agent._satiety_manager.decay_satiety()

            # Track satiety after decay
            agent._episode_tracker.track_satiety(agent.current_satiety)

            # Track health
            agent._episode_tracker.track_health(agent.env.agent_hp)

            # Temperature effects
            result, reward = self._handle_temperature_effects(agent, reward_config, params, reward)
            if result is not None:
                return result

            # Oxygen effects
            result, reward = self._handle_oxygen_effects(agent, reward_config, params, reward)
            if result is not None:
                return result

            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    f"Satiety: {agent.current_satiety:.1f}/{agent.max_satiety}",
                )

            # Starvation check
            result, reward = self._handle_starvation_check(agent, reward_config, params, reward)
            if result is not None:
                return result

            # Classical brain learning step
            if isinstance(agent.brain, ClassicalBrain):
                episode_done = bool(
                    agent._episode_tracker.steps >= max_steps or agent.env.reached_goal(),
                )
                agent.brain.learn(
                    params=params,
                    reward=reward,
                    episode_done=episode_done,
                )

            # Update the body length dynamically
            if agent.max_body_length > 0 and len(agent.env.body) < agent.max_body_length:
                agent.env.body.append(agent.env.body[-1])

            agent.brain.update_memory(reward)

            pos: GridPosition = (agent.env.agent_pos[0], agent.env.agent_pos[1])
            agent.path.append(pos)
            # Track food positions for chemotaxis validation (cell-snapped record;
            # continuous-2D sources are real-valued, the worm senses the real field).
            agent.food_history.append([(round(fx), round(fy)) for fx, fy in agent.env.foods])

            # Track step for food distance efficiency calculation
            agent._food_handler.track_step()

            # Skip the f-string construction entirely when INFO is filtered.
            # Each f-string call here is cheap individually but fitness eval
            # runs this loop ~1000 times per episode, so the saved string
            # formatting is visible in 30-60 s LSTMPPO episodes.
            if logger.isEnabledFor(logging.INFO):
                action_repr = (
                    top_action.action.value
                    if top_action.action is not None
                    else top_action.continuous
                )
                logger.info(
                    f"Step {agent._episode_tracker.steps}: Action={action_repr}, Reward={reward}",
                )

                # Log cumulative reward and average reward per step
                if agent._episode_tracker.steps > 0:
                    average_reward = agent._episode_tracker.rewards / agent._episode_tracker.steps
                    logger.info(
                        f"Cumulative reward: {agent._episode_tracker.rewards}, "
                        f"Average reward per step: {average_reward}",
                    )

            # Render current step
            agent._render_step(max_steps, render_text, show_last_frame_only=show_last_frame_only)

            # Handle Pygame window close - terminate episode early
            if agent.pygame_renderer_closed:
                logger.info("Pygame window closed by user - terminating episode.")
                return self._terminate_episode(
                    agent,
                    params,
                    reward,
                    success=False,
                    termination_reason=TerminationReason.INTERRUPTED,
                    learn=False,
                    update_memory=False,
                    food_history=(agent.food_history or None),
                )

            # Handle max steps reached
            if agent._episode_tracker.steps >= max_steps:
                logger.warning("Failed to complete episode: max steps reached.")
                return self._terminate_episode(
                    agent,
                    params,
                    reward,
                    success=False,
                    termination_reason=TerminationReason.MAX_STEPS,
                    learn=False,
                    update_memory=False,
                    food_history=(agent.food_history or None),
                )

            # Handle all food collected
            if agent._episode_tracker.foods_collected >= agent.env.foraging.target_foods_to_collect:
                logger.info(
                    "Successfully completed episode: collected target of "
                    f"{agent.env.foraging.target_foods_to_collect} food.",
                )
                return self._terminate_episode(
                    agent,
                    params,
                    reward,
                    success=True,
                    termination_reason=TerminationReason.COMPLETED_ALL_FOOD,
                    update_memory=False,
                )

        # This point is unreachable - the loop always exits via one of the return
        # statements above (max_steps check catches the final iteration)
        msg = "Unreachable code: episode loop exited without termination"
        raise RuntimeError(msg)


class ManyworldsEpisodeRunner(EpisodeRunner):
    """Runs an episode with many-worlds branching.

    This runner implements branching trajectories inspired by the many-worlds
    interpretation of quantum mechanics. At each step, multiple action branches
    are explored in parallel, creating a tree of possible futures.

    Notes
    -----
    The implementation directly accesses agent helper methods (_create_brain_params)
    and uses inline rendering for visualizing parallel universes. This specialized
    execution mode is fundamentally different from standard single-trajectory episodes.
    """

    def __init__(self) -> None:
        """Initialize the manyworlds episode runner."""

    def run(  # noqa: C901, PLR0912, PLR0915
        self,
        agent: QuantumNematodeAgent,
        reward_config: RewardConfig,
        max_steps: int,
        **kwargs: Any,  # noqa: ANN401
    ) -> EpisodeResult:
        """Run an episode with many-worlds branching.

        Complete implementation that matches agent.run_manyworlds_mode() logic exactly.

        Note: This runner intentionally accesses private agent member (_create_brain_params)
        as part of the episode execution architecture.

        Parameters
        ----------
        agent : QuantumNematodeAgent
            The agent instance (used for accessing brain, env).
        reward_config : RewardConfig
            Configuration for the reward system.
        max_steps : int
            Maximum number of steps for the episode.
        **kwargs : Any
            Additional keyword arguments:
            - config : ManyworldsModeConfig - Configuration for branching
            - show_last_frame_only : bool - Whether to show only the last frame

        Returns
        -------
        StepResult
            The result of the episode execution, including path and termination reason.
        """
        from quantumnematode.agent import ManyworldsModeConfig

        # Get configuration
        config: ManyworldsModeConfig = kwargs.get("config", ManyworldsModeConfig())  # type: ignore[assignment]
        show_last_frame_only: bool = kwargs.get("show_last_frame_only", False)

        # Many-worlds (superposition) mode relies on env.copy(), which on the
        # continuous-2D substrate would lose the continuous type + params (the
        # copy() override is future work). Fail clearly until then; single-world
        # runs are the supported path for the continuous substrate.
        from quantumnematode.env.continuous_2d import Continuous2DEnvironment

        if isinstance(agent.env, Continuous2DEnvironment):
            msg = "Many-worlds mode is not yet supported on the continuous-2D substrate."
            raise NotImplementedError(msg)

        # Initialize many-worlds mode
        agent.env.current_direction = Direction.UP

        if show_last_frame_only:
            if os.name == "nt":  # For Windows
                os.system("cls")  # noqa: S605, S607
            else:  # For macOS and Linux
                os.system("clear")  # noqa: S605, S607

        # Render the initial grid
        grid = agent.env.render()
        for frame in grid:
            print(frame)  # noqa: T201
            logger.debug(frame)
        print("#1")  # noqa: T201

        time.sleep(config.render_sleep_seconds)  # Wait before the next render

        logger.info(
            "Many-worlds mode enabled. "
            f"Visualizing top {config.top_n_actions} decisions at each step.",
        )
        superpositions = [(agent.brain.copy(), agent.env.copy(), agent.path.copy())]

        reward = 0.0
        for _ in range(max_steps):
            total_superpositions = len(superpositions)
            i = 0
            for brain_copy, env_copy, path_copy in superpositions:
                reward = agent.calculate_reward(
                    reward_config,
                    env_copy,
                    path_copy,
                    max_steps=max_steps,
                    stuck_position_count=0,  # Many-worlds mode doesn't track stuck positions
                )
                agent._episode_tracker.track_reward(reward)

                params = agent._create_brain_params()
                actions = brain_copy.run_brain(
                    params=params,
                    reward=reward,
                    input_data=None,
                    top_only=False,
                    top_randomize=True,
                )

                if config.top_n_randomize:
                    # Use seeded RNG from environment for reproducibility
                    rng = env_copy.rng
                    probs = np.array([a.probability for a in actions], dtype=float)
                    probs_sum = probs.sum()
                    if probs_sum > 0:
                        norm_probs = probs / probs_sum
                    else:
                        norm_probs = np.ones_like(probs) / len(probs)
                    actions_arr = np.array(actions)
                    top_actions_and_probs = rng.choice(
                        actions_arr,
                        p=norm_probs,
                        size=config.top_n_actions,
                        replace=True,
                    )
                    top_actions = [a.action for a in top_actions_and_probs if a.action is not None]
                else:
                    top_actions = [a.action for a in actions if a.action is not None][
                        : config.top_n_actions
                    ]

                # Update the body length dynamically
                if agent.max_body_length > 0 and len(env_copy.body) < agent.max_body_length:
                    env_copy.body.append(env_copy.body[-1])

                if len(superpositions) < config.max_superpositions and top_actions:
                    new_env = env_copy.copy()
                    new_path = path_copy.copy()
                    new_brain = agent.brain.copy()
                    runner_up_action = top_actions[1] if len(top_actions) > 1 else top_actions[0]
                    if runner_up_action is not None:
                        new_env.move_agent(runner_up_action)
                        agent._episode_tracker.track_step()
                        new_brain.update_memory(reward)
                        new_pos: GridPosition = (new_env.agent_pos[0], new_env.agent_pos[1])
                        new_path.append(new_pos)
                        superpositions.append((new_brain, new_env, new_path))

                if env_copy.reached_goal():
                    continue

                if top_actions:
                    env_copy.move_agent(top_actions[0])
                    agent._episode_tracker.track_step()
                    brain_copy.update_memory(reward)
                    copy_pos: GridPosition = (env_copy.agent_pos[0], env_copy.agent_pos[1])
                    path_copy.append(copy_pos)

                i += 1
                if i >= total_superpositions:
                    break

            if show_last_frame_only:
                if os.name == "nt":  # For Windows
                    os.system("cls")  # noqa: S605, S607
                else:  # For macOS and Linux
                    os.system("clear")  # noqa: S605, S607

            # Render all grids for superpositions at each step
            row = []
            labels = []
            label_padding_first = " " * 8
            label_padding_all = " " * agent.env.grid_size
            for i, (_, env_copy, _) in enumerate(superpositions):
                grid = env_copy.render()
                label = (
                    f"#{i + 1} <= #{i // 2 + 1}{label_padding_all}"
                    if i > 0
                    else f"#{i + 1}{label_padding_first}{label_padding_all}"
                )
                row.append(grid)
                labels.append(label)

                # Print the row when reaching MAX_COLUMNS or the last grid
                if (i + 1) % config.max_columns == 0 or i == len(superpositions) - 1:
                    for line_set in zip(*row, strict=False):
                        # Render side by side
                        print("\t".join(line_set))  # noqa: T201
                    # Add labels below the grids
                    print("\t".join(labels))  # noqa: T201
                    # Add spacing between rows
                    print("\n")  # noqa: T201
                    row = []  # Reset the row buffer
                    labels = []  # Reset the labels buffer

            if len(superpositions) < config.max_superpositions:
                time.sleep(config.render_sleep_seconds)  # Wait before the next render

            # Stop if all superpositions have reached their goal
            if all(env_copy.reached_goal() for _, env_copy, _ in superpositions):
                msg = "Successfully completed episode: all superpositions have reached their goal."
                logger.info(msg)
                print(msg)  # noqa: T201
                # Return the path from the first superposition (primary path)
                return EpisodeResult(
                    agent_path=superpositions[0][2],
                    termination_reason=TerminationReason.GOAL_REACHED,
                    food_history=None,
                )
        msg = (
            "Failed to complete episode: "
            "Many-worlds mode completed as maximum number of steps reached."
        )
        logger.warning(msg)
        print(msg)  # noqa: T201
        # Return the path from the first superposition (primary path)
        return EpisodeResult(
            agent_path=superpositions[0][2],
            termination_reason=TerminationReason.MAX_STEPS,
            food_history=None,
        )
