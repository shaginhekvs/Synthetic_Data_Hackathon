# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Sequential Environment Server Implementation.

This module wraps Cartpole, MountainCarContinuous, LunarLanderContinuous, and BipedalWalker-v2 environments
and interleaves their execution using seeded random selection.
"""

import logging
import uuid
import random
from typing import Any, Dict, Optional, List

import numpy as np

from core.env_server import Environment

from ..models import SequentialAction, SequentialObservation, SequentialState
from ...cartpole_environment.server.cartpole_environment import CartpoleEnvironment
from ...cartpole_environment.models import CartpoleAction, CartpoleObservation
from ...mountaincarcontinuous_environment.server.mountaincarcontinuous_environment import MountainCarContinuousEnvironment
from ...mountaincarcontinuous_environment.models import MountainCarContinuousAction, MountainCarContinuousObservation
from ...lunarlander_environment.server.lunarlander_environment import LunarLanderEnvironment
from ...lunarlander_environment.models import LunarLanderAction, LunarLanderObservation
from ...gym_environment.server.gymnasium_environment import GymnasiumEnvironment
from ...gym_environment.models import GymAction, GymObservation

# Set up logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


class SequentialEnvironment(Environment):
    """
    Sequential Environment wrapper that interleaves Cartpole, MountainCarContinuous, and LunarLander.

    This environment randomly selects which sub-environment to step each time, using a seeded
    random number generator for reproducibility. The observation concatenates a one-hot phase
    vector with the sub-environment observation.

    Args:
        render_mode: Render mode for visualization ("human", "rgb_array", None).
        max_steps: Maximum steps per episode before truncation.
        seed: Random seed for phase selection reproducibility.
    """

    PhaseType = str

    PHASE_CARTPOLE = "cartpole"
    PHASE_MOUNTAINCAR = "mountaincar"
    PHASE_LUNARLANDER = "lunarlander"
    PHASE_BIPEDALWALKER = "bipedalwalker"

    PHASES = [PHASE_CARTPOLE, PHASE_MOUNTAINCAR, PHASE_LUNARLANDER, PHASE_BIPEDALWALKER]

    def __init__(
        self,
        render_mode: Optional[str] = None,
        max_steps: int = 1000,
        seed: Optional[int] = None,
    ):
        """Initialize Sequential environment."""
        super().__init__()

        self.render_mode = render_mode
        self.max_steps = max_steps
        self.seed = seed

        # Initialize random number generator
        self.rng = random.Random(self.seed)

        # Create sub-environments (share the seed)
        logger.info("Creating sub-environments with seed=%s", seed)
        self.cartpole_env = CartpoleEnvironment(render_mode=render_mode, seed=seed, max_steps=max_steps)
        self.mountaincar_env = MountainCarContinuousEnvironment(render_mode=render_mode, seed=seed, max_steps=max_steps)
        self.lunarlander_env = LunarLanderEnvironment(render_mode=render_mode, seed=seed, max_steps=max_steps)
        self.bipedalwalker_env = GymnasiumEnvironment(env_id="BipedalWalker-v3", render_mode=render_mode, seed=seed, max_steps=max_steps)

        # Track which environments have reached done state
        self.cartpole_done = False
        self.mountaincar_done = False
        self.lunarlander_done = False
        self.bipedalwalker_done = False

        # Store last observation from each sub-environment
        self.cartpole_last_obs = None
        self.mountaincar_last_obs = None
        self.lunarlander_last_obs = None
        self.bipedalwalker_last_obs = None

        # Initialize state
        self._state = SequentialState(
            max_steps=max_steps,
            seed=seed,
        )

        logger.info("SequentialEnvironment initialized successfully")

    def reset(self) -> SequentialObservation:
        """
        Reset all sub-environments and return initial observation.

        Returns:
            Initial observation with randomly selected first phase.
        """
        logger.info("Resetting sequential environment")

        # Reset all sub-environments
        self.cartpole_env.reset()
        self.mountaincar_env.reset()
        self.lunarlander_env.reset()
        self.bipedalwalker_env.reset()

        # Reset completion tracking
        self.cartpole_done = False
        self.mountaincar_done = False
        self.lunarlander_done = False
        self.bipedalwalker_done = False

        # Reset returns observations, not states - we need to store those
        cartpole_reset_obs = self.cartpole_env.reset()
        mountaincar_reset_obs = self.mountaincar_env.reset()
        lunarlander_reset_obs = self.lunarlander_env.reset()
        bipedalwalker_reset_obs = self.bipedalwalker_env.reset()

        # Store the initial observations from each sub-environment reset
        self.cartpole_last_obs = cartpole_reset_obs
        self.mountaincar_last_obs = mountaincar_reset_obs
        self.lunarlander_last_obs = lunarlander_reset_obs
        self.bipedalwalker_last_obs = bipedalwalker_reset_obs

        # Reset state tracking
        self._state.episode_id = str(uuid.uuid4())
        self._state.step_count = 0
        self._state.episode_length = 0
        self._state.total_reward = 0.0
        self._state.cartpole_done = False
        self._state.mountaincar_done = False
        self._state.lunarlander_done = False
        self._state.bipedalwalker_done = False

        # Select first phase randomly
        active_phase = self._select_phase()
        self._state.active_phase = active_phase
        self._state.next_environment = active_phase  # Next is the same as current for first step

        # Create observation with the actual observation state
        observation = self._make_observation_from_phase(active_phase, reward=0.0, done=False)

        logger.info("Sequential environment reset - Episode %s started", self._state.episode_id)
        logger.info("First phase: %s", active_phase)

        return observation

    def step(self, action: SequentialAction) -> SequentialObservation:
        """
        Take a step in the randomly selected sub-environment.

        Args:
            action: SequentialAction containing actions for all sub-environments.

        Returns:
            Observation after action execution.
        """
        if not isinstance(action, SequentialAction):
            raise ValueError(f"Expected SequentialAction, got {type(action)}")

        # Get current active phase (selected in previous step/reset)
        active_phase = self._state.active_phase

        logger.info("Stepping in phase: %s", active_phase)

        # Execute action in active sub-environment
        sub_obs = self._step_sub_env(active_phase, action)

        # Store the last observation from the sub-environment that was stepped
        if active_phase == self.PHASE_CARTPOLE:
            self.cartpole_last_obs = sub_obs
        elif active_phase == self.PHASE_MOUNTAINCAR:
            self.mountaincar_last_obs = sub_obs
        elif active_phase == self.PHASE_LUNARLANDER:
            self.lunarlander_last_obs = sub_obs
        elif active_phase == self.PHASE_BIPEDALWALKER:
            self.bipedalwalker_last_obs = sub_obs

        # Check if active phase completed
        done_flags = {
            self.PHASE_CARTPOLE: sub_obs.done if active_phase == self.PHASE_CARTPOLE else self.cartpole_done,
            self.PHASE_MOUNTAINCAR: sub_obs.done if active_phase == self.PHASE_MOUNTAINCAR else self.mountaincar_done,
            self.PHASE_LUNARLANDER: sub_obs.done if active_phase == self.PHASE_LUNARLANDER else self.lunarlander_done,
            self.PHASE_BIPEDALWALKER: sub_obs.done if active_phase == self.PHASE_BIPEDALWALKER else self.bipedalwalker_done,
        }

        # Update completion status
        self.cartpole_done = done_flags[self.PHASE_CARTPOLE]
        self.mountaincar_done = done_flags[self.PHASE_MOUNTAINCAR]
        self.lunarlander_done = done_flags[self.PHASE_LUNARLANDER]
        self.bipedalwalker_done = done_flags[self.PHASE_BIPEDALWALKER]

        # Update state
        self._state.step_count += 1
        self._state.episode_length += 1
        self._state.total_reward += sub_obs.reward
        self._state.cartpole_done = self.cartpole_done
        self._state.mountaincar_done = self.mountaincar_done
        self._state.lunarlander_done = self.lunarlander_done
        self._state.bipedalwalker_done = self.bipedalwalker_done

        # Select next phase (for next step)
        next_phase = self._select_phase()
        self._state.active_phase = next_phase
        self._state.next_environment = next_phase  # The phase that will be stepped next time

        # Check episode termination conditions
        episode_done = (self._state.step_count >= self.max_steps or
                       all([self.cartpole_done, self.mountaincar_done, self.lunarlander_done, self.bipedalwalker_done]))

        # Create observation with state from next phase for next action decision
        observation = self._make_observation_from_phase(next_phase, reward=sub_obs.reward, done=episode_done)

        logger.info("Step %d: phase=%s, reward=%.3f, done=%s, next_phase=%s",
                   self._state.step_count, active_phase, sub_obs.reward, episode_done, next_phase)

        if episode_done:
            logger.info("Episode %s ended - length=%d, total_reward=%.3f",
                       self._state.episode_id, self._state.episode_length, self._state.total_reward)
            if self._state.step_count >= self.max_steps:
                logger.info("Episode ended due to max_steps reached")

        return observation

    @property
    def state(self) -> SequentialState:
        """Get current environment state."""
        return self._state

    def _select_phase(self) -> PhaseType:
        """
        Randomly select a phase that hasn't completed yet.

        Returns:
            Selected phase name.
        """
        available_phases = []
        if not self.cartpole_done:
            available_phases.append(self.PHASE_CARTPOLE)
        if not self.mountaincar_done:
            available_phases.append(self.PHASE_MOUNTAINCAR)
        if not self.lunarlander_done:
            available_phases.append(self.PHASE_LUNARLANDER)
        if not self.bipedalwalker_done:
            available_phases.append(self.PHASE_BIPEDALWALKER)

        if not available_phases:
            # All phases done, return any (episode should end)
            return self._state.active_phase if self._state.active_phase else self.PHASE_CARTPOLE

        return self.rng.choice(available_phases)

    def _step_sub_env(self, phase: PhaseType, action: SequentialAction) -> Any:
        """
        Step the appropriate sub-environment based on phase.

        Args:
            phase: Which phase to step.
            action: SequentialAction containing sub-actions.

        Returns:
            Observation from sub-environment.
        """
        if phase == self.PHASE_CARTPOLE:
            if action.cartpole_action is None:
                raise ValueError("cartpole_action must be provided for cartpole phase")
            cartpole_action = CartpoleAction(action_id=action.cartpole_action)
            return self.cartpole_env.step(cartpole_action)

        elif phase == self.PHASE_MOUNTAINCAR:
            if action.mountaincar_action is None:
                raise ValueError("mountaincar_action must be provided for mountaincar phase")
            mountaincar_action = MountainCarContinuousAction(engine_force=action.mountaincar_action)
            return self.mountaincar_env.step(mountaincar_action)

        elif phase == self.PHASE_LUNARLANDER:
            if action.lunarlander_action is None or len(action.lunarlander_action) != 2:
                raise ValueError("lunarlander_action must be a list of 2 floats for lunarlander phase")
            lunarlander_action = LunarLanderAction(
                main_engine=action.lunarlander_action[0],
                lateral_engine=action.lunarlander_action[1]
            )
            return self.lunarlander_env.step(lunarlander_action)

        elif phase == self.PHASE_BIPEDALWALKER:
            logger.info(f"BipedalWalker action received: {action.bipedalwalker_action}")
            logger.info(f"BipedalWalker action type: {type(action.bipedalwalker_action)}")
            if action.bipedalwalker_action is not None:
                logger.info(f"BipedalWalker action length: {len(action.bipedalwalker_action)}")
                logger.info(f"BipedalWalker action elements: {[type(x) for x in action.bipedalwalker_action]}")

            if action.bipedalwalker_action is None or len(action.bipedalwalker_action) != 4:
                raise ValueError("bipedalwalker_action must be a list of 4 floats for bipedalwalker phase")
            bipedalwalker_action = GymAction(action=action.bipedalwalker_action)
            return self.bipedalwalker_env.step(bipedalwalker_action)

        else:
            raise ValueError(f"Unknown phase: {phase}")

    def _get_last_obs_state(self, phase: PhaseType) -> List[float]:
        """
        Get the state list from the last observation of a specific phase.

        Args:
            phase: Which phase to get observation state for.

        Returns:
            List containing the observation state values, or empty if no observation.
        """
        last_obs = None
        if phase == self.PHASE_CARTPOLE:
            last_obs = self.cartpole_last_obs
        elif phase == self.PHASE_MOUNTAINCAR:
            last_obs = self.mountaincar_last_obs
        elif phase == self.PHASE_LUNARLANDER:
            last_obs = self.lunarlander_last_obs
        elif phase == self.PHASE_BIPEDALWALKER:
            last_obs = self.bipedalwalker_last_obs

        if last_obs and hasattr(last_obs, 'state'):
            return last_obs.state if isinstance(last_obs.state, list) else last_obs.state.tolist()
        return []

    def _make_observation_from_phase(self, phase: PhaseType, reward: float, done: bool) -> SequentialObservation:
        """
        Create a SequentialObservation based on the given phase.

        Args:
            phase: The phase for which to create the observation.
            reward: Reward from this step.
            done: Whether the episode is done.

        Returns:
            SequentialObservation with the appropriate sub-environment state.
        """
        # Create one-hot phase vector
        phase_vector = [0.0, 0.0, 0.0, 0.0]
        if phase == self.PHASE_CARTPOLE:
            phase_vector[0] = 1.0
        elif phase == self.PHASE_MOUNTAINCAR:
            phase_vector[1] = 1.0
        elif phase == self.PHASE_LUNARLANDER:
            phase_vector[2] = 1.0
        elif phase == self.PHASE_BIPEDALWALKER:
            phase_vector[3] = 1.0

        # Get the sub-environment state for this phase
        sub_state = self._get_last_obs_state(phase)

        observation = SequentialObservation(
            state=phase_vector + sub_state,
            phase=phase,
            sub_observation=sub_state,
            done=done,
            reward=reward,
            episode_length=self._state.episode_length,
            total_reward=self._state.total_reward,
            metadata={
                "render_mode": self.render_mode,
                "max_steps": self.max_steps,
                "seed": self.seed,
                "phase": phase,
                "next_environment": self._state.next_environment,
            },
        )

        return observation

    def close(self) -> None:
        """Close all sub-environments and clean up resources."""
        logger.info("Closing SequentialEnvironment")
        if hasattr(self.cartpole_env, 'close'):
            self.cartpole_env.close()
        if hasattr(self.mountaincar_env, 'close'):
            self.mountaincar_env.close()
        if hasattr(self.lunarlander_env, 'close'):
            self.lunarlander_env.close()
        if hasattr(self.bipedalwalker_env, 'close'):
            self.bipedalwalker_env.close()
        logger.info("SequentialEnvironment closed successfully")
