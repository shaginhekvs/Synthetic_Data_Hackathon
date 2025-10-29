# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Cartpole Environment Server Implementation.

This module wraps Gymnasium's CartPole-v1 environment and exposes it
via the OpenEnv Environment interface with comprehensive logging.
"""

import logging
import uuid
from typing import Any, Dict, Optional

import gymnasium as gym
import numpy as np

from core.env_server import Environment

from ..models import CartpoleAction, CartpoleObservation, CartpoleState

# Set up logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


class CartpoleEnvironment(Environment):
    """
    Cartpole Environment wrapper for OpenEnv.

    This environment wraps Gymnasium's CartPole-v1 environment and provides
    a clean interface for RL training with comprehensive logging.

    The CartPole task involves balancing a pole on a cart by applying left or right forces.
    The episode ends when the pole falls beyond a threshold angle or the cart moves
    too far from the center.

    Args:
        render_mode: Render mode for visualization ("human", "rgb_array", None).
        max_steps: Maximum steps per episode (default: 500, minimum: 500).
        seed: Random seed for reproducibility.

    Example:
        >>> env = CartpoleEnvironment(render_mode="human", max_steps=1000)
        >>> obs = env.reset()
        >>> print(obs.state)  # [cart_pos, cart_vel, pole_angle, pole_vel]
        >>> obs = env.step(CartpoleAction(action_id=1))  # Push right
        >>> print(obs.reward, obs.done)
    """

    def __init__(
        self,
        render_mode: Optional[str] = None,
        max_steps: int = 10000,
        seed: Optional[int] = None,
    ):
        """Initialize Cartpole environment."""
        super().__init__()

        self.render_mode = render_mode
        self.max_steps = max(10000, max_steps)  # Ensure minimum 500 steps
        self.seed = seed

        # Create gymnasium environment
        logger.info(f"Creating CartPole-v1 environment with render_mode={render_mode}, max_steps={self.max_steps}, seed={seed}")
        self.env = gym.make("CartPole-v1", render_mode=render_mode)

        # Standardize episode length (Gymnasium wrapper)
        if max_steps > 0:
            self.env = gym.wrappers.TimeLimit(self.env, max_episode_steps=self.max_steps)

        # Set seed if provided
        if seed is not None:
            self.env.reset(seed=seed)
            logger.info(f"Environment seeded with seed={seed}")

        # Initialize state
        self._state = CartpoleState(
            render_mode=render_mode,
            max_steps=self.max_steps,
            seed=seed,
        )

        logger.info("CartpoleEnvironment initialized successfully")

    def reset(self) -> CartpoleObservation:
        """
        Reset the environment and return initial observation.

        Returns:
            Initial observation for the agent.
        """
        logger.info("Resetting environment")

        # Reset gymnasium environment
        self.seed = self.seed + 1
        obs, info = self.env.reset(seed=self.seed)

        # Reset state tracking
        self._state.episode_id = str(uuid.uuid4())
        self._state.step_count = 0
        self._state.episode_length = 0
        self._state.total_reward = 0.0

        # Create observation
        observation = self._make_observation(obs, reward=0.0, done=False)

        logger.info(f"Environment reset - Episode {self._state.episode_id} started")
        logger.info(f"Initial state: {obs}")

        return observation

    def step(self, action: CartpoleAction) -> CartpoleObservation:
        """
        Execute agent's action and return resulting observation.

        Args:
            action: CartpoleAction containing the action_id to execute.

        Returns:
            Observation after action execution.

        Raises:
            ValueError: If action is not a CartpoleAction or action_id is invalid.
        """
        if not isinstance(action, CartpoleAction):
            raise ValueError(f"Expected CartpoleAction, got {type(action)}")

        # Validate action_id
        if action.action_id not in [0, 1]:
            raise ValueError(f"Invalid action_id: {action.action_id}. Valid actions: [0, 1]")

        logger.info(f"Taking action: {action.action_id} (0=left, 1=right)")

        # Execute action in gymnasium environment
        obs, reward, terminated, truncated, info = self.env.step(int(action.action_id))

        # Update state tracking
        self._state.step_count += 1
        self._state.episode_length += 1
        self._state.total_reward += reward

        # Determine if episode is done
        done = terminated or truncated

        # Create observation
        observation = self._make_observation(obs, reward, done)

        # Log step details
        logger.info(f"Step {self._state.step_count}: reward={reward}, terminated={terminated}, truncated={truncated}")
        logger.info(f"New state: {obs}")
        logger.info(f"Episode progress: length={self._state.episode_length}, total_reward={self._state.total_reward}")

        if done:
            logger.info(f"Episode {self._state.episode_id} ended - length={self._state.episode_length}, total_reward={self._state.total_reward}")
            if terminated:
                logger.info("Episode ended due to termination (pole fell or cart out of bounds)")
            if truncated:
                logger.info(f"Episode ended due to truncation (max_steps={self.max_steps} reached)")

        return observation

    @property
    def state(self) -> CartpoleState:
        """Get current environment state."""
        return self._state

    def _make_observation(self, obs: np.ndarray, reward: float, done: bool) -> CartpoleObservation:
        """
        Create a CartpoleObservation from current environment state.

        Args:
            obs: Numpy array observation from gymnasium.
            reward: Reward from the step.
            done: Whether the episode is done.

        Returns:
            CartpoleObservation for the agent.
        """
        # Convert numpy array to list for JSON serialization
        state_list = obs.tolist() if hasattr(obs, 'tolist') else list(obs)

        # Create observation
        observation = CartpoleObservation(
            state=state_list,
            done=done,
            reward=reward,
            episode_length=self._state.episode_length,
            total_reward=self._state.total_reward,
            metadata={
                "render_mode": self.render_mode,
                "max_steps": self.max_steps,
                "seed": self.seed,
                "action_meanings": ["left", "right"],
            },
        )

        return observation

    def close(self) -> None:
        """Close the environment and clean up resources."""
        logger.info("Closing CartpoleEnvironment")
        if hasattr(self.env, 'close'):
            self.env.close()
        logger.info("CartpoleEnvironment closed successfully")
