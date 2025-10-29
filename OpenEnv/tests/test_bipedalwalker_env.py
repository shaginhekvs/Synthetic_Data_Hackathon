#!/usr/bin/env python3
"""
BipedalWalker Strategy Tester

This script mirrors the structure of `test_cartpole_env.py` but targets
the `BipedalWalker-v3` environment via the generic `GymEnvironment` HTTP client.

It provides simple strategies and runs multiple episodes to gather basic
performance metrics (episode length, total reward, action distribution).

Usage:
  - Ensure the OpenEnv server exposing a BipedalWalker env is running at
    the configured `base_url` (default: http://localhost:9000).
  - Run the script to print per-episode and aggregate statistics.

This file is intentionally runnable rather than a pytest unit test so it
can be used for interactive strategy experimentation similar to the cartpole
script in the repo.
"""

import time
import random
import math
from typing import List

from envs.gym_environment import client as gym_client_module
from envs.gym_environment.client import GymEnvironment, GymAction

# Configure the HTTP endpoint for the OpenEnv server
BASE_URL = "http://localhost:9000"
ENV_ID = "BipedalWalker-v3"
REQUEST_TIMEOUT_S = 1000


def zero_strategy(state: List[float]) -> List[float]:
    """Always return zero torques (do-nothing strategy)."""
    return [0.0, 0.0, 0.0, 0.0]


def random_strategy(state: List[float]) -> List[float]:
    """Return a random continuous action in the valid range [-1, 1]."""
    return [random.uniform(-1.0, 1.0) for _ in range(4)]


def small_heuristic(state: List[float]) -> List[float]:
    """A tiny heuristic that attempts small corrective torques based on hull angle.

    The BipedalWalker observation contains many values; index 2/3 are commonly
    related to hull angle and angular velocity in many Gym versions, but this
    heuristic is intentionally simple and low-risk: push slightly in the
    direction of the hull angle to attempt to remain upright.
    """
    if not state or len(state) < 3:
        return [0.0, 0.0, 0.0, 0.0]
    angle = float(state[2])
    # small proportional controller -> map angle to motor torques
    torque = max(min(-0.5 * angle, 1.0), -1.0)
    return [torque, torque, torque, torque]


def test_single_strategy(client: GymEnvironment, strategy_func, strategy_name, num_episodes=5):
    print(f"\n🧪 Testing {strategy_name} strategy for {ENV_ID} ({num_episodes} episodes)")
    episode_lengths = []
    episode_rewards = []
    actions_taken = []

    for ep in range(num_episodes):
        res = client.reset()
        done = False
        total_reward = 0.0
        length = 0

        print(f"\n📍 Episode {ep+1}/{num_episodes} - initial state length={len(res.observation.state) if res.observation.state is not None else 'N/A'}")

        while not done:
            state = res.observation.state
            action = strategy_func(state)
            actions_taken.append(action)

            res = client.step(GymAction(action=action, return_frame=False))

            total_reward += res.reward or 0.0
            length += 1
            done = res.done

            # print brief progress
            if length % 200 == 0:
                print(f"   step={length}, reward={total_reward:.2f}")

        episode_lengths.append(length)
        episode_rewards.append(total_reward)

        print(f"   ✅ Episode finished: length={length}, total_reward={total_reward:.2f}")

    avg_len = sum(episode_lengths) / len(episode_lengths)
    avg_reward = sum(episode_rewards) / len(episode_rewards)
    print("\n" + "=" * 60)
    print(f"{strategy_name} results: avg_length={avg_len:.2f}, avg_reward={avg_reward:.2f}")
    return {
        "strategy_name": strategy_name,
        "avg_length": avg_len,
        "avg_reward": avg_reward,
        "episode_lengths": episode_lengths,
        "episode_rewards": episode_rewards,
    }


def compare_strategies(client: GymEnvironment, num_episodes=5):
    print("\n🎯 Comparing strategies for BipedalWalker")
    results_zero = test_single_strategy(client, zero_strategy, "Zero", num_episodes)
    results_random = test_single_strategy(client, random_strategy, "Random", num_episodes)
    results_heuristic = test_single_strategy(client, small_heuristic, "Heuristic", num_episodes)

    print("\n🏆 Summary")
    for r in (results_zero, results_random, results_heuristic):
        print(f" - {r['strategy_name']}: avg_len={r['avg_length']:.2f}, avg_reward={r['avg_reward']:.2f}")

    return results_zero, results_random, results_heuristic


def main():
    print(f"Connecting to OpenEnv Gym server at {BASE_URL} (env={ENV_ID})")
    # small wait for server readiness if started manually
    time.sleep(1)

    client = GymEnvironment(base_url=BASE_URL)

    try:
        # quick smoke test
        res = client.reset()
        print(f"Initial state (len): {len(res.observation.state) if res.observation.state is not None else 'N/A'}")

        # run comparison
        compare_strategies(client, num_episodes=3)

    except Exception as e:
        print(f"Error during strategy testing: {e}")
        print("Make sure the server is running and the env is available at the configured base_url.")
    finally:
        client.close()


if __name__ == "__main__":
    main()
