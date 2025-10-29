#!/usr/bin/env python3
"""
Simple example demonstrating Atari Environment usage.

This example shows how to:
1. Connect to an Atari environment
2. Reset the environment
3. Take random actions
4. Process observations

Usage:
    # First, start the server:
    python -m envs.atari_env.server.app

    # Then run this script:
    python examples/atari_simple.py
"""

import numpy as np
from envs.atari_env import AtariEnv, AtariAction


def main():
    """Run a simple Atari episode."""
    # Connect to the Atari environment server
    print("Connecting to Atari environment...")
    env = AtariEnv.from_docker_image("ghcr.io/meta-pytorch/openenv-atari-env:latest")

    try:
        # Reset the environment
        print("\nResetting environment...")
        result = env.reset()
        print(f"Screen shape: {result.observation.screen_shape}")
        print(f"Legal actions: {result.observation.legal_actions}")
        print(f"Lives: {result.observation.lives}")

        # Run a few steps with random actions
        print("\nTaking random actions...")
        episode_reward = 0
        steps = 0

        for step in range(100):
            # Random action
            action_id = np.random.choice(result.observation.legal_actions)
            action_id = int(action_id)
            
            # Take action
            result = env.step(AtariAction(action_id=action_id))

            episode_reward += result.reward or 0
            steps += 1

            # Print progress
            if step % 10 == 0:
                print(
                    f"Step {step}: reward={result.reward:.2f}, "
                    f"lives={result.observation.lives}, done={result.done}"
                )

            if result.done:
                print(f"\nEpisode finished after {steps} steps!")
                break

        print(f"\nTotal episode reward: {episode_reward:.2f}")

        # Get environment state
        state = env.state()
        print(f"\nEnvironment state:")
        print(f"  Game: {state.game_name}")
        print(f"  Episode: {state.episode_id}")
        print(f"  Steps: {state.step_count}")
        print(f"  Obs type: {state.obs_type}")

    finally:
        # Cleanup
        print("\nClosing environment...")
        env.close()
        print("Done!")


if __name__ == "__main__":
    main()
