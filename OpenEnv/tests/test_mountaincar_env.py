#!/usr/bin/env python3
"""
MountainCarContinuous Environment Client Script with Strategy Testing.

This script tests different control strategies for the MountainCarContinuous environment,
similar to the cartpole strategy testing script.
"""

import time
import requests
import random
import numpy as np
from envs.mountaincarcontinuous_environment import MountainCarContinuousEnv, MountainCarContinuousAction

# Configuration
base_url = "http://localhost:8050"  # MountainCarContinuous server port
request_timeout_s = 1000  # seconds

def smart_mountaincar_strategy(state):
    """
    Smart MountainCarContinuous control strategy.

    The goal is to build momentum by swinging back and forth, then use that momentum
    to climb the hill. This strategy applies maximum force in the direction that
    will help build potential energy.

    Args:
        state: List of 2 floats [position, velocity]
            - position: car position along the track (-1.2 to 0.6)
            - velocity: car velocity (-0.07 to 0.07)

    Returns:
        engine_force: float between -1.0 and 1.0
    """
    position, velocity = state

    # If we're moving right and at low position, keep going right to build speed
    # If we're moving left and at low position, keep going left to build speed
    # If we're at high position, try to go right to reach the goal

    if position < -0.5:
        # We're in the left valley, build momentum by going left when moving right
        # and right when moving left
        if velocity > 0:
            return -1.0  # Going right, apply left force to swing back
        else:
            return 1.0   # Going left, apply right force to swing back
    else:
        # We're on the right side, try to go right to reach the goal
        return 1.0

def decent_mountaincar_strategy(state):
    """
    A decent but not perfect strategy for MountainCarContinuous.

    Uses a simple heuristic: if velocity is positive, apply positive force,
    if velocity is negative, apply negative force. This helps maintain momentum
    but doesn't optimally build potential energy.

    Args:
        state: List of 2 floats [position, velocity]

    Returns:
        engine_force: float between -1.0 and 1.0
    """
    position, velocity = state

    # Simple momentum-based strategy
    if velocity > 0:
        return 1.0   # Keep going right
    else:
        return -1.0  # Go left to build momentum

def shitty_mountaincar_strategy(state):
    """
    A deliberately bad strategy that always applies maximum right force.

    This will likely get stuck in the right valley and never reach the goal.

    Args:
        state: Current state (ignored)

    Returns:
        engine_force: Always 1.0 (full right force)
    """
    return 1.0

def random_mountaincar_strategy(state):
    """
    A random strategy that applies random forces.

    Args:
        state: Current state (ignored)

    Returns:
        engine_force: Random float between -1.0 and 1.0
    """
    return random.uniform(-1.0, 1.0)

def test_single_strategy(client, strategy_func, strategy_name, num_episodes=5, max_steps_per_episode=1000, use_random_seeds=True):
    """
    Test a single strategy over multiple episodes.

    Args:
        client: MountainCarContinuousEnv client
        strategy_func: Function that takes state and returns engine_force
        strategy_name: Name of the strategy for display
        num_episodes: Number of episodes to test
        max_steps_per_episode: Maximum steps per episode before truncation
        use_random_seeds: Whether to use random seeds for each episode

    Returns:
        dict: Performance metrics
    """
    print(f"\n🧪 Testing {strategy_name} Strategy over {num_episodes} episodes...")
    print("=" * 60)

    episode_lengths = []
    episode_rewards = []
    final_positions = []
    success_count = 0

    for episode in range(num_episodes):
        # Use random seed for each episode if requested
        seed = random.randint(0, 10000) if use_random_seeds else None

        # Reset environment with seed
        result = client.reset()
        episode_reward = 0
        episode_length = 0
        done = False
        max_position = result.observation.state[0]

        print(f"\n📍 Episode {episode + 1}/{num_episodes} (seed={seed})")
        print(f"   Initial state: {result.observation.state}")

        while not done and episode_length < max_steps_per_episode:
            # Get current state
            current_state = result.observation.state

            # Apply strategy to get engine force
            engine_force = strategy_func(current_state)
            engine_force = np.clip(engine_force, -1.0, 1.0)  # Ensure valid range

            # Take action
            result = client.step(MountainCarContinuousAction(
                engine_force=engine_force,
                seed=seed
            ))

            episode_reward += result.reward or 0
            episode_length += 1
            done = result.done
            max_position = max(max_position, current_state[0])

            # Print progress every 100 steps
            if episode_length % 100 == 0:
                print(f"   Step {episode_length}: pos={current_state[0]:.3f}, vel={current_state[1]:.3f}, force={engine_force:.2f}, reward={result.reward}")

        # Record episode results
        episode_lengths.append(episode_length)
        episode_rewards.append(episode_reward)
        final_positions.append(result.observation.state[0])

        # Check if episode was successful (reached goal position >= 0.45)
        if result.observation.state[0] >= 0.45:
            success_count += 1
            print(f"   🎉 SUCCESS! Reached goal position: {result.observation.state[0]:.3f}")
        else:
            print(f"   ❌ Failed to reach goal. Final position: {result.observation.state[0]:.3f}")

        print(f"   ✅ Episode {episode + 1} finished!")
        print(f"   Length: {episode_length} steps")
        print(f"   Total reward: {episode_reward:.2f}")
        print(f"   Max position reached: {max_position:.3f}")

    # Calculate performance metrics
    avg_length = sum(episode_lengths) / len(episode_lengths)
    avg_reward = sum(episode_rewards) / len(episode_rewards)
    avg_final_position = sum(final_positions) / len(final_positions)
    max_length = max(episode_lengths)
    min_length = min(episode_lengths)
    success_rate = success_count / num_episodes

    print("\n" + "=" * 60)
    print(f"📊 {strategy_name.upper()} STRATEGY RESULTS")
    print("=" * 60)
    print(f"Episodes tested: {num_episodes}")
    print(f"Average episode length: {avg_length:.2f} steps")
    print(f"Average episode reward: {avg_reward:.2f}")
    print(f"Average final position: {avg_final_position:.3f}")
    print(f"Best episode length: {max_length} steps")
    print(f"Worst episode length: {min_length} steps")
    print(f"Success rate: {success_rate:.1%} ({success_count}/{num_episodes})")

    # Performance assessment
    if success_rate >= 0.8:
        print("🎉 Excellent! Strategy consistently reaches the goal.")
    elif success_rate >= 0.5:
        print("👍 Good! Strategy reaches the goal more often than not.")
    elif success_rate >= 0.2:
        print("😐 Fair. Strategy occasionally reaches the goal.")
    else:
        print("😞 Poor. Strategy rarely reaches the goal.")

    return {
        'strategy_name': strategy_name,
        'avg_length': avg_length,
        'avg_reward': avg_reward,
        'avg_final_position': avg_final_position,
        'max_length': max_length,
        'min_length': min_length,
        'success_rate': success_rate,
        'success_count': success_count
    }

def compare_strategies(client, num_episodes=5):
    """
    Compare all strategies side by side.

    Args:
        client: MountainCarContinuousEnv client
        num_episodes: Number of episodes to test for each strategy
    """
    print("\n" + "🎯" * 80)
    print("STRATEGY COMPARISON: MOUNTAINCAR CONTINUOUS CONTROL")
    print("🎯" * 80)

    strategies = [
        (smart_mountaincar_strategy, "Smart"),
        (decent_mountaincar_strategy, "Decent"),
        (shitty_mountaincar_strategy, "Shitty"),
        (random_mountaincar_strategy, "Random")
    ]

    all_results = []

    for strategy_func, strategy_name in strategies:
        print("\n" + "-" * 80)
        results = test_single_strategy(
            client, strategy_func, strategy_name, num_episodes, use_random_seeds=True
        )
        all_results.append(results)

    print("\n" + "🏆" * 80)
    print("FINAL COMPARISON")
    print("🏆" * 80)

    print(f"{'Strategy':<12} {'Avg Length':<12} {'Avg Reward':<12} {'Success Rate':<12} {'Avg Final Pos':<15}")
    print("-" * 80)
    for results in all_results:
        print(f"{results['strategy_name']:<12} {results['avg_length']:<12.2f} {results['avg_reward']:<12.2f} {results['success_rate']:<12.1%} {results['avg_final_position']:<15.3f}")

    # Determine winner
    best_strategy = max(all_results, key=lambda x: x['success_rate'])
    print(f"\n🎉 WINNER: {best_strategy['strategy_name']} strategy!")
    print(f"   Success rate: {best_strategy['success_rate']:.1%}")
    print(f"   Average final position: {best_strategy['avg_final_position']:.3f}")

    return all_results

def test_basic_functionality(client):
    """Test basic MountainCarContinuous functionality."""
    print("\n🔧 Testing basic functionality...")

    # Reset
    state = client.reset()
    print("✅ Reset successful")
    print(f"   Initial state: {state.observation.state}")

    # Test different engine forces
    test_forces = [-1.0, -0.5, 0.0, 0.5, 1.0]
    for force in test_forces:
        print(f"\nTesting engine force {force}...")
        state = client.step(MountainCarContinuousAction(engine_force=force))
        print(f"   Result: reward={state.reward}, done={state.done}")
        print(f"   New state: {state.observation.state}")

    print("✅ Basic functionality test completed!")

print("Starting MountainCarContinuous Environment Client with Strategy Testing...")
print(f"Connecting to: {base_url}")

# Wait for server to start
print("Waiting 5 seconds for server to start...")
time.sleep(5)

try:
    # Create the environment client
    client = MountainCarContinuousEnv(
        base_url=base_url,
        request_timeout_s=request_timeout_s
    )

    # Quick smoke test - check if server is responding
    print("\nTesting connection...")
    try:
        # Test basic functionality first
        test_basic_functionality(client)

        # Compare all strategies
        results = compare_strategies(client, num_episodes=3)

        print("\n✅ MountainCarContinuous strategy testing completed successfully!")

    except Exception as e:
        print(f"❌ Error during testing: {e}")
        print("Make sure the MountainCarContinuous server is running:")
        print("  uvicorn envs.mountaincarcontinuous_environment.server.app:app --host 0.0.0.0 --port 8010")

except Exception as e:
    print(f"❌ Failed to connect to server: {e}")
    print("Please ensure the MountainCarContinuous server is running on port 8010")

print("\nScript completed!")
