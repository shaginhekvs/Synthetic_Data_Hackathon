#!/usr/bin/env python3
"""
LunarLander Environment Client Script with Strategy Testing.

This script tests different control strategies for the LunarLander environment,
similar to the cartpole and mountaincar strategy testing scripts.
"""

import time
import requests
import random
import numpy as np
from envs.lunarlander_environment import LunarLanderEnv, LunarLanderAction

# Configuration
base_url = "http://localhost:8070"  # LunarLander server port
request_timeout_s = 1000  # seconds

def smart_lunarlander_strategy(state):
    """
    Smart LunarLander control strategy.

    Uses PID-like control to stabilize and land safely. Integrates position,
    velocity, and angle to apply appropriate thrust.

    Args:
        state: List of 8 floats [x_pos, y_pos, x_vel, y_vel, angle, ang_vel, left_contact, right_contact]

    Returns:
        tuple: (main_engine [-1,1], lateral_engine [-1,1])
    """
    x_pos, y_pos, x_vel, y_vel, angle, ang_vel, left_contact, right_contact = state

    # Vertical control (main engine) - gentle descent
    target_y_vel = -1.0  # Slow downward velocity target
    y_error = target_y_vel - y_vel
    main_engine = np.clip(y_error * 0.5, -1.0, 1.0)  # Proportional control

    # If we're too fast downward or close to ground, increase thrust
    if y_vel < -2.0 or y_pos < 0.5:
        main_engine = np.clip(main_engine + 0.3, -1.0, 1.0)

    # Lateral control - center on landing zone
    target_x = 0.0  # Land at center
    x_error = target_x - x_pos
    lateral_engine = np.clip(x_error * 0.05, -1.0, 1.0)

    # Reduce thrust if nearly upright to stabilize
    if abs(angle) < 0.1 and abs(ang_vel) < 0.1:
        main_engine *= 0.7

    return main_engine, lateral_engine

def decent_lunarlander_strategy(state):
    """
    A decent but not perfect strategy for LunarLander.

    Applies thrust based on simple heuristics: more thrust when falling,
    lateral thrust to correct position.

    Args:
        state: Current state (8 floats)

    Returns:
        tuple: (main_engine [-1,1], lateral_engine [-1,1])
    """
    x_pos, y_pos, x_vel, y_vel, angle, ang_vel, left_contact, right_contact = state

    # Main engine: proportional to downward velocity
    main_engine = np.clip(-y_vel * 0.3, -1.0, 1.0)

    # Lateral engine: push toward center
    lateral_engine = np.clip(-x_pos * 0.1, -1.0, 1.0)

    return main_engine, lateral_engine

def shitty_lunarlander_strategy(state):
    """
    A deliberately bad strategy that only uses maximum main thrust.

    This will likely crash the lander by not controlling lateral movement
    and potentially overwhelming the vertical descent.

    Args:
        state: Current state (ignored)

    Returns:
        tuple: Always (1.0, 0.0) - full main thrust, no lateral
    """
    return 1.0, 0.0

def random_lunarlander_strategy(state):
    """
    A random strategy that applies random thrusts.

    Args:
        state: Current state (ignored)

    Returns:
        tuple: Random main_engine [0,1] and lateral_engine [-1,1]
    """
    return random.uniform(0.0, 1.0), random.uniform(-1.0, 1.0)

def test_single_strategy(client, strategy_func, strategy_name, num_episodes=5, max_steps_per_episode=1000, use_random_seeds=True):
    """
    Test a single strategy over multiple episodes.

    Args:
        client: LunarLanderEnv client
        strategy_func: Function that takes state and returns (main_engine, lateral_engine)
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
    final_x_positions = []
    success_count = 0

    for episode in range(num_episodes):
        # Use random seed for each episode if requested
        seed = random.randint(0, 10000) if use_random_seeds else None

        # Reset environment with seed
        result = client.reset()
        episode_reward = 0
        episode_length = 0
        done = False
        max_height = result.observation.state[1]

        print(f"\n📍 Episode {episode + 1}/{num_episodes} (seed={seed})")
        print(f"   Initial state: pos=[{result.observation.state[0]:.3f}, {result.observation.state[1]:.3f}] vel=[{result.observation.state[2]:.3f}, {result.observation.state[3]:.3f}]")

        while not done and episode_length < max_steps_per_episode:
            # Get current state
            current_state = result.observation.state

            # Apply strategy to get engine powers
            main_engine, lateral_engine = strategy_func(current_state)
            main_engine = np.clip(main_engine, 0.0, 1.0)
            lateral_engine = np.clip(lateral_engine, -1.0, 1.0)

            # Take action
            result = client.step(LunarLanderAction(
                main_engine=main_engine,
                lateral_engine=lateral_engine,
                seed=seed
            ))

            episode_reward += result.reward or 0
            episode_length += 1
            done = result.done
            max_height = max(max_height, current_state[1])

            # Print progress every 50 steps
            if episode_length % 50 == 0:
                x_pos, y_pos, x_vel, y_vel, angle, ang_vel, left, right = current_state
                print(f"   Step {episode_length:3d}: pos=[{x_pos:+.2f}, {y_pos:.2f}] vel=[{x_vel:+.2f}, {y_vel:+.2f}] angle={angle:.2f} thrust=[{main_engine:.2f}, {lateral_engine:+.2f}]")

        # Record episode results
        episode_lengths.append(episode_length)
        episode_rewards.append(episode_reward)
        final_x_positions.append(result.observation.state[0])
        final_positions.append(result.observation.state[1])

        # Check if episode was successful (landed safely between flags)
        # Success criteria: landed upright, low velocity, between x=-0.2 to 0.2
        x_pos, y_pos, x_vel, y_vel, angle, ang_vel, left_contact, right_contact = result.observation.state
        landed_safely = (y_pos <= 0.05 and y_pos >= 0) and (left_contact == 1 or right_contact == 1) and abs(angle) < 0.2 and abs(x_vel) < 1.0 and abs(y_vel) < 1.0

        if landed_safely:
            success_count += 1
            print(f"   🎉 SUCCESS! Landed safely at x={x_pos:.3f}, stable angle and velocity")
        else:
            print(f"   ❌ Failed to land safely. Final: x={x_pos:.3f}, y={y_pos:.3f}, angle={angle:.3f}")

        print(f"   ✅ Episode {episode + 1} finished!")
        print(f"   Length: {episode_length} steps")
        print(f"   Total reward: {episode_reward:.2f}")
        print(f"   Max height reached: {max_height:.3f}")

    # Calculate performance metrics
    avg_length = sum(episode_lengths) / len(episode_lengths)
    avg_reward = sum(episode_rewards) / len(episode_rewards)
    avg_final_x = sum(final_x_positions) / len(final_x_positions)
    avg_final_y = sum(final_positions) / len(final_positions)
    max_length = max(episode_lengths)
    min_length = min(episode_lengths)
    success_rate = success_count / num_episodes

    print("\n" + "=" * 60)
    print(f"📊 {strategy_name.upper()} STRATEGY RESULTS")
    print("=" * 60)
    print(f"Episodes tested: {num_episodes}")
    print(f"Average episode length: {avg_length:.2f} steps")
    print(f"Average episode reward: {avg_reward:.2f}")
    print(f"Average final position: x={avg_final_x:.3f}, y={avg_final_y:.3f}")
    print(f"Best episode length: {max_length} steps")
    print(f"Worst episode length: {min_length} steps")
    print(f"Success rate: {success_rate:.1%} ({success_count}/{num_episodes})")

    # Performance assessment
    if success_rate >= 0.8:
        print("🎉 Excellent! Strategy lands safely most of the time.")
    elif success_rate >= 0.5:
        print("👍 Good! Strategy lands safely more often than not.")
    elif success_rate >= 0.2:
        print("😐 Fair. Strategy lands occasionally.")
    else:
        print("😞 Poor. Strategy rarely lands safely.")

    return {
        'strategy_name': strategy_name,
        'avg_length': avg_length,
        'avg_reward': avg_reward,
        'avg_final_x': avg_final_x,
        'avg_final_y': avg_final_y,
        'max_length': max_length,
        'min_length': min_length,
        'success_rate': success_rate,
        'success_count': success_count
    }

def compare_strategies(client, num_episodes=3):
    """
    Compare all strategies side by side.

    Args:
        client: LunarLanderEnv client
        num_episodes: Number of episodes to test for each strategy
    """
    print("\n" + "🚀" * 80)
    print("STRATEGY COMPARISON: LUNAR LANDER CONTROL")
    print("🚀" * 80)

    strategies = [
        (smart_lunarlander_strategy, "Smart"),
        (decent_lunarlander_strategy, "Decent"),
        (shitty_lunarlander_strategy, "Shitty"),
        (random_lunarlander_strategy, "Random")
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

    print(f"{'Strategy':<12} {'Avg Length':<12} {'Avg Reward':<12} {'Success Rate':<12} {'Avg Final X':<12}")
    print("-" * 80)
    for results in all_results:
        print(f"{results['strategy_name']:<12} {results['avg_length']:<12.2f} {results['avg_reward']:<12.2f} {results['success_rate']:<12.1%} {results['avg_final_x']:<12.3f}")

    # Determine winner
    best_strategy = max(all_results, key=lambda x: x['success_rate'])
    print(f"\n🎉 WINNER: {best_strategy['strategy_name']} strategy!")
    print(f"   Success rate: {best_strategy['success_rate']:.1%}")
    print(f"   Average reward: {best_strategy['avg_reward']:.2f}")

    return all_results

def test_basic_functionality(client):
    """Test basic LunarLander functionality."""
    print("\n🔧 Testing basic functionality...")

    # Reset
    state = client.reset()
    print("✅ Environment reset successful")
    print(f"   Initial state length: {len(state.observation.state)} (should be 8)")
    print(f"   Initial position: [{state.observation.state[0]:.3f}, {state.observation.state[1]:.3f}]")

    # Test different engine combinations
    test_actions = [
        (0.0, 0.0),    # No thrust
        (0.5, 0.0),    # Medium main thrust
        (1.0, 0.0),    # Full main thrust
        (0.5, 0.5),    # Main + right thrust
        (0.5, -0.5),   # Main + left thrust
    ]

    for main, lateral in test_actions:
        print(f"\nTesting action: main={main}, lateral={lateral}")
        state = client.step(LunarLanderAction(main_engine=main, lateral_engine=lateral))
        print(f"   Result: reward={state.reward:.2f}, done={state.done}")
        x_pos, y_pos, x_vel, y_vel, angle, ang_vel, left, right = state.observation.state
        print(f"   Position: [{x_pos:.3f}, {y_pos:.3f}], Velocity: [{x_vel:.3f}, {y_vel:.3f}]")

        if state.done:
            print("   - Episode ended!")
            break

    print("✅ Basic functionality test completed!")

print("Starting LunarLander Environment Client with Strategy Testing...")
print(f"Connecting to: {base_url}")

# Wait for server to start
print("Waiting 5 seconds for server to start...")
time.sleep(5)

try:
    # Create the environment client
    client = LunarLanderEnv(
        base_url=base_url,
        request_timeout_s=request_timeout_s
    )

    # Quick smoke test - check if server is responding
    print("\nTesting connection...")
    try:
        # Test basic functionality first
        test_basic_functionality(client)

        # Compare all strategies
        results = compare_strategies(client, num_episodes=2)

        print("\n✅ LunarLander strategy testing completed successfully!")

    except Exception as e:
        print(f"❌ Error during testing: {e}")
        print("Make sure the LunarLander server is running:")
        print("  uvicorn envs.lunarlander_environment.server.app:app --host 0.0.0.0 --port 8070")

except Exception as e:
    print(f"❌ Failed to connect to server: {e}")
    print("Please ensure the LunarLander server is running on port 8070")

print("\nScript completed!")
