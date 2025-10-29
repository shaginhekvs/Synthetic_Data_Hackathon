#!/usr/bin/env python3
"""
Corrected Cartpole Environment Client Script with Strategy Testing.

This is a corrected version of the user's original script with the following fixes:
1. Correct import path: envs.cartpole_environment (not envs.cartpole_env)
2. Removed non-existent info() method call
3. Corrected base_url to 8000 (standard port)
4. Removed unnecessary httpx import
5. Added proper error handling
6. Added cartpole strategy testing
"""

import time
import requests  # For health check if needed
import random
from envs.cartpole_environment import CartpoleEnv, CartpoleAction

# Configuration
base_url = "http://localhost:8030"  # Corrected port
request_timeout_s = 1000  # seconds

def cartpole_strategy(state):
    """
    Cartpole control strategy.

    Args:
        state: List of 4 floats [x, dx, angle, dangle]
            - x: cart position
            - dx: cart velocity
            - angle: pole angle (radians)
            - dangle: pole angular velocity

    Returns:
        action_id: 0 (push left) or 1 (push right)
    """
    # state: [x, dx, angle, dangle]
    x, dx, ang, dang = state
    score = ang + 0.1 * dang
    return 1 if score > 0.0 else 0

def shitty_strategy(state):
    """
    A deliberately bad strategy that always pushes left.

    Args:
        state: Current state (ignored)

    Returns:
        action_id: Always 0 (push left)
    """
    return 0

def test_single_strategy(client, strategy_func, strategy_name, num_episodes=10, use_random_seeds=True):
    """
    Test a single strategy over multiple episodes.

    Args:
        client: CartpoleEnv client
        strategy_func: Function that takes state and returns action_id
        strategy_name: Name of the strategy for display
        num_episodes: Number of episodes to test
        use_random_seeds: Whether to use random seeds for each episode

    Returns:
        dict: Performance metrics
    """
    print(f"\n🧪 Testing {strategy_name} Strategy over {num_episodes} episodes...")
    print("=" * 60)

    episode_lengths = []
    episode_rewards = []
    actions_taken = []

    for episode in range(num_episodes):
        # Use random seed for each episode if requested
        seed = random.randint(0, 10000) if use_random_seeds else None

        # Reset environment with seed
        result = client.reset()
        episode_reward = 0
        episode_length = 0
        done = False

        print(f"\n📍 Episode {episode + 1}/{num_episodes} (seed={seed})")
        print(f"   Initial state: {result.observation.state}")

        while not done:
            # Get current state
            current_state = result.observation.state

            # Apply strategy to get action
            action_id = strategy_func(current_state)
            actions_taken.append(action_id)

            # Take action
            result = client.step(CartpoleAction(action_id=action_id, seed=seed))

            episode_reward += result.reward or 0
            episode_length += 1
            done = result.done

            # Print progress every 50 steps
            if episode_length % 50 == 0:
                print(f"   Step {episode_length}: state={current_state}, action={action_id}, reward={result.reward}")

        # Record episode results
        episode_lengths.append(episode_length)
        episode_rewards.append(episode_reward)

        print(f"   ✅ Episode {episode + 1} finished!")
        print(f"   Length: {episode_length} steps")
        print(f"   Total reward: {episode_reward:.2f}")

    # Calculate performance metrics
    avg_length = sum(episode_lengths) / len(episode_lengths)
    avg_reward = sum(episode_rewards) / len(episode_rewards)
    max_length = max(episode_lengths)
    min_length = min(episode_lengths)

    # Action distribution
    action_0_count = actions_taken.count(0)
    action_1_count = actions_taken.count(1)
    total_actions = len(actions_taken)

    print("\n" + "=" * 60)
    print(f"📊 {strategy_name.upper()} STRATEGY RESULTS")
    print("=" * 60)
    print(f"Episodes tested: {num_episodes}")
    print(f"Average episode length: {avg_length:.2f} steps")
    print(f"Average episode reward: {avg_reward:.2f}")
    print(f"Best episode length: {max_length} steps")
    print(f"Worst episode length: {min_length} steps")
    print(f"Total actions taken: {total_actions}")
    print(f"Left actions (0): {action_0_count} ({100*action_0_count/total_actions:.1f}%)")
    print(f"Right actions (1): {action_1_count} ({100*action_1_count/total_actions:.1f}%)")

    # Performance assessment
    if avg_length > 100:
        print("🎉 Excellent! Strategy achieves good balance control.")
    elif avg_length > 50:
        print("👍 Good! Strategy shows reasonable control.")
    elif avg_length > 20:
        print("😐 Fair. Strategy needs improvement.")
    else:
        print("😞 Poor. Strategy performs worse than random.")

    return {
        'strategy_name': strategy_name,
        'avg_length': avg_length,
        'avg_reward': avg_reward,
        'max_length': max_length,
        'min_length': min_length,
        'action_distribution': {'left': action_0_count, 'right': action_1_count}
    }

def compare_strategies(client, num_episodes=10):
    """
    Compare both strategies side by side.

    Args:
        client: CartpoleEnv client
        num_episodes: Number of episodes to test for each strategy
    """
    print("\n" + "🎯" * 80)
    print("STRATEGY COMPARISON: CARTPOLE CONTROL")
    print("🎯" * 80)

    # Test smart strategy
    smart_results = test_single_strategy(
        client, cartpole_strategy, "Smart", num_episodes, use_random_seeds=True
    )

    print("\n" + "-" * 80)

    # Test shitty strategy
    shitty_results = test_single_strategy(
        client, shitty_strategy, "Shitty", num_episodes, use_random_seeds=True
    )

    print("\n" + "🏆" * 80)
    print("FINAL COMPARISON")
    print("🏆" * 80)

    print(f"{'Strategy':<15} {'Avg Length':<12} {'Avg Reward':<12} {'Best':<8} {'Worst':<8}")
    print("-" * 70)
    print(f"{smart_results['strategy_name']:<15} {smart_results['avg_length']:<12.2f} {smart_results['avg_reward']:<12.2f} {smart_results['max_length']:<8} {smart_results['min_length']:<8}")
    print(f"{shitty_results['strategy_name']:<15} {shitty_results['avg_length']:<12.2f} {shitty_results['avg_reward']:<12.2f} {shitty_results['max_length']:<8} {shitty_results['min_length']:<8}")

    # Determine winner
    if smart_results['avg_length'] > shitty_results['avg_length']:
        print(f"\n🎉 WINNER: {smart_results['strategy_name']} strategy!")
        print(f"   Improvement: {smart_results['avg_length'] - shitty_results['avg_length']:.2f} steps on average")
    else:
        print(f"\n😞 WINNER: {shitty_results['strategy_name']} strategy (something's wrong!)")
        print(f"   Difference: {shitty_results['avg_length'] - smart_results['avg_length']:.2f} steps")

    return smart_results, shitty_results

def test_strategy_performance(client, num_episodes=10):
    """
    Legacy function - now delegates to compare_strategies.

    Args:
        client: CartpoleEnv client
        num_episodes: Number of episodes to test

    Returns:
        dict: Performance metrics for smart strategy
    """
    smart_results, shitty_results = compare_strategies(client, num_episodes)
    return smart_results

def test_basic_functionality(client):
    """Test basic cartpole functionality."""
    print("\n🔧 Testing basic functionality...")

    # Reset
    state = client.reset()
    print("✅ Reset successful")
    print(f"   Initial state: {state.observation.state}")

    # Test action 0 (push left)
    print("\nTesting action 0 (push left)...")
    state = client.step(CartpoleAction(action_id=0))
    print(f"   Result: reward={state.reward}, done={state.done}")
    print(f"   New state: {state.observation.state}")

    # Test action 1 (push right)
    print("\nTesting action 1 (push right)...")
    state = client.step(CartpoleAction(action_id=1))
    print(f"   Result: reward={state.reward}, done={state.done}")
    print(f"   New state: {state.observation.state}")

    print("✅ Basic functionality test completed!")

print("Starting Cartpole Environment Client with Strategy Testing...")
print(f"Connecting to: {base_url}")

# Wait for OpenEnv server to start
print("Waiting 5 seconds for server to start...")
time.sleep(5)

try:
    # Create the environment client
    openenv_process = CartpoleEnv(
        base_url=base_url,
        request_timeout_s=request_timeout_s
    )

    # Quick smoke test - check if server is responding
    print("\nTesting connection...")
    try:
        # Test basic functionality first
        test_basic_functionality(openenv_process)

        # Test the strategy
        metrics = test_strategy_performance(openenv_process, num_episodes=5)

        print("\n✅ Cartpole strategy testing completed successfully!")

    except Exception as e:
        print(f"❌ Error during testing: {e}")
        print("Make sure the cartpole server is running:")
        print("  uvicorn envs.cartpole_environment.server.app:app --host 0.0.0.0 --port 8000")

except Exception as e:
    print(f"❌ Failed to connect to server: {e}")
    print("Please ensure the cartpole server is running on port 8000")

print("\nScript completed!")
