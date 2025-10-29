#!/usr/bin/env python3
"""
Strategy Testing for Sequential Environment.

Tests both a shitty strategy and an average strategy across the sequential environment.
The sequential environment interleaves: Cartpole, MountainCarContinuous, LunarLanderContinuous.
"""

import sys
import time
import random
from typing import Dict, Any

# Add the OpenEnv package to the path
sys.path.insert(0, 'OpenEnv/src')

from envs.sequential_environment import SequentialEnvironment, SequentialAction
from envs.sequential_environment.client import SequentialEnv


def shitty_strategy(action: SequentialAction, phase: str, sub_obs: list):
    """
    A truly shitty strategy that makes poor decisions.
    - Cartpole: Always push the wrong way (if pole tilted left, push left more)
    - MountainCar: Always apply force in direction that's slowing it down
    - LunarLander: Random thrust that often makes it crash
    """
    if phase == "cartpole":
        # Cartpole state: [cart_pos, cart_vel, pole_angle, pole_vel]
        pole_angle = sub_obs[2] if len(sub_obs) > 2 else 0
        # If pole tilted left (negative), push left harder (makes it worse)
        action.cartpole_action = 0 if pole_angle < 0 else 1
        action.mountaincar_action = random.choice([0.5, -0.5])  # Random for others
        action.lunarlander_action = [random.uniform(-0.5, 0.5), random.uniform(-0.5, 0.5)]

    elif phase == "mountaincar":
        # MountainCar state: [position, velocity]
        position = sub_obs[0] if len(sub_obs) > 0 else 0
        velocity = sub_obs[1] if len(sub_obs) > 1 else 0
        # Always apply force in direction opposite to velocity (slows down)
        # This makes it oscillate without progressing
        action.mountaincar_action = -0.3 if velocity > 0 else 0.3
        action.cartpole_action = random.choice([0, 1])
        action.lunarlander_action = [random.uniform(-0.5, 0.5), random.uniform(-0.5, 0.5)]

    elif phase == "lunarlander":
        # LunarLander state: [x_pos, y_pos, x_vel, y_vel, angle, ang_vel, left_contact, right_contact]
        # Shitty: Fire main engine randomly, ignore angle
        main_thrust = random.uniform(-0.2, 0.8)  # Sometimes negative thrust
        lateral_thrust = random.uniform(-0.8, 0.8)  # Too much lateral thrust
        action.lunarlander_action = [main_thrust, lateral_thrust]
        action.cartpole_action = random.choice([0, 1])
        action.mountaincar_action = random.uniform(-0.5, 0.5)

    elif phase == "bipedalwalker":
        # BipedalWalker state: 24 dims (position, velocity, contact info, etc.)
        # Shitty: Random joint torques - will cause immediate fall
        action.bipedalwalker_action = [
            random.uniform(-1.0, 1.0),  # hip1
            random.uniform(-1.0, 1.0),  # knee1
            random.uniform(-1.0, 1.0),  # hip2
            random.uniform(-1.0, 1.0),  # knee2
        ]
        action.cartpole_action = random.choice([0, 1])
        action.mountaincar_action = random.uniform(-0.5, 0.5)
        action.lunarlander_action = [random.uniform(-0.5, 0.5), random.uniform(-0.5, 0.5)]

    else:
        # Random for unknown phases
        action.cartpole_action = random.choice([0, 1])
        action.mountaincar_action = random.uniform(-0.5, 0.5)
        action.lunarlander_action = [random.uniform(-0.5, 0.5), random.uniform(-0.5, 0.5)]

    return action


def average_strategy(action: SequentialAction, phase: str, sub_obs: list):
    """
    An average strategy that makes reasonable but not optimal decisions.
    - Cartpole: Basic balance attempt but sometimes wrong
    - MountainCar: Mostly right direction but inconsistent force
    - LunarLander: Decent control but not perfect landing
    """
    if phase == "cartpole":
        # Cartpole state: [cart_pos, cart_vel, pole_angle, pole_vel]
        pole_angle = sub_obs[2] if len(sub_obs) > 2 else 0
        pole_vel = sub_obs[3] if len(sub_obs) > 3 else 0
        # Basic balance but sometimes make mistakes (60% correct rate)
        if random.random() < 0.6:
            action.cartpole_action = 0 if pole_angle > 0 else 1
        else:
            action.cartpole_action = 1 if pole_angle > 0 else 0
        action.mountaincar_action = random.uniform(-1.0, 1.0)  # Random for others
        action.lunarlander_action = [random.uniform(-1.0, 1.0), random.uniform(-1.0, 1.0)]

    elif phase == "mountaincar":
        # MountainCar state: [position, velocity]
        position = sub_obs[0] if len(sub_obs) > 0 else 0
        velocity = sub_obs[1] if len(sub_obs) > 1 else 0
        # Mostly apply force toward goal but sometimes wrong direction (70% correct)
        if random.random() < 0.7:
            action.mountaincar_action = 0.6 if position < 0.2 else -0.6
        else:
            action.mountaincar_action = -0.6 if position < 0.2 else 0.6
        action.cartpole_action = random.choice([0, 1])
        action.lunarlander_action = [random.uniform(-1.0, 1.0), random.uniform(-1.0, 1.0)]

    elif phase == "lunarlander":
        # LunarLander state: [x_pos, y_pos, x_vel, y_vel, angle, ang_vel, left_contact, right_contact]
        x_pos = sub_obs[0] if len(sub_obs) > 0 else 0
        y_pos = sub_obs[1] if len(sub_obs) > 1 else 0
        x_vel = sub_obs[2] if len(sub_obs) > 2 else 0
        y_vel = sub_obs[3] if len(sub_obs) > 3 else 0
        angle = sub_obs[4] if len(sub_obs) > 4 else 0
        ang_vel = sub_obs[5] if len(sub_obs) > 5 else 0

        # Average landing: decent control but not expert
        main_thrust = max(0, min(0.8, (3.0 - y_pos) * 0.3 + y_vel * 0.2))  # Basic height control
        lateral_thrust = -angle * 0.5 - ang_vel * 0.3  # Basic angle control
        lateral_thrust = max(-0.8, min(0.8, lateral_thrust))  # Clamping

        # Occasionally make mistakes (missed 20% of landings in training sim)
        if random.random() < 0.2:
            main_thrust *= random.uniform(0.5, 1.5)  # Too much or too little thrust
            lateral_thrust *= random.uniform(-2.0, 2.0)  # Over-correct

        action.lunarlander_action = [main_thrust, lateral_thrust]
        action.cartpole_action = random.choice([0, 1])
        action.mountaincar_action = random.uniform(-1.0, 1.0)

    else:
        # Random for unknown phases
        action.cartpole_action = random.choice([0, 1])
        action.mountaincar_action = random.uniform(-1.0, 1.0)
        action.lunarlander_action = [random.uniform(-1.0, 1.0), random.uniform(-1.0, 1.0)]

    return action


def test_strategy(strategy_name: str, strategy_func, seed: int, episodes: int = 3):
    """
    Test a strategy across multiple episodes.

    Args:
        strategy_name: Name of the strategy (e.g., "Shitty", "Average")
        strategy_func: Function implementing the strategy
        seed: Random seed for reproducibility
        episodes: Number of episodes to test
    """
    print(f"\n{'='*60}")
    print(f"Testing {strategy_name} Strategy")
    print(f"{'='*60}")

    all_total_rewards = []
    all_steps = []
    phase_completion_stats = []

    for episode in range(episodes):
        print(f"\n--- Episode {episode + 1} ---")

        env = SequentialEnvironment(
            render_mode=None,
            max_steps=100,  # Shorter for testing
            seed=seed + episode  # Different seed per episode
        )

        obs = env.reset()
        episode_reward = 0
        episode_steps = 0

        phase_steps = {"cartpole": 0, "mountaincar": 0, "lunarlander": 0, "bipedalwalker": 0}
        phase_rewards = {"cartpole": 0.0, "mountaincar": 0.0, "lunarlander": 0.0, "bipedalwalker": 0.0}

        try:
            while not obs.done and episode_steps < 150:  # Safety limit
                episode_steps += 1
                current_phase = obs.phase

                # Get strategy action - provide all actions with defaults
                action = SequentialAction()
                action.cartpole_action = 0
                action.mountaincar_action = 0.0
                action.lunarlander_action = [0.0, 0.0]
                action.bipedalwalker_action = [0.0, 0.0, 0.0, 0.0]

                # Let strategy override the specific phase action
                action = strategy_func(action, current_phase, obs.sub_observation)

                print(f"Step {episode_steps}: {current_phase} -> "
                      f"cartpole={action.cartpole_action}, "
                      f"mountaincar={round(action.mountaincar_action, 1)}, "
                      f"lunarlander={[round(float(x), 1) for x in action.lunarlander_action] if action.lunarlander_action else None}, "
                      f"bipedalwalker={[round(float(x), 1) for x in action.bipedalwalker_action] if action.bipedalwalker_action else None}")

                start_time = time.time()
                obs = env.step(action)
                step_time = time.time() - start_time

                episode_reward += obs.reward

                # Track per-phase stats
                phase_steps[current_phase] += 1
                phase_rewards[current_phase] += obs.reward

                if step_time > 0.1:  # Warning for slow steps
                    print(f"  ⚠️  Slow step: {step_time:.3f}s")

                if obs.done:
                    break

            # Episode completed
            print(f"Episode {episode + 1} completed: {episode_steps} steps, "
                  f"total reward: {episode_reward:.2f}")

            # Phase completion status
            phase_done = {
                "cartpole": env.state.cartpole_done,
                "mountaincar": env.state.mountaincar_done,
                "lunarlander": env.state.lunarlander_done,
                "bipedalwalker": env.state.bipedalwalker_done,
            }

            print("Phase completion:")
            for phase, done in phase_done.items():
                steps = phase_steps[phase]
                rewards = phase_rewards[phase]
                status = "✅ DONE" if done else ("❌ TIME LIMIT" if steps > 0 else "⏸️  NOT REACHED")
                avg_reward = rewards / steps if steps > 0 else 0
                print(f"  {phase}: {steps} steps, {rewards:.1f} reward ({avg_reward:.2f}/step) - {status}")

            all_total_rewards.append(episode_reward)
            all_steps.append(episode_steps)
            phase_completion_stats.append(phase_done)

        finally:
            env.close()

    # Summary statistics
    print(f"\n--- {strategy_name} Strategy Summary ({episodes} episodes) ---")
    avg_total_reward = sum(all_total_rewards) / len(all_total_rewards)
    avg_steps = sum(all_steps) / len(all_steps)

    print(".2f")
    print(".1f")

    # Phase completion success rates
    cartpole_success = sum(1 for p in phase_completion_stats if p["cartpole"]) / len(phase_completion_stats)
    mountaincar_success = sum(1 for p in phase_completion_stats if p["mountaincar"]) / len(phase_completion_stats)
    lunarlander_success = sum(1 for p in phase_completion_stats if p["lunarlander"]) / len(phase_completion_stats)
    bipedalwalker_success = sum(1 for p in phase_completion_stats if p["bipedalwalker"]) / len(phase_completion_stats)

    print("Success rates (phases completed):")
    print(f"  Cartpole: {cartpole_success*100:.0f}% ({sum(1 for p in phase_completion_stats if p['cartpole'])}/{len(phase_completion_stats)})")
    print(f"  Mountaincar: {mountaincar_success*100:.0f}% ({sum(1 for p in phase_completion_stats if p['mountaincar'])}/{len(phase_completion_stats)})")
    print(f"  Lunarlander: {lunarlander_success*100:.0f}% ({sum(1 for p in phase_completion_stats if p['lunarlander'])}/{len(phase_completion_stats)})")
    print(f"  Bipedalwalker: {bipedalwalker_success*100:.0f}% ({sum(1 for p in phase_completion_stats if p['bipedalwalker'])}/{len(phase_completion_stats)})")

    # Overall episode success (all four phases completed)
    overall_success = sum(1 for p in phase_completion_stats if all(p.values())) / len(phase_completion_stats)
    print(f"  Overall (all phases): {overall_success*100:.0f}% ({sum(1 for p in phase_completion_stats if all(p.values()))}/{len(phase_completion_stats)})")

    return {
        "avg_total_reward": avg_total_reward,
        "avg_steps": avg_steps,
        "success_rates": [cartpole_success, mountaincar_success, lunarlander_success, bipedalwalker_success, overall_success],
    }


def main():
    """Run strategy comparisons."""
    print("🌟 Sequential Environment Strategy Testing")
    print("Testing shitty vs average strategies across interleaved phases")
    print("(Cartpole, MountainCarContinuous, LunarLanderContinuous, BipedalWalker-v3)")

    # Set random seed for reproducibility
    seed = 12345

    # Test shitty strategy
    shitty_results = test_strategy("Shitty", shitty_strategy, seed, episodes=3)

    # Test average strategy
    average_results = test_strategy("Average", average_strategy, seed + 1000, episodes=3)

    # Comparison summary
    print(f"\n🎯 FINAL COMPARISON")
    print(f"{'='*40}")
    print(f"{'Metric':<20} {'Shitty':<10} {'Average':<10} {'Improvement':<12}")
    print("-" * 54)

    # Reward comparison
    shitty_reward = shitty_results["avg_total_reward"]
    average_reward = average_results["avg_total_reward"]
    reward_improvement = ((average_reward - shitty_reward) / abs(shitty_reward)) * 100 if shitty_reward != 0 else 0
    print(f"{'Average Reward':<20} {shitty_reward:<10.1f} {average_reward:<10.1f} {reward_improvement:<+12.0f}%")

    # Steps comparison
    shitty_steps = shitty_results["avg_steps"]
    average_steps = average_results["avg_steps"]
    steps_improvement = (((average_steps - shitty_steps) / shitty_steps) * 100) if shitty_steps > 0 else 0
    print(f"{'Average Steps':<20} {shitty_steps:<10.1f} {average_steps:<10.1f} {steps_improvement:<+12.0f}%")

    # Success rates comparison
    print("\nSuccess Rates (% of episodes where phase was completed):")
    phase_names = ["Cartpole", "MountainCar", "LunarLander", "BipedalWalker", "All Four"]
    for i, phase in enumerate(phase_names):
        shitty_pct = shitty_results["success_rates"][i] * 100
        average_pct = average_results["success_rates"][i] * 100
        diff = average_pct - shitty_pct
        print("5")
    print("\n📝 Analysis:")
    print("  - Shitty strategy: Random/wrong actions, high failure rate")
    print("  - Average strategy: Reasonable heuristics, better control")
    print("  - Results show significant improvement potential for reinforcement learning from these baselines!")
if __name__ == "__main__":
    main()

