# Minimal Gymnasium CartPole factory with a few env-var knobs.
import os
import gymnasium as gym

def make_env():
    env_id = os.getenv("ENV_ID", "BipedalWalker-v3")
    seed = int(os.getenv("SEED", "123"))
    max_steps = int(os.getenv("MAX_EPISODE_STEPS", "500"))
    render = os.getenv("RENDER_MODE", "none")  # "none" or "rgb_array"
    render_mode = None if render == "none" else "rgb_array"
    hardcore = os.getenv("HARDCORE", "0")  # "none" or "rgb_array"
    hardcore = hardcore == "1"
    print(env_id, seed, max_steps, render, hardcore)


    env = gym.make(env_id, render_mode=render_mode)

    # Standardize episode length (Gymnasium wrapper)
    if max_steps > 0:
        env = gym.wrappers.TimeLimit(env, max_episode_steps=max_steps)

    # Seeding
    obs, _ = env.reset(seed=seed)
    return env
