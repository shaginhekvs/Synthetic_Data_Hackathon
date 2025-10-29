"""
CartPole Environment for OpenEnv.

Example:
    >>> from envs.cartpole_env import CartPoleEnv, CartPoleAction
    >>> env = CartPoleEnv.from_env()  # expects OPENENV_URL or defaults to http://localhost:8000
    >>> s = env.reset()
    >>> s = env.step(CartPoleAction(action=1))
    >>> print(s.reward, s.done)
    >>> env.close()
"""
from .client import Env
from .models import Action, Observation, State

__all__ = ["Env", "Action", "Observation", "State"]
