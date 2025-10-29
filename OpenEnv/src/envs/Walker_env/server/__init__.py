# Copyright (c) Meta Platforms, Inc. ...
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
CartPole Environment (server-side).

Thin wrapper around Gymnasium's CartPole for OpenEnv servers.
"""

from typing import Any, Dict, Tuple
import numpy as np
from envs.Walker_env.factory import make_env

class Environment:
    def __init__(self) -> None:
        self.env = make_env()
        self._obs, _ = self.env.reset()

    # ---- API ----
    def info(self) -> Dict[str, Any]:
        os = self.env.observation_space
        aspace = self.env.action_space
        obs_desc = {"shape": list(getattr(os, "shape", [])),
                    "dtype": str(getattr(os, "dtype", "float32"))}
        if aspace.__class__.__name__.lower().startswith("discrete"):
            act_desc = {"type": "discrete", "n": int(aspace.n)}
        else:
            low = getattr(aspace, "low", [])
            high = getattr(aspace, "high", [])
            act_desc = {"type": "box",
                        "low": (low.tolist() if isinstance(low, np.ndarray) else low),
                        "high": (high.tolist() if isinstance(high, np.ndarray) else high)}
        max_steps = getattr(self.env, "_max_episode_steps",
                            getattr(getattr(self.env, "spec", None), "max_episode_steps", 0)) or 0
        env_id = str(getattr(getattr(self.env, "spec", None), "id", "CartPole-v1"))
        return {
            "id": env_id,
            "observation_space": obs_desc,
            "action_space": act_desc,
            "max_episode_steps": int(max_steps),
        }

    def reset(self) -> Tuple[np.ndarray, Dict[str, Any]]:
        self._obs, info = self.env.reset()
        return self._obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        done = bool(terminated or truncated)
        if done:
            obs, _ = self.env.reset()
        self._obs = obs
        return obs, float(reward), done, info

    def close(self):
        try:
            self.env.close()
        except Exception:
            pass
