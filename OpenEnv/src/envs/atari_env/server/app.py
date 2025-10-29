# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
FastAPI application for the Atari Environment.

This module creates an HTTP server that exposes Atari games
over HTTP endpoints, making them compatible with HTTPEnvClient.

Usage:
    # Development (with auto-reload):
    uvicorn envs.atari_env.server.app:app --reload --host 0.0.0.0 --port 8000

    # Production:
    uvicorn envs.atari_env.server.app:app --host 0.0.0.0 --port 8000 --workers 4

    # Or run directly:
    python -m envs.atari_env.server.app

Environment variables:
    ATARI_GAME: Game name to serve (default: "pong")
    ATARI_OBS_TYPE: Observation type (default: "rgb")
    ATARI_FULL_ACTION_SPACE: Use full action space (default: "false")
    ATARI_MODE: Game mode (optional)
    ATARI_DIFFICULTY: Game difficulty (optional)
    ATARI_REPEAT_ACTION_PROB: Sticky action probability (default: "0.0")
    ATARI_FRAMESKIP: Frameskip (default: "4")
"""

import os

from core.env_server import create_app

from ..models import AtariAction, AtariObservation
from .atari_environment import AtariEnvironment

# Get configuration from environment variables
game_name = os.getenv("ATARI_GAME", "pong")
obs_type = os.getenv("ATARI_OBS_TYPE", "rgb")
full_action_space = os.getenv("ATARI_FULL_ACTION_SPACE", "false").lower() == "true"
repeat_action_prob = float(os.getenv("ATARI_REPEAT_ACTION_PROB", "0.0"))
frameskip = int(os.getenv("ATARI_FRAMESKIP", "4"))

# Optional parameters
mode = os.getenv("ATARI_MODE")
difficulty = os.getenv("ATARI_DIFFICULTY")

# Convert to int if specified
mode = int(mode) if mode is not None else None
difficulty = int(difficulty) if difficulty is not None else None

# Create the environment instance
env = AtariEnvironment(
    game_name=game_name,
    obs_type=obs_type,
    full_action_space=full_action_space,
    mode=mode,
    difficulty=difficulty,
    repeat_action_probability=repeat_action_prob,
    frameskip=frameskip,
)

# Create the FastAPI app with web interface and README integration
app = create_app(env, AtariAction, AtariObservation, env_name="atari_env")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
