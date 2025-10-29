# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
FastAPI application for the MountainCarContinuous Environment.

This module creates an HTTP server that exposes MountainCarContinuous-v0 environment
over HTTP endpoints, making it compatible with HTTPEnvClient.

Usage:
    # Development (with auto-reload):
    uvicorn envs.mountaincarcontinuous_environment.server.app:app --reload --host 0.0.0.0 --port 8000

    # Production:
    uvicorn envs.mountaincarcontinuous_environment.server.app:app --host 0.0.0.0 --port 8000 --workers 4

    # Or run directly:
    python -m envs.mountaincarcontinuous_environment.server.app

Environment variables:
    MOUNTAINCARCONTINUOUS_RENDER_MODE: Render mode ("human", "rgb_array", None) (default: None)
    MOUNTAINCARCONTINUOUS_MAX_STEPS: Maximum steps per episode (default: "999")
    MOUNTAINCARCONTINUOUS_SEED: Random seed for reproducibility (optional)
"""

import os

from core.env_server import create_app

from ..models import MountainCarContinuousAction, MountainCarContinuousObservation
from .mountaincarcontinuous_environment import MountainCarContinuousEnvironment

# Get configuration from environment variables
render_mode = os.getenv("MOUNTAINCARCONTINUOUS_RENDER_MODE")
max_steps = int(os.getenv("MOUNTAINCARCONTINUOUS_MAX_STEPS", "999"))
seed_str = os.getenv("MOUNTAINCARCONTINUOUS_SEED")

# Convert seed to int if specified
seed = int(seed_str) if seed_str is not None else None

# Create the environment instance
env = MountainCarContinuousEnvironment(
    render_mode=render_mode,
    max_steps=max_steps,
    seed=seed,
)

# Create the FastAPI app with web interface and README integration
app = create_app(env, MountainCarContinuousAction, MountainCarContinuousObservation, env_name="mountaincarcontinuous_env")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8010)
