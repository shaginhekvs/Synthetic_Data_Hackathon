# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
FastAPI application for the Sequential Environment.

This module creates an HTTP server that exposes the Sequential Environment
over HTTP endpoints, making it compatible with HTTPEnvClient.

Usage:
    # Development (with auto-reload):
    uvicorn envs.sequential_environment.server.app:app --reload --host 0.0.0.0 --port 8000

    # Production:
    uvicorn envs.sequential_environment.server.app:app --host 0.0.0.0 --port 8000 --workers 4

    # Or run directly:
    python -m envs.sequential_environment.server.app

Environment variables:
    SEQUENTIAL_RENDER_MODE: Render mode ("human", "rgb_array", None) (default: None)
    SEQUENTIAL_MAX_STEPS: Maximum steps per episode (default: "1000")
    SEQUENTIAL_SEED: Random seed for reproducibility (optional)
"""

import os

from core.env_server import create_app

from ..models import SequentialAction, SequentialObservation
from .sequential_environment import SequentialEnvironment

# Get configuration from environment variables
render_mode = os.getenv("SEQUENTIAL_RENDER_MODE")
max_steps = int(os.getenv("SEQUENTIAL_MAX_STEPS", "1000"))
seed_str = os.getenv("SEQUENTIAL_SEED")

# Convert seed to int if specified
seed = int(seed_str) if seed_str is not None else None

# Create the environment instance
env = SequentialEnvironment(
    render_mode=render_mode,
    max_steps=max_steps,
    seed=seed,
)

# Create the FastAPI app with web interface and README integration
app = create_app(env, SequentialAction, SequentialObservation, env_name="sequential_env")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8050)

