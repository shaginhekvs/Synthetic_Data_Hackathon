# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
FastAPI application for the LunarLander Environment.

This module creates an HTTP server that exposes LunarLander-v3 environment
over HTTP endpoints, making it compatible with HTTPEnvClient.

Usage:
    # Development (with auto-reload):
    uvicorn envs.lunarlander_environment.server.app:app --reload --host 0.0.0.0 --port 8070

    # Production:
    uvicorn envs.lunarlander_environment.server.app:app --host 0.0.0.0 --port 8070 --workers 4

    # Or run directly:
    python -m envs.lunarlander_environment.server.app

Environment variables:
    LUNARLANDER_RENDER_MODE: Render mode ("human", "rgb_array", None) (default: None)
    LUNARLANDER_MAX_STEPS: Maximum steps per episode (default: "1000")
    LUNARLANDER_SEED: Random seed for reproducibility (optional)
    PORT: Server port (default: "8070")
"""

import os

from core.env_server import create_fastapi_app

from ..models import LunarLanderAction, LunarLanderObservation
from .lunarlander_environment import LunarLanderEnvironment

# Get configuration from environment variables
render_mode = os.getenv("LUNARLANDER_RENDER_MODE")
max_steps = int(os.getenv("LUNARLANDER_MAX_STEPS", "1000"))
seed_str = os.getenv("LUNARLANDER_SEED")

# Convert seed to int if specified
seed = int(seed_str) if seed_str is not None else None

# Create the environment instance
env = LunarLanderEnvironment(
    render_mode=render_mode,
    max_steps=max_steps,
    seed=seed,
)

# Create the FastAPI app
app = create_fastapi_app(env, LunarLanderAction, LunarLanderObservation)

if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "8070"))
    print(f"Starting LunarLander environment server on port {port}")
    print(f"Environment: LunarLander-v3")
    print(f"Max steps: {env.max_steps}")
    print(f"Render mode: {env.render_mode or 'None'}")
    print(f"Seed: {env.seed}")

    uvicorn.run(app, host="0.0.0.0", port=port)
