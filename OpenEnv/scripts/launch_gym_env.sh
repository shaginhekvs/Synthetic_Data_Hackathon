#!/bin/bash

# Variables
PORT=9000
KEEPALIVE=3000


GYM_ENVIRONMENT_ID="BipedalWalker-v3" \
GYM_RENDER_MODE="rgb_array" \
python3 -m uvicorn envs.gym_environment.server.app:app \
    --port "$PORT" \
    --timeout-keep-alive "$KEEPALIVE"
