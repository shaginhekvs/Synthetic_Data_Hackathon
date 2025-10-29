#!/bin/bash
# Cartpole Server Start Script - Port 8020

cd OpenEnv

# Set environment variables for cartpole configuration
export CARTPOLE_RENDER_MODE=None      # Optional: "human", "rgb_array", or None
export CARTPOLE_MAX_STEPS="10000"          # Maximum steps per episode
export CARTPOLE_SEED="42"                # Optional: Random seed

# Start the server on port 8020
echo "Starting Cartpole Environment Server on port 8030..."
echo "Configuration:"
echo "  Render Mode: $CARTPOLE_RENDER_MODE"
echo "  Max Steps: $CARTPOLE_MAX_STEPS"
echo "  Seed: $CARTPOLE_SEED"
echo ""
echo "Server will be available at: http://localhost:8030"
echo "API endpoints:"
echo "  POST /reset - Reset environment"
echo "  POST /step  - Execute action"
echo "  GET  /state - Get current state"
echo "  GET  /health - Health check"
echo ""

# Start uvicorn server
uvicorn envs.cartpole_environment.server.app:app \
    --host 0.0.0.0 \
    --port 8030 \
    --reload \
    --log-level info