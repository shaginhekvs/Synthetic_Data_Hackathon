#!/bin/bash
# MountainCarContinuous Server Start Script - Port 8030

cd OpenEnv

# Set environment variables for mountaincarcontinuous configuration
export MOUNTAINCARCONTINUOUS_RENDER_MODE=None      # Optional: "human", "rgb_array", or None
export MOUNTAINCARCONTINUOUS_MAX_STEPS="1000"          # Maximum steps per episode
export MOUNTAINCARCONTINUOUS_SEED="42"                # Optional: Random seed

# Start the server on port 8050
echo "🚀 Starting MountainCarContinuous Environment Server on port 8050..."
echo "Configuration:"
echo "  Render Mode: $MOUNTAINCARCONTINUOUS_RENDER_MODE"
echo "  Max Steps: $MOUNTAINCARCONTINUOUS_MAX_STEPS"
echo "  Seed: $MOUNTAINCARCONTINUOUS_SEED"
echo ""
echo "Server will be available at: http://localhost:8050"
echo "API endpoints:"
echo "  POST /reset - Reset environment"
echo "  POST /step  - Execute action"
echo "  GET  /state - Get current state"
echo "  GET  /      - Web interface"
echo ""

# Start uvicorn server
uvicorn envs.mountaincarcontinuous_environment.server.app:app \
    --host 0.0.0.0 \
    --port 8050 \
    --reload \
    --log-level info
