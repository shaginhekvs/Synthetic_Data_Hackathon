#!/bin/bash
# LunarLander Server Start Script - Port 8070

cd OpenEnv

# Set environment variables for lunarlander configuration
export LUNARLANDER_RENDER_MODE=None      # Optional: "human", "rgb_array", or None
export LUNARLANDER_MAX_STEPS="10000"          # Maximum steps per episode
export LUNARLANDER_SEED="42"                # Optional: Random seed
export PORT="8070"                          # Server port

# Start the server on port 8070
echo "Starting LunarLander Environment Server on port $PORT..."
echo "Configuration:"
echo "  Render Mode: $LUNARLANDER_RENDER_MODE"
echo "  Max Steps: $LUNARLANDER_MAX_STEPS"
echo "  Seed: $LUNARLANDER_SEED"
echo ""
echo "Server will be available at: http://localhost:$PORT"
echo "API endpoints:"
echo "  POST /reset - Reset environment"
echo "  POST /step  - Execute action"
echo "  GET  /state - Get current state"
echo "  GET  /health - Health check"
echo ""
echo "Actions:"
echo "  main_engine: [-1.0, 1.0] (thrust power)"
echo "  lateral_engine: [-1.0, 1.0] (lateral thrust: negative=left, positive=right)"
echo ""
echo "State (8 values):"
echo "  [x_pos, y_pos, x_vel, y_vel, angle, ang_vel, left_contact, right_contact]"
echo ""
echo "Goal: Land safely between the flags (x ≈ 0) with low velocity and upright orientation"
echo ""

# Start uvicorn server
uvicorn envs.lunarlander_environment.server.app:app \
    --host 0.0.0.0 \
    --port $PORT \
    --reload \
    --log-level info
