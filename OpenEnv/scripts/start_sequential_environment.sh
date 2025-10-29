#!/bin/bash
# Sequential Environment Server Start Script - Port 8060

cd OpenEnv

# Set environment variables for sequential environment configuration
export SEQUENTIAL_RENDER_MODE=None        # Optional: "human", "rgb_array", or None
export SEQUENTIAL_MAX_STEPS="10000"       # Maximum steps per episode
export SEQUENTIAL_SEED="42"               # Optional: Random seed for reproducibility
export PORT="8060"                        # Server port

# Start the server on port 8060
echo "Starting Sequential Environment Server on port $PORT..."
echo "Configuration:"
echo "  Render Mode: $SEQUENTIAL_RENDER_MODE"
echo "  Max Steps: $SEQUENTIAL_MAX_STEPS"
echo "  Seed: $SEQUENTIAL_SEED"
echo ""
echo "Server will be available at: http://localhost:$PORT"
echo "API endpoints:"
echo "  POST /reset - Reset environment"
echo "  POST /step  - Execute action"
echo "  GET  /state - Get current state"
echo "  GET  /health - Health check"
echo ""
echo "This environment interleaves Cartpole, MountainCarContinuous, and LunarLanderContinuous."
echo "At each step, it randomly selects which sub-environment to step in."
echo ""
echo "Actions (provide all, but only active phase's action is used):"
echo "  cartpole_action: 0 or 1 (push left/right)"
echo "  mountaincar_action: [-1.0, 1.0] (engine force)"
echo "  lunarlander_action: [main_engine, lateral_engine] both in [-1.0, 1.0]"
echo ""
echo "State (one-hot phase vector + sub-environment state):"
echo "  [cartpole_phase, mountaincar_phase, lunarlander_phase, ...sub_env_state]"
echo ""
echo "Phase Selection: Random seeded selection from active (non-done) phases."
echo "Episode ends when all sub-environments reach done=True OR max_steps reached."
echo ""

# Start uvicorn server
uvicorn envs.sequential_environment.server.app:app \
    --host 0.0.0.0 \
    --port $PORT \
    --reload \
    --log-level info

