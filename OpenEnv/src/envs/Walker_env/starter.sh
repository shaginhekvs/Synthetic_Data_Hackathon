# Variables
PORT=8000
KEEPALIVE=3000
WORKDIR="/shared-docker/OpenEnv"

# Run uvicorn with custom environment variables
cd "$WORKDIR" || exit

PYTHONPATH="$WORKDIR/src" \
ENV_ID="BipedalWalker-v3" \
python3 -m uvicorn envs.Walker_env.server.app:app \
    --port "$PORT" \
    --timeout-keep-alive "$KEEPALIVE"