from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Any, Dict, List, Union
import numpy as np
import traceback
import logging
from envs.Walker_env.server import Environment  # <-- new import

app = FastAPI(title=" OpenEnv-style API")
logger = logging.getLogger(__name__)

class StepRequest(BaseModel):
    action: Union[int, List[float], Dict[str, Any]]

class StepResponse(BaseModel):
    observation: List[float]
    reward: float
    done: bool
    info: Dict[str, Any] = Field(default_factory=dict)

class InfoResponse(BaseModel):
    id: str
    observation_space: Dict[str, Any]
    action_space: Dict[str, Any]
    max_episode_steps: int

_env: Environment | None = None

def env() -> Environment:
    global _env
    if _env is None:
        _env = Environment()
    return _env

def _obs_to_list(obs):
    return [float(v) for v in (obs if isinstance(obs, (list, tuple)) else np.asarray(obs).flatten())]

@app.get("/health")
def health(): return {"status": "ok"}

from fastapi import Response
from PIL import Image
import io

@app.get("/render")
def render():
    # Gymnasium: call env.render() for an rgb frame when render_mode="rgb_array"
    frame = _env.env.render()  # numpy array HxWx3
    if frame is None:
        return Response(status_code=204)
    im = Image.fromarray(frame)
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return Response(content=buf.getvalue(), media_type="image/png")


@app.get("/info", response_model=InfoResponse)
def info():
    meta = env().info()
    return InfoResponse(**meta)

@app.post("/reset", response_model=StepResponse)
def reset():
    obs, _ = env().reset()
    return StepResponse(observation=_obs_to_list(obs), reward=0.0, done=False, info={})

@app.post("/step", response_model=StepResponse)
def step(req: StepRequest):
    action = req.action
    if isinstance(action, dict):
        action = action.get("action", 0)
    try:
        obs, reward, done, info = env().step(action)
    except Exception as e:
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=400, detail=str(e))
    return StepResponse(observation=_obs_to_list(obs), reward=reward, done=done, info=info or {})
