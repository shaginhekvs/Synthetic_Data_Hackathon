import os
from typing import Optional, Dict, Any, Union
import httpx
from .models import Action, Observation,State

class Env:
    """
    Simple HTTP client for the  OpenEnv server.
    Endpoints expected:
      GET  /info
      POST /reset           -> {observation, reward, done, info}
      POST /step {"action"} -> {observation, reward, done, info}
    """

    def __init__(self, base_url: str, timeout: Optional[httpx.Timeout] = None):
        self.base_url = base_url.rstrip("/")
        self._client = httpx.Client(timeout=timeout or httpx.Timeout(connect=5.0, read=300.0, write=300.0, pool=30.0))

    # -------- convenience constructors ----------
    @classmethod
    def from_env(cls, *, env_var: str = "OPENENV_URL") -> "Env":
        url = os.environ.get(env_var, "http://localhost:8000")
        return cls(url)

    @classmethod
    def from_docker_image(cls, image: str, port: int = 8000, **run_kwargs) -> "Env":
        """
        Optional helper parity with AtariEnv: run a container exposing the server.
        You can implement actual docker calls here if you want;
        for now, we assume you've launched it elsewhere and just return the client.
        """
        base_url = f"http://localhost:{port}"
        return cls(base_url)

    # ------------- API methods ------------------
    def info(self) -> Dict[str, Any]:
        r = self._client.get(f"{self.base_url}/info")
        r.raise_for_status()
        return r.json()

    def reset(self) -> State:
        r = self._client.post(f"{self.base_url}/reset", json={})
        r.raise_for_status()
        data = r.json()
        obs = Observation(values=list(map(float, data["observation"])))
        return State(observation=obs, reward=float(data.get("reward", 0.0)),
                             done=bool(data.get("done", False)), info=data.get("info", {}) or {})

    def step(self, action: Union[int, Action]) -> State:
        if isinstance(action, Action):
            payload = action.dict()
        else:
            payload = {"action": int(action)}
        print("PAYLOAD",payload)
        r = self._client.post(f"{self.base_url}/step", json=payload)
        r.raise_for_status()
        data = r.json()
        obs = Observation(values=list(map(float, data["observation"])))
        return State(observation=obs, reward=float(data["reward"]),
                             done=bool(data["done"]), info=data.get("info", {}) or {})

    def close(self) -> None:
        self._client.close()
