"""
Utilities to load frozen opponent models and adapt them to the ExternalOpponentPolicy
interface used by OpenSpielEnvironment.

Supports:
- Loading a PyTorch model class + checkpoint (state_dict)
- Loading a TorchScript artifact (torch.jit.load)
- A small wrapper that exposes predict(observations) and select_action(legal_actions, obs)
- A HotSwapOpponent wrapper that can reload a checkpoint path on request

Usage:
    from envs.openspiel_env.server.opponent_model_loader import (
        PyTorchOpponentAdapter, make_pytorch_opponent, HotSwapOpponent
    )
    model = PyTorchOpponentAdapter(ckpt_path="path/to/checkpoint.pt", model_ctor=MyModelClass)
    policy = get_opponent_policy(model)   # uses ExternalOpponentPolicy if present
    env = OpenSpielEnvironment(..., opponent_policy=policy)
"""
from __future__ import annotations

from typing import Any, Callable, Optional, Sequence, Dict
import os
import torch
import threading
import time
import random
import logging

logger = logging.getLogger(__name__)


class PyTorchOpponentAdapter:
    """
    Adapter around a PyTorch policy model to provide:
      - .predict(observations) -> action_id | logits/scores
      - .select_action(legal_actions, observations) -> action_id (optional)
    Constructor can accept either:
      - model_ctor: a callable that returns an nn.Module (recommended)
      - or leave model_ctor None and load a torch.jit.ScriptModule from ckpt_path.

    Simple example mapping (tic-tac-toe): treats observations["info_state"] as numeric vector.
    You will need to adapt preprocess() to match your model's expected input.
    """

    def __init__(
        self,
        ckpt_path: str,
        model_ctor: Optional[Callable[[], torch.nn.Module]] = None,
        device: str = "cpu",
        preprocess: Optional[Callable[[Dict[str, Any]], torch.Tensor]] = None,
        postprocess: Optional[Callable[[Any, Sequence[int], Dict[str, Any]], int]] = None,
    ):
        self.ckpt_path = ckpt_path
        self.device = torch.device(device)
        self.model_ctor = model_ctor
        self.preprocess = preprocess or self._default_preprocess
        self.postprocess = postprocess or self._default_postprocess
        self._lock = threading.RLock()
        self._model: Optional[torch.nn.Module] = None
        self._load_model()

    def _load_model(self):
        with self._lock:
            if self.model_ctor is not None:
                # instantiate model and load state_dict
                model = self.model_ctor()
                model.to(self.device)
                state = torch.load(self.ckpt_path, map_location=self.device)
                # state might be a dict with 'state_dict' key from some frameworks
                if isinstance(state, dict) and "state_dict" in state and isinstance(state["state_dict"], dict):
                    state = state["state_dict"]
                try:
                    model.load_state_dict(state)
                except Exception:
                    # if loading fails, try direct load (maybe ckpt is a ScriptModule or full model)
                    try:
                        model = torch.jit.load(self.ckpt_path, map_location=self.device)
                    except Exception as e:
                        logger.exception("Failed to load model via state_dict or torchscript: %s", e)
                        raise
                model.eval()
                self._model = model
            else:
                # Try loading as a TorchScript module
                model = torch.jit.load(self.ckpt_path, map_location=self.device)
                model.eval()
                self._model = model

    def reload(self, ckpt_path: Optional[str] = None):
        """Reload model weights from ckpt_path (if provided) or self.ckpt_path."""
        if ckpt_path:
            self.ckpt_path = ckpt_path
        logger.info("Reloading opponent model from %s", self.ckpt_path)
        self._load_model()

    def _default_preprocess(self, observations: Dict[str, Any]) -> torch.Tensor:
        """
        Default preprocess: try to use observations['info_state'] if present,
        otherwise fall back to flattening numeric observations.
        Adapt this to your model's input shape.
        """
        info = observations.get("info_state", None)
        if info is None:
            # try to convert whole observations dict to tensor (best-effort)
            # Not ideal — user should supply a proper preprocess function.
            flat = []
            for k, v in observations.items():
                if isinstance(v, (int, float)):
                    flat.append(float(v))
            if not flat:
                # fallback single zero
                return torch.zeros(1, device=self.device)
            return torch.tensor(flat, dtype=torch.float32, device=self.device).unsqueeze(0)
        # If info is already numeric sequence
        try:
            t = torch.tensor(list(info), dtype=torch.float32, device=self.device).unsqueeze(0)
        except Exception:
            # fallback: scalar zero
            t = torch.zeros(1, device=self.device).unsqueeze(0)
        return t

    def _default_postprocess(self, model_out: Any, legal_actions: Sequence[int], observations: Dict[str, Any]) -> int:
        """
        Interprets the model output: supports
          - integer directly
          - tensor logits or numpy array -> choose argmax among legal actions
          - dict mapping action->score
        """
        if model_out is None:
            return int(random.choice(list(legal_actions)))
        # If it's a scalar integer
        if isinstance(model_out, (int,)):
            return int(model_out)
        # Torch Tensor
        if torch.is_tensor(model_out):
            scores = model_out.detach().cpu().numpy().ravel().tolist()
            # choose legal action with max score
            try:
                best = max(legal_actions, key=lambda a: scores[a] if a < len(scores) else float("-inf"))
                return int(best)
            except Exception:
                return int(random.choice(list(legal_actions)))
        # numpy array-like or list
        try:
            import numpy as _np
            if _np is not None and hasattr(model_out, "shape"):
                scores = _np.asarray(model_out).ravel().tolist()
                best = max(legal_actions, key=lambda a: scores[a] if a < len(scores) else float("-inf"))
                return int(best)
        except Exception:
            pass
        # dict mapping action->score
        if isinstance(model_out, dict):
            try:
                best = max(legal_actions, key=lambda a: model_out.get(a, float("-inf")))
                return int(best)
            except Exception:
                pass
        # fallback
        return int(random.choice(list(legal_actions)))

    def predict(self, observations: Dict[str, Any]) -> Any:
        """Run forward pass and return raw model output (int/logits/dict)."""
        with self._lock:
            if self._model is None:
                raise RuntimeError("Model not loaded")
            x = self.preprocess(observations)
            with torch.no_grad():
                out = self._model(x)
            return out

    def select_action(self, legal_actions: Sequence[int], observations: Dict[str, Any]) -> int:
        """Return an integer action id, validated against legal_actions."""
        out = self.predict(observations)
        return self.postprocess(out, legal_actions, observations)

    # expose postprocess for get_opponent_policy compatibility
    @property
    def postprocess(self):
        return self._default_postprocess


def make_pytorch_opponent(ckpt_path: str, model_ctor: Optional[Callable[[], torch.nn.Module]] = None, **kwargs) -> PyTorchOpponentAdapter:
    """
    Convenience factory to create a PyTorchOpponentAdapter.
    Pass the returned object into get_opponent_policy(...) or directly into OpenSpielEnvironment(opponent_policy=...).
    """
    return PyTorchOpponentAdapter(ckpt_path=ckpt_path, model_ctor=model_ctor, **kwargs)


class HotSwapOpponent:
    """
    Wrapper around a PyTorchOpponentAdapter (or similar object) that allows hot-swapping
    the checkpoint path in-place without restarting the training process.

    Example usage:
        adapter = make_pytorch_opponent("ckpt_v1.pt", model_ctor=MyModel)
        hot = HotSwapOpponent(adapter)
        policy = get_opponent_policy(hot)  # hot.select_action will proxy to adapter
        # Later, to swap:
        hot.reload("ckpt_v2.pt")
    """

    def __init__(self, adapter: PyTorchOpponentAdapter):
        self._adapter = adapter
        self._lock = threading.RLock()

    def select_action(self, legal_actions: Sequence[int], observations: Dict[str, Any]) -> int:
        with self._lock:
            return self._adapter.select_action(legal_actions, observations)

    def reload(self, ckpt_path: Optional[str] = None):
        with self._lock:
            self._adapter.reload(ckpt_path)