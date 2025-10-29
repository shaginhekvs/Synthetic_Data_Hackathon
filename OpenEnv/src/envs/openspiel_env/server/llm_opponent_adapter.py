"""
llm_opponent_adapter.py

Adapter that turns an LLM HTTP endpoint into a callable compatible with
ExternalOpponentPolicy (i.e., callable(legal_actions, observations) -> action_id).

Supports:
- Hugging Face Inference API (token-based)
- Any custom HTTP endpoint that accepts JSON and returns an int action
- Deterministic reply guidance (temperature=0) via prompt and endpoint params
- Timeout, retries, and fallback handling

Usage:
    from envs.openspiel_env.server.llm_opponent_adapter import make_hf_http_llm_policy
    policy = make_hf_http_llm_policy("unsloth/Llama-3.2-3B", backend="hf", hf_api_key="xxx", timeout=1.0)
    env = OpenSpielEnvironment(..., opponent_policy=policy)
"""
from __future__ import annotations

import os
import time
import json
import random
import logging
from typing import Any, Dict, Iterable, Optional, Sequence

import requests

logger = logging.getLogger(__name__)
DEFAULT_TIMEOUT = 1.0
DEFAULT_RETRIES = 1

# We instruct the LLM to reply with a single integer (action id) only.
TTT_PROMPT_TEMPLATE = """
You are playing Tic-Tac-Toe as player {player_id}. The board info_state (from the player's perspective) is:
{info_state}

Legal moves (indices) are: {legal_actions}

Respond with exactly ONE integer: the index of the chosen move (one of the legal moves). Do not add any extra text or punctuation.
"""

def _build_prompt(game_name: str, player_id: int, legal_actions: Sequence[int], observations: Dict[str, Any]) -> str:
    # For tic_tac_toe we use info_state; for other games adjust accordingly
    info_state = observations.get("info_state", observations)
    return TTT_PROMPT_TEMPLATE.format(player_id=player_id, info_state=info_state, legal_actions=list(legal_actions)).strip()

def _parse_int_from_text(s: str) -> Optional[int]:
    # tolerant parse: try to extract first integer
    if not s:
        return None
    s = s.strip()
    # if exact integer
    try:
        return int(s)
    except Exception:
        pass
    # try find digits
    import re
    m = re.search(r"-?\d+", s)
    if m:
        try:
            return int(m.group(0))
        except Exception:
            return None
    return None

def _hf_inference_request(model_id: str, prompt: str, hf_api_key: str, timeout: float = DEFAULT_TIMEOUT, **kwargs) -> str:
    """
    Calls Hugging Face Inference API text generation endpoint.
    Returns the text content produced (string).

    NOTE: You must have appropriate access to the model on HF (some models require a paid plan).
    """
    url = f"https://api-inference.huggingface.co/models/{model_id}"
    headers = {"Authorization": f"Bearer {hf_api_key}", "Accept": "application/json"}
    payload = {
        "inputs": prompt,
        "options": {"wait_for_model": True},
        # Use generation params (temperature=0 for determinism)
        "parameters": {"max_new_tokens": 32, "do_sample": False, "temperature": 0.0},
    }
    resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    # HF inference returns [{'generated_text': '...'}] or other shapes; handle common ones
    if isinstance(data, dict) and "error" in data:
        raise RuntimeError(f"HF inference error: {data['error']}")
    # If list of generations
    if isinstance(data, list) and len(data) > 0:
        txt = data[0].get("generated_text") or data[0].get("text") or ""
        return txt
    # If dict with 'generated_text'
    if isinstance(data, dict) and "generated_text" in data:
        return data["generated_text"]
    # If dict with 'outputs' or similar
    if isinstance(data, dict) and "outputs" in data:
        out = data["outputs"]
        if isinstance(out, list) and len(out) > 0:
            return out[0].get("generated_text", "")
    # fallback: stringify response
    return json.dumps(data)

def _generic_http_request(url: str, payload: Dict[str, Any], timeout: float = DEFAULT_TIMEOUT, headers: Optional[Dict[str,str]] = None) -> str:
    resp = requests.post(url, json=payload, headers=headers or {}, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    # expect {"action": <int>} or {"result": "<int>"} or raw text
    if isinstance(data, dict):
        for key in ("action", "result", "choice"):
            if key in data:
                return str(data[key])
        # try first value
        vals = list(data.values())
        if vals:
            return str(vals[0])
    # fallback: return text
    return json.dumps(data)

def make_hf_http_llm_policy(
    model_id: str,
    hf_api_key: Optional[str] = None,
    timeout: float = DEFAULT_TIMEOUT,
    max_retries: int = DEFAULT_RETRIES,
    fallback: str = "random",
    game_name: str = "tic_tac_toe",
):
    """
    Returns a callable(legal_actions, observations) -> action_id that uses HF Inference API.

    Example:
        policy = make_hf_http_llm_policy("unsloth/Llama-3.2-3B", hf_api_key=os.environ['HF_TOKEN'])
    """
    if hf_api_key is None:
        hf_api_key = os.environ.get("HF_API_KEY") or os.environ.get("HF_TOKEN")
        if hf_api_key is None:
            raise ValueError("hf_api_key not provided and HF_API_KEY not in environment")

    def http_policy(legal_actions: Sequence[int], observations: Dict[str,Any]):
        # map to player id if available in observations
        player_id = observations.get("current_player", 0)
        prompt = _build_prompt(game_name, player_id, legal_actions, observations)
        last_exc = None
        for attempt in range(max_retries):
            try:
                txt = _hf_inference_request(model_id, prompt, hf_api_key, timeout=timeout)
                action = _parse_int_from_text(txt)
                if action is not None and action in legal_actions:
                    return int(action)
                # invalid parse or not legal -> continue to fallback after retries
                logger.warning("HF returned unparsable or illegal action: %r; text=%r", action, txt)
                last_exc = RuntimeError("invalid action returned by HF")
            except Exception as e:
                last_exc = e
                logger.exception("HF request failed: %s", e)
                time.sleep(0.1)
        # fallback
        logger.warning("HF opponent falling back after error: %s", last_exc)
        if fallback == "random":
            return int(random.choice(list(legal_actions)))
        if fallback == "first":
            return int(list(legal_actions)[0])
        return int(random.choice(list(legal_actions)))

    return http_policy

def make_generic_http_llm_policy(
    endpoint_url: str,
    timeout: float = DEFAULT_TIMEOUT,
    max_retries: int = DEFAULT_RETRIES,
    fallback: str = "random",
    game_name: str = "tic_tac_toe",
    headers: Optional[Dict[str,str]] = None,
    payload_key: str = "payload",
):
    """
    Adapter for a generic HTTP LLM endpoint. The endpoint must accept JSON and return JSON.
    It will be called with {"legal_actions": [...], "obs": <info_state>} by default.
    """
    def http_policy(legal_actions: Sequence[int], observations: Dict[str,Any]):
        payload = {"legal_actions": list(legal_actions), "obs": observations.get("info_state", observations)}
        last_exc = None
        for attempt in range(max_retries):
            try:
                txt = _generic_http_request(endpoint_url, payload, timeout=timeout, headers=headers)
                action = _parse_int_from_text(txt)
                if action is not None and action in legal_actions:
                    return int(action)
                last_exc = RuntimeError("invalid action from endpoint")
            except Exception as e:
                last_exc = e
                logger.exception("HTTP LLM policy error: %s", e)
                time.sleep(0.1)
        logger.warning("Generic HTTP LLM opponent falling back after error: %s", last_exc)
        if fallback == "random":
            return int(random.choice(list(legal_actions)))
        if fallback == "first":
            return int(list(legal_actions)[0])
        return int(random.choice(list(legal_actions)))

    return http_policy