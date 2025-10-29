#!/usr/bin/env python3
"""
OpenSpiel Train CLI (ModeManager.run_train)

Train a learner against either a heuristic opponent or an LLM opponent.

Opponent options (--opponent):
- random | first | last      -> built-in heuristics
- hf                         -> Hugging Face Inference API (HTTP)
- http                       -> Generic HTTP JSON endpoint
- pycall                     -> Python callable or class via import path (module:attr)
- unsloth-local              -> Local Unsloth model (in-process)

Examples:
  # Heuristic opponent (random)
  python examples/openspiel_train_cli.py --game tic_tac_toe --episodes 50 --opponent random

  # HF Inference API LLM opponent (requires HF token)
  python examples/openspiel_train_cli.py --game tic_tac_toe --episodes 20 \
    --opponent hf --model-id unsloth/Llama-3.2-3B --hf-token $HF_TOKEN

  # Generic HTTP endpoint
  python examples/openspiel_train_cli.py --game tic_tac_toe --episodes 20 \
    --opponent http --url http://localhost:8001/llm-move

  # Local Unsloth model as opponent
  python examples/openspiel_train_cli.py --game tic_tac_toe --episodes 20 \
    --opponent unsloth-local --model-id unsloth/Llama-3.2-3B

  # Custom Python callable opponent
  # where mypkg.mypolicies has def greedy_ttt(legal, obs) -> int
  python examples/openspiel_train_cli.py --game tic_tac_toe --episodes 50 \
    --opponent pycall --target mypkg.mypolicies:greedy_ttt
"""
from __future__ import annotations

import argparse
import importlib
import os
import random
import re
import sys
from typing import Any, Callable, Dict, Optional, Sequence

# Ensure repo root on path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from envs.openspiel_env.server.mode_manager import ModeManager
from envs.openspiel_env.server.llm_opponent_adapter import (
    make_hf_http_llm_policy,
    make_generic_http_llm_policy,
)

# ----------------------------
# Learner stub (replace with your learner)
# ----------------------------
class RandomLearner:
    """Minimal learner: picks random legal action; no learning hook."""
    def select_action(self, obs):
        legal = getattr(obs, "legal_actions", None) or []
        return random.choice(legal) if legal else 0

# ----------------------------
# Utilities to build opponents
# ----------------------------
def load_pycall(target: str) -> Any:
    """
    Load a callable or class instance from "module:attr".
    If attr is a class, instantiate with no args.
    """
    if ":" not in target:
        raise ValueError("pycall target must be 'module.path:attr'")
    mod_name, attr_name = target.split(":", 1)
    mod = importlib.import_module(mod_name)
    obj = getattr(mod, attr_name)
    if isinstance(obj, type):
        return obj()  # instantiate class
    return obj  # function or object

def make_unsloth_local_llm_policy(model_id: str, device: Optional[str] = None, max_new_tokens: int = 32, temperature: float = 0.0):
    """
    Build a simple in-process Unsloth LLM opponent policy:
      callable(legal_actions, observations) -> action_id
    The prompt asks for a single integer from legal_actions.
    """
    # Lazy imports so people without Unsloth can still use heuristics
    from unsloth import FastLanguageModel
    from transformers import AutoTokenizer
    import torch

    tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    if tok.pad_token_id is None:
        tok.add_special_tokens({"pad_token": "[PAD]"})
    model, _ = FastLanguageModel.from_pretrained(model_id, max_seq_length=768, load_in_4bit=False)
    try:
        model.to(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        model.eval()
        for p in model.parameters():
            p.requires_grad = False
    except Exception:
        pass

    int_re = re.compile(r"-?\d+")

    def _build_prompt(game_name: str, player_id: int, legal_actions: Sequence[int], observations: Dict[str, Any]) -> str:
        info_state = observations.get("info_state", observations)
        return (
            f"You are playing {game_name} as player {player_id}.\n"
            f"info_state={info_state}\n"
            f"legal_actions={list(legal_actions)}\n"
            "Respond with exactly ONE integer: one of the legal action ids.\n"
        )

    def _parse_action(text: str, legal: Sequence[int]) -> Optional[int]:
        if not text:
            return None
        m = int_re.search(text.strip())
        if not m:
            return None
        try:
            x = int(m.group(0))
            return x if x in legal else None
        except Exception:
            return None

    def policy(legal_actions: Sequence[int], observations: Dict[str, Any]) -> int:
        if not legal_actions:
            return 0
        player_id = observations.get("current_player", 0)
        game_name = observations.get("game_name", "game")
        prompt = _build_prompt(game_name, player_id, legal_actions, observations)
        inputs = tok(prompt, return_tensors="pt")
        try:
            # Move inputs to model device if possible
            dev = next(model.parameters()).device
            inputs = {k: v.to(dev) for k, v in inputs.items()}
        except Exception:
            pass
        try:
            out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=(temperature > 0.0),
                                 temperature=temperature, eos_token_id=tok.eos_token_id, pad_token_id=tok.pad_token_id)
            text = tok.decode(out[0], skip_special_tokens=True)
        except Exception:
            return int(random.choice(list(legal_actions)))
        # Keep only the completion part if it starts with prompt
        if text.startswith(prompt):
            text = text[len(prompt):]
        a = _parse_action(text, legal_actions)
        return int(a) if a is not None else int(random.choice(list(legal_actions)))

    return policy

def build_opponent(args: argparse.Namespace) -> Any:
    t = (args.opponent or "").strip().lower()
    if t in ("random", "first", "last"):
        return t  # ModeManager wraps simple strings

    if t == "hf":
        if not args.model_id:
            raise ValueError("--model-id is required for --opponent hf")
        token = args.hf_token or os.environ.get("HF_API_KEY") or os.environ.get("HF_TOKEN")
        if not token:
            raise ValueError("HF token not found. Pass --hf-token or set HF_API_KEY/HF_TOKEN env.")
        return make_hf_http_llm_policy(
            model_id=args.model_id,
            hf_api_key=token,
            timeout=args.http_timeout,
            max_retries=args.http_retries,
            game_name=args.game,
        )

    if t == "http":
        if not args.url:
            raise ValueError("--url is required for --opponent http")
        headers = {}
        for kv in args.http_header or []:
            if ":" not in kv:
                raise ValueError(f"Invalid --http-header '{kv}'. Expected key:value")
            k, v = kv.split(":", 1)
            headers[k.strip()] = v.strip()
        return make_generic_http_llm_policy(
            endpoint_url=args.url,
            timeout=args.http_timeout,
            max_retries=args.http_retries,
            fallback="random",
            game_name=args.game,
            headers=headers or None,
        )

    if t == "pycall":
        if not args.target:
            raise ValueError("--target is required for --opponent pycall (module:attr)")
        return load_pycall(args.target)

    if t == "unsloth-local":
        if not args.model_id:
            raise ValueError("--model-id is required for --opponent unsloth-local")
        return make_unsloth_local_llm_policy(
            model_id=args.model_id,
            device=args.device,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
        )

    raise ValueError(f"Unknown opponent type: {t}")

# ----------------------------
# CLI
# ----------------------------
def main():
    p = argparse.ArgumentParser(description="Train a learner vs heuristic or LLM opponent (ModeManager.run_train)")
    p.add_argument("--game", type=str, default="tic_tac_toe", help="OpenSpiel game (e.g., tic_tac_toe, kuhn_poker)")
    p.add_argument("--episodes", type=int, default=50, help="Number of training episodes")
    p.add_argument("--max-steps", type=int, default=200, help="Safety cap per episode")
    p.add_argument("--agent-player", type=int, default=0, help="Which player the learner controls")

    # Opponent selection
    p.add_argument("--opponent", type=str, required=True, help="random|first|last|hf|http|pycall|unsloth-local")

    # HF / HTTP shared
    p.add_argument("--model-id", type=str, help="Model id for hf/unsloth-local")
    p.add_argument("--hf-token", type=str, default=None, help="HF API token for hf")
    p.add_argument("--url", type=str, help="Endpoint URL for http")
    p.add_argument("--http-timeout", type=float, default=1.0)
    p.add_argument("--http-retries", type=int, default=1)
    p.add_argument("--http-header", action="append", help="Extra HTTP headers key:value (repeatable)")

    # unsloth-local tuning
    p.add_argument("--device", type=str, default=None, help="Device for local Unsloth model (auto if omitted)")
    p.add_argument("--max-new-tokens", type=int, default=32)
    p.add_argument("--temperature", type=float, default=0.0)

    # pycall
    p.add_argument("--target", type=str, help="module:attr for pycall opponent")

    # Learner choice (optional future extension). For now we use RandomLearner.
    args = p.parse_args()

    # Build opponent
    opponent = build_opponent(args)

    # Build learner (replace with your RL learner if needed)
    learner = RandomLearner()

    # Train
    mm = ModeManager(game_name=args.game, agent_player=args.agent_player)
    mm.run_train(
        learner_agent=learner,
        opponent_policy=opponent,
        episodes=args.episodes,
        max_steps_per_episode=args.max_steps,
    )
    print("Training run completed.")

if __name__ == "__main__":
    main()