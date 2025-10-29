#!/usr/bin/env python3
"""
Run train or battle with an LLM opponent (unsloth/Llama-3.2-3B).

This example demonstrates:
- using Hugging Face Inference API as the LLM opponent
- using a custom HTTP endpoint (e.g. local TGI/vLLM server)
- train mode: learner trains vs frozen LLM opponent (OpenSpielEnvironment)
- battle mode: LLM vs other LLM or model (MultiAgent)

Usage examples:
  # using Hugging Face Inference API
  export HF_API_KEY="hf_xxx"
  python examples/run_train_or_battle_llm.py --mode train --llm backend=hf model=unsloth/Llama-3.2-3B

  # using a local TGI/vLLM endpoint
  python examples/run_train_or_battle_llm.py --mode battle --llm backend=http url=http://localhost:8080/choose_action

Notes:
- For stable, deterministic responses set temperature=0 on the server side or include it in HF parameters.
- Remote LLMs are slow — use small timeout and fallback to avoid blocking training.
"""
import argparse
import os
import random
import time
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from envs.openspiel_env.server.mode_manager import ModeManager
from envs.openspiel_env.server.llm_opponent_adapter import make_hf_http_llm_policy, make_generic_http_llm_policy
from envs.openspiel_env.server.opponent_policies import get_opponent_policy
from envs.openspiel_env.server.opponent_model_loader import HotSwapOpponent

# Dummy learner for demonstration
class DummyLearner:
    def select_action(self, obs):
        return random.choice(obs.legal_actions)
    def learn(self, obs, action, reward, next_obs, done):
        pass

def build_llm_policy_from_args(args) -> callable:
    backend = args.llm_backend
    if backend == "hf":
        model_id = args.model
        hf_key = args.hf_api_key or os.environ.get("HF_API_KEY") or os.environ.get("HF_TOKEN")
        if not hf_key:
            raise RuntimeError("HF API key required for HF backend (set HF_API_KEY env)")
        return make_hf_http_llm_policy(model_id, hf_api_key=hf_key, timeout=args.timeout, max_retries=args.retries, fallback=args.fallback, game_name=args.game)
    elif backend == "http":
        url = args.url
        return make_generic_http_llm_policy(url, timeout=args.timeout, max_retries=args.retries, fallback=args.fallback, game_name=args.game)
    else:
        raise ValueError("Unknown llm backend")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train","battle"], default="train")
    parser.add_argument("--game", default="tic_tac_toe")
    parser.add_argument("--llm-backend", dest="llm_backend", choices=["hf","http"], default="hf")
    parser.add_argument("--model", default="unsloth/Llama-3.2-3B")   # for HF
    parser.add_argument("--url", default="http://localhost:8080/choose_action")  # for custom HTTP
    parser.add_argument("--hf-api-key", default=None)
    parser.add_argument("--timeout", type=float, default=1.0)
    parser.add_argument("--retries", type=int, default=1)
    parser.add_argument("--fallback", choices=["random","first","last"], default="random")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--matches", type=int, default=50)
    args = parser.parse_args()

    mm = ModeManager(game_name=args.game)
    if args.mode == "train":
        learner = DummyLearner()
        llm_policy_callable = build_llm_policy_from_args(args)
        opponent_policy = get_opponent_policy(llm_policy_callable)
        print("Running TRAIN mode: learner vs LLM opponent")
        mm.run_train(learner_agent=learner, opponent_policy=opponent_policy, episodes=args.episodes)
    else:
        # Battle: LLM vs LLM (or LLM vs DummyFrozenModel)
        llm_policy_a = build_llm_policy_from_args(args)
        llm_policy_b = build_llm_policy_from_args(args)  # could be different flags / endpoints
        # For MultiAgent battle we need objects with select_action(legal_actions, obs)
        # we can wrap the callables into simple objects:
        class CallableWrapper:
            def __init__(self, fn):
                self.fn = fn
            def select_action(self, legal_actions, obs):
                return self.fn(legal_actions, obs)

        model_a = CallableWrapper(llm_policy_a)
        model_b = CallableWrapper(llm_policy_b)
        print("Running BATTLE mode: LLM vs LLM")
        stats = mm.run_battle(model_a, model_b, matches=args.matches)
        print("Battle stats:", stats)

if __name__ == "__main__":
    main()