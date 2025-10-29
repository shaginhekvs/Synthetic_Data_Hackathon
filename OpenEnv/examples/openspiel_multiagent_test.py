
from typing import List, Any, Optional, Dict, Tuple
import argparse
import random
import time
import gc
import subprocess
import json
import os
import difflib
import logging

import numpy as np
import pyspiel
import torch

logger = logging.getLogger("openspiel_multiagent_test")
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
logger.addHandler(handler)
logger.setLevel(logging.INFO)


# ----------------------------
# Agent interface and adapters
# ----------------------------
class AgentInterface:
    """Minimal agent interface for the harness. Implement act(), optionally observe() and update()."""
    def act(self, obs: Any, legal_actions: List[int], player: int, state: pyspiel.State) -> int:
        raise NotImplementedError
    def observe(self, transition: Dict): pass
    def update(self): pass


class RandomAgent(AgentInterface):
    def act(self, obs, legal_actions, player, state):
        return random.choice(legal_actions) if legal_actions else 0


# ----------------------------
# Helpers: model loading
# ----------------------------
def safe_information_state_string(state: pyspiel.State, player: int) -> Optional[str]:
    try:
        if hasattr(state, "information_state_string"):
            return state.information_state_string(player)
    except Exception:
        pass
    return None


def _load_fast_language_model(
    model_name: str,
    max_seq_length: int = 768,
    load_in_4bit: bool = False,
    lora_rank: int = 0,
    verbose: bool = False,
    device: Optional[str] = None,
) -> Tuple[Any, Any]:
    """Load an Unsloth FastLanguageModel and return (model, tokenizer) with generate_text ready."""
    if verbose:
        logger.info("[LLM LOAD] attempting to import FastLanguageModel for '%s'", model_name)
    try:
        from unsloth import FastLanguageModel
    except ImportError:
        raise ImportError("Could not import FastLanguageModel. Please `pip install unsloth`.")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        load_in_4bit=load_in_4bit,
    )

    # 🔥 Add this line — ensures .generate_text() exists
    try:
        FastLanguageModel.for_inference(model)
        if verbose:
            logger.info("[LLM LOAD] for_inference() applied to '%s'", model_name)
    except Exception as e:
        logger.warning("[LLM LOAD] for_inference() failed or unnecessary: %s", e)

    # Move to device if applicable
    if device == "cuda":
        try:
            model = model.to("cuda")
        except Exception:
            inner = getattr(model, "model", None) or getattr(model, "transformer", None)
            if inner is not None:
                inner.to("cuda")

    if verbose:
        logger.info("[LLM LOAD] final model type: %s has generate_text=%s", type(model), hasattr(model, "generate_text"))
    return model, tokenizer


# ----------------------------
# LLM Agent
# ----------------------------
class UnslothLLMAgent(AgentInterface):
    def __init__(
        self,
        model_name: Optional[str] = None,
        max_seq_length: int = 768,
        lora_rank: int = 0,
        device: Optional[str] = None,
        model: Optional[Any] = None,
        tokenizer: Optional[Any] = None,
        verbose: bool = False,
        fail_on_load_error: bool = False,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model_name = model_name
        self.max_seq_length = max_seq_length
        self.lora_rank = lora_rank
        self.verbose = verbose
        self.fail_on_load_error = fail_on_load_error

        if self.model is None and self.model_name is not None:
            try:
                self._load()
            except Exception:
                if self.fail_on_load_error:
                    raise
                if self.verbose:
                    logger.exception("UnslothLLMAgent failed to load model '%s'", self.model_name)

    def _load(self):
        self.model, self.tokenizer = _load_fast_language_model(
            model_name=self.model_name,
            max_seq_length=self.max_seq_length,
            lora_rank=self.lora_rank,
            load_in_4bit=False,
            device=self.device,
            verbose=self.verbose,
        )

    def format_observation_to_prompt(self, obs, player: int, state: pyspiel.State) -> str:
        info = safe_information_state_string(state, player)
        la = _get_legal_actions(state, player)
        if info:
            return f"Player {player} state:\n{info}\n\nLegal actions: {la}\nRespond with the numeric action id only."
        try:
            if hasattr(state, "observation_string"):
                obs_str = state.observation_string(player)
                return f"Player {player} observation:\n{obs_str}\n\nLegal actions: {la}\nRespond with the numeric action id only."
        except Exception:
            pass
        return f"Player {player} observation unavailable.\nLegal actions: {la}\nRespond with the numeric action id only."

    def _prompt_to_action(self, response: str, legal_actions: List[int]) -> int:
        for token in response.strip().split():
            try:
                act = int(token)
                if act in legal_actions:
                    return act
            except Exception:
                continue
        return random.choice(legal_actions) if legal_actions else 0

    def act(self, obs, legal_actions, player, state):
        prompt = self.format_observation_to_prompt(obs, player, state)
        if self.verbose:
            logger.info("[LLM ACT] prompt for player %s: %s", player, prompt)
        try:
            if hasattr(self.model, "generate_text"):
                text = self.model.generate_text(self.tokenizer, prompt, max_new_tokens=64)
                if not isinstance(text, str):
                    text = getattr(text, "text", str(text))
            elif hasattr(self.model, "generate"):
                inputs = self.tokenizer(prompt, return_tensors="pt")
                if self.device == "cuda":
                    inputs = {k: v.to("cuda") for k, v in inputs.items()}
                output = self.model.generate(**inputs, max_new_tokens=64)
                text = self.tokenizer.decode(output[0], skip_special_tokens=True)
            else:
                raise RuntimeError("Model has neither 'generate_text' nor 'generate'")
            if self.verbose:
                logger.info("[LLM ACT] raw response: %s", repr(text))
        except Exception as e:
            logger.exception("[LLM ACT] model generation failed: %s", e)
            return random.choice(legal_actions) if legal_actions else 0
        return self._prompt_to_action(text, legal_actions)

    def unload(self):
        for fn in ("unload", "close", "shutdown", "dispose"):
            if hasattr(self.model, fn):
                try:
                    getattr(self.model, fn)()
                except Exception:
                    pass
        self.model = None
        self.tokenizer = None


# ----------------------------
# Observation utilities
# ----------------------------
def _get_legal_actions(state: pyspiel.State, player: Optional[int] = None) -> List[int]:
    try:
        return state.legal_actions() if player is None else state.legal_actions(player)
    except Exception:
        return []


def _get_observation(state: pyspiel.State, player: int, prefer_information_state: bool = True):
    if prefer_information_state:
        try:
            return state.information_state_tensor(player)
        except Exception:
            pass
    try:
        return state.observation_tensor(player)
    except Exception:
        pass
    return None


# ----------------------------
# Episode runner
# ----------------------------
def run_episode(game_name, agents, seed=None, prefer_information_state=True, train_mode=False, max_steps=None):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    game = pyspiel.load_game(game_name)
    state = game.new_initial_state()
    step = 0
    while not state.is_terminal():
        if max_steps and step >= max_steps:
            break
        if state.is_chance_node():
            a, p = zip(*state.chance_outcomes())
            state.apply_action(random.choices(a, weights=p)[0])
            step += 1
            continue
        cur = state.current_player()
        obs = _get_observation(state, cur, prefer_information_state)
        legal = _get_legal_actions(state)
        action = agents[cur].act(obs, legal, cur, state)
        if action not in legal:
            action = random.choice(legal) if legal else 0
        state.apply_action(action)
        step += 1
    returns = list(state.returns())
    max_ret = max(returns) if returns else None
    winners = [i for i, r in enumerate(returns) if r == max_ret] if max_ret is not None else []
    return {"returns": returns, "winners": winners, "length": step}


def run_episodes(game_name, agents, num_episodes=100, seed=None, prefer_information_state=True, max_steps_per_episode=None):
    wins = [0] * len(agents)
    for ep in range(num_episodes):
        r = run_episode(game_name, agents, seed=(seed + ep if seed else None),
                        prefer_information_state=prefer_information_state,
                        max_steps=max_steps_per_episode)
        for w in r["winners"]:
            wins[w] += 1
    winrates = [w / num_episodes for w in wins]
    return {"game": game_name, "num_episodes": num_episodes, "winrates": winrates}


# ----------------------------
# CLI entrypoint
# ----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--game", required=True)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--use-llm", action="store_true")
    parser.add_argument("--model-name", type=str, default="unsloth/Llama-3.2-3B")
    parser.add_argument("--model-names", type=str, default="")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if args.verbose:
        logger.setLevel(logging.DEBUG)

    g = pyspiel.load_game(args.game)
    n = g.num_players()
    model_names = [s.strip() for s in args.model_names.split(",") if s.strip()] if args.model_names else []
    mapping = [model_names[i] if i < len(model_names) else args.model_name for i in range(n)]
    print("Player -> model mapping:")
    for i, m in enumerate(mapping):
        print(f"  player {i}: {m}")

    agents = []
    if args.use_llm:
        for i, m in enumerate(mapping):
            agents.append(UnslothLLMAgent(model_name=m, device=args.device, verbose=args.verbose))
    else:
        agents = [RandomAgent() for _ in range(n)]

    res = run_episodes(args.game, agents, num_episodes=args.episodes)
    print("Results:", res)


if __name__ == "__main__":
    main()
