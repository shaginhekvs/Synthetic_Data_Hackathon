#!/usr/bin/env python3
"""
General OpenSpiel multi-agent test harness for OpenEnv with per-player model configuration.

Place this file in: examples/openspiel_multiagent_test.py

Features:
- Per-player model configuration (--model-names)
- Optionally share a single model across players (--share-model)
- Robust observation handling: tries information_state_string, falls back to observation_string or observation_tensor
- ROCm-aware offload/cleanup helper
- Disables torch._inductor async compile (best-effort) to avoid shutdown hangs during debugging
- --verbose for loader / prompt / response debugging
- --fail-on-load-error to stop on LLM load failures (instead of falling back to RandomAgent)
- --dump-dir to save per-episode summaries (JSONL)
"""

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

    def observe(self, transition: Dict):
        pass

    def update(self):
        pass


class RandomAgent(AgentInterface):
    def act(self, obs, legal_actions, player, state):
        if not legal_actions:
            return 0
        return random.choice(legal_actions)


# ----------------------------
# Helpers: model loading & safe observation access
# ----------------------------
def safe_information_state_string(state: pyspiel.State, player: int) -> Optional[str]:
    """Return information_state_string if available, otherwise None (catch OpenSpiel exceptions)."""
    try:
        if hasattr(state, "information_state_string"):
            return state.information_state_string(player)
    except Exception:
        return None
    return None


def _load_fast_language_model(
    model_name: str,
    max_seq_length: int = 768,
    load_in_4bit: bool = False,
    lora_rank: int = 0,
    verbose: bool = False,
) -> Tuple[Any, Any]:
    """
    Helper to load an unsloth.FastLanguageModel and return (model, tokenizer).
    Raises ImportError/Exception if imports fail.
    """
    if verbose:
        logger.info("[LLM LOAD] attempting to import FastLanguageModel for '%s'", model_name)
    FastLanguageModel = None
    try:
        from unsloth import FastLanguageModel
    except ImportError:
        raise ImportError(
            "Could not import FastLanguageModel (unsloth/fastlm). "
            "Install 'unsloth' or provide a compatible loader."
        )
    try:
        if verbose:
            logger.info("[LLM LOAD] calling from_pretrained('%s')", model_name)
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=max_seq_length,
            load_in_4bit=load_in_4bit,
        )
        if verbose:
            logger.info("[LLM LOAD] from_pretrained succeeded for '%s'", model_name)
    except Exception as e:
        if verbose:
            logger.exception("[LLM LOAD] from_pretrained failed for '%s': %s", model_name, e)
        raise

    # Optional PEFT/LoRA
    if hasattr(FastLanguageModel, "get_peft_model") and lora_rank and lora_rank > 0:
        if verbose:
            logger.info("[LLM LOAD] applying PEFT/LoRA r=%s", lora_rank)
        model = FastLanguageModel.get_peft_model(
            model,
            r=lora_rank,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        )

    return model, tokenizer


class UnslothLLMAgent(AgentInterface):
    """
    Adapter for unsloth.FastLanguageModel-based agents.

    Accepts either:
    - model_name: to load its own model
    - OR a pre-loaded model+tokenizer pair (shared across agents)

    Parameters:
    - verbose: print prompt/response logs
    - fail_on_load_error: if True, raise on load failure; if False, fall back to RandomAgent (handled by caller)
    """

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
                # Let caller decide fallback; we'll re-raise or leave model None
                if self.verbose:
                    logger.exception("UnslothLLMAgent failed to load model '%s'", self.model_name)

    def _load(self):
        if not self.model_name:
            raise ValueError("No model_name provided to load an LLM")
        self.model, self.tokenizer = _load_fast_language_model(
            model_name=self.model_name,
            max_seq_length=self.max_seq_length,
            lora_rank=self.lora_rank,
            load_in_4bit=False,
            verbose=self.verbose,
        )

    def format_observation_to_prompt(self, obs, player: int, state: pyspiel.State) -> str:
        """
        Robust prompt formatting:
        - Try information_state_string safely.
        - Fallback to observation_string.
        - Fallback to observation_tensor -> stringify.
        - Enumerate legal actions to make parsing deterministic.
        """
        # 1) Try information_state_string (safe)
        info = safe_information_state_string(state, player)
        la = _get_legal_actions(state, player)
        if info:
            return (
                f"Player {player} information state:\n{info}\n\n"
                f"Legal actions (action ids): {la}\n"
                "Please respond with the numeric action id only."
            )

        # 2) observation_string
        try:
            if hasattr(state, "observation_string"):
                obs_str = state.observation_string(player)
                if obs_str:
                    return (
                        f"Player {player} observation:\n{obs_str}\n\n"
                        f"Legal actions (action ids): {la}\n"
                        "Please respond with the numeric action id only."
                    )
        except Exception:
            pass

        # 3) observation_tensor -> stringify
        try:
            if hasattr(state, "observation_tensor"):
                ot = state.observation_tensor(player)
                try:
                    ot_text = ot.tolist() if hasattr(ot, "tolist") else str(ot)
                except Exception:
                    ot_text = str(ot)
                return (
                    f"Player {player} observation (tensor):\n{ot_text}\n\n"
                    f"Legal actions (action ids): {la}\n"
                    "Please respond with the numeric action id only."
                )
        except Exception:
            pass

        # Last resort
        return (
            f"Player {player} observation: (no readable observation available).\n"
            f"Legal actions (action ids): {la}\n"
            "Please respond with the numeric action id only."
        )

    def _prompt_to_action(self, response: str, legal_actions: List[int]) -> int:
        # Parse numeric tokens and map to legal actions
        tokens = response.strip().split()
        for t in tokens:
            try:
                a = int(t)
                if a in legal_actions:
                    return a
            except Exception:
                continue
        # fallback
        return random.choice(legal_actions) if legal_actions else 0

    def act(self, obs, legal_actions, player, state):
        prompt = self.format_observation_to_prompt(obs, player, state)
        if self.verbose:
            logger.info("[LLM ACT] prompt for player %s: %s", player, prompt)
        try:
            # Try a few common generation method names
            if hasattr(self.model, "generate_text"):
                resp = self.model.generate_text(prompt, max_length=64)
                text = resp if isinstance(resp, str) else getattr(resp, "text", str(resp))
            elif hasattr(self.model, "generate"):
                out = self.model.generate(prompt, max_new_tokens=64)
                text = out if isinstance(out, str) else getattr(out, "text", str(out))
            else:
                out = self.model(prompt)
                text = out if isinstance(out, str) else getattr(out, "text", str(out))
            if self.verbose:
                logger.info("[LLM ACT] raw response: %s", repr(text))
        except Exception as e:
            if self.verbose:
                logger.exception("[LLM ACT] model generation failed: %s", e)
            # fallback to random legal action
            return random.choice(legal_actions) if legal_actions else 0

        return self._prompt_to_action(text, legal_actions)

    def observe(self, transition: Dict):
        pass

    def update(self):
        pass

    def unload(self):
        """Explicit unload hook to help offload helpers free memory."""
        for fn in ("unload", "close", "shutdown", "dispose"):
            if hasattr(self.model, fn):
                try:
                    getattr(self.model, fn)()
                except Exception:
                    pass
        try:
            del self.model
        except Exception:
            pass
        try:
            del self.tokenizer
        except Exception:
            pass
        self.model = None
        self.tokenizer = None


# ----------------------------
# Utilities: observations & actions
# ----------------------------
def _get_legal_actions(state: pyspiel.State, player: Optional[int] = None) -> List[int]:
    try:
        if player is None:
            return state.legal_actions()
        else:
            return state.legal_actions(player)
    except Exception:
        try:
            return state.legal_actions(state.current_player())
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
    try:
        return state.observation_string(player)
    except Exception:
        pass
    return None


def _sample_chance_action(state: pyspiel.State) -> int:
    outcomes = state.chance_outcomes()
    if not outcomes:
        raise RuntimeError("Chance node with no outcomes")
    actions, probs = zip(*outcomes)
    probs = np.array(probs, dtype=float)
    probs = probs / probs.sum()
    choice = np.random.choice(len(actions), p=probs)
    return actions[int(choice)]


# ----------------------------
# Episode runner
# ----------------------------
def run_episode(
    game_name: str,
    agents: List[AgentInterface],
    seed: Optional[int] = None,
    prefer_information_state: bool = True,
    train_mode: bool = False,
    max_steps: Optional[int] = None,
) -> Dict:
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    game = pyspiel.load_game(game_name)
    state = game.new_initial_state()

    transitions = []
    step = 0
    while not state.is_terminal():
        if max_steps is not None and step >= max_steps:
            break

        # Guard the per-step logic so OpenSpiel API differences don't crash the harness
        try:
            if state.is_chance_node():
                action = _sample_chance_action(state)
                state.apply_action(action)
                step += 1
                continue

            if state.is_simultaneous_node():
                num_players = game.num_players()
                joint_actions = []
                for p in range(num_players):
                    obs = _get_observation(state, p, prefer_information_state=prefer_information_state)
                    legal = _get_legal_actions(state, p)
                    action = agents[p].act(obs, legal, p, state)
                    if action not in legal:
                        action = random.choice(legal) if legal else 0
                    joint_actions.append(action)
                try:
                    state.apply_actions(joint_actions)
                except AttributeError:
                    for a in joint_actions:
                        state.apply_action(a)
                step += 1
                continue

            # Normal (turn-based) node
            cur = state.current_player()
            obs = _get_observation(state, cur, prefer_information_state=prefer_information_state)
            legal = _get_legal_actions(state)
            action = agents[cur].act(obs, legal, cur, state)
            if action not in legal:
                action = random.choice(legal) if legal else 0
            pre_obs = obs
            state.apply_action(action)
            step += 1

            if train_mode:
                transitions.append(
                    {
                        "player": cur,
                        "obs": pre_obs,
                        "action": action,
                        "state_after": state.clone() if hasattr(state, "clone") else None,
                        "time": time.time(),
                    }
                )
        except Exception as e:
            logger.exception("[run_episode] caught exception at step %s: %s", step, e)
            # Terminate this episode cleanly
            break

    returns = list(state.returns())
    max_ret = max(returns) if returns else None
    winners = [i for i, r in enumerate(returns) if r == max_ret] if max_ret is not None else []

    if train_mode:
        final_returns = returns
        for t in transitions:
            player = t["player"]
            try:
                agents[player].observe(
                    {
                        "obs": t["obs"],
                        "action": t["action"],
                        "final_return": final_returns[player] if final_returns else None,
                        "done": True,
                    }
                )
            except Exception:
                pass
        for a in agents:
            try:
                a.update()
            except Exception:
                pass

    return {
        "game": game_name,
        "returns": returns,
        "winners": winners,
        "length": step,
        "transitions": transitions,
    }


def run_episodes(
    game_name: str,
    agents: List[AgentInterface],
    num_episodes: int = 100,
    seed: Optional[int] = None,
    prefer_information_state: bool = True,
    train_mode: bool = False,
    max_steps_per_episode: Optional[int] = None,
) -> Dict:
    wins = [0] * len(agents)
    sums = [0.0] * len(agents)
    records = []
    for ep in range(num_episodes):
        ep_seed = None if seed is None else seed + ep
        r = run_episode(
            game_name,
            agents,
            seed=ep_seed,
            prefer_information_state=prefer_information_state,
            train_mode=train_mode,
            max_steps=max_steps_per_episode,
        )
        records.append(r)
        for i, ret in enumerate(r["returns"]):
            sums[i] += ret
        for w in r["winners"]:
            wins[w] += 1
    mean_returns = [s / num_episodes for s in sums]
    winrates = [w / num_episodes for w in wins]
    return {
        "game": game_name,
        "num_episodes": num_episodes,
        "mean_returns": mean_returns,
        "winrates": winrates,
        "records": records,
    }


# ----------------------------
# ROCm-aware offload helper
# ----------------------------
def offload_model_rocm(model_obj, tokenizer_obj=None, wait_seconds: float = 0.5, verbose: bool = True):
    """
    Best-effort cleanup for ROCm-backed PyTorch models and common wrapper APIs.
    Call this after finishing with model_obj.
    """
    if verbose:
        logger.info("[offload] start")

    for fn in ("unload", "close", "shutdown", "dispose"):
        if hasattr(model_obj, fn):
            try:
                if verbose:
                    logger.info("[offload] calling model.%s()", fn)
                getattr(model_obj, fn)()
            except Exception as e:
                if verbose:
                    logger.info("[offload] model.%s() raised: %s", fn, e)

    inner = getattr(model_obj, "model", None) or getattr(model_obj, "transformer", None) or model_obj
    try:
        if verbose:
            logger.info("[offload] moving inner model to CPU via .to('cpu')")
        inner.to("cpu")
    except Exception as e:
        if verbose:
            logger.info("[offload] moving to CPU failed/ignored: %s", e)

    try:
        del model_obj
    except Exception:
        pass
    if tokenizer_obj is not None:
        try:
            del tokenizer_obj
        except Exception:
            pass

    gc.collect()
    time.sleep(wait_seconds)

    try:
        import torch as _torch  # type: ignore
        if _torch.cuda.is_available():
            try:
                _torch.cuda.empty_cache()
                if verbose:
                    logger.info("[offload] called torch.cuda.empty_cache()")
                try:
                    if verbose:
                        logger.info("[offload] torch.cuda.memory_summary():\n%s", _torch.cuda.memory_summary())
                except Exception:
                    pass
            except Exception as e:
                if verbose:
                    logger.info("[offload] torch.cuda.empty_cache() failed: %s", e)
    except Exception:
        if verbose:
            logger.info("[offload] torch not available; skipped torch cleanup")

    try:
        out = subprocess.check_output(["rocm-smi"], stderr=subprocess.STDOUT, text=True)
        if verbose:
            logger.info("[offload] rocm-smi output:\n%s", out)
    except FileNotFoundError:
        if verbose:
            logger.info("[offload] rocm-smi not found on PATH; skip")
    except subprocess.CalledProcessError as e:
        if verbose:
            logger.info("[offload] rocm-smi returned non-zero code; output:\n%s", e.output)

    if verbose:
        logger.info("[offload] done")


# ----------------------------
# CLI entrypoint
# ----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--game", type=str, required=False, help="OpenSpiel game name (kuhn_poker, chess, tic_tac_toe, ...)")
    parser.add_argument("--episodes", type=int, default=100, help="Number of episodes to run")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--prefer_info_state", action="store_true", dest="prefer_info_state", help="Prefer information_state over observation")
    parser.add_argument("--train", action="store_true", help="Enable train mode (agents.observe/update called)")
    parser.add_argument("--use-llm", action="store_true", help="Use UnslothLLMAgent for players (example)")
    parser.add_argument("--model-name", type=str, default="unsloth/gpt-oss-20b", help="Default model name for LLM agent")
    parser.add_argument("--model-names", type=str, default="", help="Comma-separated model names for each player (overrides --model-name when length==num_players)")
    parser.add_argument("--share-model", action="store_true", help="If set, load a single shared model instance and share across all LLM agents")
    parser.add_argument("--lora-ranks", type=str, default="", help="Comma-separated LoRA ranks per player or single value")
    parser.add_argument("--list-games", action="store_true", help="Print available OpenSpiel games and exit")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose loader / prompt / response logging")
    parser.add_argument("--fail-on-load-error", action="store_true", help="If set, fail immediately when an LLM fails to load")
    parser.add_argument("--dump-dir", type=str, default="", help="Directory to dump per-episode records (JSONL).")
    parser.add_argument("--max-steps", type=int, default=None, help="Max steps per episode (safety)")
    args = parser.parse_args()

    if args.verbose:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)

    # Get available games from pyspiel and normalize names
    games = []
    try:
        for gt in pyspiel.registered_games():
            name = getattr(gt, "short_name", None)
            if callable(name):
                try:
                    name = name()
                except Exception:
                    name = None
            if name is None:
                name = str(gt)
            games.append(name)
    except Exception:
        games = []

    if args.list_games:
        print("Available OpenSpiel games:")
        print("\n".join(sorted(games)))
        return

    if not args.game:
        print("Error: --game must be provided (or use --list-games to view available games).")
        return

    if args.game not in games:
        suggestions = difflib.get_close_matches(args.game, games, n=5, cutoff=0.5)
        print(f"Error: Unknown game '{args.game}'.")
        if suggestions:
            print("Did you mean:", ", ".join(suggestions))
        print("Available games:")
        print("\n".join(sorted(games)))
        return

    # Load the game
    try:
        g = pyspiel.load_game(args.game)
    except Exception as e:
        print(f"Error loading game {args.game}: {e}")
        return

    n = g.num_players()

    # parse model-names and lora-ranks
    model_names_list = [s.strip() for s in args.model_names.split(",") if s.strip()] if args.model_names else []
    lora_ranks_list = []
    if args.lora_ranks:
        try:
            lora_ranks_list = [int(s.strip()) for s in args.lora_ranks.split(",") if s.strip()]
        except Exception:
            logger.warning("Could not parse --lora-ranks; ignoring LoRA ranks.")
            lora_ranks_list = []

    if model_names_list and len(model_names_list) != n:
        logger.warning(
            "Provided --model-names has %d entries but the game '%s' expects %d players. Using round-robin mapping.",
            len(model_names_list),
            args.game,
            n,
        )

    # show mapping of player -> chosen model (for debugging)
    mapping = []
    for i in range(n):
        if len(model_names_list) == n:
            m = model_names_list[i]
        elif len(model_names_list) > 0:
            m = model_names_list[i % len(model_names_list)]
        else:
            m = args.model_name
        mapping.append((i, m))
    print("Player -> model mapping:")
    for p, m in mapping:
        print(f"  player {p}: {m}")

    # Build agents
    agents: List[AgentInterface] = []
    shared_model = None
    shared_tokenizer = None

    if args.use_llm:
        # Optionally load a shared model
        if args.share_model:
            chosen_model_name = model_names_list[0] if len(model_names_list) >= 1 else args.model_name
            chosen_lora = lora_ranks_list[0] if len(lora_ranks_list) >= 1 else 0
            try:
                shared_model, shared_tokenizer = _load_fast_language_model(
                    model_name=chosen_model_name,
                    max_seq_length=768,
                    lora_rank=chosen_lora,
                    load_in_4bit=False,
                    verbose=args.verbose,
                )
            except Exception as e:
                logger.exception("Failed to load shared model '%s': %s", chosen_model_name, e)
                if args.fail_on_load_error:
                    raise
                print("Falling back to RandomAgent for all players.")
                agents = [RandomAgent() for _ in range(n)]
                shared_model = None

        # Per-player creation
        if not agents:  # if fallback not triggered
            for i in range(n):
                if args.share_model and shared_model is not None:
                    try:
                        agents.append(
                            UnslothLLMAgent(
                                model=shared_model,
                                tokenizer=shared_tokenizer,
                                model_name=None,
                                verbose=args.verbose,
                                fail_on_load_error=args.fail_on_load_error,
                            )
                        )
                    except Exception as e:
                        logger.exception("Failed to create agent %s with shared model: %s", i, e)
                        agents.append(RandomAgent())
                else:
                    # pick per-player model name or default
                    if len(model_names_list) == n:
                        mname = model_names_list[i]
                    elif len(model_names_list) > 0:
                        mname = model_names_list[i % len(model_names_list)]
                    else:
                        mname = args.model_name
                    # pick lora rank
                    lora_rank = lora_ranks_list[i] if len(lora_ranks_list) == n else (lora_ranks_list[0] if len(lora_ranks_list) == 1 else 0)
                    try:
                        agents.append(
                            UnslothLLMAgent(
                                model_name=mname,
                                lora_rank=lora_rank,
                                verbose=args.verbose,
                                fail_on_load_error=args.fail_on_load_error,
                            )
                        )
                    except Exception as e:
                        logger.exception("Failed to create UnslothLLMAgent for player %s (model %s): %s", i, mname, e)
                        if args.fail_on_load_error:
                            raise
                        agents.append(RandomAgent())
    else:
        agents = [RandomAgent() for _ in range(n)]

    # Run episodes
    res = run_episodes(
        args.game,
        agents,
        num_episodes=args.episodes,
        seed=args.seed,
        prefer_information_state=args.prefer_info_state,
        train_mode=args.train,
        max_steps_per_episode=args.max_steps,
    )
    num_players = len(agents) 
    wins_exclusive = [0] * num_players
    draws = 0
    
    records = res.get("records", [])
    for rec in records:
        winners = rec.get("winners", [])
        if winners is None:
            continue
        if len(winners) == 1:
            wins_exclusive[winners[0]] += 1
        else:
            # treat multi-winner (ties) as a draw here
            draws += 1
    
    # Avoid division by zero if res["num_episodes"] is missing
    num_eps = res.get("num_episodes", len(records) if records else 1)
    winrates_exclusive = [w / num_eps for w in wins_exclusive]
    print("Exclusive winrates:", winrates_exclusive, "draws:", draws)
    print("Summary:", {"game": res["game"], "num_episodes": res["num_episodes"], "mean_returns": res["mean_returns"], "winrates": res["winrates"]})

    # Optionally dump per-episode records
    if args.dump_dir:
        os.makedirs(args.dump_dir, exist_ok=True)
        out_path = os.path.join(args.dump_dir, f"{args.game}_records.jsonl")
        with open(out_path, "w") as f:
            for rec in res.get("records", []):
                # Keep JSON-serializable: transitions may contain state clones that aren't serializable;
                # we strip transitions down to simple fields (player, action, maybe time).
                safe_trans = []
                for t in rec.get("transitions", []):
                    safe_trans.append({"player": t.get("player"), "action": t.get("action"), "time": t.get("time")})
                out = {
                    "game": rec.get("game"),
                    "returns": rec.get("returns"),
                    "winners": rec.get("winners"),
                    "length": rec.get("length"),
                    "transitions": safe_trans,
                }
                f.write(json.dumps(out) + "\n")
        print("Wrote records to", out_path)

    # Offload LLM models if present. If a shared model was used, offload only once.
    seen_models = set()
    for a in agents:
        if hasattr(a, "unload"):
            try:
                a.unload()
            except Exception:
                pass
        model_obj = getattr(a, "model", None)
        tok_obj = getattr(a, "tokenizer", None)
        if model_obj is None and shared_model is not None:
            model_obj = shared_model
            tok_obj = shared_tokenizer
        if model_obj is not None:
            mid = id(model_obj)
            if mid in seen_models:
                continue
            seen_models.add(mid)
            try:
                offload_model_rocm(model_obj, tok_obj, verbose=args.verbose)
            except Exception as e:
                logger.exception("Offload helper failed for model id %s: %s", mid, e)


if __name__ == "__main__":
    main()