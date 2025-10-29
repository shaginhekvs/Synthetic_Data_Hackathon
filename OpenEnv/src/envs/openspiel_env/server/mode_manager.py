#!/usr/bin/env python3
"""
ModeManager: orchestrates two modes for OpenEnv OpenSpiel usage:

- train mode: one learner trains vs a frozen opponent policy (can be callable/LLM).
- battle mode: two frozen policies play head-to-head; returns win/draw stats.

Key details:
- Works with the current OpenSpielEnvironment even though it expects simple policy strings,
  by overriding env.opponent_policy_fn after construction with a richer OpponentPolicy object.
- Tries to use MultiAgentOpenSpielEnvironment if available; otherwise falls back to a
  built-in pyspiel loop for battle mode.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, Optional
import random

# Prefer the extended policies (supports callables/HTTP/LLM); fall back to simple policies.
try:
    from .opponent_policies_kazumatest import get_opponent_policy, OpponentPolicy  # extended
except Exception:  # pragma: no cover
    from .opponent_policies import get_opponent_policy, OpponentPolicy  # simple

from .openspiel_environment import OpenSpielEnvironment
from ..models import OpenSpielAction

# Multi-agent wrapper is optional; if missing, battle mode will use a pyspiel fallback.
try:
    from .multi_agent_environment import MultiAgentOpenSpielEnvironment  # expected class
except Exception:  # pragma: no cover
    MultiAgentOpenSpielEnvironment = None  # type: ignore


class ModeManager:
    """
    Orchestrates training vs frozen opponent (train mode) and head-to-head evaluation (battle mode).

    Usage:
        mm = ModeManager(game_name="tic_tac_toe", agent_player=0)
        mm.run_train(learner_agent, opponent_policy="random", episodes=100)
        mm.run_battle(model_a, model_b, matches=100)
    """

    def __init__(self, game_name: str = "tic_tac_toe", agent_player: int = 0, game_params: Optional[Dict[str, Any]] = None):
        self.game_name = game_name
        self.agent_player = agent_player
        self.game_params = game_params or {}

    # -----------------------
    # TRAIN MODE (one learner trains vs frozen opponent)
    # -----------------------
    def run_train(
        self,
        learner_agent: Any,
        opponent_policy: Any,
        episodes: int = 100,
        max_steps_per_episode: int = 200,
        on_episode_end: Optional[Callable[[int, Dict[str, Any]], None]] = None,
        hot_swap: Any = None,  # kept for API compatibility; not required
    ) -> None:
        """
        Run training loop where learner_agent is being trained and opponent_policy is frozen.

        Args:
            learner_agent: object exposing select_action(observation) -> action_id and optional learn(...) hook.
            opponent_policy: string|callable|model|dict accepted by get_opponent_policy(...)
            episodes: number of episodes to run
            max_steps_per_episode: safety cap per episode
            on_episode_end: optional callback called as on_episode_end(ep_idx, metrics_dict)
            hot_swap: kept for compatibility; you can pass an object with reload() if desired.
        """
        # Wrap opponent policy using the (extended) factory (handles callables/models/dicts)
        opp_policy_obj = get_opponent_policy(opponent_policy)

        # Construct env with a dummy simple policy to satisfy constructor, then override.
        env = OpenSpielEnvironment(
            game_name=self.game_name,
            agent_player=self.agent_player,
            opponent_policy="random",  # placeholder to pass constructor validation
            game_params=self.game_params,
        )
        # Override with the richer policy object so the environment actually uses it.
        if hasattr(env, "opponent_policy_fn"):
            env.opponent_policy_fn = opp_policy_obj  # type: ignore[attr-defined]

        for ep in range(episodes):
            res = env.reset()
            obs = res.observation
            done = res.done
            total_reward = 0.0
            steps = 0

            while not done and steps < max_steps_per_episode:
                # learner must implement select_action(obs) -> action_id
                if hasattr(learner_agent, "select_action"):
                    action_id = learner_agent.select_action(obs)
                else:
                    # Best-effort: allow callable(obs) -> action or random fallback
                    try:
                        action_id = int(learner_agent(obs))
                    except Exception:
                        legal = getattr(obs, "legal_actions", None) or []
                        action_id = random.choice(legal) if legal else 0

                step_res = env.step(OpenSpielAction(action_id=action_id, game_name=self.game_name))
                next_obs = step_res.observation
                reward = step_res.reward or 0.0
                total_reward += reward
                done = step_res.done

                # optional learn hook
                if hasattr(learner_agent, "learn"):
                    try:
                        learner_agent.learn(obs, action_id, reward, next_obs, done)
                    except TypeError:
                        # tolerate different signatures
                        try:
                            learner_agent.learn(obs, action_id, reward)
                        except Exception:
                            pass

                obs = next_obs
                steps += 1

            metrics = {"episode": ep, "steps": steps, "total_reward": total_reward}
            if on_episode_end:
                on_episode_end(ep, metrics)

            # Optional external hot-swap hook
            if hot_swap is not None and hasattr(hot_swap, "reload"):
                try:
                    hot_swap.reload()
                except Exception:
                    pass

        # Best-effort close
        try:
            env.close()
        except Exception:
            pass

    # -----------------------
    # BATTLE MODE (both models frozen; evaluate matches)
    # -----------------------
    def run_battle(
        self,
        model_a: Any,
        model_b: Any,
        matches: int = 100,
        max_steps_per_match: int = 500,
        report_every: int = 10,
    ) -> Dict[str, Any]:
        """
        Run head-to-head matches between two frozen models and return aggregated stats.

        Args:
            model_a, model_b: objects implementing select_action(legal_actions, observations) -> action_id
                              or predict(observations) -> action id/logits; strings/callables accepted.
            matches: number of matches to run
            max_steps_per_match: safety cap per match
            report_every: interval for printing progress

        Returns:
            dict with: wins_a, wins_b, draws, avg_length, matches
        """
        # Try the multi-agent wrapper if available
        if MultiAgentOpenSpielEnvironment is not None:
            env = MultiAgentOpenSpielEnvironment(game_name=self.game_name, game_params=self.game_params)
            num_players = getattr(env, "num_players", 2)

            def _act(model, legal_actions, obs):
                if hasattr(model, "select_action"):
                    return model.select_action(legal_actions, obs)
                if hasattr(model, "predict"):
                    out = model.predict(obs)
                    if isinstance(out, int):
                        return out
                    if isinstance(out, dict):
                        return max(legal_actions, key=lambda a: out.get(a, float("-inf")))
                    try:
                        seq = list(out)
                        return max(legal_actions, key=lambda a: seq[a] if a < len(seq) else float("-inf"))
                    except Exception:
                        pass
                if callable(model):
                    try:
                        out = model(legal_actions, obs)
                        return int(out) if out in legal_actions else random.choice(list(legal_actions))
                    except Exception:
                        pass
                return random.choice(list(legal_actions))

            wins_a = wins_b = draws = 0
            lengths = []

            for m in range(matches):
                obs_dict = env.reset_multi()
                done = any(o["done"] for o in obs_dict.values())
                steps = 0
                while not done and steps < max_steps_per_match:
                    cur = obs_dict[0]["current_player"]
                    legal = obs_dict[cur]["legal_actions"]
                    if cur == 0:
                        action = _act(model_a, legal, obs_dict[cur])
                    else:
                        action = _act(model_b, legal, obs_dict[cur])
                    obs_dict = env.step_one(cur, int(action))
                    steps += 1
                    done = any(o["done"] for o in obs_dict.values())

                r0 = obs_dict[0].get("reward", 0.0) or 0.0
                r1 = obs_dict[1].get("reward", 0.0) or 0.0
                if r0 > r1:
                    wins_a += 1
                    result = "A"
                elif r1 > r0:
                    wins_b += 1
                    result = "B"
                else:
                    draws += 1
                    result = "D"
                lengths.append(steps)
                if report_every and (m + 1) % report_every == 0:
                    print(f"[battle] match {m+1}/{matches}: {result} (len={steps})")

            return {
                "wins_a": wins_a,
                "wins_b": wins_b,
                "draws": draws,
                "avg_length": (sum(lengths) / len(lengths)) if lengths else 0.0,
                "matches": matches,
            }

        # -----------------------
        # Fallback: direct pyspiel loop (no MultiAgentOpenSpielEnvironment available)
        # -----------------------
        import numpy as np  # local import
        import pyspiel

        game = pyspiel.load_game(self.game_name)

        def _legal_actions(state, player=None):
            try:
                return list(state.legal_actions(player)) if player is not None else list(state.legal_actions())
            except Exception:
                return list(state.legal_actions())

        def _obs_for_player(state, player: int):
            # minimal observation dict similar to MultiAgentOpenSpielEnvironment
            try:
                info = state.information_state_tensor(player)
            except Exception:
                try:
                    info = state.observation_tensor(player)
                except Exception:
                    info = None
            return {
                "current_player": state.current_player() if not state.is_terminal() else -1,
                "legal_actions": _legal_actions(state, player),
                "info_state": list(info) if info is not None else None,
                "reward": None,
                "done": state.is_terminal(),
            }

        def _act(model, legal_actions, obs):
            if hasattr(model, "select_action"):
                return model.select_action(legal_actions, obs)
            if hasattr(model, "predict"):
                out = model.predict(obs)
                if isinstance(out, int):
                    return out
                if isinstance(out, dict):
                    return max(legal_actions, key=lambda a: out.get(a, float("-inf")))
                try:
                    seq = list(out)
                    return max(legal_actions, key=lambda a: seq[a] if a < len(seq) else float("-inf"))
                except Exception:
                    pass
            if callable(model):
                try:
                    out = model(legal_actions, obs)
                    return int(out) if out in legal_actions else random.choice(list(legal_actions))
                except Exception:
                    pass
            return random.choice(list(legal_actions))

        def _sample_chance(state):
            while state.is_chance_node():
                outcomes = state.chance_outcomes()
                actions, probs = zip(*outcomes)
                probs = np.asarray(probs, dtype=float)
                probs = probs / probs.sum()
                idx = int(np.random.choice(len(actions), p=probs))
                state.apply_action(int(actions[idx]))

        wins_a = wins_b = draws = 0
        lengths = []
        for m in range(matches):
            state = game.new_initial_state()
            steps = 0
            _sample_chance(state)
            while not state.is_terminal() and steps < max_steps_per_match:
                if state.is_simultaneous_node():
                    joint = []
                    for p in range(game.num_players()):
                        obs = _obs_for_player(state, p)
                        legal = _legal_actions(state, p)
                        act = _act(model_a if p == 0 else model_b, legal, obs)
                        if act not in legal and legal:
                            act = legal[0]
                        joint.append(int(act))
                    try:
                        state.apply_actions(joint)
                    except AttributeError:
                        for a in joint:
                            state.apply_action(a)
                    _sample_chance(state)
                    steps += 1
                    continue

                cur = state.current_player()
                obs = _obs_for_player(state, cur)
                legal = _legal_actions(state, cur)
                act = _act(model_a if cur == 0 else model_b, legal, obs)
                if act not in legal and legal:
                    act = legal[0]
                state.apply_action(int(act))
                _sample_chance(state)
                steps += 1

            returns = list(state.returns())
            r0 = returns[0] if returns else 0.0
            r1 = returns[1] if returns else 0.0
            if r0 > r1:
                wins_a += 1
                result = "A"
            elif r1 > r0:
                wins_b += 1
                result = "B"
            else:
                draws += 1
                result = "D"
            lengths.append(steps)
            if report_every and (m + 1) % report_every == 0:
                print(f"[battle:fallback] match {m+1}/{matches}: {result} (len={steps})")

        return {
            "wins_a": wins_a,
            "wins_b": wins_b,
            "draws": draws,
            "avg_length": (sum(lengths) / len(lengths)) if lengths else 0.0,
            "matches": matches,
        }