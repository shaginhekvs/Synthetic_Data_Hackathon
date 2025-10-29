"""
MultiAgentOpenSpielEnvironment: a minimal multi-agent runner around PySpiel.

Exposes:
- reset_multi() -> per-player observation dict
- step_one(player_id, action_id) -> per-player observation dict

Observation dict per player id includes:
  {
    "current_player": int,
    "legal_actions": List[int],
    "info_state": List[float] | List[int] | None,
    "reward": float | None,      # terminal only
    "done": bool,
  }
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
import numpy as np

try:
    import pyspiel
except ImportError as e:
    raise ImportError(
        "OpenSpiel is not installed. Install from https://github.com/google-deepmind/open_spiel"
    ) from e


class MultiAgentOpenSpielEnvironment:
    def __init__(self, game_name: str = "tic_tac_toe", game_params: Optional[Dict[str, Any]] = None):
        self.game_name = game_name
        self.game_params = game_params or {}
        # Load game (game_params support varies across games; safest is to load by name)
        self.game = pyspiel.load_game(self.game_name)
        self.num_players = self.game.num_players()
        self.state: pyspiel.State = self.game.new_initial_state()

    # ------------- helpers -------------
    def _sample_chance(self, state: pyspiel.State) -> None:
        """Resolve chance nodes by sampling according to outcome distribution."""
        while state.is_chance_node():
            outcomes = state.chance_outcomes()  # List[(action, prob)]
            if not outcomes:
                raise RuntimeError("Chance node with no outcomes")
            actions, probs = zip(*outcomes)
            probs = np.asarray(probs, dtype=float)
            probs = probs / probs.sum()
            idx = int(np.random.choice(len(actions), p=probs))
            state.apply_action(int(actions[idx]))

    def _legal_actions(self, state: pyspiel.State, player: int) -> List[int]:
        try:
            return list(state.legal_actions(player))
        except Exception:
            # Some games allow querying only for current_player
            try:
                return list(state.legal_actions())
            except Exception:
                return []

    def _info_state(self, state: pyspiel.State, player: int):
        # Prefer information_state_tensor; fall back to observation_tensor/string
        try:
            return list(state.information_state_tensor(player))
        except Exception:
            pass
        try:
            return list(state.observation_tensor(player))
        except Exception:
            pass
        try:
            return state.information_state_string(player)
        except Exception:
            pass
        try:
            return state.observation_string(player)
        except Exception:
            pass
        return None

    def _obs_dict(self, state: pyspiel.State) -> Dict[int, Dict[str, Any]]:
        done = state.is_terminal()
        current_player = -1 if done else state.current_player()
        rewards = list(state.returns()) if done else [0.0] * self.num_players
        obs: Dict[int, Dict[str, Any]] = {}
        for p in range(self.num_players):
            obs[p] = {
                "current_player": current_player,
                "legal_actions": self._legal_actions(state, p),
                "info_state": self._info_state(state, p),
                "reward": rewards[p] if done else None,
                "done": done,
            }
        return obs

    # ------------- public API -------------
    def reset_multi(self) -> Dict[int, Dict[str, Any]]:
        self.state = self.game.new_initial_state()
        # Resolve any initial chance nodes
        self._sample_chance(self.state)
        return self._obs_dict(self.state)

    def step_one(self, player_id: int, action_id: int) -> Dict[int, Dict[str, Any]]:
        # If terminal, just echo observation
        if self.state.is_terminal():
            return self._obs_dict(self.state)

        # If simultaneous node, require all players' actions — simple fallback:
        if self.state.is_simultaneous_node():
            # For simplicity, we treat this as applying only the provided player's action and
            # choosing "noop/first legal" for others. You can extend this to accept a joint action.
            joint = [0] * self.num_players
            for p in range(self.num_players):
                if p == player_id:
                    joint[p] = int(action_id)
                else:
                    legal = self._legal_actions(self.state, p)
                    joint[p] = int(legal[0]) if legal else 0
            try:
                self.state.apply_actions(joint)
            except AttributeError:
                for a in joint:
                    self.state.apply_action(int(a))
        else:
            # Turn-based: apply the given player's action
            self.state.apply_action(int(action_id))

        # Resolve any chance nodes that appear after the action
        self._sample_chance(self.state)
        return self._obs_dict(self.state)

    def close(self) -> None:
        # Nothing specific to close for pyspiel, but keep for API symmetry
        pass