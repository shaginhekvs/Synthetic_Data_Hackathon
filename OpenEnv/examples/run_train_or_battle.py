#!/usr/bin/env python3
"""
Example runner that demonstrates both Train and Battle modes.

Usage:
    python examples/run_train_or_battle.py --mode train
    python examples/run_train_or_battle.py --mode battle

This example uses dummy models; replace them with your real learner and frozen models.
"""
import argparse
import random
import time
from pathlib import Path
import sys
# make sure repo src is importable (if running from repo root)
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from envs.openspiel_env.server.mode_manager import ModeManager
from envs.openspiel_env.server.opponent_model_loader import make_pytorch_opponent, HotSwapOpponent
from envs.openspiel_env.server.opponent_policies_kazumatest import get_opponent_policy
from envs.openspiel_env.server.opponent_model_loader import PyTorchOpponentAdapter

# Dummy learner for train mode
class DummyLearner:
    def select_action(self, obs):
        # obs.legal_actions is expected as list
        return random.choice(obs.legal_actions)

    def learn(self, obs, action, reward, next_obs, done):
        # placeholder: pretend to update
        pass

# Dummy frozen model for battle mode (implements select_action)
class DummyFrozenModel:
    def __init__(self, player_id=0):
        self.player_id = player_id

    def select_action(self, legal_actions, observations):
        # naive: choose last legal action deterministically
        return legal_actions[-1]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "battle"], default="train")
    parser.add_argument("--game", default="tic_tac_toe")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--matches", type=int, default=100)
    args = parser.parse_args()

    mm = ModeManager(game_name=args.game)

    if args.mode == "train":
        print("Running in TRAIN mode: learner trains vs frozen opponent")
        learner = DummyLearner()

        # Example: load frozen opponent model from checkpoint via adapter (point to a real file)
        # For demo we'll use a DummyFrozenModel and wrap via get_opponent_policy
        frozen = DummyFrozenModel(player_id=1)
        opp_policy = get_opponent_policy(frozen)

        def on_ep_end(ep, metrics):
            print(f"[train] ep={ep} steps={metrics['steps']} reward={metrics['total_reward']:.2f}")

        mm.run_train(
            learner_agent=learner,
            opponent_policy=opp_policy,
            episodes=args.episodes,
            on_episode_end=on_ep_end,
            hot_swap=None,  # optionally pass HotSwapOpponent instance
        )

    else:
        print("Running in BATTLE mode: evaluate two frozen models")
        model_a = DummyFrozenModel(player_id=0)
        model_b = DummyFrozenModel(player_id=1)
        stats = mm.run_battle(model_a, model_b, matches=args.matches, report_every=20)
        print("Battle results:", stats)

if __name__ == "__main__":
    main()