#!/usr/bin/env python3
"""
OpenSpiel Battle CLI

Pit two frozen policies against each other using ModeManager.run_battle.

Supported opponent types per side (--a-policy / --b-policy):
- random | first | last
- hf          (Hugging Face Inference API; requires --*-model-id and HF token)
- http        (Generic HTTP JSON endpoint; requires --*-url)
- pycall      (Python callable or class via import path: module:attr; returns int or scores)

Examples:
  # Random vs Random (Tic-Tac-Toe), 100 matches
  python examples/openspiel_battle_cli.py --game tic_tac_toe --matches 100 --a-policy random --b-policy random

  # HF LLM vs Random (requires HF_TOKEN env)
  python examples/openspiel_battle_cli.py --game tic_tac_toe --matches 50 \
    --a-policy hf --a-model-id unsloth/Llama-3.2-3B --b-policy random

  # Generic HTTP endpoint vs first
  python examples/openspiel_battle_cli.py --game tic_tac_toe --matches 50 \
    --a-policy http --a-url http://localhost:8001/llm-move --b-policy first

  # Python callable vs Python callable
  # where mypkg.mypolicies has def greedy_ttt(legal, obs) -> int
  python examples/openspiel_battle_cli.py --game tic_tac_toe --matches 200 --swap-sides \
    --a-policy pycall --a-target mypkg.mypolicies:greedy_ttt \
    --b-policy pycall --b-target mypkg.mypolicies:center_first_then_corners
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from typing import Any, Callable, Dict, Optional, Sequence

# Ensure repo root on path if running from nested locations
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# ModeManager and adapters
from envs.openspiel_env.server.mode_manager import ModeManager
from envs.openspiel_env.server.llm_opponent_adapter import (
    make_hf_http_llm_policy,
    make_generic_http_llm_policy,
)


def load_pycall(target: str) -> Any:
    """
    Load a callable or class instance from "module:attr" or "module.sub:attr".
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


def build_policy(
    which: str,
    args: argparse.Namespace,
) -> Any:
    """
    Builds an opponent policy (string or callable) acceptable by ModeManager.
    which: "a" or "b" (selects side-specific flags)
    """
    policy_type = getattr(args, f"{which}_policy")
    policy_type = (policy_type or "").strip().lower()

    if policy_type in ("random", "first", "last"):
        # Hand off simple strings; ModeManager will wrap appropriately
        return policy_type

    if policy_type == "hf":
        model_id = getattr(args, f"{which}_model_id")
        if not model_id:
            raise ValueError(f"--{which}-model-id is required when --{which}-policy hf")
        # Token: prefer CLI arg, then env HF_API_KEY/HF_TOKEN
        hf_token = args.hf_token or os.environ.get("HF_API_KEY") or os.environ.get("HF_TOKEN")
        if not hf_token:
            raise ValueError("HF token not found. Pass --hf-token or set HF_API_KEY/HF_TOKEN env.")
        return make_hf_http_llm_policy(
            model_id=model_id,
            hf_api_key=hf_token,
            timeout=args.http_timeout,
            max_retries=args.http_retries,
            game_name=args.game,
        )

    if policy_type == "http":
        url = getattr(args, f"{which}_url")
        if not url:
            raise ValueError(f"--{which}-url is required when --{which}-policy http")
        headers = {}
        if args.http_header:
            # allow multiple --http-header key:value
            for kv in args.http_header:
                if ":" not in kv:
                    raise ValueError(f"Invalid --http-header '{kv}'. Expected key:value")
                k, v = kv.split(":", 1)
                headers[k.strip()] = v.strip()
        return make_generic_http_llm_policy(
            endpoint_url=url,
            timeout=args.http_timeout,
            max_retries=args.http_retries,
            fallback="random",
            game_name=args.game,
            headers=headers or None,
        )

    if policy_type == "pycall":
        target = getattr(args, f"{which}_target")
        if not target:
            raise ValueError(f"--{which}-target is required when --{which}-policy pycall")
        obj = load_pycall(target)
        # Accept function(legal, obs) -> int OR object with select_action/predict
        return obj

    raise ValueError(f"Unknown policy type for --{which}-policy: {policy_type}")


def combine_battle_stats(stats1: Dict[str, Any], stats2: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Combine two battle stats dicts where stats2 is from a side-swapped run.
    In stats2, 'wins_a' corresponds to player 0 in that run (which was B from the original pairing).
    So totals are:
      A_wins = stats1['wins_a'] + stats2['wins_b']
      B_wins = stats1['wins_b'] + stats2['wins_a']
      draws  = stats1['draws']   + stats2['draws']
      avg_length = weighted by matches
    """
    if not stats2:
        return stats1

    total_matches = (stats1.get("matches", 0) or 0) + (stats2.get("matches", 0) or 0)
    a_wins = (stats1.get("wins_a", 0) or 0) + (stats2.get("wins_b", 0) or 0)
    b_wins = (stats1.get("wins_b", 0) or 0) + (stats2.get("wins_a", 0) or 0)
    draws = (stats1.get("draws", 0) or 0) + (stats2.get("draws", 0) or 0)

    # Weighted average for lengths
    l1 = (stats1.get("avg_length", 0.0) or 0.0) * (stats1.get("matches", 0) or 0)
    l2 = (stats2.get("avg_length", 0.0) or 0.0) * (stats2.get("matches", 0) or 0)
    avg_len = (l1 + l2) / total_matches if total_matches > 0 else 0.0

    return {
        "wins_a": a_wins,
        "wins_b": b_wins,
        "draws": draws,
        "avg_length": avg_len,
        "matches": total_matches,
    }


def main():
    p = argparse.ArgumentParser(description="OpenSpiel Battle CLI using ModeManager.run_battle")
    p.add_argument("--game", type=str, default="tic_tac_toe", help="OpenSpiel game name (e.g., tic_tac_toe, kuhn_poker)")
    p.add_argument("--matches", type=int, default=100, help="Matches for the primary (A as player0) run")
    p.add_argument("--swap-sides", action="store_true", help="Also run with sides swapped and combine stats")
    p.add_argument("--matches-swapped", type=int, default=None, help="Optional matches for swapped run (defaults to --matches)")

    # Per-side policy selection
    p.add_argument("--a-policy", type=str, required=True, help="Policy type for side A: random|first|last|hf|http|pycall")
    p.add_argument("--b-policy", type=str, required=True, help="Policy type for side B: random|first|last|hf|http|pycall")

    # HF-specific params (shared token arg; per-side model ids)
    p.add_argument("--a-model-id", type=str, help="HF model id for side A when --a-policy hf")
    p.add_argument("--b-model-id", type=str, help="HF model id for side B when --b-policy hf")
    p.add_argument("--hf-token", type=str, default=None, help="HF API token (fallback to HF_API_KEY/HF_TOKEN env)")

    # Generic HTTP endpoint params
    p.add_argument("--a-url", type=str, help="HTTP endpoint for side A when --a-policy http")
    p.add_argument("--b-url", type=str, help="HTTP endpoint for side B when --b-policy http")
    p.add_argument("--http-timeout", type=float, default=1.0, help="HTTP timeout (seconds) for hf/http policies")
    p.add_argument("--http-retries", type=int, default=1, help="HTTP retries for hf/http policies")
    p.add_argument("--http-header", action="append", help="Optional extra HTTP headers key:value (can be repeated)")

    # Python callable/class
    p.add_argument("--a-target", type=str, help="Import path for side A when --a-policy pycall (module:attr)")
    p.add_argument("--b-target", type=str, help="Import path for side B when --b-policy pycall (module:attr)")

    # Misc
    p.add_argument("--report-every", type=int, default=10, help="Report interval during runs")
    p.add_argument("--max-steps", type=int, default=500, help="Safety cap per match")
    p.add_argument("--json", action="store_true", help="Emit JSON summary only")

    args = p.parse_args()

    pol_a = build_policy("a", args)
    pol_b = build_policy("b", args)

    # Run battles
    mm = ModeManager(game_name=args.game, agent_player=0)

    print(f"[battle] Game={args.game} matches={args.matches} A(player0) vs B(player1)")
    s1 = mm.run_battle(pol_a, pol_b, matches=args.matches, max_steps_per_match=args.max_steps, report_every=args.report_every)

    s2 = None
    if args.swap_sides:
        m2 = args.matches if args.matches_swapped is None else args.matches_swapped
        print(f"[battle] Swapped sides run: B(player0) vs A(player1), matches={m2}")
        s2 = mm.run_battle(pol_b, pol_a, matches=m2, max_steps_per_match=args.max_steps, report_every=args.report_every)

    combined = combine_battle_stats(s1, s2)

    if args.json:
        print(json.dumps({"primary": s1, "swapped": s2, "combined": combined}, indent=2))
        return

    print("\nPrimary run (A as player0):")
    print(json.dumps(s1, indent=2))
    if s2 is not None:
        print("\nSwapped run (B as player0):")
        print(json.dumps(s2, indent=2))
        print("\nCombined totals (side-balanced):")
        print(json.dumps(combined, indent=2))


if __name__ == "__main__":
    main()
