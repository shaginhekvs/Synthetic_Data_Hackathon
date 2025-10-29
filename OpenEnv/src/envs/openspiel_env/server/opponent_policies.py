"""
Opponent policies used by the OpenSpiel OpenEnv wrapper.

This file is additive: it preserves the existing simple policies (random/first/last)
and adds ExternalOpponentPolicy so opponents can be arbitrary callables,
in-process models, or remote APIs (e.g., an LLM HTTP endpoint).

OpenSpielEnvironment already calls get_opponent_policy(opponent_policy) so you
can pass a callable/model into its constructor:
  env = OpenSpielEnvironment(game_name="tic_tac_toe", opponent_policy=my_callable)
"""
from __future__ import annotations

import random
import logging
import concurrent.futures
from typing import Any, Callable, Dict, Iterable, Optional, Sequence, Union

import requests  # used by example wrappers; keep imported for convenience

logger = logging.getLogger(__name__)


class OpponentPolicy:
    """Base interface for opponent policies used by OpenSpielEnvironment."""

    def select_action(self, legal_actions: Sequence[int], observations: Dict[str, Any]) -> int:
        raise NotImplementedError()


class RandomPolicy(OpponentPolicy):
    def select_action(self, legal_actions: Sequence[int], observations: Dict[str, Any]) -> int:
        if not legal_actions:
            raise RuntimeError("RandomPolicy: no legal actions")
        return random.choice(list(legal_actions))


class FirstPolicy(OpponentPolicy):
    def select_action(self, legal_actions: Sequence[int], observations: Dict[str, Any]) -> int:
        if not legal_actions:
            raise RuntimeError("FirstPolicy: no legal actions")
        return list(legal_actions)[0]


class LastPolicy(OpponentPolicy):
    def select_action(self, legal_actions: Sequence[int], observations: Dict[str, Any]) -> int:
        if not legal_actions:
            raise RuntimeError("LastPolicy: no legal actions")
        return list(legal_actions)[-1]


def _call_with_timeout(fn: Callable[..., Any], args: tuple = (), kwargs: Optional[dict] = None, timeout: float = 0.5):
    """Run fn(*args, **kwargs) with timeout using ThreadPoolExecutor and return result or raise."""
    kwargs = kwargs or {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
        fut = ex.submit(fn, *args, **kwargs)
        return fut.result(timeout=timeout)


class ExternalOpponentPolicy(OpponentPolicy):
    """
    Wrap a callable or model object so it behaves like an OpponentPolicy.

    Args:
        policy: Callable or model object. Supported shapes:
            - callable(legal_actions, observations) -> action_id
            - object.select_action(legal_actions, observations) -> action_id
            - object.predict(observations) -> action_id or logits/probs/text
        timeout_s: seconds to wait for the policy to respond. On timeout we fallback.
        fallback: "random" | "first" | "last" | callable (callable must accept legal_actions)
        postprocess: optional function(post_output, legal_actions, observations) -> int to map model outputs to action_id
        max_retries: number of retries on transient errors (exceptions) before fallback
    """

    def __init__(
        self,
        policy: Any,
        timeout_s: float = 0.5,
        fallback: Union[str, Callable[[Iterable[int]], int]] = "random",
        postprocess: Optional[Callable[[Any, Sequence[int], Dict[str, Any]], int]] = None,
        max_retries: int = 1,
    ):
        if not (callable(policy) or hasattr(policy, "select_action") or hasattr(policy, "predict")):
            raise ValueError("policy must be a callable or have select_action/predict")
        self._policy = policy
        self.timeout_s = float(timeout_s)
        self._postprocess = postprocess
        self.max_retries = int(max_retries)

        # Configure fallback
        if isinstance(fallback, str):
            s = fallback.lower()
            if s == "random":
                self._fallback_fn = lambda legal: int(random.choice(list(legal)))
            elif s in ("first", "fixed_first"):
                self._fallback_fn = lambda legal: int(list(legal)[0])
            elif s in ("last", "fixed_last"):
                self._fallback_fn = lambda legal: int(list(legal)[-1])
            else:
                raise ValueError(f"Unknown fallback string: {fallback}")
        elif callable(fallback):
            self._fallback_fn = fallback
        else:
            raise ValueError("fallback must be string or callable")

    def _invoke_policy(self, legal_actions: Sequence[int], observations: Dict[str, Any]) -> Any:
        """Call underlying policy using supported interfaces."""
        # Prefer object.select_action
        if hasattr(self._policy, "select_action") and callable(getattr(self._policy, "select_action")):
            return self._policy.select_action(legal_actions, observations)
        # Then object.predict(observations)
        if hasattr(self._policy, "predict") and callable(getattr(self._policy, "predict")):
            return self._policy.predict(observations)
        # Finally, assume it's a callable with signature (legal_actions, observations)
        return self._policy(legal_actions, observations)

    def select_action(self, legal_actions: Sequence[int], observations: Dict[str, Any]) -> int:
        """Return a legal action id, with timeout/retry/fallback and validation."""
        if not legal_actions:
            # nothing to pick
            raise RuntimeError("ExternalOpponentPolicy: no legal actions to select from")

        last_exc = None
        for attempt in range(self.max_retries):
            try:
                out = _call_with_timeout(self._invoke_policy, args=(legal_actions, observations), timeout=self.timeout_s)
                # If postprocess provided, use it
                if self._postprocess is not None:
                    try:
                        action = int(self._postprocess(out, legal_actions, observations))
                    except Exception as e:
                        logger.exception("postprocess failed: %s", e)
                        action = None
                else:
                    action = self._interpret_output(out, legal_actions, observations)
                # Validate
                if action in legal_actions:
                    return int(action)
                else:
                    logger.warning("Opponent returned invalid action %r (legal=%r). Falling back.", action, legal_actions)
                    return int(self._fallback_fn(legal_actions))
            except concurrent.futures.TimeoutError:
                last_exc = TimeoutError(f"ExternalOpponentPolicy timed out after {self.timeout_s}s")
                logger.warning("%s", last_exc)
            except Exception as e:
                last_exc = e
                logger.exception("ExternalOpponentPolicy error on attempt %d: %s", attempt + 1, e)

        # If here: retries exhausted -> fallback
        logger.warning("ExternalOpponentPolicy falling back after error: %s", last_exc)
        return int(self._fallback_fn(legal_actions))

    def _interpret_output(self, out: Any, legal_actions: Sequence[int], observations: Dict[str, Any]) -> Optional[int]:
        """Try to map a model output to an action id."""
        # If model returns an int
        if isinstance(out, int):
            return out
        # If model returns a dict mapping action->score
        if isinstance(out, dict):
            try:
                best = max(legal_actions, key=lambda a: out.get(a, float("-inf")))
                return int(best)
            except Exception:
                pass
        # If model returns list/tuple of scores indexed by action id
        if isinstance(out, (list, tuple)):
            try:
                best = max(legal_actions, key=lambda a: out[a] if 0 <= a < len(out) else float("-inf"))
                return int(best)
            except Exception:
                pass
        # If model returns string (e.g., from LLM): try to parse int
        if isinstance(out, str):
            try:
                candidate = int(out.strip())
                return candidate
            except Exception:
                # try to find digits
                import re
                m = re.search(r"\d+", out)
                if m:
                    try:
                        return int(m.group(0))
                    except Exception:
                        pass
        # Unknown output: fallback will handle
        return None


def get_opponent_policy(opponent_policy: Any) -> OpponentPolicy:
    """
    Factory that returns an OpponentPolicy.

    Accepts:
      - OpponentPolicy instance -> returned as-is
      - string: "random", "first", "last"
      - callable: wrapped in ExternalOpponentPolicy
      - dict config: {"type":"external", "policy":callable_or_model, "timeout_s":..., ...}
    """
    if isinstance(opponent_policy, OpponentPolicy):
        return opponent_policy

    if isinstance(opponent_policy, dict):
        t = opponent_policy.get("type", "").lower()
        if t in ("external", "callable", ""):
            policy_obj = opponent_policy.get("policy")
            timeout_s = opponent_policy.get("timeout_s", 0.5)
            fallback = opponent_policy.get("fallback", "random")
            post = opponent_policy.get("postprocess", None)
            retries = opponent_policy.get("max_retries", 1)
            return ExternalOpponentPolicy(policy_obj, timeout_s=timeout_s, fallback=fallback, postprocess=post, max_retries=retries)
        # allow specifying simple string via dict
        opponent_policy = opponent_policy.get("name", opponent_policy)

    if isinstance(opponent_policy, str):
        s = opponent_policy.lower()
        if s == "random":
            return RandomPolicy()
        if s in ("first", "fixed_first"):
            return FirstPolicy()
        if s in ("last", "fixed_last"):
            return LastPolicy()
        raise ValueError(f"Unknown opponent policy string: {opponent_policy}")

    if callable(opponent_policy):
        return ExternalOpponentPolicy(opponent_policy)

    raise ValueError("Unsupported opponent_policy type. Must be string, callable, OpponentPolicy or dict.")