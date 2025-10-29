# A simple REINFORCE-based self-play trainer for OpenSpiel 2048 via the existing HTTP server.
# - Same policy (same model instance) is used for all players (self-play).
# - Works for single-player (2048) and is easy to adapt to multi-player: we prompt
#   with current_player_id and let the very same policy pick for each player turn.
# - Uses a constrained action head by filtering logits to tokens {"0","1","2","3"}.
#
# Notes:
# - This is a minimal, readable trainer intended to get you started. For production,
#   you may want to add PPO, KL control, advantage normalization, batching, etc.

from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any
import math
import itertools
import time
import torch
from torch.optim import AdamW
from transformers import LogitsProcessor, LogitsProcessorList
from envs.openspiel_env import OpenSpielEnv
from envs.openspiel_env.models import OpenSpielAction

DIGIT_ACTIONS = ["0", "1", "2", "3"]

class RestrictToActionDigits(LogitsProcessor):
    def __init__(self, allowed_token_ids: List[int]):
        self.allowed = set(allowed_token_ids)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        # Mask out all tokens except the allowed ones
        mask = torch.full_like(scores, float("-inf"))
        # scores shape: (batch, vocab)
        for b in range(scores.size(0)):
            mask[b, list(self.allowed)] = 0.0
        return scores + mask

@dataclass
class TrainConfig:
    base_url: str = "http://localhost:8000"  # OpenEnv server URL
    game_name: str = "2048"
    episodes: int = 200
    max_steps_per_episode: int = 200
    learning_rate: float = 1e-5
    gamma: float = 0.99
    entropy_coef: float = 0.0  # Add small entropy bonus if desired
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    # prompt formatting
    system_prefix: str = (
        "You are a 2048 policy. Given a board as a 2D list of numbers, output only one token among "
        '"0","1","2","3" corresponding to up(0), right(1), down(2), left(3). No explanations.'
    )

def board_from_obs(obs) -> List[List[int]]:
    import numpy as np
    n = len(obs.info_state)
    size = int(math.sqrt(n))
    arr = np.array(obs.info_state, dtype=int)
    return [arr[i*size:(i+1)*size].tolist() for i in range(size)]

def build_prompt(board: List[List[int]], current_player_id: Optional[int] = None, system_prefix: Optional[str] = None) -> str:
    # Minimal prompt; include player id if multi-player
    sp = system_prefix or ""
    pi = f"\ncurrent_player_id: {current_player_id}" if current_player_id is not None else ""
    return f"{sp}\nboard: {board}{pi}\naction:"

def map_actions_to_token_ids(tokenizer) -> Dict[str, int]:
    # Ensure that the exact single-character tokens "0","1","2","3" exist distinctly
    ids = {}
    for a in DIGIT_ACTIONS:
        tok = tokenizer.encode(a, add_special_tokens=False)
        # Prefer single-token encoding for speed and simplicity
        if len(tok) == 1:
            ids[a] = tok[0]
        else:
            # Fallback: ensure we can still match the first token (rare)
            ids[a] = tok[0]
    return ids

def sample_action_and_logprob(
    model, tokenizer, prompt: str, action_token_ids: Dict[str, int], device: str
) -> Tuple[int, float]:
    model.eval()
    with torch.no_grad():
        enc = tokenizer(prompt, return_tensors="pt").to(device)
        outputs = model(**enc)
        logits = outputs.logits[:, -1, :]  # last token
        # Restrict to action tokens
        allowed = list(action_token_ids.values())
        mask = torch.full_like(logits, float("-inf"))
        mask[0, allowed] = 0.0
        masked = logits + mask
        probs = torch.softmax(masked, dim=-1)
        # sample
        act_id = torch.multinomial(probs, num_samples=1).item()
        # if sampled outside allowed due to numeric issues, pick argmax allowed
        if act_id not in allowed:
            act_id = allowed[torch.argmax(probs[0, allowed]).item()]
        logp = torch.log(torch.clamp(probs[0, act_id], min=1e-12)).item()
        # Map back to 0..3
        inv = {v: int(k) for k, v in action_token_ids.items()}
        action = inv.get(act_id, 0)
        return action, logp

def supervised_logprob(
    model, tokenizer, prompt: str, chosen_digit: int, action_token_ids: Dict[str, int], device: str
) -> torch.Tensor:
    # Compute log-prob of a specific chosen action under current model (for REINFORCE gradient)
    enc = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = model(**enc)
    logits = outputs.logits[:, -1, :]  # last token
    allowed = list(action_token_ids.values())
    mask = torch.full_like(logits, float("-inf"))
    mask[0, allowed] = 0.0
    masked = logits + mask
    log_probs = torch.log_softmax(masked, dim=-1)
    act_token_id = action_token_ids[str(chosen_digit)]
    return log_probs[0, act_token_id]

def reinforce_returns(rewards: List[float], gamma: float) -> List[float]:
    g = 0.0
    ret = []
    for r in reversed(rewards):
        g = r + gamma * g
        ret.append(g)
    return list(reversed(ret))

def train_selfplay_2048(
    model,
    tokenizer,
    cfg: TrainConfig,
    optimizer: Optional[torch.optim.Optimizer] = None,
    lora_only: bool = True,
):
    """
    Self-play REINFORCE fine-tuning on 2048 via OpenEnv server.
    - Uses the SAME model instance for all players (self-play).
    - One-step action token: "0","1","2","3".
    - Performs on-policy updates after every episode.

    Tip: Start with LoRA enabled (in loader) and lora_only=True so you only train adapters.
    """
    device = cfg.device
    model.to(device)

    if optimizer is None:
        # Only optimize LoRA (or all params if no LoRA present)
        if lora_only:
            params = [p for n, p in model.named_parameters() if p.requires_grad and "lora_" in n]
            if not params:
                params = [p for p in model.parameters() if p.requires_grad]
        else:
            params = [p for p in model.parameters() if p.requires_grad]
        optimizer = AdamW(params, lr=cfg.learning_rate)

    # Prepare env client
    env = OpenSpielEnv(base_url=cfg.base_url, request_timeout_s=300.0)

    # Precompute action token ids
    action_token_ids = map_actions_to_token_ids(tokenizer)

    ema_baseline = 0.0
    ema_beta = 0.9

    for ep in range(1, cfg.episodes + 1):
        # Rollout
        rollout_prompts: List[str] = []
        rollout_actions: List[int] = []
        rollout_logps: List[float] = []
        rollout_rewards: List[float] = []

        reset = env.reset(game_name=cfg.game_name)
        obs = reset.observation
        total_reward = 0.0
        steps = 0

        while not obs.done and steps < cfg.max_steps_per_episode:
            board = board_from_obs(obs)
            prompt = build_prompt(board, getattr(obs, "current_player_id", None), cfg.system_prefix)
            action, logp = sample_action_and_logprob(model, tokenizer, prompt, action_token_ids, device)

            rollout_prompts.append(prompt)
            rollout_actions.append(action)
            rollout_logps.append(logp)

            result = env.step(OpenSpielAction(action_id=action, game_name=cfg.game_name))
            obs = result.observation
            r = result.reward or 0.0
            total_reward += r
            rollout_rewards.append(r)
            steps += 1

        # Compute returns and advantage (with EMA baseline)
        returns = reinforce_returns(rollout_rewards, cfg.gamma)
        mean_return = float(sum(returns) / max(1, len(returns)))
        ema_baseline = ema_beta * ema_baseline + (1 - ema_beta) * mean_return

        # Optimize policy: L = -sum (logpi * (G - b)) - entropy coef * H
        model.train()
        loss_terms = []
        ent_terms = []

        for prompt, act, Gt in zip(rollout_prompts, rollout_actions, returns):
            logp_tensor = supervised_logprob(model, tokenizer, prompt, act, action_token_ids, device)
            advantage = Gt - ema_baseline
            loss_terms.append(-logp_tensor * advantage)

            # Optional entropy on masked vocabulary
            if cfg.entropy_coef > 0.0:
                enc = tokenizer(prompt, return_tensors="pt").to(device)
                out = model(**enc)
                logits = out.logits[:, -1, :]
                allowed = list(action_token_ids.values())
                mask = torch.full_like(logits, float("-inf"))
                mask[0, allowed] = 0.0
                masked = logits + mask
                log_probs = torch.log_softmax(masked, dim=-1)
                probs = torch.softmax(masked, dim=-1)
                entropy = -(probs * log_probs).sum()
                ent_terms.append(-cfg.entropy_coef * entropy)

        loss = torch.stack(loss_terms).mean()
        if ent_terms:
            loss = loss + torch.stack(ent_terms).mean()

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        print(f"[Episode {ep}/{cfg.episodes}] steps={steps} return={total_reward:.2f} loss={loss.item():.4f} baseline={ema_baseline:.2f}")

    print("Training complete.")