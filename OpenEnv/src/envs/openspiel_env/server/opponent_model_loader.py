"""
Unified opponent model loader.

Replaces ad-hoc loading for opponent 1 / opponent 2 by centralizing logic so both
opponents are loaded the same way. Supports optional PEFT/LoRA attachment,
unique adapter names / checkpoints per opponent, and safe device placement.

Usage:
    from opponent_model_loader import load_opponent_model

    m1, t1 = load_opponent_model(
        "opponent1",
        model_id="???",  # for a quick local test; use your real model_id in production
        peft_kwargs={"r": 8, "target_modules": ["q_proj", "v_proj"], "lora_alpha": 16},
        peft_adapter_name="opponent1_adapter",
        load_in_4bit=False,  # set True only if your environment supports bitsandbytes 4-bit
    )

    m2, t2 = load_opponent_model("opponent2", model_id="???", ...)
"""
from typing import Dict, Optional, Tuple
import torch

try:
    from unsloth import FastLanguageModel
except Exception:
    # If your project uses a different wrapper, replace the line above with the correct import.
    raise

from transformers import AutoTokenizer


def _ensure_tokenizer_padding(tokenizer):
    if tokenizer.pad_token_id is None:
        # Add a pad token if missing to avoid generation/training issues
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    return tokenizer


def load_model_and_tokenizer(
    model_id: str,
    *,
    device: Optional[str] = None,
    load_in_4bit: bool = True,
    max_seq_length: int = 768,
    peft_kwargs: Optional[Dict] = None,
    peft_adapter_name: Optional[str] = None,
    peft_checkpoint: Optional[str] = None,
    dtype: Optional[torch.dtype] = None,
) -> Tuple[FastLanguageModel, AutoTokenizer]:
    """
    Load a base model and tokenizer, then optionally attach a PEFT (LoRA) adapter.

    - model_id: model repo/id
    - device: "cuda" or "cpu" (None -> auto choose)
    - load_in_4bit: whether to use 4-bit loading (disabled on ROCm if unsupported)
    - peft_kwargs: dict passed to get_peft_model (e.g. {'r': 8, 'target_modules': [...]})
    - peft_adapter_name: explicit name for the adapter (helps avoid name collisions)
    - peft_checkpoint: path/identifier to a saved adapter to load (optional)
    - dtype: torch dtype for model creation (optional)
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    tokenizer = _ensure_tokenizer_padding(tokenizer)

    # Load the base model (FastLanguageModel wrapper expected in your project)
    model = FastLanguageModel.from_pretrained(
        model_id,
        load_in_4bit=load_in_4bit,
        max_seq_length=max_seq_length,
        torch_dtype=dtype,
    )

    # Attach PEFT/LoRA if requested
    if peft_kwargs:
        # If the PEFT API supports named adapters, pass adapter name to avoid collisions.
        if peft_adapter_name:
            peft_kwargs = dict(peft_kwargs)
            peft_kwargs.setdefault("adapter_name", peft_adapter_name)

        model = FastLanguageModel.get_peft_model(model, **peft_kwargs)

        # If a checkpoint for the adapter is provided, try to load it.
        if peft_checkpoint:
            # Try multiple load methods depending on the PEFT API surface
            try:
                # Preferred: the wrapper's adapter load API (example)
                model.load_adapter(peft_checkpoint, adapter_name=peft_adapter_name)
            except Exception:
                try:
                    # Fallback: load state dict (if checkpoint is a raw .pt)
                    sd = torch.load(peft_checkpoint, map_location="cpu")
                    model.load_state_dict(sd, strict=False)
                except Exception:
                    # Let the caller see the failure; don't silently swallow.
                    raise

    # Safe device placement: some wrappers manage placement internally; ignore failures.
    try:
        model.to(device)
    except Exception:
        pass

    return model, tokenizer


def load_opponent_model(
    opponent_name: str,
    model_id: str,
    *,
    device: Optional[str] = None,
    load_in_4bit: bool = True,
    max_seq_length: int = 768,
    peft_kwargs: Optional[Dict] = None,
    peft_adapter_name: Optional[str] = None,
    peft_checkpoint: Optional[str] = None,
    dtype: Optional[torch.dtype] = None,
) -> Tuple[FastLanguageModel, AutoTokenizer]:
    """
    Convenience wrapper to load an opponent's model with a clear name.
    Ensures both opponents are loaded via the same logic.
    """
    # Use the opponent name in the adapter name if none was supplied to make adapters unique.
    if peft_kwargs and peft_adapter_name is None:
        peft_adapter_name = f"{opponent_name}_adapter"

    model, tokenizer = load_model_and_tokenizer(
        model_id,
        device=device,
        load_in_4bit=load_in_4bit,
        max_seq_length=max_seq_length,
        peft_kwargs=peft_kwargs,
        peft_adapter_name=peft_adapter_name,
        peft_checkpoint=peft_checkpoint,
        dtype=dtype,
    )

    return model, tokenizer


# Quick local test helper when running the file directly.
if __name__ == "__main__":
    test_model = "unsloth/Llama-3.2-3B"
    common_peft = {
        "r": 4,
        "target_modules": ["q_proj", "v_proj"],
        "lora_alpha": 8,
    }

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Loading opponent1 (test)...")
    m1, t1 = load_opponent_model("opponent1", test_model, device=device, load_in_4bit=False, peft_kwargs=common_peft)
    print("Loaded opponent1. Tokenizer length:", len(t1))

    print("Loading opponent2 (test)...")
    m2, t2 = load_opponent_model("opponent2", test_model, device=device, load_in_4bit=False, peft_kwargs=common_peft)
    print("Loaded opponent2. Tokenizer length:", len(t2))

    print("Test load finished.")