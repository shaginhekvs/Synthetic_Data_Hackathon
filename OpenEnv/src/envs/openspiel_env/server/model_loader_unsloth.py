#!/usr/bin/env python3
from __future__ import annotations
from typing import Optional, Tuple, Any

def load_unsloth_model(
    model_name: str,
    *,
    max_seq_length: int = 768,
    load_in_4bit: bool = False,
    lora_rank: int = 0,
    verbose: bool = False,
) -> Tuple[Any, Any]:
    """
    Centralized Unsloth model loader.
    Mirrors your previous _load_fast_language_model helper and returns (model, tokenizer).
    """
    FastLanguageModel = None
    try:
        from unsloth import FastLanguageModel
    except ImportError as e:
        raise ImportError(
            "Could not import FastLanguageModel (unsloth). Install 'unsloth' to use Unsloth models."
        ) from e

    if verbose:
        print(f"[unsloth-loader] from_pretrained('{model_name}')")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        load_in_4bit=load_in_4bit,
    )

    # Optional PEFT/LoRA
    if hasattr(FastLanguageModel, "get_peft_model") and lora_rank and lora_rank > 0:
        if verbose:
            print(f"[unsloth-loader] applying PEFT/LoRA r={lora_rank}")
        model = FastLanguageModel.get_peft_model(
            model,
            r=lora_rank,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        )

    return model, tokenizer