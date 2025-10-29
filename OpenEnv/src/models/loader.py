# Optional dependencies: unsloth, peft, bitsandbytes
# This loader supports:
# - Unsloth models via FastLanguageModel (if use_unsloth=True)
# - Any Hugging Face model via transformers AutoModelForCausalLM
# - Optional LoRA adaptation for either backend
#
# Example:
#   from src.models.loader import load_language_model
#   model, tokenizer = load_language_model(
#       model_name="unsloth/gpt-oss-20b", use_unsloth=True, load_in_4bit=True
#   )
#   # or
#   model, tokenizer = load_language_model(
#       model_name="microsoft/Phi-3.5-mini-instruct",
#       use_unsloth=False, load_in_4bit=True
#   )

from typing import Optional, Tuple, Dict, Any
import os
import torch

def _bf16_if_available() -> torch.dtype:
    try:
        if torch.cuda.is_available():
            cap = torch.cuda.get_device_capability()
            # bf16 widely supported on modern accelerators; ROCm supports bf16 too
            return torch.bfloat16
    except Exception:
        pass
    return torch.float16

def _maybe_import_unsloth():
    try:
        from unsloth import FastLanguageModel
        return FastLanguageModel
    except Exception as e:
        raise RuntimeError(
            "Unsloth not installed but `use_unsloth=True` was requested. "
            "Install unsloth or set use_unsloth=False. Error: %s" % str(e)
        )

def _maybe_import_peft():
    try:
        from peft import LoraConfig, get_peft_model, TaskType
        return LoraConfig, get_peft_model, TaskType
    except Exception:
        return None, None, None

def _resolve_tokenizer_padding(tokenizer):
    # Ensure pad token exists for batch generation/training
    if tokenizer.pad_token is None:
        # Prefer eos_token as pad if available
        if tokenizer.eos_token:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            # Create a new pad token if none exists
            tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
    return tokenizer

def load_language_model(
    model_name: str,
    use_unsloth: bool = False,
    load_in_4bit: bool = True,
    max_seq_length: int = 2048,
    lora: bool = True,
    lora_rank: int = 8,
    lora_alpha: Optional[int] = None,
    target_modules: Optional[list] = None,
    gradient_checkpointing: bool = True,
    torch_dtype: Optional[torch.dtype] = None,
    attn_implementation: Optional[str] = None,  # "flash_attention_2" if desired
    device_map: str = "auto",
    tokenizer_kwargs: Optional[Dict[str, Any]] = None,
    model_kwargs: Optional[Dict[str, Any]] = None,
) -> Tuple[torch.nn.Module, Any]:
    """
    Unified loader for Unsloth or Hugging Face models with optional LoRA.

    Returns:
      model, tokenizer
    """
    tokenizer_kwargs = tokenizer_kwargs or {}
    model_kwargs = model_kwargs or {}

    if torch_dtype is None:
        torch_dtype = _bf16_if_available()

    if use_unsloth:
        FastLanguageModel = _maybe_import_unsloth()
        # Unsloth path
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            load_in_4bit=load_in_4bit,
            max_seq_length=max_seq_length,
            **model_kwargs,
        )
        if lora:
            if target_modules is None:
                target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
            if lora_alpha is None:
                lora_alpha = lora_rank * 2
            model = FastLanguageModel.get_peft_model(
                model,
                r=lora_rank,
                lora_alpha=lora_alpha,
                target_modules=target_modules,
                use_gradient_checkpointing="unsloth" if gradient_checkpointing else False,
                random_state=3407,
            )
        # Unsloth tokenizer already aligned
        tokenizer = _resolve_tokenizer_padding(tokenizer)
        return model, tokenizer

    # Hugging Face transformers path
    from transformers import AutoTokenizer, AutoModelForCausalLM

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, **tokenizer_kwargs)
    tokenizer = _resolve_tokenizer_padding(tokenizer)

    # 4-bit quantization if bitsandbytes available
    quantization_config = None
    if load_in_4bit:
        try:
            from transformers import BitsAndBytesConfig
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch_dtype,
            )
        except Exception:
            quantization_config = None

    # Model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        device_map=device_map,
        quantization_config=quantization_config,
        attn_implementation=attn_implementation,
        **model_kwargs,
    )

    # Resize embeddings if we added new special tokens
    if len(tokenizer) > model.get_input_embeddings().weight.shape[0]:
        model.resize_token_embeddings(len(tokenizer))

    # Optional LoRA for HF models
    if lora:
        LoraConfig, get_peft_model, TaskType = _maybe_import_peft()
        if LoraConfig is None:
            raise RuntimeError(
                "peft not installed but LoRA requested. Install `peft` or set lora=False."
            )
        if target_modules is None:
            # Typical for decoder-only models
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        if lora_alpha is None:
            lora_alpha = lora_rank * 2

        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=0.05,
            target_modules=target_modules,
            bias="none",
        )
        model = get_peft_model(model, peft_config)

        if gradient_checkpointing:
            model.gradient_checkpointing_enable()

    return model, tokenizer