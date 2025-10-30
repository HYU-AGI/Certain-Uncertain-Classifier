from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

try:
    from openai_harmony import load_harmony_encoding, HarmonyEncodingName  # type: ignore
    _HARMONY = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
    _HARMONY_EOS: Optional[List[int]] = _HARMONY.stop_tokens_for_assistant_actions()
except Exception:
    _HARMONY = None
    _HARMONY_EOS = None


SUPPORTED_MODELS = {
    "gpt-oss-20b",
    "qwen3-8b",
    "qwen3-14b",
    "deepseek-v2-16b",
    "llama-3b",
    "llama-8b",
    "mistral-7b",
}


@dataclass
class Args:
    model_name: str
    model_id: str
    cache_dir: Optional[str] = None
    max_new_tokens: int = 512
    do_sample: bool = True
    top_p: Optional[float] = 0.95
    top_k: Optional[int] = 50
    temperature: float = 0.7


def _inject_reasoning_low(messages):
    has_low = any(m.get("role") == "system" and "Reasoning:" in m.get("content", "") for m in messages)
    if not has_low:
        return [{"role": "system", "content": "Reasoning: low"}] + messages
    return messages


def _extract_harmony_final(decoded: str) -> Optional[str]:
    start_tag = "<|channel|>final<|message|>"
    end_tag = "<|end|>"
    if start_tag in decoded:
        part = decoded.split(start_tag, 1)[-1]
        if end_tag in part:
            part = part.split(end_tag, 1)[0]
        return part.strip()
    start_alt = "<|final|>"
    if start_alt in decoded:
        part = decoded.split(start_alt, 1)[-1]
        if end_tag in part:
            part = part.split(end_tag, 1)[0]
        return part.strip()
    return None


def load_model(args: Args) -> Tuple[Any, Any]:
    if args.model_name not in SUPPORTED_MODELS:
        raise ValueError(f"Model {args.model_name} is not supported.")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_id,
        trust_remote_code=True,
        cache_dir=args.cache_dir,
    )

    model_kwargs: Dict[str, Any] = dict(
        torch_dtype=torch.bfloat16,
        device_map="auto",
        cache_dir=args.cache_dir,
        trust_remote_code=True,
        use_kernels=False
    )

    try:
        model = AutoModelForCausalLM.from_pretrained(args.model_id, **model_kwargs)
    except TypeError:
        model_kwargs.pop("use_kernels", None)
        model = AutoModelForCausalLM.from_pretrained(args.model_id, **model_kwargs)

    return model, tokenizer


def generate_response(
    model: Any,
    tokenizer: Any,
    messages,
    args: Args,
    forced_prefix: Optional[str] = None,
) -> Tuple[str, Dict[str, int]]:
    try:
        model.generation_config = GenerationConfig.from_pretrained(args.model_id)
    except Exception:
        pass

    if args.model_name == "deepseek-v2-16b":
        try:
            model.config.use_cache = False
        except Exception:
            pass

    inputs = _apply_chat_template(tokenizer, args.model_name, messages).to(model.device)

    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else getattr(model.generation_config, "eos_token_id", None)
    if isinstance(pad_id, (list, tuple)):
        pad_id = pad_id[0]
    if pad_id is not None:
        model.generation_config.pad_token_id = pad_id

    prefix_len = 0
    if forced_prefix:
        if args.model_name.startswith("gpt-oss"):
            forced = "<|channel|>final<|message|>" + forced_prefix
        else:
            forced = forced_prefix
        pref_ids = tokenizer(forced, add_special_tokens=False, return_tensors="pt")["input_ids"].to(model.device)
        prefix_len = pref_ids.shape[1]
        inputs["input_ids"] = torch.cat([inputs["input_ids"], pref_ids], dim=1)
        if "attention_mask" in inputs:
            pref_mask = torch.ones_like(pref_ids)
            inputs["attention_mask"] = torch.cat([inputs["attention_mask"], pref_mask], dim=1)

    prompt_len = inputs["input_ids"].shape[1]

    gen_kwargs: Dict[str, Any] = dict(max_new_tokens=args.max_new_tokens)
    if args.do_sample:
        gen_kwargs.update(
            dict(
                do_sample=True,
                top_p=args.top_p if args.top_p is not None else 0.95,
                top_k=args.top_k if args.top_k is not None else 50,
                temperature=None if args.temperature == 0 else args.temperature,
            )
        )
    else:
        gen_kwargs.update(dict(do_sample=False))

    if args.model_name.startswith("gpt-oss") and _HARMONY_EOS:
        gen_kwargs["eos_token_id"] = _HARMONY_EOS

    outputs = model.generate(**inputs, **gen_kwargs)

    if args.model_name.startswith("gpt-oss"):
        decoded_full = tokenizer.decode(outputs[0], skip_special_tokens=False)
        final_only = _extract_harmony_final(decoded_full)
        if final_only is None:
            start = prompt_len - prefix_len if forced_prefix else prompt_len
            generated = tokenizer.decode(outputs[0][start:], skip_special_tokens=True).strip()
        else:
            generated = final_only.strip()
    else:
        start = prompt_len - prefix_len if forced_prefix else prompt_len
        generated = tokenizer.decode(outputs[0][start:], skip_special_tokens=True).strip()

    stats = {
        "input_len": prompt_len,
        "output_len": int(outputs[0].shape[0] - prompt_len),
        "prefix_len": prefix_len,
    }
    return generated, stats


def _apply_chat_template(tokenizer, model_name: str, messages, *, as_text: bool = False):
    if model_name.startswith("gpt-oss"):
        messages = _inject_reasoning_low(messages)
        return tokenizer.apply_chat_template(
            messages, tokenize=not as_text, add_generation_prompt=True,
            return_tensors=None if as_text else "pt",
            return_dict=not as_text
        )
    elif model_name.startswith("qwen3"):
        return tokenizer.apply_chat_template(
            messages, tokenize=not as_text, enable_thinking=False, add_generation_prompt=True,
            return_tensors=None if as_text else "pt",
            return_dict=not as_text
        )
    else:
        return tokenizer.apply_chat_template(
            messages, tokenize=not as_text, add_generation_prompt=True,
            return_tensors=None if as_text else "pt",
            return_dict=not as_text
        )
    