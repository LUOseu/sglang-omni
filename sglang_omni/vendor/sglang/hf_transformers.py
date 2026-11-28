"""Hugging Face utilities moved out of srt.utils after SGLang 0.5.23."""

from importlib.util import find_spec

if find_spec("sglang.srt.hf_transformers") is None:
    from sglang.srt.utils.hf_transformers import get_config
    from sglang.srt.utils.hf_transformers.tokenizer import get_tokenizer
else:
    from sglang.srt.hf_transformers import get_config
    from sglang.srt.hf_transformers.tokenizer import get_tokenizer


def dspine_context_length(model_path: str, revision: str | None) -> int:
    """Read the native DSpine backbone capacity after SGLang converts the checkpoint."""
    config = get_config(model_path, trust_remote_code=False, revision=revision)
    return int(config.max_position_embeddings)


__all__ = ["dspine_context_length", "get_tokenizer"]
