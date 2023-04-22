import transformers
from transformers import (
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    PretrainedConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    GPT2TokenizerFast
)
from ..rwkv_lm.rwkv_hf import RWKVModel, RWKVConfig

# Ordered by preference
_DECODER_ONLY_SUFFIXES = [
    "CausalLM",
    "LMHeadModel",
]
# Includes encoder-decoder models
_AUTOREGRESSIVE_SUFFIXES = ["ConditionalGeneration"] + _DECODER_ONLY_SUFFIXES


def instantiate_model(model_str: str, **kwargs) -> PreTrainedModel:
    """Instantiate a model string with the appropriate `Auto` class."""
    if model_str.startswith("rwkv"):
        return RWKVModel()

    model_cfg = AutoConfig.from_pretrained(model_str)
    archs = model_cfg.architectures
    if not isinstance(archs, list):
        return AutoModel.from_pretrained(model_str, **kwargs)

    for suffix in _AUTOREGRESSIVE_SUFFIXES:
        # Check if any of the architectures in the config end with the suffix.
        # If so, return the corresponding model class.
        for arch_str in archs:
            if arch_str.endswith(suffix):
                model_cls = getattr(transformers, arch_str)
                return model_cls.from_pretrained(model_str, **kwargs)

    return AutoModel.from_pretrained(model_str, **kwargs)


def instantiate_tokenizer(model_str: str, **kwargs) -> PreTrainedTokenizerBase:
    """Instantiate a tokenizer, using the fast one iff it exists."""
    if model_str.startswith("rwkv"):
        return GPT2TokenizerFast(tokenizer_file="elk/rwkv_lm/20B_tokenizer.json")

    try:
        return AutoTokenizer.from_pretrained(model_str, use_fast=True, **kwargs)
    except Exception as e:
        if kwargs.get("verbose", True):
            print(f"Falling back to slow tokenizer; fast one failed to load: '{e}'")

        return AutoTokenizer.from_pretrained(model_str, use_fast=False, **kwargs)

def instantiate_config(model_str: str, **kwargs) -> PretrainedConfig:
    """Instantiate a config."""
    if model_str.startswith("rwkv"):
        return RWKVConfig()

    return AutoConfig.from_pretrained(model_str, **kwargs)


def is_autoregressive(model_cfg: PretrainedConfig, include_enc_dec: bool) -> bool:
    """Check if a model config is autoregressive."""
    archs = model_cfg.architectures
    if not isinstance(archs, list):
        return False

    suffixes = _AUTOREGRESSIVE_SUFFIXES if include_enc_dec else _DECODER_ONLY_SUFFIXES
    return any(arch_str.endswith(suffix) for arch_str in archs for suffix in suffixes)
