import transformers
from transformers import (
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    PretrainedConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

# Ordered by preference
_AUTOREGRESSIVE_SUFFIXES = [
    # Encoder-decoder models
    "ConditionalGeneration",
    # Autoregressive models
    "CausalLM",
    "LMHeadModel",
]


def instantiate_model(model_str: str, **kwargs) -> PreTrainedModel:
    """Instantiate a model string with the appropriate `Auto` class."""
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
    try:
        return AutoTokenizer.from_pretrained(model_str, use_fast=True, **kwargs)
    except Exception as e:
        if kwargs.get("verbose", True):
            print(f"Falling back to slow tokenizer; fast one failed to load: '{e}'")

        return AutoTokenizer.from_pretrained(model_str, use_fast=False, **kwargs)


def is_autoregressive(model_cfg: PretrainedConfig) -> bool:
    """Check if a model config is autoregressive."""
    archs = model_cfg.architectures
    if not isinstance(archs, list):
        return False

    return any(
        arch_str.endswith(suffix)
        for arch_str in archs
        for suffix in _AUTOREGRESSIVE_SUFFIXES
    )
