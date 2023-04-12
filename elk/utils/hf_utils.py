import transformers
from transformers import AutoConfig, AutoModel, PretrainedConfig, PreTrainedModel

from ..rnn.elmo import ElmoModel
from .typing import assert_type

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
    if model_str.startswith("elmo"):
        return ElmoModel.from_pretrained(model_str)

    model_cfg = AutoConfig.from_pretrained(model_str)
    archs = assert_type(list, model_cfg.architectures)

    for suffix in _AUTOREGRESSIVE_SUFFIXES:
        # Check if any of the architectures in the config end with the suffix.
        # If so, return the corresponding model class.
        for arch_str in archs:
            if arch_str.endswith(suffix):
                model_cls = getattr(transformers, arch_str)
                return model_cls.from_pretrained(model_str, **kwargs)

    return AutoModel.from_pretrained(model_str, **kwargs)


def is_autoregressive(model_cfg: PretrainedConfig) -> bool:
    """Check if a model config is autoregressive."""
    archs = assert_type(list, model_cfg.architectures)
    return any(
        arch_str.endswith(suffix)
        for arch_str in archs
        for suffix in _AUTOREGRESSIVE_SUFFIXES
    )
