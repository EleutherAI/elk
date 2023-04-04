from .typing import assert_type
from transformers import AutoConfig, PreTrainedModel
from typing import Type
import transformers


def get_model_class(model_str: str) -> Type[PreTrainedModel]:
    """Get the appropriate model class for a model string."""
    model_cfg = AutoConfig.from_pretrained(model_str)
    archs = assert_type(list, model_cfg.architectures)

    # Ordered by preference
    suffixes = [
        # Fine-tuned for classification
        "SequenceClassification",
        # Encoder-decoder models
        "ConditionalGeneration",
        # Autoregressive models
        "CausalLM",
        "LMHeadModel",
    ]

    for suffix in suffixes:
        # Check if any of the architectures in the config end with the suffix.
        # If so, return the corresponding model class.
        for arch_str in archs:
            if arch_str.endswith(suffix):
                return getattr(transformers, arch_str)

    raise ValueError(
        f"'{model_str}' does not have any supported architectures: {archs}"
    )
