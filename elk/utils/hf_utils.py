import torch
import transformers
from transformers import (
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    PretrainedConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from .data_utils import prevent_name_conflicts

# Ordered by preference
_DECODER_ONLY_SUFFIXES = [
    "CausalLM",
    "LMHeadModel",
]
# Includes encoder-decoder models
_AUTOREGRESSIVE_SUFFIXES = ["ConditionalGeneration"] + _DECODER_ONLY_SUFFIXES
# These modules shouldn't be split across devices because they have residual connections
# TODO: expand this list, preferably without hard-coding it
NO_SPLIT_MODULE_CLASSES = ["LlamaDecoderLayer"]
IGNORE_PATTERNS = ["*.bin", "*.bin.index.json"]  # only use the safetensors


def instantiate_model(
    model_str: str,
    device: str | torch.device = "cpu",
    **kwargs,
) -> PreTrainedModel:
    """Instantiate a model string with the appropriate `Auto` class."""

    device = torch.device(device)

    with prevent_name_conflicts():
        model_cfg = AutoConfig.from_pretrained(model_str)

        # When the torch_dtype is None, this generally means the model is fp32, because
        # the config was probably created before the `torch_dtype` field was added.
        fp32_weights = model_cfg.torch_dtype in (None, torch.float32)

        # Required by `bitsandbytes` to load in 8-bit.
        if kwargs.get("load_in_8bit"):
            # Sanity check: we probably shouldn't be loading in 8-bit if the checkpoint
            # is in fp32. `bitsandbytes` only supports mixed fp16/int8 inference, and
            # we can't guarantee that there won't be overflow if we downcast to fp16.
            if fp32_weights:
                raise ValueError("Cannot load in 8-bit if weights are fp32")

            dtype = torch.float16

        # CPUs generally don't support anything other than fp32.
        elif device.type == "cpu":
            dtype = torch.float32

        # If the model is fp32 but bf16 is available, convert to bf16.
        # Usually models with fp32 weights were actually trained in bf16, and
        # converting them doesn't hurt performance.
        elif fp32_weights and torch.cuda.is_bf16_supported():
            dtype = torch.bfloat16
            print("Weights seem to be fp32, but bf16 is available. Loading in bf16.")
        else:
            dtype = "auto"

        model_cls = AutoModel
        archs = model_cfg.architectures
        if isinstance(archs, list):
            for suffix in _AUTOREGRESSIVE_SUFFIXES:
                # Check if any of the architectures in the config end with the suffix.
                # If so, return the corresponding model class.
                for arch_str in archs:
                    if arch_str.endswith(suffix):
                        model_cls = getattr(transformers, arch_str)
                        break
                if model_cls is not AutoModel:
                    break

        kwargs["device_map"] = {"": device}
        kwargs["torch_dtype"] = dtype
        return model_cls.from_pretrained(model_str, **kwargs)


def instantiate_tokenizer(model_str: str, **kwargs) -> PreTrainedTokenizerBase:
    """Instantiate a tokenizer, using the fast one iff it exists."""
    with prevent_name_conflicts():
        try:
            return AutoTokenizer.from_pretrained(model_str, use_fast=True, **kwargs)
        except Exception as e:
            if kwargs.get("verbose", True):
                print(f"Falling back to slow tokenizer; fast one failed: '{e}'")

            return AutoTokenizer.from_pretrained(model_str, use_fast=False, **kwargs)


def is_autoregressive(model_cfg: PretrainedConfig, include_enc_dec: bool) -> bool:
    """Check if a model config is autoregressive."""
    archs = model_cfg.architectures
    if not isinstance(archs, list):
        return False

    suffixes = _AUTOREGRESSIVE_SUFFIXES if include_enc_dec else _DECODER_ONLY_SUFFIXES
    return any(arch_str.endswith(suffix) for arch_str in archs for suffix in suffixes)


def get_max_memory(devices: tuple[torch.device, ...]) -> dict[int | str, int | str]:
    """Get the maximum memory available on a tuple of devices."""
    # Make sure CUDA is initialized on each GPU to have the right memory info.
    for d in devices:
        _ = torch.tensor([0], device=d)
    return {d.index: torch.cuda.mem_get_info(d.index)[0] for d in devices}
