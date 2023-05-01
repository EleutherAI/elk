from dataclasses import dataclass
from typing import Sequence, TYPE_CHECKING

import torch
from transformers import PreTrainedModel
from elk.extraction.llama.device_map import get_llama_65b_8bit_device_map
from elk.utils import select_usable_devices, instantiate_model

if TYPE_CHECKING:
    from elk import Extract


@dataclass
class Llama65bDeviceConfig:
    first_device: str
    second_device: str


def select_devices_or_llama_65b_configs(
    model_name: str,
    num_gpus: int,
    min_memory: int | None = None,
) -> Sequence[str | Llama65bDeviceConfig]:
    if "llama-65b" not in model_name:
        return select_usable_devices(num_gpus, min_memory=min_memory)
    else:
        print(
            f"You've selected a llama-65b model, which requires at least two GPUs."
            f"Each GPU must have at least 40 GiB of memory."
        )
        print("Note that we will force the model to use 8-bit")
        assert num_gpus >= 2, "llama-65b models require at least two GPUs"
        # how many pairs of 2 gpus are specified?
        num_pairs = num_gpus // 2
        print(f"Will create {num_pairs} llama workers ")
        forty_gb = 42_949_672_960
        devices = select_usable_devices(num_gpus, min_memory=forty_gb)
        # split the devices into pairs
        configs = []
        while len(configs) < num_pairs:
            first_device = devices.pop()
            second_device = devices.pop()
            configs.append(
                Llama65bDeviceConfig(
                    first_device=first_device, second_device=second_device
                )
            )
        print(f"Created {len(configs)} llama workers")

        return configs


def instantiate_model_or_llama(
    cfg: "Extract", device_config: str | Llama65bDeviceConfig, **kwargs
) -> PreTrainedModel:
    is_llama_65b = isinstance(device_config, Llama65bDeviceConfig)
    first_device = device_config.first_device if is_llama_65b else device_config
    if cfg.int8 or is_llama_65b:
        # Required by `bitsandbytes`
        dtype = torch.float16
    elif device_config == "cpu":
        dtype = torch.float32
    else:
        dtype = "auto"
    if is_llama_65b:
        model = instantiate_model(
            cfg.model,
            device_map=get_llama_65b_8bit_device_map(
                first_device=first_device, second_device=device_config.second_device
            ),
            load_in_8bit=True,
            torch_dtype=dtype,
            **kwargs,
        )
    else:
        model = instantiate_model(
            cfg.model,
            device_map={"": first_device},
            load_in_8bit=cfg.int8,
            torch_dtype=dtype,
            **kwargs,
        )
    return model
