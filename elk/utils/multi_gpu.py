from contextlib import nullcontext, redirect_stdout
from dataclasses import dataclass
from typing import TYPE_CHECKING, Type

import torch
from accelerate import infer_auto_device_map, init_empty_weights
from torch import dtype
from torch.nn import Module
from transformers import PreTrainedModel

from elk.utils import instantiate_model, select_usable_devices
from elk.utils.gpu_utils import get_available_memory_for_devices

if TYPE_CHECKING:
    from elk import Extract


@dataclass
class ModelDevices:
    # The devices to instantiate a single model on
    first_device: str
    other_devices: list[str]

    @property
    def is_single_gpu(self) -> bool:
        return len(self.other_devices) == 0

    @property
    def used_devices(self) -> list[str]:
        return [self.first_device] + self.other_devices


def instantiate_model_with_devices(
    cfg: "Extract", device_config: ModelDevices, is_verbose: bool, **kwargs
) -> PreTrainedModel:
    first_device = device_config.first_device
    if cfg.int8:
        # Required by `bitsandbytes`
        torch_dtype = torch.float16
    elif device_config == "cpu":
        torch_dtype = torch.float32
    else:
        torch_dtype = "auto"

    # TODO: Maybe we should ensure the device map is the same
    # for all the extract processes? This is because the device map
    # can affect performance highly and its annoying if one process
    # is using a different device map than the others.
    device_map = (
        {"": first_device}
        if device_config.is_single_gpu
        else create_device_map(
            model_str=cfg.model,
            use_8bit=cfg.int8,
            torch_dtype=torch_dtype,
            model_devices=device_config,
            verbose=is_verbose,
        )
    )
    if is_verbose:
        print(
            f"Using {len(device_config.used_devices)} gpu(s) to"
            f" instantiate a single model."
        )
    with redirect_stdout(None) if not is_verbose else nullcontext():
        model = instantiate_model(
            cfg.model,
            device_map=device_map,
            load_in_8bit=cfg.int8,
            torch_dtype=torch_dtype,
            **kwargs,
        )
    return model


def create_device_map(
    model_str: str,
    use_8bit: float,
    torch_dtype: dtype | str,
    model_devices: ModelDevices,
    verbose: bool,
) -> dict[str, str]:
    """Creates a device map for a model running on multiple GPUs."""
    with init_empty_weights():
        # Need to first instantiate an empty model to get the layer class
        model = instantiate_model(model_str=model_str, torch_dtype=torch_dtype)

    # e.g. {"cuda:0": 16000, "cuda:1": 16000}
    max_memory_all_devices: dict[str, int] = get_available_memory_for_devices()
    # now let's get the available memory for the devices we want to use
    used_devices = model_devices.used_devices
    max_memory_used_devices: dict[str, int | float] = {
        device: max_memory_all_devices[device] for device in used_devices
    }
    # Decrease the memory potentially used by the first device
    # because we're going to create additional tensors on it
    max_memory_used_devices[model_devices.first_device] = (
        max_memory_used_devices[model_devices.first_device] * 0.6
    )
    if use_8bit:
        print("Using 8bit")
    # If 8bit, multiply the memory by 2
    # This is because we instantiated our empty model in (probably) float16
    # We aren't able to instantiate an empty model in 8bit currently
    devices_accounted_8bit = (
        {
            device: max_memory_used_devices[device] * 2
            for device in max_memory_used_devices
        }
        if use_8bit
        else max_memory_used_devices
    )

    # Make sure that the transformer layer is not split
    # because that contains residual connections
    # See https://huggingface.co/docs/accelerate/usage_guides/big_modeling
    # Otherwise we get an error like this:
    # RuntimeError: Expected all tensors to be on the same device,
    # but found at least two devices, cuda:0 and cuda1
    maybe_transformer_class: Type[Module] | None = get_transformer_layer_cls(model)
    dont_split = [maybe_transformer_class.__name__] if maybe_transformer_class else []
    autodevice_map = infer_auto_device_map(
        model, no_split_module_classes=dont_split, max_memory=devices_accounted_8bit
    )

    if verbose:
        print(f"Autodevice map: {autodevice_map}")
    assert "disk" not in autodevice_map.values(), (
        f"Unable to fit the model {model} into the given memory for {used_devices}."
        f" Try increasing gpus_per_model?"
    )
    return autodevice_map


def select_devices_multi_gpus(
    gpus_per_model: int,
    num_gpus: int,
    min_memory: int | None = None,
) -> list[ModelDevices]:
    if gpus_per_model == 1:
        devices = select_usable_devices(num_gpus, min_memory=min_memory)
        return [
            ModelDevices(first_device=devices, other_devices=[]) for devices in devices
        ]
    else:
        # how many models can we create?
        models_to_create = num_gpus // gpus_per_model
        print(
            f"Allocating devices for {models_to_create} models with {gpus_per_model}"
            f" GPUs each"
        )
        devices = select_usable_devices(num_gpus, min_memory=min_memory)
        configs = split_devices_into_model_devices(
            devices=devices,
            gpus_per_model=gpus_per_model,
            models_to_create=models_to_create,
        )
        print(f"Models will be instantiated on {configs}")
        return configs


def get_transformer_layer_cls(model: torch.nn.Module) -> Type[torch.nn.Module] | None:
    """Get the class of the transformer layer used by the given model."""
    total_params = sum(p.numel() for p in model.parameters())
    for module in model.modules():
        if isinstance(module, torch.nn.ModuleList):
            module_params = sum(p.numel() for p in module.parameters())
            if module_params > total_params / 2:
                type_of_cls = type(module[0])
                print(f"Found transformer layer of type {type_of_cls}")
                return type_of_cls

    return None


def split_devices_into_model_devices(
    devices: list[str], gpus_per_model: int, models_to_create: int
) -> list[ModelDevices]:
    assert len(devices) >= gpus_per_model * models_to_create
    configs = []
    while len(configs) < models_to_create:
        first_device = devices.pop(0)
        other_devices = devices[: gpus_per_model - 1]
        devices = devices[gpus_per_model - 1 :]
        configs.append(ModelDevices(first_device, other_devices))
    return configs
