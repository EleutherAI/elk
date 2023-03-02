"""Utilities that use PyNVML to get GPU usage info, and select GPUs accordingly."""

import os
import pynvml
import torch
import warnings


def select_usable_devices(max_gpus: int = -1, *, min_memory: int = 0) -> list[str]:
    """Select a set of devices that have at least `min_memory` bytes of free memory.

    When there are more than enough GPUs to satisfy the request, the GPUs with the
    most free memory will be selected. With default arguments, this function will
    simply return a list of all available GPU indices. A human-readable message will
    be printed to stdout indicating which GPUs were selected.

    This function uses the PyNVML library to get usage info. Unfortunately, PyNVML does
    not respect the `CUDA_VISIBLE_DEVICES` environment variable, while PyTorch does.
    Correctly converting PyNVML device indices to PyTorch indices is nontrivial and was
    only recently (commit `dc4f2af` on 9 Feb. 2023) implemented in PyTorch `master`. We
    can't depend on PyTorch nightly and we also don't want to copy-paste the code here.

    For now, we simply return `list(range(max_gpus))` whenever `CUDA_VISIBLE_DEVICES`
    is set. Arguably this is expected behavior. If the user set `CUDA_VISIBLE_DEVICES`,
    they probably want to use all & only those GPUs.

    Args:
        num_gpus: Maximum number of GPUs to select. If negative, all available GPUs
            meeting the criteria will be selected.
        min_memory: Minimum amount of free memory (in bytes) required to select a GPU.

    Returns:
        A list of suitable PyTorch device strings, in ascending numerical order.

    Raises:
        ValueError: If `max_gpus` is greater than the number of visible GPUs.
    """
    # Trivial case: no GPUs requested or available
    num_visible = torch.cuda.device_count()
    if max_gpus == 0 or num_visible == 0:
        return ["cpu"]

    # Sanity checks
    if max_gpus > num_visible:
        raise ValueError(
            f"Requested {max_gpus} GPUs, but only {num_visible} are visible."
        )
    elif max_gpus < 0:
        max_gpus = num_visible

    # No limits, so try to use all installed GPUs
    if max_gpus == num_visible and min_memory <= 0:
        print(f"Using all {num_visible} GPUs.")
        return [f"cuda:{i}" for i in range(max_gpus)]

    # The user set CUDA_VISIBLE_DEVICES and also requested a specific number of GPUs.
    # The environment variable takes precedence, so we'll just use all visible GPUs.
    count_msg = "all" if max_gpus == num_visible else f"first {max_gpus}"
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        warnings.warn(
            f"Smart GPU selection not supported when CUDA_VISIBLE_DEVICES is set. "
            f"Will use {count_msg} visible devices."
        )
        return [f"cuda:{i}" for i in range(max_gpus)]

    # pynvml.nvmlInit() will raise if we're using non-NVIDIA GPUs
    try:
        pynvml.nvmlInit()
    except pynvml.NVMLError:
        warnings.warn(
            f"Unable to initialize PyNVML; are you using non-NVIDIA GPUs? Will use "
            f"{count_msg} visible devices."
        )
        return [f"cuda:{i}" for i in range(max_gpus)]

    try:
        # PyNVML and PyTorch device indices should agree when CUDA_VISIBLE_DEVICES is
        # not set. We need them to agree so that the PyNVML indices match the PyTorch
        # indices, and we don't have to do any complex error-prone conversions.
        num_installed = pynvml.nvmlDeviceGetCount()
        assert num_installed == num_visible, "PyNVML and PyTorch disagree on GPU count"

        # List of (-free memory, GPU index) tuples. Sorted descending by free memory,
        # then ascending by GPU index.
        memories_and_indices = sorted(
            (
                -int(pynvml.nvmlDeviceGetMemoryInfo(handle).free),
                pynvml.nvmlDeviceGetIndex(handle),
            )
            for handle in map(pynvml.nvmlDeviceGetHandleByIndex, range(num_installed))
        )
        usable_indices = [
            index for neg_mem, index in memories_and_indices if -neg_mem >= min_memory
        ]
    finally:
        # Make sure we always shut down PyNVML
        pynvml.nvmlShutdown()

    # Indices are sorted descending by free memory, so we want the first `max_gpus`
    # items. For printing purposes, though, we sort the indices numerically.
    selection = sorted(usable_indices[:max_gpus])

    # Did we get the maximum number of GPUs requested?
    if len(selection) == max_gpus:
        print(f"Using {len(selection)} of {num_visible} GPUs: {selection}")
    else:
        print(f"Using {len(selection)} of {max_gpus} requested GPUs: {selection}")
        print(
            f"{num_visible - len(selection)} GPUs have insufficient free memory "
            f"({min_memory / 10 ** 9:.2f} GB needed)."
        )

    if len(selection) > 0:
        return [f"cuda:{i}" for i in selection]
    else:
        return ["cpu"]
