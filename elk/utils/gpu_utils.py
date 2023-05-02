"""Utilities that use PyNVML to get GPU usage info, and select GPUs accordingly."""

import os
import time
import warnings
from functools import cache

import pynvml
import torch

from .typing import assert_type


# We cache the results primarily so that we don't display "Using N of M GPUs..."
# multiple times during the same run. This does sort of assume that once we identify
# a GPU as being available, it will remain available for the duration of the run.
# This seems to be a reasonable assumption because PyTorch tends to hold onto VRAM
# for later use once it's been allocated. Calling torch.cuda.empty_cache() might break
# this assumption, but we never do that.
@cache
def select_usable_devices(
    num_gpus: int = -1, *, min_memory: int | None = None
) -> list[str]:
    """Select a set of devices that have at least `min_memory` bytes of free memory.
    Blocks until at least `num_gpus` devices are available.

    When there are more than enough GPUs to satisfy the request, the GPUs with the
    most free memory will be selected. With default arguments, this function will
    simply return a list of all available GPU indices. A human-readable message will
    be printed to stdout indicating which GPUs were selected.

    This function uses the PyNVML library to get usage info. Unfortunately, PyNVML does
    not respect the `CUDA_VISIBLE_DEVICES` environment variable, while PyTorch does.
    Correctly converting PyNVML device indices to PyTorch indices is nontrivial and was
    only recently (commit `dc4f2af` on 9 Feb. 2023) implemented in PyTorch `master`. We
    can't depend on PyTorch nightly and we also don't want to copy-paste the code here.

    For now, we simply return `list(range(num_gpus))` whenever `CUDA_VISIBLE_DEVICES`
    is set. Arguably this is expected behavior. If the user set `CUDA_VISIBLE_DEVICES`,
    they probably want to use all & only those GPUs.

    Args:
        num_gpus: Number of GPUs to select. If negative, all available GPUs
            meeting the criteria will be selected.
        min_memory: Minimum amount of free memory (in bytes) required to select a GPU.
            If None, `min_memory` is set to 90% of the per-GPU memory.

    Returns:
        A list of suitable PyTorch device strings, in ascending numerical order, with
        exactly `num_gpus` elements.

    Raises:
        ValueError: If `num_gpus` is greater than the number of visible GPUs.
    """
    # Trivial case: no GPUs requested or available
    num_visible = torch.cuda.device_count()
    if num_gpus == 0 or num_visible == 0:
        return ["cpu"]

    # Sanity checks
    if num_gpus > num_visible:
        raise ValueError(
            f"Requested {num_gpus} GPUs, but only {num_visible} are visible."
        )
    elif num_gpus < 0:
        num_gpus = num_visible

    # No limits, so try to use all installed GPUs
    if num_gpus == num_visible and min_memory == 0:
        print(f"Using all {num_visible} GPUs.")
        return [f"cuda:{i}" for i in range(num_gpus)]

    # The user set CUDA_VISIBLE_DEVICES and also requested a specific number of GPUs.
    # The environment variable takes precedence, so we'll just use all visible GPUs.
    count_msg = "all" if num_gpus == num_visible else f"first {num_gpus}"
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        warnings.warn(
            f"Smart GPU selection not supported when CUDA_VISIBLE_DEVICES is set. "
            f"Will use {count_msg} visible devices."
        )
        return [f"cuda:{i}" for i in range(num_gpus)]

    # Initialize PyNVML
    try:
        pynvml.nvmlInit()
    except pynvml.NVMLError:
        warnings.warn(
            f"Unable to initialize PyNVML; are you using non-NVIDIA GPUs? Will use "
            f"{count_msg} visible devices."
        )
        return [f"cuda:{i}" for i in range(num_gpus)]

    try:
        # PyNVML and PyTorch device indices should agree when CUDA_VISIBLE_DEVICES is
        # not set. We need them to agree so that the PyNVML indices match the PyTorch
        # indices, and we don't have to do any complex error-prone conversions.
        num_installed = pynvml.nvmlDeviceGetCount()
        assert num_installed == num_visible, "PyNVML and PyTorch disagree on GPU count"

        # Set default value for `min_memory`
        if min_memory is None:
            min_device_ram = min(
                (
                    assert_type(
                        int,
                        pynvml.nvmlDeviceGetMemoryInfo(
                            pynvml.nvmlDeviceGetHandleByIndex(idx)
                        ).total,
                    )
                    for idx in range(num_installed)
                )
            )
            min_memory = int(0.9 * min_device_ram)

        # Get free memory for each GPU
        num_tries = 0
        while True:
            # check if at least `num_gpus` GPUs have at least `min_memory`
            # bytes of free memory

            try:
                # List of (-free memory, GPU index) tuples. Sorted descending by
                # free memory, then ascending by GPU index.
                memories_and_indices = sorted(
                    (
                        -int(pynvml.nvmlDeviceGetMemoryInfo(handle).free),
                        pynvml.nvmlDeviceGetIndex(handle),
                    )
                    for handle in map(
                        pynvml.nvmlDeviceGetHandleByIndex, range(num_installed)
                    )
                )
                usable_indices = [
                    index
                    for neg_mem, index in memories_and_indices
                    if -neg_mem >= min_memory
                ]
                if len(usable_indices) >= num_gpus:
                    break
                elif num_tries % 60 == 0:  # Print every 10 minutes
                    print(
                        f"Waiting for {num_gpus} GPUs with "
                        f"at least {min_memory / 10 ** 9:.2f} GB "
                        f"of free memory. {len(usable_indices)} GPUs "
                        "currently available."
                    )
            except Exception as e:
                warnings.warn(
                    f"Unable to query GPU memory: {e}. Will try again in 10 seconds."
                )

            # Wait a bit before trying again
            time.sleep(10)
            num_tries += 1
    finally:
        # make sure to shut down PyNVML
        pynvml.nvmlShutdown()

    # Indices are sorted descending by free memory, so we want the first `num_gpus`
    # items. For printing purposes, though, we sort the indices numerically.
    selection = sorted(usable_indices[:num_gpus])

    assert len(selection) == num_gpus
    print(f"Using {len(selection)} of {num_visible} GPUs: {selection}")

    return [f"cuda:{i}" for i in selection]
