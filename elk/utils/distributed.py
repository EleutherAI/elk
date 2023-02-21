"""Utilities for code that works both inside and outside a distributed environment."""

from torch import nn, Tensor
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist


def maybe_all_gather(x: Tensor) -> Tensor:
    """Concatenate `x` across all ranks along dim 0 if needed.

    Does nothing if `torch.distributed.is_initialized() is False`.
    """

    if not dist.is_initialized():
        return x

    buffer = x.new_empty([dist.get_world_size() * x.shape[0], *x.shape[1:]])
    dist.all_gather_into_tensor(buffer, x)
    return buffer


def maybe_barrier() -> None:
    """Wait for all ranks to reach this point if needed.

    Does nothing if `torch.distributed.is_initialized() is False`.
    """
    if dist.is_initialized():
        dist.barrier()


def maybe_all_reduce(x: Tensor) -> Tensor:
    """Average the tensor across all ranks if needed.

    Does nothing if `torch.distributed.is_initialized() is False`.
    """

    if not dist.is_initialized():
        return x

    dist.all_reduce(x, op=dist.ReduceOp.SUM)
    x /= dist.get_world_size()
    return x


def maybe_ddp_wrap(model: nn.Module) -> nn.Module:
    """Wrap `model` with `DistributedDataParallel` if needed.

    Does nothing if `torch.distributed.is_initialized() is False`.
    """
    return DDP(model, device_ids=[dist.get_rank()]) if dist.is_initialized() else model
