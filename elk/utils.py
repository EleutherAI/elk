from .files import elk_cache_dir
from torch import Tensor
from torch.nn.parallel import DistributedDataParallel as DDP
from typing import Callable, Mapping, TypeVar
import torch.distributed as dist
import torch.nn as nn
import os


def exists_hiddens(args) -> bool:
    cache_dir = elk_cache_dir() / args.name
    return os.path.exists(cache_dir / "train_hiddens") and os.path.exists(
        cache_dir / "validation_hiddens"
    )


def maybe_all_gather(x: Tensor) -> Tensor:
    if not dist.is_initialized():
        return x

    buffer = x.new_empty([dist.get_world_size() * x.shape[0], *x.shape[1:]])
    dist.all_gather_into_tensor(buffer, x)
    return buffer


def maybe_all_reduce(x: Tensor) -> Tensor:
    if not dist.is_initialized():
        return x

    dist.all_reduce(x, op=dist.ReduceOp.SUM)
    x /= dist.get_world_size()
    return x


def maybe_ddp_wrap(model: nn.Module) -> nn.Module:
    if not dist.is_initialized():
        return model

    return DDP(model, device_ids=[dist.get_rank()])


TreeType = TypeVar("TreeType")


# Port of jax.tree_util.tree_map to PyTorch. In JAX, "pytrees" are nested collections
# (lists, tuples, dicts) of tensors which can be mapped over.
# See <https://jax.readthedocs.io/en/latest/pytrees.html>.
def pytree_map(func: Callable, tree: TreeType) -> TreeType:
    """
    Recursively apply a function to all tensors in a pytree, returning the results
    in a new pytree with the same structure.

    Examples:
    >>> pytree_map(lambda x: x + 1, {"x": 7, "y": 42})
    {'x': 8, 'y': 43}
    """
    # Recursive case
    if isinstance(tree, Mapping):
        return {k: pytree_map(func, v) for k, v in tree.items()}  # type: ignore

    if isinstance(tree, list):
        return [pytree_map(func, v) for v in tree]  # type: ignore

    if isinstance(tree, tuple):
        return tuple(pytree_map(func, v) for v in tree)  # type: ignore

    # Stopping condition
    return func(tree)
