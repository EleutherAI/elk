"""Port of `jax.tree_util` to PyTorch.

In JAX, "pytrees" are nested collections (lists, tuples, dicts) of tensors which can be
mapped over. See <https://jax.readthedocs.io/en/latest/pytrees.html>.
"""

from typing import Callable, Mapping, TypeVar

TreeType = TypeVar("TreeType")


def pytree_map(func: Callable, tree: TreeType) -> TreeType:
    """
    Recursively apply a function to all objects in a pytree, returning the results
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
