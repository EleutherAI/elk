import multiprocessing as mp
from typing import Callable, Sequence, Optional

from elk.utils.typing import B


def eval_thunk(func: Callable[[], B]) -> B:
    """Evaluates a thunk and returns the result."""
    return func()


def evaluate_with_processes(
    sequence: Sequence[Callable[[], B]],
    pool: Optional["mp.pool.Pool"],  # type: ignore
) -> list[B]:
    """Evaluates thunks for and return a list of the results."""
    mapper = pool.imap_unordered if pool else map
    return list(mapper(eval_thunk, sequence))
