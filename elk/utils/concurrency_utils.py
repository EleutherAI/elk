from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Sequence

from elk.utils.multiprocessing_utils import A, B


def map_threadpool(
    items: Sequence[A], func: Callable[[A], B], threadpool: ThreadPoolExecutor
) -> list[B]:
    """
    Map a function over a sequence of items using a threadpool
    The output is guaranteed to be in the same order as the input items
    """
    futures = [threadpool.submit(func, item) for item in items]
    results = []
    for fut in futures:
        results.append(fut.result())
    return results
