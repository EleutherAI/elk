import csv
from functools import partial
from typing import List, Tuple
from omegaconf import DictConfig
import multiprocessing as mp

from elk.utils.gpu_utils import select_usable_devices


def execute_in_parallel(func, cfg, ds, layers: List[int]) -> List[Tuple[int, float, float, float, float]]:
    devices = select_usable_devices(cfg.max_gpus)
    num_devices = len(devices)

    with mp.Pool(num_devices) as pool, open(path, "w") as f:
        fn = partial(
            func, cfg, ds, devices=devices, world_size=num_devices
        )
        writer = csv.writer(f)
        writer.writerow(cols)

        mapper = pool.imap_unordered if num_devices > 1 else map
        row_buf = []
        try:
            for i, *stats in tqdm(mapper(fn, layers), total=len(layers)):
                row_buf.append([i] + [f"{s:.4f}" for s in stats])
        finally:
            # Make sure the CSV is written even if we crash or get interrupted
            for row in sorted(row_buf):
                writer.writerow(row)




