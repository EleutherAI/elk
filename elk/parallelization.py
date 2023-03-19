import csv
import multiprocessing as mp
from functools import partial
from typing import List

from tqdm.auto import tqdm

from elk.utils.gpu_utils import select_usable_devices


def run_on_layers(func, cols, eval_output_path, cfg, ds, layers: List[int]):
    devices = select_usable_devices(cfg.max_gpus)
    num_devices = len(devices)

    with mp.Pool(num_devices) as pool, open(eval_output_path, "w") as f:
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




