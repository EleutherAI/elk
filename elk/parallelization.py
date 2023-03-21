from abc import ABC
import csv
import multiprocessing as mp
from functools import partial
from pathlib import Path
from typing import Callable, List, Optional, Union

from tqdm.auto import tqdm
from elk.utils.gpu_utils import select_usable_devices

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from elk.evaluation.evaluate import Eval
    from elk.training.train import Elicit

def run_on_layers(func: Callable, cols: List[str], out_dir: Path, cfg: Union["Elicit", "Eval"], ds, layers: List[int]):
    devices = select_usable_devices(cfg.max_gpus)
    num_devices = len(devices)

    with mp.Pool(num_devices) as pool, open(out_dir / "eval.csv", "w") as f:
        fn = partial(
            func, cfg, ds, out_dir, devices=devices, world_size=num_devices
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




