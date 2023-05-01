import argparse
import random
from threading import Thread

import torch
from accelerate import infer_auto_device_map
from tqdm import tqdm

from elk.extraction import PromptConfig
from elk.extraction.extraction import (
    Extract,
    temp_extract_input_ids_cached,
)
from elk.inference_server.fsdp import (
    get_transformer_layer_cls,
)
from elk.utils import instantiate_model


def inference_worker(model, input_ids_queue, use_tqdm=False):
    if use_tqdm:
        input_ids_queue = tqdm(input_ids_queue, desc="Inference")

    for input_id_args in input_ids_queue:
        input_id_args = input_id_args.to(0)
        with torch.no_grad():
            model(input_id_args)


def main(args):
    model_str = args.model
    num_gpus = args.num_gpus
    min_gpu_mem = args.min_gpu_mem
    num_threads = args.threads

    cfg = Extract(model=model_str, prompts=PromptConfig(datasets=["imdb"]))

    print("Extracting input ids...")
    input_ids_list = temp_extract_input_ids_cached(
        cfg=cfg, device="cpu", split_type="train"
    ) + temp_extract_input_ids_cached(cfg=cfg, device="cpu", split_type="val")

    input_ids_list = random.sample(input_ids_list, len(input_ids_list))
    print("Number of input ids:", len(input_ids_list))
    WORLD_SIZE = num_gpus

    print("Instantiating model...")
    model = instantiate_model(model_str, torch_dtype="auto")

    layer_cls = get_transformer_layer_cls(model)

    device_map = infer_auto_device_map(
        model,
        no_split_module_classes={layer_cls},
        max_memory={
            rank: min_gpu_mem if rank != 0 else min_gpu_mem / 2
            for rank in range(WORLD_SIZE)
        },
    )

    device_map["lm_head"] = 0
    print("Device map:", device_map)
    model = instantiate_model(model_str, torch_dtype="auto", device_map=device_map)

    input_ids_chunks = [input_ids_list[i::num_threads] for i in range(num_threads)]

    threads = []
    for i in range(num_threads):
        input_ids_queue = input_ids_chunks[i]
        use_tqdm = i == 0
        t = Thread(target=inference_worker, args=(model, input_ids_queue, use_tqdm))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference with specified model")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help='Model string, e.g., "huggyllama/llama-13b"',
    )
    parser.add_argument(
        "--num_gpus", type=int, default=8, help="Number of GPUs to run on"
    )
    default_bytes = 20 * 1024 * 1024 * 1024
    parser.add_argument(
        "--min_gpu_mem", type=int, default=default_bytes, help="Min GPU memory per GPU"
    )
    parser.add_argument(
        "--threads", type=int, default=2, help="Number of threads to run"
    )
    args = parser.parse_args()

    main(args)