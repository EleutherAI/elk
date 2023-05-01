import argparse

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


def main(args):
    model_str = args.model
    num_gpus = args.num_gpus
    min_gpu_mem = args.min_gpu_mem
    cfg = Extract(
        model=model_str,
        prompts=PromptConfig(datasets=["imdb"])
        # run on all layers, tiny-gpt only has 2 layers
    )
    print("Extracting input ids...")
    input_ids_list = temp_extract_input_ids_cached(
        cfg=cfg, device="cpu", split_type="train"
    ) + temp_extract_input_ids_cached(cfg=cfg, device="cpu", split_type="val")
    print("Number of input ids:", len(input_ids_list))
    WORLD_SIZE = num_gpus

    print("Instantiating model...")
    model = instantiate_model(model_str, torch_dtype="auto")

    layer_cls = get_transformer_layer_cls(model)

    device_map = infer_auto_device_map(
        model,
        no_split_module_classes={layer_cls},
        max_memory={
            # lesser mem for device 0 so we can run larger batch sizes
            # and because we are going to assign lm_head to it
            rank: min_gpu_mem if rank != 0 else min_gpu_mem / 2
            for rank in range(WORLD_SIZE)
        },
    )
    # explicitly set the lm head to be device 0 so
    # see https://github.com/huggingface/accelerate/issues/362
    device_map["lm_head"] = 0
    print("Device map:", device_map)
    model = instantiate_model(model_str, torch_dtype="auto", device_map=device_map)
    input_ids_to_run = tqdm(input_ids_list, desc="Inference")
    for input_id_args in input_ids_to_run:
        # GPU 0 is the input guy.. i guess?
        input_id_args = input_id_args.to(0)
        with torch.no_grad():
            # do nothing
            model(input_id_args)


if __name__ == "__main__":
    # e.g. python llama_fsdp.py --model huggyllama/llama-13b
    parser = argparse.ArgumentParser(description="Run inference with specified model")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help='Model string, e.g., "huggyllama/llama-13b"',
    )
    # --num_gpus default 8
    parser.add_argument(
        "--num_gpus", type=int, default=8, help="Number of GPUs to run on"
    )
    # 20GiB in bytes
    default_bytes = 20 * 1024 * 1024 * 1024
    parser.add_argument(
        "--min_gpu_mem", type=int, default=default_bytes, help="Min GPU memory per GPU"
    )
    args = parser.parse_args()

    main(args)
