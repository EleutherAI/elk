import argparse

import torch
from accelerate import infer_auto_device_map

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
        max_memory={rank: "30GiB" for rank in range(WORLD_SIZE)},
    )
    print("Device map:", device_map)
    model = instantiate_model(model_str, torch_dtype="auto", device_map=device_map)

    for input_id_args in input_ids_list:
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
    args = parser.parse_args()

    main(args)
