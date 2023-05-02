import argparse
import random
import time
from threading import Thread

import torch
from accelerate import infer_auto_device_map, init_empty_weights
from tqdm import tqdm

from elk.extraction import PromptConfig
from elk.extraction.extraction import (
    Extract,
    temp_extract_input_ids_cached,
)
from elk.utils import instantiate_model
from llama_overwrite import overwrite_30b, overwrite_65b


def pad_tensors(tensors, device, pad_value=0):
    max_len = max([t.size(-1) for t in tensors])
    padded_tensors = []
    attention_masks = []
    for _t in tensors:
        t = _t.to(device)
        pad_len = max_len - t.size(-1)
        padded_tensor = torch.cat(
            [torch.full((1, pad_len), pad_value, dtype=t.dtype, device=device), t],
            dim=-1,
        )
        attention_mask = torch.cat(
            [
                torch.zeros((1, pad_len), dtype=torch.bool, device=device),
                torch.ones_like(t),
            ],
            dim=-1,
        )
        padded_tensors.append(padded_tensor)
        attention_masks.append(attention_mask)
    return torch.cat(padded_tensors, dim=0), torch.cat(attention_masks, dim=0)


def batch_ids(
    input_ids_unbatched: list[torch.Tensor], batch_size: int
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    input_ids_unbatched_sorted = sorted(input_ids_unbatched, key=lambda x: x.size(-1))
    output = []
    input_buffer = []
    for input_id_args in input_ids_unbatched_sorted:
        input_buffer.append(input_id_args)
        if len(input_buffer) == batch_size:
            batch_input_ids, attention_mask = pad_tensors(input_buffer, device=0)
            output.append((batch_input_ids, attention_mask))
            input_buffer = []
    if input_buffer:  # Process remaining input_ids in the buffer
        batch_input_ids, attention_mask = pad_tensors(input_buffer, device=0)
        output.append((batch_input_ids, attention_mask))
    return output


def inference_worker(
    model,
    batched_input_ids: list[tuple[torch.Tensor, torch.Tensor]],
    use_tqdm=False,
):
    batched_input_ids_use_tqdm: list[tuple[torch.Tensor, torch.Tensor]] = (
        tqdm(batched_input_ids, desc="Inference") if use_tqdm else batched_input_ids
    )

    for input_ids, attention_mask in batched_input_ids_use_tqdm:
        with torch.no_grad():
            model(input_ids, attention_mask=attention_mask)


def main(args):
    model_str = args.model
    min_gpu_mem = args.min_gpu_mem
    num_threads = args.threads
    use_8bit = args.use_8bit
    batch_size = args.batch_size
    use_llama_override: bool = args.use_llama_override
    print("Batch size:", batch_size)

    cfg = Extract(model=model_str, prompts=PromptConfig(datasets=["imdb"]))

    print("Extracting input ids...")
    input_ids_list = temp_extract_input_ids_cached(
        cfg=cfg, device="cpu", split_type="train"
    ) + temp_extract_input_ids_cached(cfg=cfg, device="cpu", split_type="val")
    # bring all the tensors to device 0

    print("Number of input ids:", len(input_ids_list))
    device_tensors = [t.to(0) for t in input_ids_list]
    device_tensors_batched = batch_ids(device_tensors, batch_size=batch_size)
    # shuffle so we can tqdm more accurately
    device_tensors_batched = random.sample(
        device_tensors_batched, len(device_tensors_batched)
    )
    print("Number of batches:", len(device_tensors_batched))

    print("Instantiating model...")
    used_dtype = torch.float16 if use_8bit else "auto"

    with init_empty_weights():
        # Kinda dumb but you need to first insantiate on the CPU to get the layer class
        model = instantiate_model(model_str, torch_dtype=used_dtype)

    # Hack to take into account that its 8bit
    min_gpu_mem * 2 if use_8bit else min_gpu_mem
    autodevice_map = infer_auto_device_map(model)
    print("Auto device map:", autodevice_map)

    device_map_override = (
        (
            overwrite_30b
            if "30b" in model_str
            else overwrite_65b
            if "65b" in model_str
            else {}
        )
        if use_llama_override
        else {}
    )

    # autodevice_map["lm_head"] = 0
    print("Device map overwrite:", device_map_override)
    # Then instantiate on the GPU
    model = instantiate_model(
        model_str,
        torch_dtype=used_dtype,
        device_map=device_map_override or autodevice_map,
        load_in_8bit=use_8bit,
    )
    time_start = time.time()
    time_end = time.time()
    # in minutes
    print("Compilation time:", (time_end - time_start) / 60)

    input_ids_chunks = [
        device_tensors_batched[i::num_threads] for i in range(num_threads)
    ]

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
    default_bytes = 40 * 1024 * 1024 * 1024
    parser.add_argument(
        "--min_gpu_mem", type=int, default=default_bytes, help="Min GPU memory per GPU"
    )
    parser.add_argument(
        "--threads", type=int, default=2, help="Number of threads to run"
    )
    parser.add_argument(
        "--use_8bit", type=bool, default=True, help="Whether to use 8bit"
    )
    parser.add_argument(
        "--batch_size", type=int, default=8, help="Batch size for inference"
    )
    parser.add_argument(
        "--use_llama_override",
        type=bool,
        default=False,
        help="Whether to torch compile",
    )
    args = parser.parse_args()

    main(args)
