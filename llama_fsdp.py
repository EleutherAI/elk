import os
import argparse
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm

from elk.extraction import PromptConfig
from elk.extraction.extraction import temp_extract_input_ids, Extract
from elk.utils import instantiate_model


def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def run_inference(rank, world_size, model, input_ids_list):
    setup(rank, world_size)
    torch.cuda.set_device(rank)

    model.to(rank)
    model.eval()

    if rank == 0:
        input_ids_list = tqdm(input_ids_list, desc="Inference")

    for input_id_args in input_ids_list:
        input_id_args = input_id_args.to(rank)
        with torch.no_grad():
            output = model(input_id_args)

    cleanup()


def main(args):
    model_str = args.model
    cfg = Extract(
        model=model_str,
        prompts=PromptConfig(datasets=["imdb"])
        # run on all layers, tiny-gpt only has 2 layers
    )
    print("Extracting input ids...")
    input_ids_list = temp_extract_input_ids(
        cfg=cfg, device="cpu", split_type="train"
    ) + temp_extract_input_ids(cfg=cfg, device="cpu", split_type="test")
    print("Instantiating model...")
    model = instantiate_model(model_str, torch_dtype="auto")
    model = FSDP(model)

    WORLD_SIZE = 8

    mp.spawn(
        run_inference,
        args=(WORLD_SIZE, model, input_ids_list),
        nprocs=WORLD_SIZE,
        join=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run FSDP inference with specified model"
    )
    parser.add_argument(
        "--model", type=str, required=True, help='Model string, e.g., "llama_7b"'
    )
    args = parser.parse_args()

    main(args)
