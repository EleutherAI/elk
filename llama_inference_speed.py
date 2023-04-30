from functools import partial

import torch
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

from transformers import LlamaTokenizer, LlamaForCausalLM
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
# from https://github.com/huggingface/transformers/issues/22687
# torchrun --nproc_per_node=2 --master_port=56718 run_forward.py
model_dir = "huggyllama/llama-13b"

import os
from time import perf_counter

local_rank = int(os.environ["LOCAL_RANK"])
local_world_size = int(os.environ["LOCAL_WORLD_SIZE"])

torch.cuda.set_device(torch.device(f"cuda:{local_rank}"))

torch.distributed.init_process_group(
    "nccl",
    rank=local_rank,
    world_size=local_world_size,
)
llama_auto_wrap_policy = partial(
    transformer_auto_wrap_policy,
    transformer_layer_cls={
        LlamaDecoderLayer,
    },
)

print(torch.cuda.current_device())

tokenizer = LlamaTokenizer.from_pretrained(model_dir)
model = LlamaForCausalLM.from_pretrained(
    model_dir, torch_dtype=torch.float16, low_cpu_mem_usage=True
)

model = FSDP(
    model,
    auto_wrap_policy=llama_auto_wrap_policy,
    device_id=torch.device(f"cuda:{local_rank}"),
    # sharding_strategy=sharding_strategy,
)
inputs = tokenizer(["Who is Dalai?"], return_tensors="pt")

print(inputs)
t1_start = perf_counter()
for _ in range(20):
    logits = model(**inputs).logits[:, -1, :]
t1_stop = perf_counter()
print("forward time:", t1_stop - t1_start)
print(torch.cuda.max_memory_allocated() / 1e9)
