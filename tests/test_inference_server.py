from concurrent.futures import ThreadPoolExecutor

import pytest
import torch
import transformers
import transformers.modeling_outputs
from datasets import Dataset
from transformers import AutoTokenizer

from elk.inference_server.fsdp_options import FSDPOptions
from elk.utils.concurrency_utils import map_threadpool
from elk.inference_server.fsdp import InferenceServer


def test_inference_server_non_fsdp():
    model_str = "sshleifer/tiny-gpt2"
    server = InferenceServer(
        model_str=model_str,
        num_workers=2,
        fsdp=FSDPOptions(fsdp_enabled=False),
        min_gpu_mem=0,
    )
    print("Started inference server")
    # encode this text into input ids
    text = "Hello, my dog is cute"
    input_ids = AutoTokenizer.from_pretrained(model_str).encode(
        text, return_tensors="pt"
    )
    # Make sure we only pass the arguments that the model expects
    inputs = dict(input_ids=input_ids)
    # make the dict a dataset, while still making it a pytorch dataset
    input_dataset = Dataset.from_dict(inputs)
    input_dataset.set_format(type="torch")
    outputs = server.map(dataset=input_dataset, closure=lambda x: x)[0]
    assert (
        type(outputs) == transformers.modeling_outputs.CausalLMOutputWithCrossAttentions
    )


def test_inference_server_propagates_error():
    """Test that the server propagates errors from workers to the main process"""
    model_str = "sshleifer/tiny-gpt2"
    server = InferenceServer(
        model_str=model_str,
        num_workers=2,
        fsdp=FSDPOptions(fsdp_enabled=False),
        min_gpu_mem=0,
    )
    # encode this text into input ids
    text = "Hello, my dog is cute"
    input_ids = AutoTokenizer.from_pretrained(model_str).encode(
        text, return_tensors="pt"
    )
    # Make sure we only pass the arguments that the model expects
    inputs = dict(wrongargument=input_ids)
    # make the dict a dataset, while still making it a pytorch dataset
    input_dataset = Dataset.from_dict(inputs)
    input_dataset.set_format(type="torch")
    # GPT2LMHeadModel.forward() got an unexpected keyword argument 'wrongargument'
    with pytest.raises(TypeError, match="got an unexpected keyword argument"):
        outputs = server.map(dataset=input_dataset, closure=lambda x: x)

@pytest.mark.gpu
def test_fsdp_same_result():
    """Test that the results from fsdp are the same as the results from a single model"""
    model_str = "sshleifer/tiny-gpt2"
    server = InferenceServer(
        model_str=model_str,
        num_workers=2,
        fsdp=FSDPOptions(fsdp_enabled=True),
        min_gpu_mem=0,
    )
    print("Started inference server")
    # encode this text into input ids
    text = "Hello, my dog is cute"
    input_ids = AutoTokenizer.from_pretrained(model_str).encode(
        text, return_tensors="pt"
    )
    # Make sure we only pass the arguments that the model expects
    inputs = dict(input_ids=input_ids)
    # make the dict a dataset, while still making it a pytorch dataset
    input_dataset = Dataset.from_dict(inputs)
    input_dataset.set_format(type="torch")
    outputs = server.map(dataset=input_dataset, closure=lambda x: x)[0]
    assert (
        type(outputs) == transformers.modeling_outputs.CausalLMOutputWithCrossAttentions
    )

@pytest.mark.gpu
def test_fsdp_multithreading():
    """Test that fsdp works with multithreading"""
    _dicts = [{"input_ids": torch.tensor([[55] * i])} for i in range(1, 10)]
    items: list[Dataset] = [
        Dataset.from_dict({"input_ids": torch.tensor([[1] * i])}) for i in range(1, 10)
    ]
    for item in items:
        item.set_format(type="torch")
    # make a threadpool
    threadpool = ThreadPoolExecutor(max_workers=2)
    # # make a server
    model_str = "sshleifer/tiny-gpt2"
    server = InferenceServer(
        model_str=model_str,
        num_workers=2,
        fsdp=FSDPOptions(fsdp_enabled=True),
        min_gpu_mem=0,
    )
    # run the function .one on the server
    outputs_server = map_threadpool(
        items=items, func=lambda x: server.one(x).logits, threadpool=threadpool
    )
    # assert that the length of the 2nd dimension of the logits is equal to the number of repeats
    for i, output in enumerate(outputs_server):
        assert output.shape[1] == i + 1
