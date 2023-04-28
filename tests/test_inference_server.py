from concurrent.futures import ThreadPoolExecutor

import pytest
import torch
import transformers
import transformers.modeling_outputs
from transformers import AutoTokenizer

from elk.inference_server.fsdp import InferenceServer, shard_seq
from elk.inference_server.fsdp_options import FSDPOptions
from elk.utils.concurrency_utils import map_threadpool


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
    outputs = server.infer(input_ids=input_ids)
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
    with pytest.raises(TypeError, match="got an unexpected keyword argument"):
        server.map([{"wrongkeyword": torch.Tensor([1, 2, 3])}], func=lambda x: x)


@pytest.mark.gpu
def test_fsdp_same_result():
    """The results should be the same as the results from a single model"""
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
    outputs = server.infer(input_ids=input_ids)
    assert (
        type(outputs) == transformers.modeling_outputs.CausalLMOutputWithCrossAttentions
    )


def test_fsdp_map():
    _dicts = [{"input_ids": torch.tensor([55] * i)} for i in range(1, 10)]
    # # make a server
    model_str = "sshleifer/tiny-gpt2"
    server = InferenceServer(
        model_str=model_str,
        num_workers=4,
        fsdp=FSDPOptions(fsdp_enabled=False),
        min_gpu_mem=0,
    )
    # run the function map on the server
    outputs_server = server.map(keywords=_dicts, func=lambda x: x)
    # the length of the 2nd dimension of the logits is equal to the number of repeats
    lengths = []
    for i, output in enumerate(outputs_server):
        lengths.append(len(output.logits))
    sorted_lengths = sorted(lengths)
    assert sorted_lengths == [1, 2, 3, 4, 5, 6, 7, 8, 9]


@pytest.mark.gpu
def test_fsdp_multithreading():
    """Test that fsdp works with multithreading"""
    _dicts = [{"input_ids": torch.tensor([55] * i)} for i in range(1, 10)]
    # make a threadpool
    threadpool = ThreadPoolExecutor(max_workers=2)
    # # make a server
    model_str = "sshleifer/tiny-gpt2"
    server = InferenceServer(
        model_str=model_str,
        num_workers=2,
        fsdp=FSDPOptions(fsdp_enabled=False),
        min_gpu_mem=0,
    )
    # run the function .one on the server
    outputs_server = map_threadpool(
        items=_dicts, func=lambda x: server.infer(**x).logits, threadpool=threadpool
    )
    # the length of the 2nd dimension of the logits is equal to the number of repeats
    lengths = [len(output) for output in outputs_server]
    assert lengths == [1, 2, 3, 4, 5, 6, 7, 8, 9]


def test_shared_seq():
    assert shard_seq([], 3) == [[], [], []]
    assert shard_seq([1, 2, 3, 4, 5, 6], 2) == [[1, 3, 5], [2, 4, 6]]
    assert shard_seq([1, 2, 3, 4, 5, 6], 4) == [[1, 5], [2, 6], [3], [4]]
    assert shard_seq([1, 2, 3, 4, 5, 6], 1) == [[1, 2, 3, 4, 5, 6]]
    assert shard_seq([1, 2, 3, 4, 5, 6], 6) == [[1], [2], [3], [4], [5], [6]]
    assert shard_seq([1, 2, 3, 4, 5, 6], 7) == [[1], [2], [3], [4], [5], [6], []]
