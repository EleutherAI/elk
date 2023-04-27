from concurrent.futures import ThreadPoolExecutor
from typing import Sequence, Callable

import torch
import transformers
import transformers.modeling_outputs
from datasets import Dataset
from transformers import AutoTokenizer

from elk.multiprocessing import A, B
from elk.utils import pytree_map
from elk.utils.fsdp import InferenceServer


def test_pytree_map():
    _ids = [15496, 11, 616, 3290, 318, 13779]
    inputs = {"input_ids": _ids}
    input_dataset = Dataset.from_dict(inputs)
    result: Dataset = pytree_map(lambda x: x, input_dataset)
    # assert we got the same thing back
    assert _ids == result["input_ids"]


def test_inference_server_normal():
    model_str = "sshleifer/tiny-gpt2"
    server = InferenceServer(model_str=model_str, num_workers=2)
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


def test_inference_server_fsdp_one():
    model_str = "sshleifer/tiny-gpt2"
    server = InferenceServer(
        model_str=model_str, num_workers=2, fsdp=True, cpu_offload=True
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


def test_inference_server_fsdp_other_map_imp():
    model_str = "sshleifer/tiny-gpt2"
    single_model = transformers.AutoModelForCausalLM.from_pretrained(model_str)
    server = InferenceServer(
        model_str=model_str, num_workers=2, fsdp=True, cpu_offload=True
    )
    print("Started inference server")
    # encode this text into input ids
    text_one = "Hello, my dog is cute"
    text_two = "Hello world!"

    input_ids_one = AutoTokenizer.from_pretrained(model_str).encode(
        text_one, return_tensors="pt"
    )
    input_ids_two = AutoTokenizer.from_pretrained(model_str).encode(
        text_two, return_tensors="pt"
    )
    # Make sure we only pass the input_ids that the model expects
    # make the dict a dataset, while still making it a pytorch dataset
    input_dataset = Dataset.from_list(
        [{"input_ids": input_ids_one}, {"input_ids": input_ids_two}]
    )
    input_dataset.set_format(type="torch")
    outputs = server.map(dataset=input_dataset, closure=lambda x: x)
    assert len(outputs) == 2
    first_output = outputs[0]
    assert (
        type(first_output)
        == transformers.modeling_outputs.CausalLMOutputWithCrossAttentions
    )
    # assert that the first output's logits is equal
    first_output_logits = first_output.logits
    second_output_logits = outputs[1].logits
    single_model_logits = single_model(input_ids=input_ids_one).logits
    try:
        assert torch.allclose(first_output_logits, single_model_logits, atol=1e-3)
    except Exception:
        print("First output logits not equal to single model logits")
        assert torch.allclose(second_output_logits, single_model_logits, atol=1e-3)


def test_inference_server_fsdp_limited():
    model_str = "sshleifer/tiny-gpt2"
    server = InferenceServer(
        model_str=model_str, num_workers=2, fsdp=False, cpu_offload=True
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
    input_dataset = Dataset.from_list([inputs, inputs])
    # limit the dataset to 1
    input_dataset = input_dataset.select(range(1))
    assert len(input_dataset) == 1
    input_dataset.set_format(type="torch")
    outputs = server.map(dataset=input_dataset, closure=lambda x: x)[0]
    assert (
        type(outputs) == transformers.modeling_outputs.CausalLMOutputWithCrossAttentions
    )


def ordered_map_threads(
    items: Sequence[A], func: Callable[[A], B], threadpool: ThreadPoolExecutor
) -> list[B]:
    """
    Map a function over a sequence of items using a threadpool
    """
    futures = [threadpool.submit(func, item) for item in items]
    results = []
    for fut in futures:
        results.append(fut.result())
    return results


def test_fsdp_multithreading():
    # make items repeat input_ids 1, but with ascending number of repeats
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
        model_str=model_str, num_workers=2, fsdp=False, cpu_offload=True
    )
    # run the function .one on the server
    outputs_server = ordered_map_threads(
        items=items, func=lambda x: server.one(x).logits, threadpool=threadpool
    )
    # assert that the length of the 2nd dimension of the logits is equal to the number of repeats
    for i, output in enumerate(outputs_server):
        assert output.shape[1] == i + 1


def test_tiny_gpt_diff_logits():
    tiny_gpt = transformers.AutoModelForCausalLM.from_pretrained("sshleifer/tiny-gpt2")
    first_input_ids = torch.tensor([[1, 2, 3]])
    second_input_ids = torch.tensor([[1, 2, 3, 4, 5]])
    first_logits = tiny_gpt(input_ids=first_input_ids, output_hidden_states=True).logits
    second_logits = tiny_gpt(
        input_ids=second_input_ids, output_hidden_states=True
    ).logits
    assert not torch.allclose(first_logits, second_logits, atol=1e-3)
