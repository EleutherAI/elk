import transformers
import transformers.modeling_outputs
from datasets import Dataset
from transformers import AutoTokenizer

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


def test_inference_server_fsdp_single():
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
    input_dataset.set_format(tyxpe="torch")
    outputs = server.map(dataset=input_dataset, closure=lambda x: x)[0]
    assert (
        type(outputs) == transformers.modeling_outputs.CausalLMOutputWithCrossAttentions
    )


def test_inference_server_fsdp_rep():
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
    input_dataset = Dataset.from_list([inputs, inputs])
    input_dataset.set_format(type="torch")
    outputs = server.map(dataset=input_dataset, closure=lambda x: x)[0]
    assert (
        type(outputs) == transformers.modeling_outputs.CausalLMOutputWithCrossAttentions
    )
