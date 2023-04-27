import torch
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
    outputs = server.map_for_non_fsdp(dataset=input_dataset, closure=lambda x: x)
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
    except AssertionError:
        print("First output logits not equal to single model logits")
        assert torch.allclose(second_output_logits, single_model_logits, atol=1e-3)


def test_inference_server_fsdp_limited():
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
    # limit the dataset to 1
    input_dataset = input_dataset.select(range(1))
    assert len(input_dataset) == 1
    input_dataset.set_format(type="torch")
    outputs = server.map(dataset=input_dataset, closure=lambda x: x)[0]
    assert (
        type(outputs) == transformers.modeling_outputs.CausalLMOutputWithCrossAttentions
    )
