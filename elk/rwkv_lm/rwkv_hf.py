import os
import torch
from huggingface_hub import hf_hub_download
from transformers import (
    PretrainedConfig,
    PreTrainedModel,
    PreTrainedTokenizer,
    BatchEncoding,
)
from transformers.modeling_outputs import CausalLMOutput

from rwkv.model import RWKV
from rwkv.utils import PIPELINE

os.environ["RWKV_JIT_ON"] = "1"
os.environ["RWKV_CUDA_ON"] = "0"


class RWKVConfig(PretrainedConfig):
    def __init__(self, hidden_size, num_hidden_layers):
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.is_encoder_decoder = False
        self.architectures = ["RWKV-LM"]

    @staticmethod
    def from_pretrained(pretrained_model_name_or_path, **kwargs):
        path_config_maps = {
            "BlinkDL/rwkv-4-pile-1b5": {
                "hidden_size": 2048,
                "num_hidden_layers": 120,
            },
            "BlinkDL/rwkv-4-pile-3b": {
                "hidden_size": 2560,
                "num_hidden_layers": 160,
            },
            "BlinkDL/rwkv-4-pile-7b": {
                "hidden_size": 4096,
                "num_hidden_layers": 160,
            },
            "BlinkDL/rwkv-4-pile-14b": {
                "hidden_size": 5120,
                "num_hidden_layers": 200,
            },
            "BlinkDL/rwkv-4-raven": {
                "hidden_size": 5120,
                "num_hidden_layers": 200,
            }
        }
        return RWKVConfig(
            hidden_size=path_config_maps[pretrained_model_name_or_path]["hidden_size"],
            num_hidden_layers=path_config_maps[pretrained_model_name_or_path]["num_hidden_layers"],
        )


class RWKVModel(PreTrainedModel):
    def __init__(self, config, weights_path, device):
        super().__init__(config)
        strategy = f"{device} fp16"
        self.model = RWKV(model=weights_path, strategy=strategy)
        self.device_object = torch.device(device)

    def forward(
        self,
        input_ids,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        labels=None,
        output_hidden_states=None,
    ):
        inputs = input_ids[0].detach().cpu()
        token, states = self.model.forward(inputs, None)
        response = CausalLMOutput(
            logits=token.detach().clone(),
            hidden_states=states
        )

        return response

    @staticmethod
    def from_pretrained(pretrained_model_name_or_path, device):
        repo_weights_paths = {
            "BlinkDL/rwkv-4-pile-1b5": "RWKV-4-Pile-1B5-20220903-8040.pth",
            "BlinkDL/rwkv-4-pile-3b": "RWKV-4-Pile-3B-20221008-8023.pth",
            "BlinkDL/rwkv-4-pile-7b": "RWKV-4-Pile-7B-20221115-8047.pth",
            "BlinkDL/rwkv-4-pile-14b": "RWKV-4-Pile-14B-20230213-8019.pth",
            "BlinkDL/rwkv-4-raven": "RWKV-4-Raven-14B-v10-Eng99%-Other1%-20230427-ctx8192.pth",
        }

        if pretrained_model_name_or_path not in repo_weights_paths:
            raise ValueError(f"Unsupported RWKV model: {pretrained_model_name_or_path}")

        weights_path = hf_hub_download(repo_id=pretrained_model_name_or_path, filename=repo_weights_paths[pretrained_model_name_or_path])
        config = RWKVConfig.from_pretrained(pretrained_model_name_or_path)
        model = RWKVModel(config, weights_path, device)
        return model


class RWKVTokenizer(PreTrainedTokenizer):
    model_max_length = 2048

    def __init__(self, vocab_file_path="elk/rwkv_lm/20B_tokenizer.json"):
        self.pipeline = PIPELINE(None, vocab_file_path)

    def __call__(
        self,
        text=None,
        return_tensors=None,
        truncation=None,
        return_offsets_mapping=None,
        text_target=None,
        add_special_tokens=None,
    ):
        input_ids = self.encode(text)
        return BatchEncoding({"input_ids": torch.tensor([input_ids])})

    def encode(self, text):
        return self.pipeline.encode(text)

    def decode(self, token_ids):
        return self.pipeline.decode(token_ids)
