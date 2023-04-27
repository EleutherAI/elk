import os
import torch
from huggingface_hub import hf_hub_download
from transformers import PretrainedConfig, PreTrainedModel, PreTrainedTokenizer, BatchEncoding
from transformers.modeling_outputs import CausalLMOutput

# The rwkv.model is the official build
# from rwkv.model import RWKV
# rwkv_hiddens is a custom implementation that exposes all the hidden states as layer states - written by Nora
from .rwkv_hiddens import RWKV
from rwkv.utils import PIPELINE

os.environ["RWKV_JIT_ON"] = "1"
os.environ["RWKV_CUDA_ON"] = "0"


class RWKVConfig(PretrainedConfig):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.hidden_size = 4096
        self.num_hidden_layers = 33
        self.is_encoder_decoder = False
        self.architectures = ["RWKV-LM"]


class RWKVModel(PreTrainedModel):
    def __init__(self, device):
        super().__init__(RWKVConfig())

        # TODO: Add support for specifying the parameter count through the HF path provided in the CLI args

        # 1.5b
        # weights_path = hf_hub_download(repo_id="BlinkDL/rwkv-4-pile-1b5", filename="RWKV-4-Pile-1B5-20220903-8040.pth")

        # 3b
        # weights_path = hf_hub_download(repo_id="BlinkDL/rwkv-4-pile-3b", filename="RWKV-4-Pile-3B-20221008-8023.pth")

        # 7b
        weights_path = hf_hub_download(
            repo_id="BlinkDL/rwkv-4-pile-7b",
            filename="RWKV-4-Pile-7B-20221115-8047.pth",
        )

        # 14b
        # weights_path = hf_hub_download(repo_id="BlinkDL/rwkv-4-pile-14b", filename="RWKV-4-Pile-14B-20230213-8019.pth")

        strategy = f"{device} bf16"
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
            # hidden_states=[states[-1]],
            # hidden_states=[state.detach() for state in output_states],
        )

        return response

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
        return BatchEncoding({
            "input_ids": torch.tensor([input_ids])
        })

    def encode(self, text):
        return self.pipeline.encode(text)

    def decode(self, token_ids):
        return self.pipeline.decode(token_ids)
