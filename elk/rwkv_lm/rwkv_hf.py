import os
from huggingface_hub import hf_hub_download
from transformers.modeling_outputs import CausalLMOutput
from transformers import PretrainedConfig, PreTrainedModel

# The rwkv.model is the official build 
# from rwkv.model import RWKV

# rwkv_hiddens is a custom implementation that exposes all the hidden states as layer states - written by Nora
from .rwkv_hiddens import RWKV

os.environ["RWKV_JIT_ON"] = "1"
os.environ["RWKV_CUDA_ON"] = "0"


class RWKVConfig(PretrainedConfig):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.hidden_size = 2048
        self.num_hidden_layers = 25
        self.is_encoder_decoder = False
        self.architectures = ["RWKV-LM"]


class RWKVModel(PreTrainedModel):
    def __init__(self):
        super().__init__(RWKVConfig())
        
        # TODO: Add support for specifying the parameter count through the HF path provided in the CLI args
        
        # 1.5b
        # weights_path = hf_hub_download(repo_id="BlinkDL/rwkv-4-pile-1b5", filename="RWKV-4-Pile-1B5-20220903-8040.pth")
        
        # 3b
        # weights_path = hf_hub_download(repo_id="BlinkDL/rwkv-4-pile-3b", filename="RWKV-4-Pile-3B-20221008-8023.pth")
        
         # 7b
        weights_path = hf_hub_download(repo_id="BlinkDL/rwkv-4-pile-7b", filename="RWKV-4-Pile-7B-20221115-8047.pth")
        
        # 14b
        # weights_path = hf_hub_download(repo_id="BlinkDL/rwkv-4-pile-14b", filename="RWKV-4-Pile-14B-20230213-8019.pth")
        
        self.model = RWKV(model=weights_path, strategy="cuda bf16")

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
        inputs = input_ids.detach().cpu()
        token, states = self.model.forward(inputs, None)
        mock_embedding_state = states[0].clone()
        output_states = [mock_embedding_state] + states
        response = CausalLMOutput(
            logits=token.detach().clone(),
            hidden_states=[state.detach() for state in output_states],
        )
        return response

    # @staticmethod
    # def from_pretrained(pretrained_model_name_or_path):
    #     weights_path = "/home/kyle/HF-MODEL/rwkv-4-pile-1b5/models--BlinkDL--rwkv-4-pile-1b5/snapshots/6ea995eaa87a17af560c9b41ce1a3d92355c5a49/RWKV-4-Pile-1B5-20220903-8040.pth"
    #     model = RWKVModel(weights_path)
    #     return model
