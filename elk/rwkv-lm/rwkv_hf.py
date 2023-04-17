import os
from rwkv.model import RWKV
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer, GPT2TokenizerFast, PreTrainedModel, PretrainedConfig

os.environ["RWKV_JIT_ON"] = '1'
os.environ["RWKV_CUDA_ON"] = '0'

class RWKVConfig(PretrainedConfig):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.hidden_size = 2048
        self.num_hidden_layers = 120
        self.is_encoder_decoder = False
        self.architectures = ["RWKV-LM"]

class RWKVModel(PreTrainedModel):
    def __init__(self):
        super().__init__(RWKVConfig())
        weights_path = "/home/kyle/HF-MODEL/rwkv-4-pile-1b5/models--BlinkDL--rwkv-4-pile-1b5/snapshots/6ea995eaa87a17af560c9b41ce1a3d92355c5a49/RWKV-4-Pile-1B5-20220903-8040.pth"
        self.model = RWKV(model=weights_path, strategy='cuda fp16')

    def forward(
        self,
        input_ids,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        labels=None,
    ):
        _, state = self.model.forward(input_ids, None)
        return state

    # @staticmethod
    # def from_pretrained(pretrained_model_name_or_path):
    #     weights_path = "/home/kyle/HF-MODEL/rwkv-4-pile-1b5/models--BlinkDL--rwkv-4-pile-1b5/snapshots/6ea995eaa87a17af560c9b41ce1a3d92355c5a49/RWKV-4-Pile-1B5-20220903-8040.pth"
    #     model = RWKVModel(weights_path)
    #     return model
