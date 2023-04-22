import tensorflow as tf
import tensorflow_hub as hub
import torch
from transformers import (
    PretrainedConfig,
    PreTrainedModel,
    PreTrainedTokenizer,
)


class ElmoConfig(PretrainedConfig):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.hidden_size = 1024
        self.num_hidden_layers = 3
        self.is_encoder_decoder = False
        self.architectures = ["Elmo"]


class TfElmoTokenizer(PreTrainedTokenizer):
    def __call__(
        self, text, return_tensors, truncation, return_offsets_mapping, text_target
    ):
        return text

    @staticmethod
    def from_pretrained(path):
        return TfElmoTokenizer()


class TfElmoModel(PreTrainedModel):
    def __init__(self):
        super().__init__(config=ElmoConfig())
        self.elmo_model = hub.load("https://tfhub.dev/google/elmo/3").signatures[
            "default"
        ]

    def forward(
        self,
        input_ids,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        labels=None,
    ):
        embeddings = self.elmo_model(tf.constant(input_ids))
        return {
            "hidden_states": [
                torch.tensor(embeddings["word_emb"].numpy()),
                torch.tensor(embeddings["lstm_outputs1"].numpy()),
                torch.tensor(embeddings["lstm_outputs2"].numpy()),
                torch.tensor(embeddings["elmo"].numpy()),
            ]
        }

    @staticmethod
    def from_pretrained(path):
        return TfElmoModel()
