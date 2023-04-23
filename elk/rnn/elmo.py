import tensorflow as tf
import tensorflow_hub as hub
import torch
from transformers import (
    AutoTokenizer,
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


class ElmoTokenizer(PreTrainedTokenizer):
    """ "
    The ELMo tokenizer is a wrapper around the GPT-2 tokenizer since much of the extraction
    pipeline depends on the input being tensors. The ELMo TF implementaiton takes a string
    input, so the tensors are decoded within the TfElmoModel instance.
    """

    def __init__(self):
        self.internal_tokenizer = AutoTokenizer.from_pretrained("gpt2")
        self.model_max_length = self.internal_tokenizer.model_max_length

    def __call__(
        self,
        text=None,
        return_tensors=None,
        truncation=None,
        return_offsets_mapping=None,
        text_target=None,
        add_special_tokens=None,
    ):
        return self.internal_tokenizer(
            text=text,
            add_special_tokens=add_special_tokens,
            return_tensors=return_tensors,
            text_target=text_target,
            truncation=truncation,
        )


class TfElmoModel(PreTrainedModel):
    """A HF wrappper around the Tensorflow ELMo model"""

    def __init__(self):
        super().__init__(config=ElmoConfig())
        self.internal_tokenizer = AutoTokenizer.from_pretrained("gpt2")
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
        output_hidden_states=None,
    ):
        nl_inputs = [
            self.internal_tokenizer.decode(sequence_tensor)
            for sequence_tensor in input_ids
        ]
        embeddings = self.elmo_model(tf.constant(nl_inputs))
        return {
            "hidden_states": [
                torch.tensor(embeddings["word_emb"].numpy()),
                torch.tensor(embeddings["lstm_outputs1"].numpy()),
                torch.tensor(embeddings["lstm_outputs2"].numpy()),
                torch.tensor(embeddings["elmo"].numpy()),
            ]
        }
