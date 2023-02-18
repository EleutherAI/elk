from ..utils import pytree_map
from .prompt_collator import Prompt, PromptCollator
from einops import rearrange
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import BatchEncoding, PreTrainedModel, PreTrainedTokenizerBase
from typing import cast, Literal, Iterator, Sequence
import torch
from dataclasses import dataclass
from datasets import Dataset
import multiprocess as mp

@dataclass
class ExtractionParameters:
    model: PreTrainedModel
    tokenizer: PreTrainedTokenizerBase
    collator: PromptCollator
    indices: int
    batch_size: int = 1
    layers: Sequence[int] = ()
    prompt_suffix: str = ""
    token_loc: Literal["first", "last", "mean"] = "last"
    use_encoder_states: bool = False


def get_device_ids(local_rank: int, local_world_size: int) -> list[int]:
    """
    Splits devices among local_world_size processes and returns their ids.
    """
    devices_per_proc = torch.cuda.device_count() // local_world_size
    device_ids = list(range(local_rank * devices_per_proc, (local_rank + 1) * devices_per_proc))
    return device_ids


def uniform_split(elements: list, num_splits: int) -> Iterator[list]:
    """
    Splits input list as evenly as possible among num_splits splits. No elements are excluded.
    """
    num_per_split = [len(elements) // num_splits] * num_splits

    remaining = len(elements) % num_splits
    for i in range(remaining):
        num_per_split[i] += 1
    
    start_idx = 0
    for split_size in num_per_split:
        yield elements[start_idx : start_idx + split_size]
        start_idx += split_size


def extract_hiddens(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    collator: PromptCollator,
    *,
    # TODO: Bring back auto-batching when we have a good way to prevent excess padding
    batch_size: int = 1,
    layers: Sequence[int] = (),
    prompt_suffix: str = "",
    token_loc: Literal["first", "last", "mean"] = "last",
    use_encoder_states: bool = False,
    num_procs: int = 1
):
    """Run inference on a model with a set of prompts, yielding the hidden states."""

    # Dataset.from_generator expects a list >= num_procs
    # This wraps given parameters into a list of ExtractionParameters with length num_procs
    all_indices = list(range(len(collator)))

    # Samples are split among processes here, instead of through Dataset.from_generator
    all_proc_indices = uniform_split(all_indices, num_procs)

    all_params = []

    for proc_indices in all_proc_indices:
        params = ExtractionParameters(
            model=model,
            tokenizer=tokenizer,
            collator=collator,
            indices=proc_indices,
            batch_size=batch_size,
            layers=layers,
            prompt_suffix=prompt_suffix,
            token_loc=token_loc,
            use_encoder_states=use_encoder_states
        )
        
        all_params.append(params)
    
    # each list needs to have length num_proc
    multiprocess_kwargs = {
        'wrapped_params': all_params,
        'wrapped_rank': list(range(num_procs)),
        'wrapped_num_procs': [num_procs] * num_procs
    }

    mp.set_start_method("spawn")

    return Dataset.from_generator(
        _extract_hiddens_process,
        gen_kwargs=multiprocess_kwargs,
        num_proc=num_procs
    )


@torch.no_grad()
def _extract_hiddens_process(
    wrapped_params: list[ExtractionParameters],
    wrapped_rank: list[int],
    wrapped_num_procs: list[int]
) -> Iterator[dict]:
    """
    Internal function for inference on a model with a set of prompts on a single process.
    To be passed to Dataset.from_generator.
    """

    # Dataset.from_generator splits input kwargs into lists
    params = wrapped_params[0]
    rank = wrapped_rank[0]
    num_procs = wrapped_num_procs[0]

    device_ids = get_device_ids(rank, num_procs)

    print(f'Process rank {rank} using {len(device_ids)} GPUs.')

    # TODO: multi-GPU processes (i.e., sharded models) support
    if len(device_ids) > 1:
        raise ValueError('Only one GPU per process is supported.')

    model = params.model.to(device_ids[0])

    num_choices = len(params.collator.labels)

    # TODO: Make this configurable or something
    # Token used to separate the question from the answer
    sep_token = params.tokenizer.sep_token or "\n"

    # TODO: Maybe also make this configurable?
    # We want to make sure the answer is never truncated
    params.tokenizer.truncation_side = "left"
    if not params.tokenizer.pad_token:
        params.tokenizer.pad_token = params.tokenizer.eos_token

    def tokenize(strings: list[str]):
        return pytree_map(
            lambda x: x.to(device_ids[0]),
            params.tokenizer(
                strings,
                padding=True,
                return_tensors="pt",
                truncation=True,
            ),
        )

    # This function returns the flattened questions and answers, and the labels for
    # each question-answer pair. After inference we need to reshape the results.
    def collate(prompts: list[Prompt]) -> tuple[BatchEncoding, list[int]]:
        choices = [
            prompt.to_string(i, sep=sep_token) + params.prompt_suffix
            for prompt in prompts
            for i in range(num_choices)
        ]
        return tokenize(choices), [prompt.label for prompt in prompts]

    def collate_enc_dec(
        prompts: list[Prompt],
    ) -> tuple[BatchEncoding, BatchEncoding, list[int]]:
        tokenized_questions = tokenize(
            [prompt.question for prompt in prompts for _ in range(num_choices)]
        )
        tokenized_answers = tokenize(
            [
                prompt.answers[i] + params.prompt_suffix
                for prompt in prompts
                for i in range(num_choices)
            ]
        )
        labels = [prompt.label for prompt in prompts]
        return tokenized_questions, tokenized_answers, labels

    def reduce_seqs(
        hiddens: list[torch.Tensor], attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Reduce sequences of hiddens into single vectors."""

        # Unflatten the hiddens
        hiddens = [rearrange(h, "(b c) l d -> b c l d", c=num_choices) for h in hiddens]

        if params.token_loc == "first":
            hiddens = [h[..., 0, :] for h in hiddens]
        elif params.token_loc == "last":
            # Because of padding, the last token is going to be at a different index
            # for each example, so we use gather.
            B, C, _, D = hiddens[0].shape
            lengths = attention_mask.sum(dim=-1).view(B, C, 1, 1)
            indices = lengths.sub(1).expand(B, C, 1, D)
            hiddens = [h.gather(index=indices, dim=-2).squeeze(-2) for h in hiddens]
        elif params.token_loc == "mean":
            hiddens = [h.mean(dim=-2) for h in hiddens]
        else:
            raise ValueError(f"Invalid token_loc: {params.token_loc}")

        if params.layers:
            hiddens = [hiddens[i] for i in params.layers]

        # [batch size, layers, num choices, hidden size]
        return torch.stack(hiddens, dim=1)

    # If this is an encoder-decoder model and we're passing the answer to the encoder,
    # we don't need to run the decoder at all. Just strip it off, making the problem
    # equivalent to a regular encoder-only model.
    is_enc_dec = model.config.is_encoder_decoder
    if is_enc_dec and params.use_encoder_states:
        # This isn't actually *guaranteed* by HF, but it's true for all existing models
        if not hasattr(model, "get_encoder") or not callable(model.get_encoder):
            raise ValueError(
                "Encoder-decoder model doesn't have expected get_encoder() method"
            )

        model = cast(PreTrainedModel, model.get_encoder())

    # Whether to concatenate the question and answer before passing to the model.
    # If False pass them to the encoder and decoder separately.
    should_concat = not is_enc_dec or params.use_encoder_states

    dl = DataLoader(
        params.collator,
        batch_size=params.batch_size,
        collate_fn=collate if should_concat else collate_enc_dec,
    )

    # Iterating over questions
    for batch in tqdm(dl):
        # Condition 1: Encoder-decoder transformer, with answer in the decoder
        if not should_concat:
            questions, answers, labels = batch
            outputs = model(
                **questions,
                **{f"decoder_{k}": v for k, v in answers.items()},
                output_hidden_states=True,
            )
            # [batch_size, num_layers, num_choices, hidden_size]
            yield {
                'hiddens': torch.stack(outputs.decoder_hidden_states, dim=2),
                'labels': labels
            }

        # Either a decoder-only transformer or a transformer encoder
        else:
            choices, labels = batch

            # Skip the input embeddings which are unlikely to be interesting
            h = model(**choices, output_hidden_states=True).hidden_states[1:]
            yield {
                'hiddens': reduce_seqs(h, choices["attention_mask"]),
                'labels': labels
            }
