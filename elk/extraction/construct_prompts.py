from pathlib import Path
from promptsource.templates import DatasetTemplates
from transformers import PreTrainedTokenizerBase
import json
import pandas as pd


root_dir = Path(__file__).parent.parent
with open(root_dir / "confusion_prefixes.json", "r") as f:
    confusion_prefixes = json.load(f)
with open(root_dir / "prompts.json", "r") as f:
    prompt_dict = json.load(f)


def collate_qa_prompts(
    question: str,
    answers: list[str],
    confusion: str,
    tokenizer: PreTrainedTokenizerBase,
):
    """Combine a question with its possible answers, returning tokenized prompts.

    Returns:
        A list containing the tokenized sequences."""
    # Add ? at the end of the question if needed.
    if confusion != "normal":
        if question[-1] == " ":
            question = question[:-1] + "?"
        elif question[-1] not in ["!", ".", "?"]:
            question = question + "?"

    question = confusion_prefixes[confusion].format(question)
    if confusion != "normal":
        question += "\nA: "

    # TODO: Should we ever actually use [SEP]? The original paper used it, but
    # intuitively it seems that this should result in worse performance because
    # it could cause the model to pay less attention to the question.
    sep = tokenizer.sep_token
    if sep is None:
        sep = "\n"

    return [tokenizer(question + answer).input_ids for answer in answers]


def get_question_answer(prompter, prompt_idx: int, dataframe, data_idx):
    prompt_name = prompter.all_template_names[prompt_idx]
    prompt_template = prompter[prompt_name]
    text_label_pair = {k: v for k, v in dataframe.loc[data_idx].items()}
    actual_label = text_label_pair["label"]

    question, answer = prompt_template.apply(text_label_pair)

    if text_label_pair["label"] == 0:
        text_label_pair["label"] = 1
        _, pos_answer = prompt_template.apply(text_label_pair)
        answer_choices = [answer, pos_answer]
    elif text_label_pair["label"] == 1:
        text_label_pair["label"] = 0
        _, neg_answer = prompt_template.apply(text_label_pair)
        answer_choices = [neg_answer, answer]
    else:
        raise ValueError("Label should be 0 or 1")

    return question, answer_choices, actual_label


def construct_prompt_dataframe(
    set_name: str, dataframe, prompt_idx, tokenizer, max_num, confusion
):
    """
    According to the prompt idx and set_name, return corresponding construction
    Will change according to model type, i.e. for BERT will add [SEP] in the middle
    Return: A dataframe, with `0`, `1`, `label`, `selection`, which should be
    saved with hidden states together
    """
    from .load_utils import get_hugging_face_load_name

    prompter = DatasetTemplates(*get_hugging_face_load_name(set_name))

    result = {
        "0": [],
        "1": [],
        "label": [],
        "selection": [],
    }

    for data_idx in range(len(dataframe)):
        # early stopping if data num meets the requirement
        if len(result["label"]) >= max_num:
            break

        # TODO: FIX THESE BUGS
        # "ag-news", "dbpedia-14" don't appear in the results
        # '"piqa", mixed up questions: don't know what's going on here yet
        question, answer_choices, actual_label = get_question_answer(
            prompter, prompt_idx, dataframe, data_idx
        )
        result["label"].append(actual_label)
        result["selection"].append(answer_choices)

        prompts = collate_qa_prompts(question, answer_choices, confusion, tokenizer)
        if any([len(prompt) > tokenizer.model_max_length for prompt in prompts]):
            continue

        # append to the result
        for i in range(2):
            result[str(i)].append(concat_data[i])

    return pd.DataFrame(result)
