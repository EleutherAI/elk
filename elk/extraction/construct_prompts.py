from pathlib import Path
from promptsource.templates import DatasetTemplates
import json
import pandas as pd


filter_length = 500

label_dict = {
    "imdb": ["negative", "positive"],  # This is for normal IMDB
    "amazon-polarity": ["negative", "positive"],
    "ag-news": ["politics", "sports", "business", "technology"],
    "dbpedia-14": [
        "company",
        "educational institution",
        "artist",
        "athlete",
        "office holder",
        "mean of transportation",
        "building",
        "natural place",
        "village",
        "animal",
        "plant",
        "album",
        "film",
        "written work",
    ],
    "cola": ["choice 1", "choice 2"],
    "rte": ["yes", "no"],  # whether entail
    "boolq": ["false", "true"],
    "qnli": ["yes", "no"],  # represent whether entail
    "piqa": ["solution 1", "solution 2"],
    "story-cloze": ["choice 1", "choice 2"],
}


root_dir = Path(__file__).parent.parent
with open(root_dir / "confusion_prefixes.json", "r") as f:
    confusion_prefixes = json.load(f)
with open(root_dir / "prompts.json", "r") as f:
    prompt_dict = json.load(f)


def check_length_exceeded(tokenizer, str_list):
    for s in str_list:
        if len(tokenizer(s).input_ids) > filter_length:
            return True
    return False


def concat_answer(question, answer, mdl_name, confusion):
    # Add a ? at the end of the question if not;
    # Add an A: before the answer.
    if confusion != "normal":
        if question[-1] == " ":
            question = question[:-1] + "?"
        elif question[-1] not in ["!", ".", "?"]:
            question = question + "?"

    question = confusion_prefixes[confusion].format(question)
    if confusion != "normal":
        question = question + "\nA: "

    if "gpt" not in mdl_name and "roberta" not in mdl_name:
        # Does not have `\n` token, should replace to '''
        # TODO: check whether this is valid
        question = question.replace("\n", " ")
    if answer == "":  # null one, don't do anything
        return question

    # for bert model, should add [SEP]
    if "deberta" in mdl_name:
        return question + " [SEP] " + answer
    elif "roberta" in mdl_name:
        return question + "</s></s>" + answer
    else:  # T5 based moel
        if question[-1] != "\n" and question[-1] != " ":
            sep = "\n" if "gpt" in mdl_name else " "
            return question + sep + answer
        return question + answer


def get_question_answer(prompter, prompt_idx, dataframe, data_idx):
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

    return question, answer_choices, actual_label


def construct_prompt_dataframe(
    set_name, dataframe, prompt_idx, mdl_name, tokenizer, max_num, confusion
):
    """
    According to the prompt idx and set_name, return corresponding construction
    Will change according to model type, i.e. for BERT will add [SEP] in the middle
    Return: A dataframe, with `null`, `0`, `1`, `label`, `selection`, which should be
    saved with hidden states together
    """
    from extraction.load_utils import get_hugging_face_load_name

    prompter = DatasetTemplates(*get_hugging_face_load_name(set_name))

    result = {
        "null": [],
        "0": [],
        "1": [],
        "label": [],
        "selection": [],
    }

    for data_idx in range(len(dataframe)):
        # early stopping if data num meets the requirement
        if len(result["null"]) >= max_num:
            break

        # TODO: FIX THESE BUGS
        # "ag-news", "dbpedia-14" don't appear in the results
        # '"piqa", mixed up questions: don't know what's going on here yet
        question, answer_choices, actual_label = get_question_answer(
            prompter, prompt_idx, dataframe, data_idx
        )
        concat_data = [
            concat_answer(question, answer, mdl_name, confusion)
            for answer in answer_choices
        ]

        if check_length_exceeded(tokenizer, concat_data):
            continue

        # append to the result
        result["null"].append(concat_answer(question, "", mdl_name, confusion))
        result["label"].append(actual_label)
        result["selection"].append(answer_choices)
        for i in range(2):
            result[str(i)].append(concat_data[i])

    return pd.DataFrame(result)
