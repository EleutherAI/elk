import subprocess
from typing import Optional

models = {
    # "Deberta": "microsoft/deberta-v2-xxlarge-mnli",
    "UQA": "allenai/unifiedqa-t5-11b",
    "GPTJ": "EleutherAI/gpt-j-6B",
    "T5": "t5-11b",
    "Roberta": "roberta-large-mnli",
    "T0": "bigscience/T0pp",
}
datasets = {
    "imdb": ["imdb"],
    "amazon-polarity": ["amazon_polarity"],
    "ag-news": ["ag_news"],
    "dbpedia-14": ["dbpedia_14"],
    "copa": ["super_glue", "copa"],
    "rte": ["super_glue", "rte"],
    "boolq": ["super_glue", "boolq"],
    "qnli": ["glue", "qnli"],
    "piqa": ["piqa"],
    "story-cloze": ["story_cloze", "2016"],
}


def run_extraction_origin():
    for model_short, model_name in models.items():
        for dataset_short, dataset_list in datasets.items():
            print("\n=====================================\n")
            print(f"Running {model_short} on {dataset_short}")
            original_args: dict[str, Optional[str]] = {
                "--max-examples": "1000",
                # "--max-examples": "30",
                "--prompts": "all",
                "--layers": "-1",
                # "--name": f"{model_short}@{dataset_short}_small",
                "--name": f"{model_short}@{dataset_short}",
                "--val-frac": "0.4",
                "--balance": "True",
                "--use-encoder-states": None,  # use encoder states when possible
            }
            if dataset_short == "story-cloze":
                original_args["--label-column"] = "answer_right_ending"

            args = [model_name, *dataset_list]
            for k, v in original_args.items():
                args.append(k)
                if v is not None:
                    args.append(v)

            subprocess.run(["python", "-m", "elk", "extract", *args])
