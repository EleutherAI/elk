import os
import random
import time
from dataclasses import dataclass
from sys import argv

import openai
from datasets import IterableDatasetDict, load_dataset
from rich import print
from tqdm import tqdm


@dataclass
class Choice:
    question: str
    answer: str


@dataclass
class Prompt:
    choices: list[Choice]

    def __len__(self):
        return len(self.choices)

    def __iter__(self):
        yield from self.choices

    def __getitem__(self, index):
        return self.choices[index]


@dataclass
class Example:
    label: int
    prompts: list[Prompt]
    template_names: list[str]


@dataclass(frozen=True)
class NeelRow:
    prompt: str
    target_true: str
    target_false: str

    @staticmethod
    def from_ds(row):
        return NeelRow(
            prompt=row["prompt"],
            target_true=row["target_true"],
            target_false=row["target_false"],
        )


def row_to_example(row, id):
    case = random.choice([0, 3])
    match case:
        case 0:
            first = f"{row['target_true']}"
            second = f" not{row['target_true']}"
            label = 0
        # case 1:
        #     first = f" not{row['target_false']}"
        #     second = f"{row['target_false']}"
        #     label = 0
        # case 2:
        #     first = f" not{row['target_true']}"
        #     second = f"{row['target_true']}"
        #     label = 1
        case 3:
            first = f"{row['target_false']}"
            second = f" not{row['target_false']}"
            label = 1

    prompts = [
        Prompt(
            choices=[
                Choice(
                    question=row["prompt"], answer=first
                ),  # model(question + answer)
                Choice(question=row["prompt"], answer=second),
            ]
        )
    ]

    example = Example(label=label, prompts=prompts, template_names=["template_null"])
    return example


def invert_example(example):
    choices = (example.prompts[0].choices[1], example.prompts[0].choices[0])
    return Example(
        label=1 if example.label == 0 else 0,
        prompts=[Prompt(choices=choices)],
        template_names=example.template_names,
    )


def get_neel_examples():
    # Load the Counterfact-Tracing dataset
    dataset = load_dataset("NeelNanda/counterfact-tracing")
    examples = [row_to_example(row, i) for (i, row) in enumerate(dataset["train"])]
    # inverted_list = [
    #     invert_example(example) if index % 2 == 0 else example
    #     for index, example in enumerate(examples)
    # ]

    return examples


def get_and_save_neel_inverted_by_lm():
    # Load dataset
    dataset: IterableDatasetDict = load_dataset("NeelNanda/counterfact-tracing")

    # Set up the OpenAI API
    openai.api_key = os.environ["OPENAI_API_KEY"]
    print("OpenAI API key:", openai.api_key)

    # Open the file in write mode
    FILE_PATH = "inverted_prompts.tsv"
    with open(FILE_PATH, "a") as f:
        # Write the header if it exists
        if os.path.getsize(FILE_PATH) == 0:
            f.write(
                "relation_id\toriginal_prompt\toriginal_target_true\t"
                "original_target_false\tinverted_prompt_true\tinverted_prompt_false\n"
            )

        # Iterate through the dataset and generate inverted prompts
        start = int(argv[1])
        end = int(argv[2])  # len(dataset['train'])
        bar = tqdm(range(start, end))
        for i in bar:
            data = dataset["train"][i]
            target_true = data["target_true"]
            target_false = data["target_false"]
            prompt = data["prompt"]
            relation_id = data["relation_id"]

            instruct = f"""
Negate the following sentences and don't say anything else:
"{prompt + target_true}"
"{prompt + target_false}"
            """
            inverted = generate_inverted_prompt(instruct)
            bar.set_description(instruct + inverted)

            try:
                inverted_true = inverted.split("\n")[0]
                inverted_false = inverted.split("\n")[1]

                # Write to the file
                f.write(
                    f"{i}\t{prompt}\t{target_true}\t{target_false}"
                    f"\t{inverted_true}\t{inverted_false}\n"
                )
                f.flush()
            except:  # noqa: E722
                print(f"Error: {relation_id} {inverted}")
                continue


def generate_inverted_prompt(prompt):
    # Call the ChatGPT API to generate inverted prompts
    prompt1 = """Negate the following sentences or questions and
    don't say anything else:
    Autonomous University of Madrid, which is located in Spain
    Autonomous University of Madrid, which is located in England
    """
    ans1 = (
        "Autonomous University of Madrid, which not located in Spain\n"
        "Autonomous University of Madrid, which not located in England"
    )
    prompt2 = prompt
    TRY_LIMIT = 3
    for try_no in range(TRY_LIMIT):
        start_time = time.time()
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt1},
                    {"role": "assistant", "content": ans1},
                    {"role": "user", "content": prompt2},
                ],
            )
            req_time = time.time() - start_time
            print(f"Request time: {req_time}")
            # Extract the generated inverted prompt from the API response
            inverted_prompt = response.choices[0].message.content.strip()
        except Exception as err:
            if try_no < TRY_LIMIT:
                req_time = time.time() - start_time
                print(
                    f"Error try {try_no}, stack trace: {err},"
                    "took {req_time},\nRetrying..."
                )
                continue
            else:
                print(f"Error try {try_no}, stack trace: {err}\nSorry, goodbye.")
                raise err
        else:
            return inverted_prompt


if __name__ == "__main__":
    get_and_save_neel_inverted_by_lm()
