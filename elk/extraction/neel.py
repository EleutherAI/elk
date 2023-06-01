import concurrent.futures
import os
import random
import time
from dataclasses import dataclass
from sys import argv

import openai
from datasets import IterableDatasetDict, load_dataset
from rich import print
from tqdm import tqdm

TRY_LIMIT = 3


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


def generate_inverted_prompts(args):
    print(f"Generating inverted prompt for {args}")
    prompt, ans_true, ans_false, i = args
    start_time = time.time()
    prompt1 = """Negate the two sentences or questions and
    don't say anything else:
    Autonomous University of Madrid, which is located in Spain
    Autonomous University of Madrid, which is located in England
    """
    ans1 = "Autonomous University of Madrid, which is not located in Spain\nAutonomous University of Madrid, which is not located in England"
    print(f"{prompt}{ans_true}\n{prompt}{ans_false}\n")
    for try_no in range(TRY_LIMIT):
        try:
            print(f'Generating prompt for {f"{prompt}[{ans_true}/{ans_false}]"}')
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt1},
                    {"role": "assistant", "content": ans1},
                    {
                        "role": "user",
                        "content": f"{prompt}{ans_true}\n{prompt}{ans_false}",
                    },
                ],
            )
            req_time = time.time() - start_time
            print(f"Request time: {req_time}")
            # Extract the generated inverted prompt from the API response
            print(response)
            inverted_prompt = response.choices[0].message.content.strip()
            inverted_true = inverted_prompt.split("\n")[0]
            inverted_false = inverted_prompt.split("\n")[1]
        except Exception as err:
            if try_no < TRY_LIMIT:
                req_time = time.time() - start_time
                print(
                    f"Error try {try_no}, stack trace: {err} took {req_time},\nRetrying..."
                )
            else:
                return "ERROR", "ERROR"
        else:
            return inverted_true, inverted_false
    return "ERROR", inverted_prompt or "ERROR"


def get_and_save_neel_inverted_by_lm():
    # Load dataset
    dataset: IterableDatasetDict = load_dataset("NeelNanda/counterfact-tracing")

    # Set up the OpenAI API
    openai.api_key = os.environ["OPENAI_API_KEY"]
    print("OpenAI API key:", openai.api_key)

    # Open the file in write mode
    FILE_PATH = "inverted_prompts.tsv"
    batch_size = 30

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
        tqdm(range(start, end))

        subset = dataset["train"].select(range(start, end))
        jobs = [
            (data["prompt"], data["target_true"], data["target_false"], i)
            for i, data in zip(range(start, end), subset)
        ]
        indices = range(start, end)
        import threading

        lock = threading.Lock()

        with concurrent.futures.ProcessPoolExecutor(max_workers=batch_size) as executor:
            # future_to_inverted = {executor.submit(generate_inverted_prompts, (data['prompt'], data['target_true'], data['target_false'], i)): data for i, data in enumerate(dataset["train"].select(range(start, end)))}
            future_to_inverted = {
                executor.submit(generate_inverted_prompts, job): (data, i)
                for job, data, i in zip(jobs, subset, indices)
            }

            for future in concurrent.futures.as_completed(future_to_inverted):
                with lock:
                    try:
                        inverted_true, inverted_false = future.result()
                    except Exception as exc:
                        lock.release()
                        print("B")
                        print(exc)
                        break
                    else:
                        (data, i) = future_to_inverted[future]
                        try:
                            target_true = data["target_true"]
                            target_false = data["target_false"]
                            prompt = data["prompt"]
                            relation_id = i
                            f.write(
                                f"{relation_id}\t{prompt}\t{target_true}\t{target_false}"
                                f"\t{inverted_true}\t{inverted_false}\n"
                            )
                            f.flush()
                        except Exception:
                            # print('A')
                            # print(f"small_i: {smallest_i}, large_i: {largest_i}, j: {j}, completed: {completed.keys()}")
                            # print(f"Queue: {completed.keys()}, next_batch: {next_batch}")
                            # print(err)

                            # write error to file
                            inverted_true = inverted_true
                            inverted_false = inverted_false
                            target_true = "ERR"
                            target_false = "ERR"
                            prompt = "ERR"
                            relation_id = i
                            f.write(
                                f"{relation_id}\t{prompt}\t{target_true}\t{target_false}"
                                f"\t{inverted_true}\t{inverted_false}\n"
                            )
                            f.flush()

if __name__ == "__main__":
    get_and_save_neel_inverted_by_lm()
