import random
from dataclasses import dataclass

from datasets import load_dataset


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


def row_to_example(row, id):
    case = random.choice([0, 1, 2, 3])
    match case:
        case 0:
            first = f"{row['target_true']}"
            second = f" not{row['target_true']}"
            label = 0
        case 1:
            first = f" not{row['target_false']}"
            second = f"{row['target_false']}"
            label = 0
        case 2:
            first = f" not{row['target_true']}"
            second = f"{row['target_true']}"
            label = 1
        case 3:
            first = f"{row['target_false']}"
            second = f" not{row['target_false']}"
            label = 1

    prompts = [
        Prompt(
            choices=[
                Choice(question=row["prompt"], answer=first),
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


if __name__ == "__main__":
    from rich import print

    examples = get_neel_examples()
    first_example = examples[0:10]
    print(first_example)
