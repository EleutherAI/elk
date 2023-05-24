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


def row_to_example(row):
    label = 0
    template_names = ["template_null"]
    prompts = [
        Prompt(
            choices=[
                Choice(question=row["prompt"], answer=row["target_true"]),
                Choice(question=row["prompt"], answer=row["target_false"]),
            ]
        )
    ]

    example = Example(label=label, prompts=prompts, template_names=template_names)
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

    # Get the first row
    first_row = dataset["train"][0]

    # Get the values for "prompt" and "answer"
    first_row["prompt"]
    first_row["target_true"]
    first_row["target_false"]

    # for row in dataset["train"]:
    #     example = row_to_example(row)
    #     print(example)

    examples = [row_to_example(row) for row in dataset["train"]]
    inverted_list = [
        invert_example(example) if index % 2 == 0 else example
        for index, example in enumerate(examples)
    ]

    return inverted_list


if __name__ == "__main__":
    examples = get_neel_examples()
    first_example = examples[0]
    print(first_example)
    prompt = first_example.prompts[0]
    print(prompt)
    prompt[0]
