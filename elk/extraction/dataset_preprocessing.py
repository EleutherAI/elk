import numpy as np
from datasets import DatasetDict, concatenate_datasets


def undersample(dataset: DatasetDict, seed: int, label_column: str = "label"):
    """
    Balance a dataset by undersampling the majority class.

    Args:
        dataset (DatasetDict): The dataset to balance.
        label_column (str, optional):
        The column containing the labels.
        Defaults to "label".

    Returns:
        DatasetDict: The balanced dataset.
    """
    labels, counts = np.unique(dataset[label_column], return_counts=True)

    subsets = []
    for label in labels:
        subsets.append(
            dataset.filter(lambda x: x[label_column] == label).select(
                range(min(counts))
            )
        )

    return concatenate_datasets(subsets).shuffle(seed)
