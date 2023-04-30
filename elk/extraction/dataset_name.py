from typing import NamedTuple

from datasets import DatasetDict


def parse_dataset_string(dataset_config_str: str) -> tuple[str, str]:
    """Extract the dataset name and config name from the dataset prompt."""
    ds_name, _, config_name = dataset_config_str.partition(":")
    return ds_name, config_name


class DatasetDictWithName(NamedTuple):
    """A Datasetwith a name attribute.
    The dataset_name is the dataset (e.g. imdb)
    that was used to create the dataset
    """

    name: str
    dataset: DatasetDict
