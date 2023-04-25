from dataclasses import dataclass
from typing import NamedTuple

from datasets import DatasetDict, DatasetInfo, get_dataset_config_info


def extract_dataset_name_and_config(dataset_config_str: str) -> tuple[str, str]:
    """Extract the dataset name and config name from the dataset prompt."""
    ds_name, _, config_name = dataset_config_str.partition(" ")
    return ds_name, config_name

class DatasetDictWithName(NamedTuple):
    """A Datasetwith a name attribute.
    The dataset_name is the dataset (e.g. imdb)
    that was used to create the dataset
    """
    name: str
    dataset: DatasetDict


@dataclass
class DatasetInfoWithName:
    """A DatasetInfo with a dataset_name attribute.
    This class exists because the DatasetInfo class doesn't always have a
    builder_name attribute, but we need a dataset_name attribute for
    the naming datasets
    """

    dataset_name: str
    info: DatasetInfo

    @staticmethod
    def from_ds_name_and_config(
        ds_name: str,
        config_name: str | None = None,
    ) -> "DatasetInfoWithName":
        """
        If the DatasetInfo doesn't have a builder_name,
        dataset_name will be set to the dataset name.
        """
        info = get_dataset_config_info(ds_name, config_name or None)
        return DatasetInfoWithName(dataset_name=info.builder_name or ds_name, info=info)
