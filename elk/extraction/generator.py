from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Callable

from datasets import (
    BuilderConfig,
    DatasetInfo,
    Features,
    GeneratorBasedBuilder,
    SplitInfo,
)
from datasets.splits import NamedSplit


@dataclass
class _GeneratorConfig(BuilderConfig):
    generator: Callable | None = None
    gen_kwargs: dict[str, Any] = field(default_factory=dict)
    features: Features | None = None

    def create_config_id(
        self, config_kwargs: dict, custom_features: Features | None
    ) -> str:
        config_kwargs = deepcopy(config_kwargs)

        # By default the values in gen_kwargs are lists of length world_size. We want
        # to erase the world_size dimension so that the config id is the same no matter
        # how many processes are used. We also remove the explicit device, rank, and
        # world_size keys.
        config_kwargs["gen_kwargs"] = {
            k: v[0]
            for k, v in config_kwargs.get("gen_kwargs", {}).items()
            if k not in ("device", "rank", "world_size")
        }
        return super().create_config_id(config_kwargs, custom_features)


@dataclass
class _SplitGenerator:
    """
    datasets.SplitGenerator but use given `split_info` instead initializing a new one
    """

    name: str
    split_info: SplitInfo
    gen_kwargs: dict = field(default_factory=dict)

    def __post_init__(self):
        self.name = str(self.name)  # Make sure we convert NamedSplits in strings
        NamedSplit(self.name)  # check that it's a valid split name


class _GeneratorBuilder(GeneratorBasedBuilder):
    """Patched version of `datasets.Generator` allowing for splits besides `train`"""

    BUILDER_CONFIG_CLASS = _GeneratorConfig
    config: _GeneratorConfig

    def __init__(
        self,
        split_name: str,
        split_info: SplitInfo,
        **kwargs,
    ):
        self.split_name = split_name
        self.split_info = split_info

        super().__init__(**kwargs)

    def _info(self):
        # Use the same builder and config name as the original builder
        return DatasetInfo(features=self.config.features)

    def _split_generators(self, dl_manager):
        return [
            _SplitGenerator(
                name=self.split_name,
                split_info=self.split_info,
                gen_kwargs=self.config.gen_kwargs,
            )
        ]

    def _generate_examples(self, **gen_kwargs):
        assert self.config.generator is not None, "generator must be specified"

        for idx, ex in enumerate(self.config.generator(**gen_kwargs)):
            yield idx, ex
