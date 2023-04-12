from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional

import datasets
from datasets import Features
from datasets.splits import NamedSplit


@dataclass
class _GeneratorConfig(datasets.BuilderConfig):
    generator: Optional[Callable] = None
    gen_kwargs: dict[str, Any] = field(default_factory=dict)
    features: Optional[datasets.Features] = None

    def create_config_id(
        self, config_kwargs: dict, custom_features: Features | None
    ) -> str:
        # These are implementation details that don't need to be hashed into the id
        # TODO: Make this customizable or something? Right now we're just hard-coding
        # these values from extraction.py. OTOH maybe it's not terrible because this is
        # a private class anyway.
        gen_kwargs = config_kwargs.get("gen_kwargs", {})
        gen_kwargs.pop("device", None)
        gen_kwargs.pop("rank", None)
        gen_kwargs.pop("world_size", None)

        return super().create_config_id(config_kwargs, custom_features)


@dataclass
class _SplitGenerator:
    """
    datasets.SplitGenerator but use given `split_info` instead initializing a new one
    """

    name: str
    split_info: datasets.SplitInfo
    gen_kwargs: Dict = field(default_factory=dict)

    def __post_init__(self):
        self.name = str(self.name)  # Make sure we convert NamedSplits in strings
        NamedSplit(self.name)  # check that it's a valid split name


class _GeneratorBuilder(datasets.GeneratorBasedBuilder):
    """Patched version of `datasets.Generator` allowing for splits besides `train`"""

    BUILDER_CONFIG_CLASS = _GeneratorConfig
    config: _GeneratorConfig

    def __init__(self, split_name: str, split_info: datasets.SplitInfo, **kwargs):
        self.split_name = split_name
        self.split_info = split_info

        super().__init__(**kwargs)

    def _info(self):
        return datasets.DatasetInfo(features=self.config.features)

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
