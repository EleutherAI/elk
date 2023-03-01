from dataclasses import dataclass
from typing import Callable, Optional, Any

import datasets


@dataclass
class _GeneratorConfig(datasets.BuilderConfig):
    generator: Optional[Callable] = None
    gen_kwargs: Optional[dict[str, Any]] = None
    features: Optional[datasets.Features] = None

    def __post_init__(self):
        assert self.generator is not None, "generator must be specified"

        if self.gen_kwargs is None:
            self.gen_kwargs = {}


class _GeneratorBuilder(datasets.GeneratorBasedBuilder):
    BUILDER_CONFIG_CLASS = _GeneratorConfig

    def __init__(self, split: str, **kwargs):
        self.split = split
        super().__init__(**kwargs)

    def _info(self):
        return datasets.DatasetInfo(features=self.config.features)  # type: ignore

    def _split_generators(self, dl_manager):
        return [
            datasets.SplitGenerator(
                name=self.split, gen_kwargs=self.config.gen_kwargs  # type: ignore
            )
        ]

    def _generate_examples(self, **gen_kwargs):
        for idx, ex in enumerate(self.config.generator(**gen_kwargs)):  # type: ignore
            yield idx, ex
