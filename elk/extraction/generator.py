from dataclasses import dataclass, field
from typing import Callable, Optional, Any

import datasets


@dataclass
class _GeneratorConfig(datasets.BuilderConfig):
    generator: Optional[Callable] = None
    gen_kwargs: dict[str, Any] = field(default_factory=dict)
    features: Optional[datasets.Features] = None


class _GeneratorBuilder(datasets.GeneratorBasedBuilder):
    """Patched version of `datasets.Generator` allowing for splits besides `train`"""

    BUILDER_CONFIG_CLASS = _GeneratorConfig
    config: _GeneratorConfig

    def __init__(self, split: str, **kwargs):
        self.split = split
        super().__init__(**kwargs)

    def _info(self):
        return datasets.DatasetInfo(features=self.config.features)

    def _split_generators(self, dl_manager):
        return [
            datasets.SplitGenerator(name=self.split, gen_kwargs=self.config.gen_kwargs)
        ]

    def _generate_examples(self, **gen_kwargs):
        assert self.config.generator is not None, "generator must be specified"
        for idx, ex in enumerate(self.config.generator(**gen_kwargs)):
            yield idx, ex
