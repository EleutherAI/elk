from dataclasses import dataclass
from pathlib import Path

import torch as t

from elk.training import CcsReporter
from elk.training.common import Reporter

SingleReporter = CcsReporter | Reporter


@dataclass
class ReporterWithInfo:  # I don't love this name but I have no choice because
    # of the other Reporter
    model: SingleReporter
    train_loss: float | None = None
    prompt_index: int | None = None


class MultiReporter:
    def __init__(self, reporter: list[ReporterWithInfo]):
        assert len(reporter) > 0, "Must have at least one reporter"
        self.reporter_w_infos: list[ReporterWithInfo] = reporter
        self.models = [r.model for r in reporter]
        train_losses = (
            [r.train_loss for r in reporter]
            if reporter[0].train_loss is not None
            else None
        )
        self.train_loss = (
            sum(train_losses) / len(train_losses) if train_losses is not None else None
        )

    def __call__(self, h):
        num_variants = h.shape[1]
        assert len(self.models) == num_variants
        credences = []
        for i, reporter in enumerate(self.models):
            credences.append(reporter(h[:, [i], :, :]))
        return t.stack(credences, dim=0).mean(dim=0)

    @staticmethod
    def load(path: Path, layer: int, device: str):
        prompt_folders = [p for p in path.iterdir() if p.is_dir()]
        reporters = [
            (
                t.load(folder / "reporters" / f"layer_{layer}.pt", map_location=device),
                int(folder.name.split("_")[-1]),  # prompt index
            )
            for folder in prompt_folders
        ]
        # we don't care about the train losses for evaluating
        return MultiReporter([ReporterWithInfo(r, None, pi) for r, pi in reporters])
