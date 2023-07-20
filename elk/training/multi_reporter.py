from dataclasses import dataclass
from pathlib import Path

import torch as t

from elk.training import CcsReporter
from elk.training.common import Reporter

AnyReporter = CcsReporter | Reporter


@dataclass
class ReporterTrainResult:
    reporter: AnyReporter
    train_loss: float | None
    prompt_index: int | None


class MultiReporter:
    def __init__(self, reporter_results: list[ReporterTrainResult]):
        self.reporter_results: list[ReporterTrainResult] = reporter_results
        self.reporters = [r.reporter for r in reporter_results]
        train_losses = (
            [r.train_loss for r in reporter_results]
            if reporter_results[0].train_loss is not None
            else None
        )
        self.train_loss = (
            sum(train_losses) / len(train_losses) if train_losses is not None else None
        )

    def __call__(self, h):
        num_variants = h.shape[1]
        assert len(self.reporters) == num_variants
        credences = []
        for i, reporter in enumerate(self.reporters):
            credences.append(reporter(h[:, [i], :, :]))
        return t.stack(credences, dim=0).mean(dim=0)

    @staticmethod
    def load(path: Path, layer: int, device: str):
        prompt_folders = [p for p in path.iterdir() if p.is_dir()]
        reporters = []
        for folder in prompt_folders:
            path = folder / "reporters" / f"layer_{layer}.pt"
            reporter = t.load(path, map_location=device)
            reporters.append(reporter)
        # TODO for now I don't care about the train losses
        return MultiReporter([ReporterTrainResult(r, None) for r in reporters])
