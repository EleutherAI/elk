import csv

import torch

from elk.training.preprocessing import load_hidden_states, normalize
from ..files import elk_cache_dir


def evaluate(args):
    """
    Note: eval is a reserved keyword in python, therefore we use evaluate instead
    """
    for hidden_state_dir in args.hidden_states:
        hiddens, labels = load_hidden_states(
            path=elk_cache_dir() / hidden_state_dir / "validation_hiddens.pt"
        )
        assert len(set(labels)) > 1

        # TODO: We actually don't need to normalize the train hidden states here (?)
        _, hiddens = normalize(hiddens, hiddens, args.normalization)

        reporters = torch.load(elk_cache_dir() / args.reporters / "reporters.pt")
        L = hiddens.shape[1]

        statistics = []
        for reporter in reporters:
            reporter.eval()

            layers = list(hiddens.unbind(1))
            layers.reverse()

            with torch.no_grad():
                for hidden_state in layers:
                    x0, x1 = hidden_state.to(args.device).float().chunk(2, dim=-1)

                    result = reporter.score(
                        (x0, x1),
                        labels.to(args.device),
                    )
                    stats = [*result]
                    stats.append(hidden_state)
                    statistics.append(stats)

        cols = [
            "layer",
            "acc",
            "cal_acc",
            "auroc",
            "normalization",
            "traind on dataset",
            "eval on dataset",
        ]

        args.eval_dir.mkdir(parents=True, exist_ok=True)
        with open(args.eval_dir / f"{hidden_state_dir}_eval.csv", "w") as f:
            writer = csv.writer(f)
            writer.writerow(cols)

            # TODO: Fix stats, there are some problems with the tensors
            for i, stats in enumerate(statistics):
                breakpoint()
                a = []
                for s in stats:
                    breakpoint()
                    a.append(s)
                breakpoint()
                # [f"{s:.4f}" for s in stats]
                writer.writerow([L - i])
