import csv

import torch

from elk.training.preprocessing import load_hidden_states, normalize

from ..files import elk_cache_dir


def evaluate_reporters(args):
    for hidden_state_dir in args.hidden_states:
        hiddens, labels = load_hidden_states(
            path=elk_cache_dir() / hidden_state_dir / "validation_hiddens.pt"
        )
        assert len(set(labels)) > 1

        _, hiddens = normalize(hiddens, hiddens, args.normalization)

        reporter_root_path = (
            elk_cache_dir() / args.name / "reporters" / args.reporter_name
        )
        reporters = torch.load(
            reporter_root_path / "reporters.pt", map_location=args.device
        )

        transfer_eval = reporter_root_path / "transfer_eval"
        transfer_eval.mkdir(parents=True, exist_ok=True)

        L = hiddens.shape[1]
        layers = list(hiddens.unbind(1))
        layers.reverse()
        csv_file = transfer_eval / f"{hidden_state_dir}.csv"

        for reporter in reporters:
            reporter.eval()

            with torch.no_grad(), open(csv_file, "w") as f:
                for layer_idx, hidden_state in enumerate(layers):
                    x0, x1 = hidden_state.to(args.device).float().chunk(2, dim=-1)

                    result = reporter.score(
                        (x0, x1),
                        labels.to(args.device),
                    )
                    stats = [*result]
                    stats += [args.normalization, args.name, hidden_state_dir]

                    writer = csv.writer(f)
                    if not csv_file.exists():
                        # write column names only once
                        cols = [
                            "layer",
                            "acc",
                            "cal_acc",
                            "auroc",
                            "normalization",
                            "name",
                            "hidden_states",
                        ]
                        writer.writerow(cols)
                    writer.writerow([L - layer_idx] + [stats])

        print("Eval file generated: ", csv_file)
