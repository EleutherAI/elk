import csv

import torch

from elk.training.preprocessing import load_hidden_states, normalize


def evaluate(args):
    """
    Note: eval is a reserved keyword in python, therefore we use evaluate instead
    """
    for hidden_states_path in args.hidden_states_path:
        hiddens, labels = load_hidden_states(path=hidden_states_path)
        assert len(set(labels)) > 1

        # TODO: We actually don't need to normalize the train hidden states here (?)
        _, hiddens = normalize(hiddens, hiddens, args.normalization)

        ccs_models = torch.load(args.ccs_models_path)
        L = hiddens.shape[1]
        
        statistics = []
        for ccs_model in ccs_models:
            ccs_model.eval()

            layers = list(hiddens.unbind(1))

            with torch.no_grad():
                for hidden_state in layers:
                    x0, x1 = hidden_state.to(args.device).float()

                    result = ccs_model.score(
                        (x0, x1),
                        labels.to(args.device),
                    )
                    stats = [*result]
                    stats.append(hidden_states_path)
                    statistics.append(stats)

        print(statistics)

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
        with open(args.eval_dir / hidden_states_path, "w") as f:
            writer = csv.writer(f)
            writer.writerow(cols)

            for i, stats in enumerate(statistics):
                writer.writerow([L - i] + [f"{s:.4f}" for s in stats])
