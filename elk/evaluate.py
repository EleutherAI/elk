from .files import elk_cache_dir
from .training.ccs import CCS
from .training.preprocessing import load_hidden_states
from argparse import ArgumentParser
from sklearn.model_selection import train_test_split
import pickle
import torch


@torch.autocast("cuda", enabled=torch.cuda.is_available())
def evaluate(args, hiddens, labels, lr_model, ccs_model: CCS):
    assert isinstance(hiddens, torch.Tensor)

    print("Evaluating logistic regression model")
    acc_lr = lr_model.score(hiddens, labels)

    print("Evaluating CCS model")
    labels = torch.tensor(labels, device=args.device)
    x0, x1 = hiddens.to(args.device).chunk(2, dim=1)
    acc_ccs, loss_ccs = ccs_model.score((x0, x1), labels)

    print(f"accuracy_ccs {acc_ccs}")
    print(f"loss_ccs {loss_ccs}")

    print(f"accuracy_lr {acc_lr}")


def main():
    parser = ArgumentParser()
    parser.add_argument("name", type=str, help="Name of the experiment")
    parser.add_argument(
        "--transfer-to",
        type=str,
        help="Name of experiment whose hidden states to evaluate on.",
    )
    parser.add_argument("--device", type=str, help="PyTorch device to use.")
    parser.add_argument("--layer", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    # Default to CUDA iff available
    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    hidden_path = elk_cache_dir() / (args.transfer_to or args.name)
    print(f"Loading hidden states from \033[1m{hidden_path}\033[0m")  # bold
    hiddens, labels = load_hidden_states(path=hidden_path / "hiddens.pt")
    _, test_hiddens, __, test_labels = train_test_split(
        hiddens, labels, random_state=args.seed, stratify=labels
    )

    path = elk_cache_dir() / args.name
    with open(path / "lr_models.pkl", "rb") as file:
        lr_models = pickle.load(file)

    ccs_models = torch.load(path / "ccs_models.pt")
    for h, lr_model, ccs_model in zip(test_hiddens.unbind(1), lr_models, ccs_models):
        evaluate(args, h, test_labels, lr_model, ccs_model)


if __name__ == "__main__":
    main()
