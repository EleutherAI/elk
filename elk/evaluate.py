from .training.ccs import CCS
from .eval.utils_evaluation import load_hidden_states
from .eval.parser import get_args
from .files import elk_cache_dir
from sklearn.model_selection import train_test_split
import pickle
import torch


def evaluate(args, lr_model, ccs_model: CCS):
    args.save_dir.mkdir(parents=True, exist_ok=True)

    hiddens, labels = load_hidden_states(
        path=elk_cache_dir() / args.name,
        reduce=args.mode,
    )
    _, test_hiddens, __, test_labels = train_test_split(
        hiddens, labels, random_state=args.seed, stratify=labels
    )
    assert isinstance(test_hiddens, torch.Tensor)

    print("Evaluating logistic regression model")
    acc_lr = lr_model.score(test_hiddens, test_labels)

    print("Evaluating CCS model")
    labels = torch.tensor(labels, device=args.device)
    x0, x1 = test_hiddens.to(args.device).chunk(2, dim=1)
    acc_ccs, loss_ccs = ccs_model.score((x0, x1), labels)

    print(f"accuracy_ccs {acc_ccs}")
    print(f"loss_ccs {loss_ccs}")

    print(f"accuracy_lr {acc_lr}")


if __name__ == "__main__":
    args = get_args()

    with open(args.trained_models_path / "lr_model.pkl", "rb") as file:
        lr_model = pickle.load(file)

    ccs_model = CCS.load(args.trained_models_path / "ccs_model.pt")
    evaluate(args, lr_model, ccs_model)
