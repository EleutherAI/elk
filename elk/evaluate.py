from .training.ccs import CCS
from .eval.utils_evaluation import load_hidden_states
from .eval.parser import get_args
from .files import elk_cache_dir
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


if __name__ == "__main__":
    args = get_args()

    hiddens, labels = load_hidden_states(
        path=elk_cache_dir() / args.name / "hiddens.pt",
        reduce=args.mode,
    )
    _, test_hiddens, __, test_labels = train_test_split(
        hiddens, labels, random_state=args.seed, stratify=labels
    )

    path = elk_cache_dir() / args.name
    with open(path / "lr_models.pkl", "rb") as file:
        lr_models = pickle.load(file)

    ccs_models = torch.load(path / "ccs_models.pt")
    for h, lr_model, ccs_model in zip(test_hiddens.unbind(1), lr_models, ccs_models):
        evaluate(args, h, test_labels, lr_model, ccs_model)
