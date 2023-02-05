from .extraction import extract_hiddens, PromptCollator
from .extraction.parser import get_args
from .files import get_memorable_cache_dir
from transformers import AutoModel, AutoTokenizer
import json
import torch


if __name__ == "__main__":
    args = get_args()

    # AutoModel should do the right thing here in nearly all cases. We don't actually
    # care what head the model has, since we are just extracting hidden states.
    print(f"Loading model '{args.model}'...")
    model = AutoModel.from_pretrained(args.model, torch_dtype="auto").to(args.device)
    print(f"Done. Model class: '{model.__class__.__name__}'")

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    save_dir = get_memorable_cache_dir()
    print(f"Saving results to \033[1m{save_dir}\033[0m")  # bold

    print("Loading datasets")
    collator = PromptCollator(*args.dataset, split="train")
    with open(save_dir / "hiddens.pt", "wb") as f:
        # Save the hidden states
        for state, label in extract_hiddens(args, model, tokenizer, collator):
            torch.save((state, label), f)

    with open(save_dir / "args.pkl", "w") as f:
        json.dump(vars(args), f)

    with open(save_dir / "model_config.pkl", "w") as f:
        json.dump(model.config.to_dict(), f)
