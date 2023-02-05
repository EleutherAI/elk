from extraction.parser import get_args
from hashlib import md5
from .extraction import extract_hiddens, PromptCollator
from .utils import elk_cache_dir
from transformers import AutoModel, AutoTokenizer
import json
import pickle
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

    run_id = md5(pickle.dumps(args)).hexdigest()
    save_dir = elk_cache_dir() / run_id
    print(f"Saving results to '{save_dir}'")

    print("Loading datasets")
    collator = PromptCollator(*args.dataset, split="train")
    with open(save_dir / "hiddens.pt", "wb") as f:
        # Save the hidden states
        for state in extract_hiddens(args, model, tokenizer, collator):
            torch.save(state, f)

    with open(save_dir / "args.pkl", "w") as f:
        json.dump(vars(args), f)

    with open(save_dir / "model_config.pkl", "w") as f:
        json.dump(model.config.to_dict(), f)
