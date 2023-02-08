from .extraction import extract_hiddens, PromptCollator
from .extraction.parser import get_args
from .files import elk_cache_dir, memorable_cache_dir
from .training.preprocessing import silence_datasets_messages
from transformers import AutoModel, AutoTokenizer
import json
import torch


def extract(args, split: str):
    frac = 1 - args.val_frac if split == "train" else args.val_frac

    collator = PromptCollator(
        *args.dataset,
        max_examples=round(args.max_examples * frac) if args.max_examples else 0,
        split=split,
        label_column=args.label_column,
        strategy=args.prompts,
    )
    print(f"Class balance for '{split}': {[f'{x:.2%}' for x in collator.label_fracs]}")
    if split == "train":
        prompt_names = collator.prompter.all_template_names
        if args.prompts == "all":
            print(f"Using {len(prompt_names)} prompts per example: {prompt_names}")
        elif args.prompts == "randomize":
            print(f"Randomizing over {len(prompt_names)} prompts: {prompt_names}")
        else:
            raise ValueError(f"Unknown prompt strategy: {args.prompts}")

    items = [
        (features.cpu(), labels)
        for features, labels in extract_hiddens(args, model, tokenizer, collator)
    ]
    with open(save_dir / f"{split}_hiddens.pt", "wb") as f:
        hidden_batches, label_batches = zip(*items)
        hiddens = torch.cat(hidden_batches)  # type: ignore
        labels = sum(label_batches, [])
        torch.save((hiddens, labels), f)


if __name__ == "__main__":
    args = get_args()

    # AutoModel should do the right thing here in nearly all cases. We don't actually
    # care what head the model has, since we are just extracting hidden states.
    print(f"Loading model '{args.model}'...")
    model = AutoModel.from_pretrained(args.model, torch_dtype="auto").to(args.device)
    print(f"Done. Model class: '{model.__class__.__name__}'")

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    save_dir = elk_cache_dir() / args.name if args.name else memorable_cache_dir()
    print(f"Saving results to \033[1m{save_dir}\033[0m")  # bold

    print("Loading datasets")
    silence_datasets_messages()

    extract(args, "train")
    extract(args, "validation")

    with open(save_dir / "args.json", "w") as f:
        json.dump(vars(args), f)

    with open(save_dir / "model_config.json", "w") as f:
        json.dump(model.config.to_dict(), f)
