from .extraction import extract_hiddens, PromptCollator
from ..files import args_to_uuid, elk_cache_dir
from ..training.preprocessing import silence_datasets_messages
from ..utils import maybe_all_gather
from transformers import AutoModel, AutoTokenizer
import json
import torch
import torch.distributed as dist


def run(args):
    rank = dist.get_rank() if dist.is_initialized() else 0

    def extract(args, split: str):
        frac = 1 - args.val_frac if split == "train" else args.val_frac

        collator = PromptCollator(
            *args.dataset,
            max_examples=round(args.max_examples * frac) if args.max_examples else 0,
            split=split,
            label_column=args.label_column,
            strategy=args.prompts,
        )
        print(f"Class balance '{split}': {[f'{x:.2%}' for x in collator.label_fracs]}")
        if split == "train":
            prompt_names = collator.prompter.all_template_names
            if args.prompts == "all":
                print(f"Using {len(prompt_names)} prompts per example: {prompt_names}")
            elif args.prompts == "randomize":
                print(f"Randomizing over {len(prompt_names)} prompts: {prompt_names}")
            else:
                raise ValueError(f"Unknown prompt strategy: {args.prompts}")

        items = [
            (features, labels)
            for features, labels in extract_hiddens(
                model,
                tokenizer,
                collator,
                layers=args.layers,
                prompt_suffix=args.prompt_suffix,
                token_loc=args.token_loc,
                use_encoder_states=args.use_encoder_states,
            )
        ]
        save_dir.mkdir(parents=True, exist_ok=True)

        with open(save_dir / f"{split}_hiddens.pt", "wb") as f:
            hidden_batches, label_batches = zip(*items)
            hiddens = maybe_all_gather(torch.cat(hidden_batches))  # type: ignore

            # Moving labels to GPU just to be able to use maybe_all_gather
            labels = torch.tensor(sum(label_batches, []), device=hiddens.device)
            labels = maybe_all_gather(labels)  # type: ignore

            if rank == 0:
                torch.save((hiddens.cpu(), labels.cpu()), f)

    # AutoModel should do the right thing here in nearly all cases. We don't actually
    # care what head the model has, since we are just extracting hidden states.
    print(f"Loading model '{args.model}'...")
    model = AutoModel.from_pretrained(args.model, torch_dtype="auto").to(args.device)
    print(f"Done. Model class: '{model.__class__.__name__}'")

    if args.use_encoder_states and not model.config.is_encoder_decoder:
        raise ValueError(
            "--use_encoder_states is only compatible with encoder-decoder models."
        )

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    # If the user didn't specify a name, we'll use a hash of the CLI args
    if not args.name:
        args.name = args_to_uuid(args)

    save_dir = elk_cache_dir() / args.name
    print(f"Saving results to \033[1m{save_dir}\033[0m")  # bold

    print("Loading datasets")
    silence_datasets_messages()

    extract(args, "train")
    extract(args, "validation")

    if rank == 0:
        with open(save_dir / "args.json", "w") as f:
            json.dump(vars(args), f)

        with open(save_dir / "model_config.json", "w") as f:
            json.dump(model.config.to_dict(), f)
