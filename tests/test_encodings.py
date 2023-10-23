from datasets import load_dataset
from transformers import AutoTokenizer

from elk.extraction import Extract, tokenize_dataset


def test_get_encodings():
    dataset_name = "imdb"
    model_path = "sshleifer/tiny-gpt2"

    seed = 42
    cfg = Extract(
        model=model_path,
        datasets=(dataset_name,),
        max_examples=(10, 10),
        template_path="_default",
        get_lm_preds=True,
        statement_column="text",
        balance=False,
        seed=seed,
    )
    split_type = "train"
    encodings = tokenize_dataset(cfg, split_type)

    tokenizer = AutoTokenizer.from_pretrained(model_path, truncation_side="left")
    ds = load_dataset(dataset_name, split=split_type)
    ds = ds.add_column("row_id", range(len(ds)))  # type: ignore
    ds = ds.shuffle(seed=seed).select(range(10))  # type: ignore

    suffix = '\n\n\nQ: Is the above statement "True" or "False"?\n\nA:'
    suffix_tokens = tokenizer.encode(suffix, add_special_tokens=False)

    def map_fn(ex: dict) -> dict:
        out_record = {
            "row_id": ex["row_id"],
            "label": ex["label"],
            "variant_id": "_default",
            "text": ex["text"] + suffix,
            "num_suffix_tokens": len(suffix_tokens),
        }
        input_ids = tokenizer(ex["text"], add_special_tokens=True)["input_ids"]
        out_record["input_ids"] = [input_ids + suffix_tokens]  # type: ignore
        answer_ids = [
            tokenizer.encode(s, add_special_tokens=False)[0] for s in ["False", "True"]
        ]
        out_record["answer_ids"] = answer_ids
        return out_record

    ds = ds.map(map_fn, batched=False, remove_columns=ds.column_names, num_proc=1)
    gt_ds = ds.filter(lambda ex: len(ex["input_ids"]) <= tokenizer.model_max_length)

    assert len(encodings) == len(gt_ds)
    assert set(encodings.column_names) == set(gt_ds.column_names)
    for col in encodings.column_names:
        assert encodings[col] == gt_ds[col]
