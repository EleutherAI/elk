import logging
from pathlib import Path

from .extraction.dataset_name import DatasetDictWithName
from .utils import select_train_val_splits


def save_debug_log(datasets: list[DatasetDictWithName], out_dir: Path) -> None:
    """
    Save a debug log to the output directory. This is useful for debugging
    training issues.
    """

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s %(levelname)s:\n%(message)s",
        filename=out_dir / "debug.log",
        filemode="w",
    )

    for ds_name, ds in datasets:
        logging.info(
            "=========================================\n"
            f"Dataset: {ds_name}\n"
            "========================================="
        )

        if len(ds) == 1:
            train_split = None
            val_split = list(ds.keys())[0]
        else:
            train_split, val_split = select_train_val_splits(ds)

        if len(ds[val_split]) == 0:
            logging.warning(f"Val split '{val_split}' is empty!")
            continue

        texts = ds[val_split][0]["texts"]
        template_ids = ds[val_split][0]["variant_ids"]
        ds[val_split][0]["label"]

        # log the train size and val size
        if train_split is not None:
            logging.info(f"Train size: {len(ds[train_split])}")
        logging.info(f"Val size: {len(ds[val_split])}")

        templates_text = f"{len(texts)} templates used:\n"
        trailing_whitespace = False
        for text, id in zip(texts, template_ids):
            templates_text += f'***---TEMPLATE "{id}"---***\n' f'"""{text}"""\n'
            if text[-1].isspace():
                trailing_whitespace = True
        if trailing_whitespace:
            logging.warning(
                "Some inputs to the model have trailing whitespace! "
                "Check that the jinja templates are not adding "
                "trailing whitespace. If `token_loc` is 'last', this "
                "will extract hidden states from the whitespace token."
            )
        logging.info(templates_text)
