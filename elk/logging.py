import logging
from .utils import select_train_val_splits


def save_debug_log(ds, out_dir):
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

    train_split, val_split = select_train_val_splits(ds)
    text_inputs = ds[val_split][0]["text_inputs"]
    template_ids = ds[val_split][0]["variant_ids"]
    label = ds[val_split][0]["label"]

    # log the train size and val size
    logging.info(f"Train size: {len(ds[train_split])}")
    logging.info(f"Val size: {len(ds[val_split])}")

    templates_text = f"{len(text_inputs)} templates used:\n"
    trailing_whitespace = False
    for (text0, text1), id in zip(text_inputs, template_ids):
        templates_text += (
            f'***---TEMPLATE "{id}"---***\n'
            f"{'false' if label else 'true'}:\n"
            f'"""{text0}"""\n'
            f"{'true' if label else 'false'}:\n"
            f'"""{text1}"""\n\n\n'
        )
        if text0[-1].isspace() or text1[-1].isspace():
            trailing_whitespace = True
    if trailing_whitespace:
        logging.warning(
            "Some inputs to the model have trailing whitespace! "
            "Check that the jinja templates are not adding "
            "trailing whitespace. If `token_loc` is 'last', this "
            "will extract hidden states from the whitespace token."
        )
    logging.info(templates_text)
