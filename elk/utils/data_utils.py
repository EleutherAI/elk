import logging


def silence_datasets_messages():
    """Silence the annoying wall of logging messages and warnings."""

    def filter_fn(log_record):
        msg = log_record.getMessage()
        return (
            "Found cached dataset" not in msg
            and "Loading cached" not in msg
            and "Using custom data configuration" not in msg
        )

    handler = logging.StreamHandler()
    handler.addFilter(filter_fn)
    logging.getLogger("datasets").addHandler(handler)
