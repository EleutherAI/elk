from typing import NoReturn


def raise_assertion_error(msg: str) -> NoReturn:
    raise AssertionError(msg)
