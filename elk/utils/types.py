from typing import NoReturn, Any, TypeVar, Type


def raise_assertion_error(msg: str) -> NoReturn:
    raise AssertionError(msg)


A = TypeVar("A")


def assert_is_instance(obj: Any, cls: Type[A]) -> A:
    assert isinstance(obj, cls), f"Expected {obj} to be an instance of {cls}"
    return obj
