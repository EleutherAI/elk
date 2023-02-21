from typing import Any, TypeVar, Type


A = TypeVar("A")


def assert_is_instance(obj: Any, cls: Type[A]) -> A:
    assert isinstance(obj, cls), f"Expected {obj} to be an instance of {cls}"
    return obj
