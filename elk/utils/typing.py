from typing import Any, Type, TypeVar, cast

import torch

T = TypeVar("T")


def assert_type(typ: Type[T], obj: Any) -> T:
    """Assert that an object is of a given type at runtime and return it."""
    if not isinstance(obj, typ):
        raise TypeError(f"Expected {typ.__name__}, got {type(obj).__name__}")

    return cast(typ, obj)


def float32_to_int16(x: torch.Tensor) -> torch.Tensor:
    """Converts float32 to float16, then reinterprets as int16."""
    downcast = x.type(torch.bfloat16)
    if not downcast.isfinite().all():
        raise ValueError("Non-finite values in tensor")

    return downcast.view(torch.int16)


def int16_to_float32(x: torch.Tensor) -> torch.Tensor:
    """Converts int16 to float16, then reinterprets as float32."""
    return x.view(torch.bfloat16).type(torch.float32)
