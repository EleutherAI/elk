from typing import cast, Any, Type, TypeVar

import torch


T = TypeVar("T")


def assert_type(typ: Type[T], obj: Any) -> T:
    """Assert that an object is of a given type at runtime and return it."""
    if not isinstance(obj, typ):
        raise TypeError(f"Expected {typ.__name__}, got {type(obj).__name__}")

    return cast(typ, obj)

def float32_to_int16(x: torch.Tensor) -> torch.Tensor:
    """Converts float32 to float16, then reinterprets as int16."""
    return x.type(torch.float16).view(torch.int16)

def int16_to_float32(x: torch.Tensor) -> torch.Tensor:
    """Converts int16 to float16, then reinterprets as float32."""
    return x.view(torch.float16).type(torch.float32)

def upcast_hiddens(hiddens: torch.Tensor) -> torch.Tensor:
    """Upcast hidden states to float32."""

    # Note: currently we're just upcasting to float32 so we don't have to deal with
    # grad scaling (which isn't supported for LBFGS), while the hidden states are
    # saved in float16 to save disk space. In the future we could try to use mixed
    # precision training in at least some cases.
    return int16_to_float32(assert_type(torch.Tensor, hiddens))