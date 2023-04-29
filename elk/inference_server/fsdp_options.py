from dataclasses import dataclass
from typing import Literal


@dataclass
class FSDPOptions:
    fsdp_enabled: bool = False
    cpu_offload: bool = False
    # See https://pytorch.org/docs/stable/multiprocessing.html
    # We tend to get alot of issues with file_descriptors, so we default to file_system
    mp_sharing_strategy: Literal["file_system", "file_descriptor"] = "file_system"
