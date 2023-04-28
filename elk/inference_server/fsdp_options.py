from dataclasses import dataclass


@dataclass
class FSDPOptions:
    cpu_offload: bool = False
