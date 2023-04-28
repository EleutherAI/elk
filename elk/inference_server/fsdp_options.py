from dataclasses import dataclass


@dataclass
class FSDPOptions:
    fsdp_enabled: bool = False
    cpu_offload: bool = False
