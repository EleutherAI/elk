from enum import Enum


class Ensembling(Enum):
    FULL = "full"
    PARTIAL = "partial"
    NONE = "none"

    @staticmethod
    def all() -> tuple["Ensembling"]:
        return tuple(Ensembling)
