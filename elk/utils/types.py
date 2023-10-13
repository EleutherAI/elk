from enum import Enum


class PromptEnsembling(Enum):
    FULL = "full"
    PARTIAL = "partial"
    NONE = "none"

    @staticmethod
    def all() -> tuple["PromptEnsembling"]:
        return tuple(PromptEnsembling)
