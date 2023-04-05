import re

from .training.losses import LOSSES


def parse_loss(terms: list[str]) -> dict[str, float]:
    """Parse the loss command line argument list into a dictionary."""
    if len(terms) == 0:
        return {"ccs_prompt_var": 1.0}
    loss_dict = dict()
    for term in terms:
        if term in loss_dict:
            raise ValueError(f"Duplicate loss term: {term}")
        # check if the term is of the form "coef*name"
        if re.match(r"^\d+(\.)?\d*\*\w+$", term):
            coef, name = term.split("*")
            coef = float(coef)
        # check if the term is of the form "name"
        elif re.match(r"^\w+$", term):
            name = term
            coef = 1.0
        else:
            raise ValueError(
                f"Invalid loss term: {term}. "
                "Loss terms should be of the form 'coef*name' or 'name'."
            )
        assert name in LOSSES, f"Unknown loss term: {name}"
        loss_dict[name] = coef
    return loss_dict
