from datasets import Dataset

from elk.utils import pytree_map


def test_pytree_map():
    _ids = [15496, 11, 616, 3290, 318, 13779]
    inputs = {"input_ids": _ids}
    input_dataset = Dataset.from_dict(inputs)
    result: Dataset = pytree_map(lambda x: x, input_dataset)
    # assert we got the same thing back
    assert _ids == result["input_ids"]
