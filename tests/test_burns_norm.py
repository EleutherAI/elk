import torch
from torch import Tensor

from elk.training.burns_norm import BurnsNorm


def correct_but_slow_normalization(x_all: Tensor, scale=True) -> Tensor:
    res = []
    xs = x_all.unbind(dim=1)

    for x in xs:
        num_elements = x.shape[0]
        x_mean: Tensor = x - x.mean(dim=0) if num_elements > 1 else x
        if scale is True:
            std = torch.linalg.norm(x_mean, axis=0) / torch.sqrt(
                torch.tensor(x_mean.shape[0], dtype=torch.float32)
            )
            avg_norm = std.mean()
            x_mean = x_mean / avg_norm
        res.append(x_mean)

    return torch.stack(res, dim=1)


def test_BurnsNorm_3d_input():
    x_all_3d = torch.randn((2, 13, 768))
    expected_output_3d = correct_but_slow_normalization(x_all_3d)
    bn = BurnsNorm()
    output_3d = bn(x_all_3d)
    diff = output_3d - expected_output_3d
    assert (diff == torch.zeros_like(diff)).all()


def test_BurnsNorm_4d_input():
    x_all_4d = torch.randn((2, 13, 2, 768))
    expected_output_4d = correct_but_slow_normalization(x_all_4d)
    bn = BurnsNorm()
    output_4d = bn(x_all_4d)
    diff = output_4d - expected_output_4d
    assert (diff == torch.zeros_like(diff)).all()
