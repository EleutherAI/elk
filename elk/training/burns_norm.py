import torch
from torch import Tensor, nn


class BurnsNorm(nn.Module):
    """ Burns et al. style normalization  """

    def forward(self, x: Tensor) -> Tensor:
        breakpoint()
        assert x.dim() == 3, "the input should have a dimension of 3." # TODO: add info about needed shape

        print("Per Prompt Normalization...")
        x: Tensor = x - torch.mean(x, dim=0)
        norm = torch.linalg.norm(x, dim=2)
        avg_norm = torch.mean(norm)
        return x / avg_norm * torch.sqrt(torch.tensor(x.shape[2], dtype=torch.float32))

