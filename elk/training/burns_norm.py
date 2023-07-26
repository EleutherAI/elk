import torch
from torch import Tensor, nn


class BurnsNorm(nn.Module):
    """Burns et al. style normalization  Minimal changes from the original code."""
    scale = False

    def forward(self, x_all: Tensor) -> Tensor:
        if not self.scale:
            # not dividing by std deviation (scaling). We just subtract the mean
            return x_all - x_all.mean(dim=0, keep_dim=True)
        else:
            res = []

            # x_all has the shape (n, v, d)
            # unbind gives us a list of (n, d)
            for x in x_all.unbind(dim=1):
                x_normalized: Tensor = x - x.mean(dim=0)
                
                # the following code block is taken from the original Burns code 
                # and uses a strange way of calculating the std 
                std = torch.linalg.norm(x_normalized, axis=0) / torch.sqrt(
                    torch.tensor(x_normalized.shape[0], dtype=torch.float32)
                )
                avg_norm = std.mean()
                res.append(x_normalized / avg_norm)

            return torch.stack(res, dim=1) #  (n, v, d)
    