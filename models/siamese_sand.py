# ---Dependencies---
import torch
import torch.nn as nn

# ---Custom---
from models import Sand


class SiameseSand(nn.Module):
    def __init__(self, n_dims):
        super().__init__()
        self.n_dims = n_dims
        self.branch = Sand(self.n_dims)

    def forward(self, features):
        f1, f2 = torch.chunk(features, 2, dim=2)
        d1, d2 = self.branch(f1), self.branch(f2)
        out = torch.cat([d1, d2], dim=2)
        return out
