import torch
import torch.nn as nn
import torch.nn.functional as F

class WaveletLayer(nn.Module):
    def __init__(self, in_channels=1, trainable=False):
        super().__init__()

        ll = torch.tensor([[1, 1],
                           [1, 1]]) / 2
        lh = torch.tensor([[1, 1],
                           [-1, -1]]) / 2
        hl = torch.tensor([[1, -1],
                           [1, -1]]) / 2
        hh = torch.tensor([[1, -1],
                           [-1, 1]]) / 2

        filters = torch.stack([ll, lh, hl, hh])      # (4,2,2)
        filters = filters.unsqueeze(1)               # (4,1,2,2)
        filters = filters.repeat(in_channels, 1, 1, 1)
        filters = filters.view(4 * in_channels, 1, 2, 2)

        if trainable:
            self.filters = nn.Parameter(filters)
        else:
            self.register_buffer("filters", filters)

        self.in_channels = in_channels

    def forward(self, x):
        return F.conv2d(
            x,
            self.filters,
            stride=2,
            groups=self.in_channels
        )
