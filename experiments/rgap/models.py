"""Paper backbone from the public repository; tiny network is test-only."""
import torch
from torch import nn
from .medium_unet import MediumUNet


class TinyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.inc = nn.Sequential(nn.Conv2d(2, 8, 3, padding=1), nn.LeakyReLU(.1),
                                 nn.Conv2d(8, 8, 3, padding=1), nn.LeakyReLU(.1))
        self.out = nn.Conv2d(8, 2, 1)

    def forward(self, x):
        return self.out(self.inc(x))


def make_model(backbone="medium", rescat=False):
    net = MediumUNet(2, 2) if backbone == "medium" else TinyNet()
    if rescat:
        # Construct the same 2-channel network first: all shared weights match
        # exactly for a paired seed. Add zero-initialized residual channels.
        parent = net.inc.double_conv if backbone == "medium" else net.inc
        conv = parent[0]
        expanded = nn.Conv2d(4, conv.out_channels, conv.kernel_size,
                             padding=conv.padding, bias=conv.bias is not None)
        with torch.no_grad():
            expanded.weight.zero_()
            expanded.weight[:, :2].copy_(conv.weight)
            if conv.bias is not None:
                expanded.bias.copy_(conv.bias)
        parent[0] = expanded
    return net
