"""
Various helper network modules
"""

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from nflib.nets_utils import LeafParam, PositionalEncoder, MADE, ResidualBlock

class PosEncMLP(nn.Module):
    """ 
    Position Encoded MLP, where the first layer performs position encoding.
    Each dimension of the input gets transformed to len(freqs)*2 dimensions
    using a fixed transformation of sin/cos of given frequencies.
    """
    def __init__(self, nin, nout, nh, freqs=(.5,1,2,4,8)):
        super().__init__()
        self.net = nn.Sequential(
            PositionalEncoder(freqs),
            MLP(nin * len(freqs) * 2, nout, nh),
        )
    def forward(self, x):
        return self.net(x)
    
# MLP section
    
class MLP(nn.Module):
    """ a simple 4-layer MLP """

    def __init__(self, nin, nout, nh, context=False):
        """ context  - int, False or zero if None""" 
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(nin + int(context), nh),
            nn.LeakyReLU(0.2),
            nn.Linear(nh, nh),
            nn.LeakyReLU(0.2),
            nn.Linear(nh, nh),
            nn.LeakyReLU(0.2),
            nn.Linear(nh, nout),
        )
    def forward(self, x, context=None):
        if context is not None:
            return self.net(torch.cat([context, x], dim=1))
        return self.net(x)

# Masked section
    
class ARMLP(nn.Module):
    """ a 4-layer auto-regressive MLP, wrapper around MADE net """

    def __init__(self, nin, nout, nh, **base_network_kwargs):
        super().__init__()
        self.net = MADE(nin, [nh, nh, nh], nout, **base_network_kwargs)
        
    def forward(self, x, context=None):
        return self.net(x)

# Residual section

class ResidualNet(nn.Module):
    """A general-purpose residual network. Works only with 1-dim inputs."""

    def __init__(self, nin, nout, nh, context=False, num_blocks=2):
        super().__init__()
        self.initial_layer = nn.Sequential(
            nn.Linear(nin + int(context), nh),
            nn.BatchNorm1d(nh, eps=1e-3),
            nn.LeakyReLU(0.2)
        )
        self.blocks = nn.ModuleList([
            ResidualBlock(
                features=nh,
                context=context,
            ) for _ in range(num_blocks)
        ])
        self.final_layer = nn.Linear(nh, nout)

    def forward(self, inputs, context=None):
        if context is None:
            temps = self.initial_layer(inputs)
        else:
            temps = self.initial_layer(
                torch.cat((inputs, context), dim=1)
            )
        for block in self.blocks:
            temps = block(temps, context=context)
        outputs = self.final_layer(temps)
        return outputs


