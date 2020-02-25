"""
Implements various flows.
Each flow is invertible so it can be forward()ed and backward()ed.
Notice that backward() is not backward as in backprop but simply inversion.
Each flow also outputs its log det J "regularization"

Reference:

NICE: Non-linear Independent Components Estimation, Dinh et al. 2014
https://arxiv.org/abs/1410.8516

Variational Inference with Normalizing Flows, Rezende and Mohamed 2015
https://arxiv.org/abs/1505.05770

Density estimation using Real NVP, Dinh et al. May 2016
https://arxiv.org/abs/1605.08803
(Laurent's extension of NICE)

Improved Variational Inference with Inverse Autoregressive Flow, Kingma et al June 2016
https://arxiv.org/abs/1606.04934
(IAF)

Masked Autoregressive Flow for Density Estimation, Papamakarios et al. May 2017 
https://arxiv.org/abs/1705.07057
"The advantage of Real NVP compared to MAF and IAF is that it can both generate data and estimate densities with one forward pass only, whereas MAF would need D passes to generate data and IAF would need D passes to estimate densities."
(MAF)

Glow: Generative Flow with Invertible 1x1 Convolutions, Kingma and Dhariwal, Jul 2018
https://arxiv.org/abs/1807.03039

"Normalizing Flows for Probabilistic Modeling and Inference"
https://arxiv.org/abs/1912.02762
(review paper)
"""

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from nflib.nets import LeafParam, MLP, ARMLP

import logging

logger = logging.getLogger('main.flows')

class AffineConstantFlow(nn.Module):
    """ 
    Scales + Shifts the flow by (learned) constants per dimension.
    In NICE paper there is a Scaling layer which is a special case of this where t is None
    """
    def __init__(self, dim, scale=True, shift=True):
        super().__init__()
        self.s = nn.Parameter(torch.randn(1, dim, requires_grad=True)) if scale else None
        self.t = nn.Parameter(torch.randn(1, dim, requires_grad=True)) if shift else None
        self.register_buffer('placeholder', torch.randn(1))
        
    def forward(self, x, context=None):
        s = self.s if self.s is not None else x.new_zeros(x.size(), device=self.placeholder.device)
        t = self.t if self.t is not None else x.new_zeros(x.size(), device=self.placeholder.device)
        z = x * torch.exp(s) + t
        log_det = torch.sum(s, dim=1)
        return z, log_det
    
    def backward(self, z, context=None):
        s = self.s if self.s is not None else z.new_zeros(z.size(), device=self.placeholder.device)
        t = self.t if self.t is not None else z.new_zeros(z.size(), device=self.placeholder.device)
        x = (z - t) * torch.exp(-s)
        log_det = torch.sum(-s, dim=1)
        return x, log_det


class ActNorm(AffineConstantFlow):
    """
    Really an AffineConstantFlow but with a data-dependent initialization,
    where on the very first batch we clever initialize the s,t so that the output
    is unit gaussian. As described in Glow paper.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.data_dep_init_done = False

    def forward(self, x, context=None):
        # first batch is used for init
        if not self.data_dep_init_done:
            assert self.s is not None and self.t is not None # for now
            self.s.data = (-torch.log(x.std(dim=0, keepdim=True))).detach()
            self.t.data = (-(x * torch.exp(self.s)).mean(dim=0, keepdim=True)).detach()
            self.data_dep_init_done = True
        return super().forward(x)


class AffineHalfFlow(nn.Module):
    """
    As seen in RealNVP, affine autoregressive flow (z = x * exp(s) + t), where half of the 
    dimensions in x are linearly scaled/transfromed as a function of the other half.
    Which half is which is determined by the parity bit.
    - RealNVP both scales and shifts (default)
    - NICE only shifts
    """
    def __init__(self, dim, parity, hidden_dim=24, scale=True, shift=True, base_network=MLP, **base_network_kwargs):
        super().__init__()
        self.dim = dim
        self.parity = parity
        self.s_cond = lambda x: x.new_zeros(x.size(0), self.dim // 2, device=x.device)
        self.t_cond = lambda x: x.new_zeros(x.size(0), self.dim // 2, device=x.device)
        if scale:
            self.s_cond = base_network(self.dim // 2, self.dim // 2, hidden_dim, **base_network_kwargs)
        if shift:
            self.t_cond = base_network(self.dim // 2, self.dim // 2, hidden_dim, **base_network_kwargs)
        
    def forward(self, x, context=None):
        x0, x1 = x[:,::2], x[:,1::2]
        if self.parity:
            x0, x1 = x1, x0
        s = self.s_cond(x0, context=context)
        t = self.t_cond(x0, context=context)
        z0 = x0 # untouched half
        z1 = torch.exp(s) * x1 + t # transform this half as a function of the other
        if self.parity:
            z0, z1 = z1, z0
        z = torch.cat([z0, z1], dim=1)
        log_det = torch.sum(s, dim=1)
        return z, log_det
    
    def backward(self, z, context=None):
        z0, z1 = z[:,::2], z[:,1::2]
        if self.parity:
            z0, z1 = z1, z0
        s = self.s_cond(z0, context)
        t = self.t_cond(z0, context)
        x0 = z0 # this was the same
        x1 = (z1 - t) * torch.exp(-s) # reverse the transform on this half
        if self.parity:
            x0, x1 = x1, x0
        x = torch.cat([x0, x1], dim=1)
        log_det = torch.sum(-s, dim=1)
        return x, log_det

class MAF(nn.Module):
    """ Masked Autoregressive Flow that uses a MADE-style network for fast forward """
    def __init__(self, dim, parity, net_class=ARMLP, hidden_dim=24, **base_network_kwargs):
        super().__init__()
        self.register_buffer('placeholder', torch.randn(1))
        self.dim = dim
        self.net = net_class(dim, dim*2, hidden_dim, **base_network_kwargs)
        self.parity = parity

    def forward(self, x, context=None):
        # here we see that we are evaluating all of z in parallel, so density estimation will be fast
        st = self.net(x, context=context)
        s, t = st.split(self.dim, dim=1)
        z = x * torch.exp(s) + t
        # reverse order, so if we stack MAFs correct things happen
        z = z.flip(dims=(1,)) if self.parity else z
        log_det = torch.sum(s, dim=1)
        return z, log_det
    
    def backward(self, z, context=None):
        # we have to decode the x one at a time, sequentially
        x = torch.zeros_like(z)
        log_det = torch.zeros(z.shape[0], device=self.placeholder.device)

        z = z.flip(dims=(1,)) if self.parity else z
        for i in range(self.dim):
            st = self.net(x.clone(), context=context) # clone to avoid in-place op errors if using IAF
            s, t = st.split(self.dim, dim=1)
            x[:, i] = (z[:, i] - t[:, i]) * torch.exp(-s[:, i])
            log_det += -s[:, i]
        return x, log_det

class IAF(MAF):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        """
        reverse the flow, giving an Inverse Autoregressive Flow (IAF) instead, 
        where sampling will be fast but density estimation slow
        """
        self.forward, self.backward = self.backward, self.forward


class Invertible1x1Conv(nn.Module):
    """ 
    As introduced in Glow paper.
    """
    
    def __init__(self, dim):
        super().__init__()
        self.register_buffer('placeholder', torch.randn(1))
        self.dim = dim
        Q = torch.nn.init.orthogonal_(torch.randn(dim, dim))
        P, L, U = torch.lu_unpack(*Q.lu())
        self.P = nn.Parameter(P, requires_grad=False) # remains fixed during optimization
        self.L = nn.Parameter(L) # lower triangular portion
        self.S = nn.Parameter(U.diag()) # "crop out" the diagonal to its own parameter
        self.U = nn.Parameter(torch.triu(U, diagonal=1)) # "crop out" diagonal, stored in S

    def _assemble_W(self):
        """ assemble W from its pieces (P, L, U, S) """
        L = torch.tril(self.L, diagonal=-1) + torch.diag(torch.ones(self.dim, device=self.placeholder.device))
        U = torch.triu(self.U, diagonal=1)
        W = self.P @ L @ (U + torch.diag(self.S))
        return W

    def forward(self, x, context=None):
        W = self._assemble_W()
        z = x @ W
        log_det = torch.sum(torch.log(torch.abs(self.S)))
        return z, log_det

    def backward(self, z, context=None):
        W = self._assemble_W()
        W_inv = torch.inverse(W)
        x = z @ W_inv
        log_det = -torch.sum(torch.log(torch.abs(self.S)))
        return x, log_det
    
class RandomPerm(nn.Module):
    # robdrynkin
    def __init__(self, dim):
        super().__init__()
        self.perm = np.random.permutation(dim)
        self.inv_perm = np.argsort(self.perm)
        
    def forward(self, x, context=None):
        return x[:, self.perm], 0

    def backward(self, z, context=None):
        return z[:, self.inv_perm], 0


# ------------------------------------------------------------------------

class NormalizingFlow(nn.Module):
    """ A sequence of Normalizing Flows is a Normalizing Flow """

    def __init__(self, flows):
        super().__init__()
        self.register_buffer('placeholder', torch.randn(1))
        self.flows = nn.ModuleList(flows)

    def forward(self, x, context=None):
        m, _ = x.shape
        log_det = torch.zeros(m, device=self.placeholder.device)
        zs = [x]
        for flow in self.flows:
            x, ld = flow.forward(x, context=context)
            log_det += ld
            zs.append(x)
        return zs, log_det

    def backward(self, z, context=None):
        m, _ = z.shape
        log_det = torch.zeros(m, device=self.placeholder.device)
        xs = [z]
        for flow in self.flows[::-1]:
            z, ld = flow.backward(z, context=context)
            log_det += ld
            xs.append(z)
        return xs, log_det

class NormalizingFlowModel(nn.Module):
    """ A Normalizing Flow Model is a (prior, flow) pair """
    
    def __init__(self, prior, flows):
        super().__init__()
        self.register_buffer('placeholder', torch.randn(1))
        self.prior = prior
        self.flow = NormalizingFlow(flows)
    
    def forward(self, x, context=None):
        zs, log_det = self.flow.forward(x, context=context)
        prior_logprob = self.prior.log_prob(zs[-1]).view(x.shape[0], -1).sum(1).to(self.placeholder.device)
        return zs, prior_logprob, log_det

    def backward(self, z, context=None):
        xs, log_det = self.flow.backward(z, context=context)
        return xs, log_det
    
    def sample(self, num_samples, context=None):
        z = self.prior.sample((num_samples,)).to(self.placeholder.device)
        xs, _ = self.backward(z, context=context)
        return xs
