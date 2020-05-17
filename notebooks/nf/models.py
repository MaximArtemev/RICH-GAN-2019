import torch
import torch.nn as nn


class NormalizingFlowModel(nn.Module):

    def __init__(self, dim, prior, flows):
        super().__init__()
        self.dim = dim
        self.prior = prior
        self.flows = nn.ModuleList(flows)

    def forward(self, x):
        m, _ = x.shape
        log_det = torch.zeros(m)
        for flow in self.flows:
            x, ld = flow.forward(x)
            log_det += ld
        z, prior_logprob = x, self.prior.log_prob(x)
        return z, prior_logprob, log_det

    def backward(self, z):
        m, _ = z.shape
        log_det = torch.zeros(m)
        for flow in self.flows[::-1]:
            z, ld = flow.backward(z)
            log_det += ld
        x = z
        return x, log_det

    def sample(self, n_samples):
        z = self.prior.sample((n_samples,self.dim))
        if z.dim() != 2: z = self.prior.sample((n_samples,))
        x, _ = self.backward(z)
        return x


class ConditionalNormalizingFlowModel(nn.Module):

    def __init__(self, dim, condition_dim, prior, flows, mu=FCNN, log_sigma=FCNN, hidden_dim=8):
        super().__init__()
        self.dim = dim
        self.condition_dim = condition_dim
        self.prior = prior
        self.flows = nn.ModuleList(flows)
        self.mu = mu(condition_dim, dim, hidden_dim)
        self.log_sigma = log_sigma(condition_dim, dim, hidden_dim)

    def forward(self, x, condition):
        m, _ = x.shape
        log_det = torch.zeros(m).to(x.device)
        for flow in self.flows:
            if hasattr(flow, 'is_conditional'):
                x, ld = flow.forward(x, condition)
            else:
                x, ld = flow.forward(x)
            log_det += ld

        mu, log_sigma = self.mu(condition), self.log_sigma(condition)
        x = (x - mu) * torch.exp(-log_sigma)
        log_det -= log_sigma.sum(dim=1)
        z, prior_logprob = x, self.prior.log_prob(x)
        return z, prior_logprob, log_det

    def backward(self, z, condition):
        m, _ = z.shape
        log_det = torch.zeros(m).to(z.device)
        mu, log_sigma = self.mu(condition), self.log_sigma(condition)
        z = z * torch.exp(log_sigma) + mu
        log_det += log_sigma.sum(dim=1)
        for flow in self.flows[::-1]:
            if hasattr(flow, 'is_conditional'):
                z, ld = flow.backward(z, condition)
            else:
                z, ld = flow.backward(z)
            log_det += ld
        x = z
        return x, log_det
