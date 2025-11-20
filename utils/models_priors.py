import torch
import gpytorch
from math import log

class KumaAlphaPrior(gpytorch.priors.Prior):
    def __init__(self):
        super().__init__()
        self.log_a_max = log(2)
        pass

    def log_prob(self, x):
        x = torch.log(x)
        loc = torch.tensor(0.).to(x)
        scale = torch.tensor(0.01).to(x)
        return torch.sum(torch.log(
            torch.distributions.Normal(loc=loc, scale=scale).log_prob(x).exp() + 0.5 / self.log_a_max
        ))

class KumaBetaPrior(gpytorch.priors.Prior):
    def __init__(self):
        super().__init__()
        self.log_b_max = log(2)
        pass

    def log_prob(self, x):
        x = torch.log(x)
        loc = torch.tensor(0.).to(x)
        scale = torch.tensor(0.01).to(x)
        return torch.sum(torch.log(
            torch.distributions.Normal(loc=loc, scale=scale).log_prob(x).exp() + 0.5 / self.log_b_max
        ))

class AngularWeightsPrior(gpytorch.priors.Prior):
    def __init__(self):
        super(AngularWeightsPrior, self).__init__()

    def log_prob(self, x):
        x = torch.log(x)
        loc = torch.tensor(0.).to(x)
        scale = torch.tensor(2.).to(x)
        return torch.distributions.Normal(loc=loc, scale=scale).log_prob(x).sum()
    
class IBNN_ReLU(gpytorch.kernels.Kernel):
    is_stationary = False

    def __init__(self, d, var_w, var_b, depth, **kwargs):
        super().__init__(**kwargs)
        self.d = d
        self.var_w = var_w
        self.var_b = var_b
        self.depth = depth

    def k(self, l, x1, x2):
        # base case
        if l == 0:
            return self.var_b + self.var_w * (x1 * x2).sum(-1) / self.d
        else:
            K_12 = self.k(l - 1, x1, x2)
            K_11 = self.k(l - 1, x1, x1)
            K_22 = self.k(l - 1, x2, x2)
            sqrt_term = torch.sqrt(K_11 * K_22)
            fraction = K_12 / sqrt_term
            epsilon = 1e-7
            theta = torch.acos(torch.clamp(fraction, min=-1 + epsilon, max=1 - epsilon))
            theta_term = torch.sin(theta) + (torch.pi - theta) * fraction
            result = self.var_b + self.var_w / (2 * torch.pi) * sqrt_term * theta_term
            return result
        
    def forward(self, x1, x2, **params):
        d2 = x2.shape[-2]
        x1_shape = tuple(x1.shape)
        d1, dim = x1_shape[-2:]
        new_shape = x1_shape[:-2] + (d1, d2, dim)
        new_x1 = x1.unsqueeze(-2).expand(new_shape)
        new_x2 = x2.unsqueeze(-3).expand(new_shape)
        result = self.k(self.depth, new_x1, new_x2)
        return result