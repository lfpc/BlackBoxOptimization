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