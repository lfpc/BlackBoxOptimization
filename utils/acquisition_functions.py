
import botorch
import torch
from utils import soft_clamp

class Custom_LogEI(botorch.acquisition.LogExpectedImprovement):
    def __init__(self, model, best_f, deterministic_fn = None, constraint_fn = None):
        self.deterministic_fn = deterministic_fn
        self.constraint_fn = constraint_fn
        self._size_x = model.bounds.shape[-1]
        super().__init__(model = model, best_f = best_f)
    
    def _mean_and_sigma(self, X, compute_sigma = True, min_var = 1e-12):
        mean, sigma = super()._mean_and_sigma(X, compute_sigma, min_var)
        return mean,sigma
        mean = -1 * mean
        if self.deterministic_fn is not None: 
            mean = self.deterministic_fn(X.reshape(-1,self._size_x),mean.reshape(-1,1)).reshape(mean.shape)
        mean = -1 * mean
        #mean = soft_clamp(mean, 1E6)
        #if self.constraint_fn is not None: 
        #    sigma = torch.where(self.constraint_fn(X.reshape(-1,self._size_x)).view(sigma.shape)>0, sigma*0.1, sigma).to(sigma.dtype).clamp(min = min_var)
        assert not torch.isnan(mean).any(), "Há valores NaN nos dados de entrada!"
        assert not torch.isinf(mean).any(), f"Há valores infinitos nos dados de entrada!, {mean.max()}, {mean.min()}"
        assert not torch.isnan(sigma).any(), "Há valores NaN nos dados de entrada!"
        assert not torch.isinf(sigma).any(), "Há valores infinitos nos dados de entrada!"
        return mean, sigma
    