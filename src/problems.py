import torch
import gzip
import h5py
import pickle
import sys
import numpy as np
import os
from multiprocessing import Pool, cpu_count
PROJECTS_DIR = os.getenv('PROJECTS_DIR', '~/projects')
sys.path.insert(1, os.path.join(PROJECTS_DIR,'BlackBoxOptimization'))
from utils import split_array, split_array_idx, get_split_indices, compute_solid_volume, make_index, apply_index, uniform_sample
import logging
import json
from functools import partial
logging.basicConfig(level=logging.WARNING)
import time
#torch.set_default_dtype(torch.float64)

class RosenbrockProblem:
    """
    Represents the classic N-dimensional Rosenbrock function.
    It's a non-convex function with a characteristic long, narrow,
    parabolic valley. The global minimum is at (1, 1, ..., 1).
    """
    def __init__(self, dim: int = 2, phi_bounds=((-2.0, 2.0)), noise: float = 0) -> None:
        if dim < 2:
            raise ValueError("Dimension for RosenbrockProblem must be >= 2")
        self.dim = dim
        # Create bounds dynamically based on dimension
        self._phi_bounds = ([phi_bounds[0]] * self.dim, [phi_bounds[1]] * self.dim)
        self.noise = noise

    def GetBounds(self, device='cpu') -> torch.Tensor:
        """Returns the bounds for the PHI parameters."""
        return torch.tensor(self._phi_bounds, device=device)

    def loss(self, phi):
        """The classic Rosenbrock function, generalized for N dimensions."""
        if phi.dim() == 1:
            phi = phi.unsqueeze(0)
        
        # Standard Rosenbrock formula for N dimensions
        return (100 * (phi[:, 1:] - phi[:, :-1].pow(2)).pow(2)).sum(dim=1) + \
               (1 - phi[:, :-1]).pow(2).sum(dim=1)
    def gradient(self, phi, analytic: bool = False):
        """
        Calculates the gradient of the loss function with respect to phi.
        """
        if phi.dim() == 1:
            phi = phi.unsqueeze(0)
        
        if analytic:
            return torch.tensor([-400*phi[:,0]*(phi[:,1]-phi[:,0]**2)-2*(1-phi[:,0]), 200*(phi[:,1]-phi[:,0]**2)])
        
        if not phi.requires_grad:
            phi.requires_grad_(True)
        
        loss_val = self.loss(phi)
        
        if loss_val.dim() > 0 and len(loss_val) > 1:
            loss_val = loss_val.sum()

        grad = torch.autograd.grad(loss_val, phi)[0]
        return grad
    
    def hessian(self, phi, analytic: bool = False):
        """
        Calculates the Hessian matrix of the loss function with respect to phi.
        """
        if analytic:
            if phi.dim() == 1:
                phi = phi.unsqueeze(0)
            x, y = phi[:, 0], phi[:, 1]
            hessian_matrix = torch.tensor([
                [1200 * x**2 - 400 * y + 2, -400 * x],
                [-400 * x, 200]
            ])
            return hessian_matrix
        if not phi.requires_grad:
            phi.requires_grad_(True)
            
        return torch.autograd.functional.hessian(self.loss, phi)

    def get_constraints(self, phi):
        return torch.tensor(0.0, device=phi.device)  # No constraints

    def __call__(self, phi: torch.tensor, x: torch.tensor = None):
        y = self.loss(phi)
        y = y.view(-1, 1)
        if self.noise and x is not None:
            y += x
        return y
    


class Rosenbrock_stochastic_hits(RosenbrockProblem):
    """
    Transforms the continuous N-dimensional Rosenbrock problem into a stochastic,
    binary framework.
    """
    def __init__(self, dim: int = 2, n_samples: int = 100_000,
                 phi_bounds=((-2.0, 2.0)),
                 x_bounds=(-3.0, 3.0),
                 reduction='mean'):
        super().__init__(dim=dim, phi_bounds=phi_bounds)
        self.x_bounds = x_bounds
        self.n_samples = n_samples
        self.reduction = reduction
        self.offset = 0.01  # To avoid zero probabilities
        
        self.y_max = self._find_max_value()
        self.normalize_factor = 1.0
        #self.normalize_factor = self._integrate_sensitivity(self.GetBounds().mean(0).view(1,self.dim))

    def _find_max_value(self):
        """
        Numerically finds the max value of the Rosenbrock function in its domain
        using Monte Carlo sampling, which is suitable for high dimensions.
        """
        n_mc_samples = 500000
        bounds = self.GetBounds()
        # Sample PHI uniformly within the N-dimensional bounds
        phis = (bounds[1] - bounds[0]) * torch.rand(n_mc_samples, self.dim) + bounds[0]
        
        y_vals = super().loss(phis)
        return torch.max(y_vals).mul(1.1).item()

    def _integrate_sensitivity(self, phi):
        """
        Normalizes the sensitivity function to have an approximate mean of 1
        using a Monte Carlo approach.
        """
        X_sample = torch.empty(200_000, 2).uniform_(*self.x_bounds).float()
        
        mean_sensitivity = self.sensitivity(phi, X_sample).mean()

        return mean_sensitivity

    def loss(self, phi):
        """Scales the Rosenbrock function to the [0, 1] range."""
        return (super().loss(phi) / self.y_max + self.offset).clamp(max=1.0)

    def sample_x(self, device='cpu'):
        """Samples the random variable X from a 2D uniform distribution."""
        return torch.empty(self.n_samples, 2).uniform_(*self.x_bounds).to(device).float()
    def get_weights(self,x):
        return torch.ones(x.shape[0], device=x.device)

    def sensitivity(self, phi, X):
        """
        A 'shifting interference fringes' sensitivity function. The orientation
        of the parallel fringes is controlled by PHI. This design ensures the
        integral over X is constant with respect to PHI.
        """
        x1 = X[:, 0]
        x2 = X[:, 1]
        
        angle = phi[:, 0]
        if self.dim > 1:
            angle = angle + phi[:, 1]

        # Rotate the coordinate system according to the angle from PHI
        c, s = torch.cos(angle), torch.sin(angle)
        x_rot = x1 * c - x2 * s

        # --- Define the static shape of the sensitivity manifold ---
        # 1. A base pattern of parallel sine-wave "fringes"
        frequency = 4.0
        interference_pattern = 0.5 * (torch.sin(frequency * x_rot) + 1)
        
        # 2. A Gaussian decay to ensure the function is zero near the boundaries
        decay_sigma = 2.0
        radial_decay = torch.exp(-((x1).pow(2) + x2.pow(2)) / (2 * decay_sigma**2))
        
        # The final sensitivity is the product of the two parts
        fringes_sensitivity = interference_pattern * radial_decay
        
        return fringes_sensitivity / self.normalize_factor
    def probability_of_hit(self, phi, X):
        """
        Calculates the hit probability P(hit|X, PHI).
        """
        return (self.sensitivity(phi, X) * self.loss(phi)).clamp(0.0,1.0)

    def simulate(self, phi, X):
        """
        Performs a stochastic simulation. A hit occurs if a random number
        is less than the hit probability P(hit|X, PHI).
        """
        x0 = torch.rand(X.shape[0], device=X.device)
        hit_probability = self.probability_of_hit(phi, X)
        return hit_probability - x0

    def _blackbox_loss(self, y):
        """Converts the boolean output of simulate into an integer (0 or 1)."""
        return (y > 0).int()

    def __call__(self, phi, x=None):
        if x is None:
            x = self.sample_x(device=phi.device)
            
        if phi.dim() == 1:
            phi = phi.unsqueeze(0)
        y = self.simulate(phi, x)

        y = self._blackbox_loss(y)
        
        if self.reduction == 'mean':
            return y.float().mean()
        elif self.reduction == 'sum':
            return y.float().sum()
        return y


class ThreeHump():
    """
    Represents the Three-Hump Camel function.
    This is a classic optimization test problem with a global minimum at (0, 0).
    The function is defined as:
    f(phi_0, phi_1) = 2*phi_0^2 - 1.05*phi_0^4 + (phi_0^6)/6 + phi_0*phi_1 + phi_1^2
    """
    def __init__(self, phi_bounds = ((-2.3,-2),(2.3,2)), noise: float = 0) -> None:
        """
        Initializes the function.
        Args:
            noise (float): A noise factor (not used in this visualization).
        """
        self.dim = 2
        self._phi_bounds = phi_bounds
        self.noise = noise
    def GetBounds(self, device = 'cpu') -> torch.Tensor:
        """Returns the bounds for the PHI parameters."""
        return torch.tensor(self._phi_bounds, device=device)

    def loss(self, phi):
        """The classic Three-Hump Camel function."""
        if phi.dim() == 1:
            phi = phi.unsqueeze(0)
        p1 = phi[:, 0]
        p2 = phi[:, 1]
        return 2*p1**2 - 1.05*p1**4 + (p1**6)/6 + p1*p2 + p2**2
    def gradient(self, phi, analytic: bool = False):
        """
        Calculates the gradient of the loss function with respect to phi.
        """
        if phi.dim() == 1:
            phi = phi.unsqueeze(0)
        
        if analytic:
            x = phi[:, 0]
            y = phi[:, 1]
            grad_x = 4*x - 4.2*x**3 + x**5 + y
            grad_y = x + 2*y
            return torch.stack([grad_x, grad_y], dim=1)
        
        # --- Autograd version ---
        phi_clone = phi.clone().requires_grad_(True)
        loss_val = self.loss(phi_clone)
        
        if loss_val.dim() > 0 and len(loss_val) > 1:
            loss_val = loss_val.sum()

        grad = torch.autograd.grad(loss_val, phi_clone)[0]
        return grad
    
    def hessian(self, phi, analytic: bool = False):
        """
        Calculates the Hessian matrix of the loss function with respect to phi.
        """
        if phi.dim() > 1:
            if phi.shape[0] > 1:
                raise ValueError("Hessian calculation is supported for a single phi vector at a time.")
            phi = phi.squeeze(0)

        if analytic:
            x = phi[0]
            H_xx = 4 - 12.6 * x**2 + 5 * x**4
            H_xy = 1.0
            H_yy = 2.0
            
            hessian_matrix = torch.tensor([
                [H_xx, H_xy],
                [H_xy, H_yy]
            ], device=phi.device)
            return hessian_matrix

        # --- Autograd version ---
        phi_clone = phi.clone().requires_grad_(True).view(-1,2)
        return torch.autograd.functional.hessian(self.loss, phi_clone)
    
    def get_constraints(self, phi):
        return torch.tensor(0.0, device=phi.device)  # No constraints in this problem

    def __call__(self, phi: torch.tensor, x: torch.tensor = None):
        """
        Evaluates the function for a batch of input vectors.
        Args:
            phi (torch.tensor): A tensor of shape (n, 2) where n is the number of points.
            x (torch.tensor, optional): Noise tensor. Defaults to None.
        Returns:
            torch.tensor: A tensor of shape (n, 1) with the function values.
        """
        # Ensure phi is a 2D tensor
        if phi.dim() == 1:
            phi = phi.unsqueeze(0)
            
        y = self.loss(phi)
        if self.noise and x is not None:
            y += x
            
        return y

class G02Problem():
    def __init__(self, dim: int = 20, phi_bounds=((-0.0, 10.0)), noise: float = 0, constrained = True) -> None:
        self.dim = dim
        # Create bounds dynamically based on dimension
        self._phi_bounds = ([phi_bounds[0]] * self.dim, [phi_bounds[1]] * self.dim)
        self.noise = noise
        self.constrained = constrained

    def GetBounds(self, device='cpu') -> torch.Tensor:
        """Returns the bounds for the PHI parameters."""
        return torch.tensor(self._phi_bounds, device=device)

    def loss(self, phi):
        if phi.dim() == 1:
            phi = phi.unsqueeze(0)
        L = torch.cos(phi).pow(4).sum(-1) + 2*torch.cos(phi).pow(2).prod(-1)
        L = L.abs() / phi.pow(2).mul(torch.arange(1, self.dim+1, device=phi.device).unsqueeze(0)).sum(-1).sqrt()
        return -L
    def gradient(self, phi, analytic: bool = False):
        """
        Calculates the gradient of the loss function with respect to phi.
        """
        if phi.dim() == 1:
            phi = phi.unsqueeze(0)
        
        if analytic:
            return 0
        
        if not phi.requires_grad:
            phi.requires_grad_(True)
        
        loss_val = self.loss(phi)
        if loss_val.dim() > 0 and len(loss_val) > 1:
            loss_val = loss_val.sum()
        grad = torch.autograd.grad(loss_val, phi)[0]
        return grad
    
    def hessian(self, phi, analytic: bool = False):
        """
        Calculates the Hessian matrix of the loss function with respect to phi.
        """
        if analytic:
            if phi.dim() == 1:
                phi = phi.unsqueeze(0)
            hessian_matrix = []
            return hessian_matrix

        if not phi.requires_grad:
            phi.requires_grad_(True)
        return torch.autograd.functional.hessian(self.loss, phi)

    def get_constraints(self, phi):
        if not self.constrained: return torch.tensor(0.0, device=phi.device)
        constraints = torch.zeros((phi.shape[0],1), device=phi.device)
        constraints = constraints + (0.75 - phi.prod(-1))
        constraints = constraints + (phi.sum(-1) - 7.5*self.dim)

        return constraints

    def __call__(self, phi: torch.tensor, x: torch.tensor = None):
        y = self.loss(phi)
        y = y.view(-1, 1)
        if self.noise and x is not None:
            y += x
        return y

class ThreeHump_stochastic_hits(ThreeHump):
    """
    Transforms the continuous Three-Hump Camel optimization problem into a
    stochastic, binary framework suitable for surrogate model training.

    The objective y(PHI) is embedded as the expected value of a binary
    hit/miss function h(PHI, X), i.e., y(PHI) = E_X[h(PHI, X)].
    """
    def __init__(self, n_samples:int = 100_000, 
                 phi_bounds = ((-2.3,-2),(2.3,2)), 
                 x_bounds = (-3,3),
                 reduction = 'mean'):
        super().__init__(phi_bounds=phi_bounds)
        self.x_bounds = x_bounds
        self.n_samples = n_samples
        self.reduction = reduction
        self.offset = 0.01  # To avoid zero probabilities
        self.y_max = self._find_max_value()
        self.normalize_factor = 1.0
        #self.normalize_factor = self._integrate_sensitivity(self.GetBounds().mean(0))

    def _find_max_value(self):
        """Numerically finds the max value of the function in its domain."""
        pp1 = torch.linspace(self._phi_bounds[0][0], self._phi_bounds[1][0], 500)
        pp2 = torch.linspace(self._phi_bounds[0][1], self._phi_bounds[1][1], 500)
        pp1, pp2 = torch.meshgrid(pp1, pp2, indexing='ij')
        phis = torch.stack([pp1.reshape(-1), pp2.reshape(-1)], dim=1)
        y_vals = super().loss(phis)
        y_vals *= 1+self.offset
        return torch.max(y_vals).mul(1.1).item()
    def _integrate_sensitivity(self, phi):
        """Normalizes the sensitivity values to ensure they sum to 1 over the domain."""
        n = 2000
        xx1 = torch.linspace(*self.x_bounds, n)
        xx1, xx2 = torch.meshgrid(xx1, xx1, indexing='ij')
        X_vis = torch.stack([xx1.reshape(-1), xx2.reshape(-1)], dim=1)
        return self.sensitivity(phi, X_vis).mean()
    
    def loss(self, phi):
        """Scales the camel function to the [0, 1] range to act as a probability."""
        return (super().loss(phi.reshape(-1,2)) / self.y_max + self.offset).clamp(max=1.0)

    def sample_x(self, device='cpu'):
        """Samples the random variable X. Here, X is uniformly distributed in [0, 1]."""
        return torch.empty(self.n_samples, 2).uniform_(*self.x_bounds).to(device)
    def get_weights(self,x):
        return torch.ones(x.shape[0], device=x.device)

    def sensitivity(self, phi, X):
        """
        The 'sensitivity' function s(X), which makes the hit probability
        dependent on the features of X.
        """
        x1 = X[:, 0]
        x2 = X[:, 1]
        radial_decay = torch.exp(-(x1**2 + x2**2) / 4.0)
        angular_part = 0.5 * (torch.sin(5 * (torch.atan2(x2, x1) + phi.sum(-1))) + 1)
        return radial_decay * angular_part / self.normalize_factor
    def probability_of_hit(self, phi, X):
        """
        Calculates the hit probability P(hit|X, PHI).
        """
        return (self.sensitivity(phi, X) * self.loss(phi)).clamp(0.0,1.0)
    def simulate(self, phi, X):
        """
        Calculates the comparison value. A hit occurs if this value > 0.
        """
        x0 = torch.rand(X.shape[0], device=X.device)
        return self.probability_of_hit(phi, X) - x0

    def _blackbox_loss(self, y):
        """
        The binary hit/miss function. This is what you train your surrogate on.
        Returns 1 (hit) or 0 (miss).
        """
        hits = (y > 0).int()
        return hits

    def __call__(self, phi, x=None):
        if x is None:
            x = self.sample_x()
        y = self.simulate(phi, x)
        y = self._blackbox_loss(y)
        if self.reduction == 'mean':
            return y.sum().float() / self.n_samples
        elif self.reduction == 'sum':
            return y.sum().float()
        return y
    
class HelicalValleyProblem:
    """
    Represents the Helical Valley Function, a benchmark for optimization algorithms.
    
    This is a 3-dimensional problem where the minimum lies at the bottom of a
    twisting, helical valley. The global minimum is at (1, 0, 0).
    
    The function's geometry is particularly challenging for first-order methods.
    """
    def __init__(self, dim: int = 3, phi_bounds=((-10.0, 10.0)), noise: float = 0) -> None:
        if dim != 3:
            raise ValueError("Helical Valley Function is defined for 3 dimensions.")
        self.dim = dim
        self._phi_bounds = ([phi_bounds[0]] * self.dim, [phi_bounds[1]] * self.dim)
        self.noise = noise

    def GetBounds(self, device='cpu') -> torch.Tensor:
        """Returns the bounds for the PHI parameters."""
        return torch.tensor(self._phi_bounds, device=device)

    def _theta(self, p0, p1):
        """Helper function to calculate the angle theta."""
        # This implementation handles the discontinuity at p0=0
        angle = torch.atan2(p1, p0) / (2 * np.pi)
        return torch.where(p0 >= 0, angle, angle + 0.5)

    def loss(self, phi):
        """The classic Helical Valley function."""
        if phi.dim() == 1:
            phi = phi.unsqueeze(0)
        
        p0, p1, p2 = phi[:, 0], phi[:, 1], phi[:, 2]
        
        theta_val = self._theta(p0, p1)
        
        term1 = 100 * (p2 - 10 * theta_val).pow(2)
        term2 = (torch.sqrt(p0.pow(2) + p1.pow(2)) - 1).pow(2)
        term3 = p2.pow(2)
        
        return term1 + term2 + term3
    def gradient(self, phi):
        """
        Calculates the gradient of the loss function with respect to phi.
        """
        if phi.dim() == 1:
            phi = phi.unsqueeze(0)
        
        
        if not phi.requires_grad:
            phi.requires_grad_(True)
        
        loss_val = self.loss(phi)
        
        if loss_val.dim() > 0 and len(loss_val) > 1:
            loss_val = loss_val.sum()

        grad = torch.autograd.grad(loss_val, phi)[0]
        return grad
    
    def hessian(self, phi, analytic: bool = False):
        """
        Calculates the Hessian matrix of the loss function with respect to phi.
        """
        if phi.dim() > 1:
            if phi.shape[0] > 1:
                raise ValueError("Hessian calculation is supported for a single phi vector at a time.")
            phi = phi.squeeze(0)

        # --- Autograd version ---
        phi_clone = phi.clone().requires_grad_(True).view(-1,3)
        return torch.autograd.functional.hessian(self.loss, phi_clone)

    def get_constraints(self, phi):
        return torch.tensor(0.0, device=phi.device)  # No constraints

    def __call__(self, phi: torch.tensor, x: torch.tensor = None):
        y = self.loss(phi)
        y = y.view(-1, 1)
        if self.noise and x is not None:
            y += x
        return y
    def get_weights(self,x):
        return torch.ones(x.shape[0], device=x.device)


class HelicalValley_stochastic_hits(HelicalValleyProblem):
    """
    Transforms the continuous Helical Valley problem into a stochastic,
    binary framework.
    """
    def __init__(self, n_samples: int = 100_000,
                 phi_bounds=((-10.0, 10.0)),
                 x_bounds=(-5.0, 5.0),
                 reduction='mean'):
        super().__init__(dim=3, phi_bounds=phi_bounds)
        self.x_bounds = x_bounds
        self.n_samples = n_samples
        self.reduction = reduction
        self.offset = 0.01  # To avoid zero probabilities
        
        self.y_max = self._find_max_value()
        self.normalize_factor = self._integrate_sensitivity(self.GetBounds().mean(0))

    def _find_max_value(self):
        """Numerically finds the max value of the function in its domain."""
        n_mc_samples = 1_000_000
        bounds = self.GetBounds()
        phis = (bounds[1] - bounds[0]) * torch.rand(n_mc_samples, self.dim) + bounds[0]
        y_vals = super().loss(phis)
        maxes = y_vals.argmax()
        return y_vals[maxes].mul(1.1).item()

    def _integrate_sensitivity(self, phi):
        """Integrates the sensitivity function over the input space."""
        X_sample = torch.empty(200000, 2).uniform_(*self.x_bounds)
        
        self.normalize_factor = 1.0
        mean_sensitivity = self.sensitivity(phi, X_sample).mean()

        return mean_sensitivity

    def loss(self, phi):
        """Scales Helical Valley function to the [0, 1] range."""
        return (super().loss(phi) / self.y_max + self.offset).clamp(max=1.0)

    def sample_x(self, device='cpu'):
        """Samples the random variable X from a 2D uniform distribution."""
        return torch.empty(self.n_samples, 2).uniform_(*self.x_bounds).to(device)

    def sensitivity(self, phi, X):
        """
        A 'spiral galaxy' sensitivity function whose orientation is controlled by PHI.
        This design ensures the integral over X is constant with respect to PHI,
        making the final objective a faithful representation of the Helical Valley problem.
        """
        if phi.dim() == 1:
            phi = phi.unsqueeze(0)
            
        x1_orig = X[:, 0]
        x2_orig = X[:, 1]
        
        # --- PHI controls the manifold's rotation ---
        # We use different components for a more complex rotation control
        angle = phi[:, 0] + phi[:, 2] # Components 0 and 2 control the angle

        # Rotate the coordinate system
        c, s = torch.cos(angle), torch.sin(angle)
        x_rot = x1_orig  - x2_orig * s
        y_rot = x1_orig * s + x2_orig * c

        # --- Define the static shape of the sensitivity manifold ---
        # This shape is now independent of PHI
        r = torch.sqrt(x_rot.pow(2) + y_rot.pow(2))
        theta = torch.atan2(y_rot, x_rot)
        
        # 1. A radial decay part
        radial_decay = torch.exp(-0.5 * r)
        
        arm_tightness = 1.0 + phi[:, 1].pow(2) # Make it positive and > 1
        num_arms = 2.0
        spiral_arms = (torch.sin(num_arms * (theta + arm_tightness * r)) + 1)
        
        spiral_sensitivity = radial_decay * spiral_arms
        
        return spiral_sensitivity / self.normalize_factor
    def probability_of_hit(self, phi, X):
        """
        Calculates the hit probability P(hit|X, PHI).
        """
        return (self.sensitivity(phi, X) * self.loss(phi)).clamp(0.0,1.0)

    def simulate(self, phi, X):
        """
        Performs a stochastic simulation. A hit occurs if a random number
        is less than the hit probability P(hit|X, PHI).
        """
        x0 = torch.rand(X.shape[0], device=X.device)
        hit_probability = self.probability_of_hit(phi, X)
        return hit_probability - x0

    def _blackbox_loss(self, y):
        """Converts the boolean output of simulate into an integer (0 or 1)."""
        return (y>0).int()

    def __call__(self, phi, x=None):
        if x is None:
            x = self.sample_x(device=phi.device)
            
        if phi.dim() == 1:
            phi = phi.unsqueeze(0)

        y = self.simulate(phi, x)
        y = self._blackbox_loss(y)
        
        if self.reduction == 'mean':
            return y.float().mean()
        elif self.reduction == 'sum':
            return y.float().sum()
        return y

class ShipMuonShield():

    idx_mag = {0: 'Z_gap[cm]', 1: 'Z_len[cm]',
               2: 'dXIn[cm]', 3: 'dXOut[cm]',
               4: 'dYIn[cm]', 5: 'dYOut[cm]',
               6: 'gapIn[cm]', 7: 'gapOut[cm]',
               8: 'ratio_yokesIn', 9: 'ratio_yokesOut',
               10: 'dY_yokeIn[cm]', 11: 'dY_yokeOut[cm]',
               12: 'XmgapIn[cm]', 13: 'XmgapOut[cm]',
               14: 'NI[A]'}
    
    SC_Ymgap = 15

    params = {
    'only_HA':[[0,115.50,50.00, 50.00, 119.00, 119.00, 2.00, 2.00, 1.00, 1.00, 50.00, 50.00, 0.00, 0.00, 1.9]],
    'hybrid_baseline': [
        [0, 120.5, 50.0, 50.0, 119.0, 119.0, 2.0, 2.0, 1.0, 1.0, 50.0, 50.0, 0.0, 0.0, 1.9],
        [40, 350.0, 45.0, 45.0, 25.0, 25.0, 35.0, 35.0, 2.67, 2.67, 120.15, 120.15, 0.0, 0.0, 5.1],
        [145, 200.0, 5.263, 55.632, 39.86, 5.278, 2.0, 2.0, 1.0, 0.9, 5.263, 50.069, 0.1, 0.1, -1.9],
        [10, 200.0, 33.016, 12.833, 123.27, 78.523, 22.414, 2.001, 0.959, 0.953, 31.663, 12.233, 30., 30., -1.9],
        [10, 195.0, 15.755, 77.293, 69.398, 152.828, 2.0, 35.402, 1.0, 0.9, 15.755, 69.564, 0.0, 0.0, -1.9]],

    'tokanut_v5': [
        [0,115.50,50.00, 50.00, 119.00, 119.00, 2.00, 2.00, 1.00, 1.00, 50.00, 50.00, 0.00, 0.00, 1.9],
        [10,225.38, 52.25, 78.36, 6.54, 9.37, 2.04, 40.36, 1.01, 1.05, 52.88, 82.20, 0.00, 0.00, 1.9],
        [10,274.43,52.47, 11.70, 41.59, 79.05, 2.19, 2.01, 1.03, 1.00, 54.15, 11.70, 0.00, 0.00, 1.9],
        [10,284.85, 33.05, 24.10, 55.13, 30.61, 90.36, 2.00, 1.00, 1.00, 33.05, 24.10, 0.11, 0.11, 1.9],
        [15,114.58, 5.62, 60.38, 31.37, 5.00, 2.00, 2.86, 1.00, 0.74, 5.59, 44.82, 0.00, 0.00, -1.9],
        [10,165.53, 90.85, 7.44, 137.45, 66.69, 10.09, 2.00, 0.88, 1.00, 80.16, 7.41, 0.00, 0.00, -1.9],
        [10,244.28,9.81, 47.60, 19.59, 164.19, 2.00, 2.05, 1.00, 0.89, 9.81, 42.16, 0.10, 0.10, -1.9]
    ],
    'Piet_solution': [
        [0, 120.5, 50.0, 50.0, 119.0, 119.0, 2.0, 2.0, 1.0, 1.0, 50.0, 50.0, 0.0, 0.0, 1.9],
        [10, 485.5, 30.0, 31.0, 27.0, 43.0, 6.0, 6.0, 4.29, 4.29, 34.0, 34.0, 0.0, 0.0, 1.9],
        [10, 245.0, 31.0, 35.0, 43.0, 56.0, 6.0, 6.0, 4.29, 3.79, 44.0, 44.0, 0.0, 0.0, 1.9],
        [10, 255.0, 3.0, 18.6, 56.0, 56.0, 6.0, 6.0, 54.8, 8.2, 44.0, 44.0, 0.0, 0.0, -1.7],
        [10, 197.0, 18.6, 31.8, 56.0, 56.0, 6.0, 6.0, 8.20, 4.47, 44.0, 44.0, 30.0, 30.0, -1.7],
        [10, 197.0, 31.8, 45.0, 56.0, 56.0, 6.0, 6.0, 4.47, 2.87, 44.0, 44.0, 0.0, 0.0, -1.7]],
    "supernut_v2": [
        [0.00, 120.50, 50.00, 50.00, 119.00, 119.00, 2.00, 2.00, 1.00, 1.00, 50.00, 50.00, 0.00, 0.00, 1.90],
        [41.00, 263.52, 45.00, 45.00, 25.00, 25.00, 65.80, 50.65, 2.44, 3.29, 107.83, 120.28, 0.00, 0.00, 5.70],
        [284.00, 249.02, 5.39, 21.56, 29.39, 17.26, 68.87, 65.04, 0.94, 0.84, 5.26, 50.07, 77.41, 77.41, -1.90],
        [10.00, 177.17, 34.20, 8.87, 100.69, 32.75, 148.15, 38.30, 0.99, 0.65, 31.66, 12.23, 51.04, 51.04, -1.90],
        [10.00, 172.25, 21.44, 62.75, 197.10, 143.48, 6.63, 22.10, 0.94, 0.94, 15.76, 69.56, 0.06, 0.06, -1.90]],
    'stellatryon_v2': [
        [0, 120.5, 50.0, 50.0, 119.0, 119.0, 2.0, 2.0, 1.0, 1.0, 50.0, 50.0, 0.0, 0.0, 1.9],#59120.14],
        [10, 500.0, 67.1, 79.92, 27.0, 43.0, 5.0, 5.0, 1.38, 1.06, 67.1, 79.92, 0.0, 0.0, 1.9],#38637.97],
        [10, 285.48, 53.12, 49.56, 43.0, 56.0, 5.03, 5.0, 2.11, 2.4, 53.12, 49.56, 0.0, 0.0, 1.9],#38266.52],
        [10, 237.53, 2.73, 3.68, 56.0, 56.0, 5.0, 5.21, 60.44, 45.63, 2.73, 3.68, 0.5, 0.5, -1.9],
        [10, 90.0, 1.0, 77.12, 56.0, 56.0, 5.27, 5.0, 140.93, 0.88, 1.0, 77.12, 30.0, 30.0, -1.9],
        [10, 238.82, 30.03, 40.0, 56.0, 56.0, 5.0, 5.01, 4.83, 3.37, 30.03, 40.0, 0.0, 0.0, -1.9]
    ],
    'stellatryon_v3': [
                        [0,  115.5,  50.00, 50.00, 119.00, 119.00, 2.00, 2.00, 1.00, 1.00, 50.00, 50.00, 0.00, 0.00, 1.90], 
                        [20, 250, 67.10, 79.92, 27.00, 43.00, 8.00, 8.00, 1.38, 1.06, 67.10, 79.92, 0.00, 0.00, 1.90], 
                        [10, 250, 53.12, 49.56, 43.00, 56.00, 8.00, 8.00, 2.11, 2.40, 53.12, 49.56, 0.00, 0.00, 1.90], 
                        [10, 250, 53.12, 49.56, 43.00, 56.00, 8.00, 8.00, 2.11, 2.40, 53.12, 49.56, 0.00, 0.00, 1.90], 
                        [10, 232.53, 2.73, 3.68, 56.00, 56.00, 8.00, 8.00, 60.44, 45.63, 2.73, 3.68, 0.50, 0.50, 0], 
                        [10, 233.82, 30.03, 40.00, 56.00, 56.00, 8.00, 8.00, 4.83, 3.37, 30.03, 40.00, 0.00, 0.00, -1.4]
    ],
    "warm_baseline": [
                [0.,120.5, 50.00,  50.00, 119.00, 119.00,   2.00,   2.00, 1.00,1.0,50.00,  50.00,0.0, 0.00, 1.9],
                  [10, 250, 72.00, 51.00, 29.00, 46.00, 10.00, 7.00, 1.00,1.0,72.00, 51.00,0.0, 0.00, 1.9],
                  [10, 250, 54.00, 38.00, 46.00, 130.00, 14.00, 9.00, 1.00,1.0,54.00, 38.00,0.0, 0.00, 1.9],
                  [10, 250, 10.00, 31.00, 35.00, 31.00, 51.00, 11.00, 1.00,1.0,10.00, 31.00,0.0, 0.00, 1.9],
                  [10, 150, 5.00, 32.00, 54.00, 24.00, 8.00, 8.00, 1.00,1.0,5.00, 32.00,0.0, 0.00, -1.9],
                  [10, 150, 22.00, 32.00, 130.00, 35.00, 8.00, 13.00, 1.00,1.0,22.00, 32.00,30.0, 30.00, -1.9],
                  [10, 251, 33.00, 77.00, 85.00, 90.00, 9.00, 26.00, 1.00,1.0,33.00, 77.00,0.0, 0.00, -1.9]
    ],
    "tokanut_v6": [[0.00, 115.50, 50.00, 50.00, 119.00, 119.00, 2.00, 2.00, 1.00, 1.00, 50.00, 50.00, 0.00, 0.00, 1.90],
                    [10.00, 204.65, 42.00, 110.11, 5.01, 10.03, 2.00, 107.67, 0.99, 1.23, 41.58, 135.41, 0.00, 0.00, 1.90],
                    [10.00, 264.06, 68.69, 13.83, 41.03, 87.47, 2.00, 2.00, 1.12, 0.99, 76.89, 13.69, 0.00, 0.00, 1.90],
                    [10.00, 260.60, 29.07, 5.00, 24.40, 31.81, 30.74, 2.00, 1.01, 0.99, 29.29, 4.95, 0.00, 0.00, 1.90],
                    [10.00, 112.93, 5.00, 74.46, 22.04, 5.01, 8.42, 5.07, 1.01, 0.66, 5.05, 49.21, 0.00, 0.00, -1.91],
                    [10.00, 139.03, 104.13, 5.02, 105.38, 41.39, 45.92, 2.00, 0.92, 1.01, 95.58, 5.07, 0.00, 0.00, -1.86],
                    [10.00, 342.17, 9.97, 47.14, 5.00, 141.48, 2.00, 2.00, 1.01, 0.95, 10.07, 44.62, 0.10, 0.10, -1.9]],
    "tokanut_v6_snd": [[0.00, 120.50, 50.00, 50.00, 119.00, 119.00, 2.00, 2.00, 1.00, 1.00, 50.00, 50.00, 0.00, 0.00, 1.90],
                    [10.00, 220.99, 24.34, 134.21, 5.00, 5.00, 2.08, 55.24, 0.99, 0.99, 24.10, 132.86, 0.00, 0.00, 1.90],
                    [10.00, 262.47, 58.75, 5.98, 5.42, 99.45, 2.00, 2.00, 1.08, 0.99, 63.45, 5.92, 0.00, 0.00, 1.90],
                    [10.00, 245.98, 24.69, 14.57, 91.24, 17.19, 53.62, 2.19, 1.57, 0.99, 38.87, 14.42, 0.00, 0.00, 1.90],
                    [20.00, 175.66, 9.80, 29.71, 67.85, 12.50, 2.00, 5.15, 1.01, 0.35, 9.90, 9.08, 0.26, 0.26, -1.90],
                    [10.00, 120.88, 51.35, 6.09, 94.40, 5.00, 7.25, 2.20, 1.01, 0.67, 51.87, 4.08, 30.00, 30.00, -1.90],
                    [10.00, 300.00, 30.00, 41.89, 71.01, 121.81, 2.00, 2.13, 1.00, 1.01, 30.13, 42.31, 0.02, 0.02, -1.90]],
    }
    
    parametrization = {
        "hybrid_idx": (
            make_index(1, [0,1,4,6,7,8,9,10,11,12]) +
            make_index(2, list(range(0,10)) + [12, 14]) +
            make_index(3, list(range(1,10)) + [12, 14]) +
            make_index(4, list(range(1,10)) + [12, 14])
        ),
        "warm_idx": (
            make_index(1, list(range(1,10)) + [12]) +
            make_index(2, list(range(1,10)) + [12]) +
            make_index(3, list(range(1,10)) + [12]) +
            make_index(4, list(range(10)) + [12,14]) +
            make_index(5, list(range(1,10)) + [12,14]) +
            make_index(6, list(range(1,10)) + [14])
        ),
        "robustness": (
            make_index(0, [1,2,8,14]) +
            make_index(1, [1,2,8,14]) +
            make_index(2, [1,2,8,14]) +
            make_index(3, [1,2,8,14]) +
            make_index(4, [1,2,8,14]) +
            make_index(5, [1,2,8,14])
        ),
        "piet_idx": (
            make_index(1, [1,2,4,3,6,7]) +
            make_index(2, [1,2,4,3,6,7]) +
            make_index(3, [1,2,4,3,6,7] + [12,14]) +
            make_index(4, [1,2,4,3,6,7] + [12,14]) +
            make_index(5, [1,2,4,3,6,7] + [14])
        ),
        "all_7":sum((make_index(i, list(range(12)) + [12,14]) for i in range(7)), []),
        "stellatryon": (
            make_index(1, [1,2,4,8,9]) +
            make_index(2, [1,2,4,8,9]) +
            make_index(3, [1,2,4,8,9]) +
            make_index(4, [1]) + # SET EQUAL TO M3
            make_index(5, [1,2,4,8,9] + [14])
        ),
    }


    
    DEFAULT_PHI = torch.tensor(params['warm_baseline'])
    n_params = 15
    #initial_phi = DEFAULT_PHI.clone()
    MUON = 13

    def __init__(self,
                 W0:float = 13E6,
                 L0:float = 29.7,
                 cores:int = 45,
                 n_samples:int = 0,
                 sensitive_plane:float = [{'dz': 0.01, 'dx': 4, 'dy': 6,'position': 82}],
                 apply_det_loss:bool = True,
                 cost_loss_fn:bool = 'exponential',
                 loss_fn:bool = 'continuous',
                 fSC_mag:bool = False,
                 uniform_fields:bool = False,
                 cavern:bool = True,
                 seed:int = None,
                 x_margin = 2.1,
                 y_margin = 3,
                 SmearBeamRadius = 5,
                 dimensions_phi = 105,
                muons_file = os.path.join(PROJECTS_DIR,'MuonsAndMatter/data/muons/subsample_biased_v4.npy'),
                fields_file = os.path.join(PROJECTS_DIR,'MuonsAndMatter/data/outputs/fields.h5'),
                extra_magnet = False,
                cut_P:float = None,
                initial_phi:torch.tensor = None,
                SND:bool = False,
                add_target:bool = True,
                decay_vessel_sensitive:bool = False,
                use_diluted:bool = False,
                parallel:bool = False,
                use_B_goal:bool = True,
                cost_as_constraint:bool = False,
                reduction:str = 'mean'
                 ) -> None:
        
        self.x_margin = x_margin
        self.y_margin = y_margin
        self.W0 = W0
        self.L0 = L0
        self.cores = cores
        self.muons_file = muons_file
        self.n_samples = n_samples
        self._sum_weights = n_samples
        self.cost_loss_fn = cost_loss_fn
        self.loss_fn = loss_fn
        self.reduction = reduction
        self.sensitive_plane = sensitive_plane
        self.fSC_mag = fSC_mag
        self.uniform_fields = uniform_fields
        self.seed = seed
        self.cavern = cavern
        self.apply_det_loss = apply_det_loss
        self.extra_magnet = extra_magnet    
        self.SmearBeamRadius = SmearBeamRadius
        self.lambda_constraints = 50
        self.cut_P = cut_P
        self.use_B_goal = use_B_goal
        self.SND = SND
        self.add_target = add_target
        self.decay_vessel_sensitive = decay_vessel_sensitive
        self.use_diluted = use_diluted
        self.parallel = parallel
        self.cost_as_constraint = cost_as_constraint    

        key = None
        if initial_phi is not None:
            self.DEFAULT_PHI = torch.as_tensor(initial_phi).view(-1,self.n_params)
            
        if isinstance(dimensions_phi,list):
            self.params_idx = torch.tensor(dimensions_phi)
        else:
            for key, indexes in self.parametrization.items():
                if len(indexes) == dimensions_phi:
                    self.params_idx = torch.tensor(indexes)
                    print(f'Using parametrization: {key}')
                    break
            else: 
                self.params_idx = torch.tensor(sum((make_index(i, list(range(self.n_params))) for i in range(len(self.DEFAULT_PHI))), []))
        self.key = key
        self.initial_phi = apply_index(self.DEFAULT_PHI, self.params_idx).flatten()
        self.dimensions_phi = self.dim = len(self.initial_phi)
        
        self.n_magnets = len(self.DEFAULT_PHI)

        self.materials_directory = os.path.join(PROJECTS_DIR,'MuonsAndMatter/data/materials')
        from muons_and_matter import run, get_field, estimate_electrical_cost, RESOL_DEF
        self.estimate_electrical_cost = estimate_electrical_cost
        self.run_muonshield = run
        self.run_magnet = get_field
        self.resol = RESOL_DEF
        self.fields_file = fields_file

    def sample_x(self,phi=None, idx = None):
        print('Sampling muons')
        t1 = time.time()
        if idx is None: idx = slice(None)
        else: idx = slice(*idx)
        if self.muons_file.endswith('.npy'):
            mmap_array = np.load(self.muons_file, mmap_mode='r')
            x = mmap_array[idx].copy()
            if 0<self.n_samples<x.shape[0]: 
                indices = np.random.choice(x.shape[0], self.n_samples, replace=False)
                x = x[indices]
        elif self.muons_file.endswith('.h5'):
            if self.n_samples == 0:
                with h5py.File(self.muons_file, "r") as f:
                    n = f["px"].shape[0]
            else: n = self.n_samples
            start, stop, step = idx.indices(n)
            n = (stop - start + (step - 1)) // step
            read_slice = slice(start, stop, step) 
            
            x = np.empty((n, 8), dtype=np.float32)
            with h5py.File(self.muons_file, "r") as f:
                n_total = f["px"].shape[0]
                
                for j, feat in enumerate(["px", "py", "pz", "x", "y", "z", "pdg", "weight"]):
                    col = np.empty(n, dtype=np.float32)  
                    if 0 < n < n_total:
                        f[feat].read_direct(col, read_slice)
                    else:
                        f[feat].read_direct(col, idx)      
                    
                    x[:, j] = col     
        print(f'Sampling muons took {time.time()-t1:.2f} seconds')    
        x = torch.from_numpy(x)                  
        self._sum_weights = x[:, -1].sum()    
        return x
    def get_weights(self, x):
        return x[:, -1]
    def simulate_mag_fields(self,phi:torch.tensor, cores:int = 7):
        phi = self.add_fixed_params(phi)
        z_gap, dZ, dXIn, dXOut, dYIn, dYOut, gapIn, gapOut, ratio_yokesIn, ratio_yokesOut, \
        dY_yokeIn, dY_yokeOut, XmgapIn, XmgapOut = phi[:, :14].T
        length = (dZ.sum().mul(2) + z_gap.sum()).item()
        dX_all = torch.max(
            dXIn * (1 + ratio_yokesIn) + gapIn + XmgapIn,
            dXOut * (1 + ratio_yokesOut) + gapOut + XmgapOut
        )
        dY_all = torch.max(dYIn + dY_yokeIn, dYOut + dY_yokeOut)
        max_x = dX_all.max().item()
        max_y = dY_all.max().item()
        max_x = int((max_x // self.resol[0]) * self.resol[0])
        max_y = int((max_y // self.resol[1]) * self.resol[1])
        d_space = ((0,max_x+30), (0,max_y+30), (-50, int(((length+200) // self.resol[2]) * self.resol[2])))
        self.run_magnet(True,phi.cpu().numpy(),file_name = self.fields_file,d_space = d_space, cores = cores, fSC_mag = self.fSC_mag, use_diluted = self.use_diluted,NI_from_B_goal = self.use_B_goal)

    def simulate(self,phi:torch.tensor,muons = None, return_all = False, simulate_fields = True): 
        phi = self.add_fixed_params(phi)
        if muons is None: muons = self.sample_x()
        self._sum_weights = muons[:, -1].sum()
        workloads = split_array(muons,self.cores)
        assert phi.shape[1] == 15, f"Expected phi to have 15 columns, got {phi.shape}"
        if simulate_fields and (not self.uniform_fields): 
            print('SIMULATING MAGNETIC FIELDS')
            self.simulate_mag_fields(phi)
        run_partial = partial(self.run_muonshield, 
                      params=phi.cpu().numpy(), 
                      return_cost=False, 
                      fSC_mag=self.fSC_mag, 
                      sensitive_film_params=self.sensitive_plane, 
                      add_cavern=self.cavern, 
                      simulate_fields=False, 
                      field_map_file=self.fields_file, 
                      return_nan=return_all, 
                      seed=self.seed, 
                      draw_magnet=False, 
                      SmearBeamRadius=self.SmearBeamRadius, 
                      add_target=self.add_target, 
                      keep_tracks_of_hits=False, 
                      extra_magnet=self.extra_magnet,
                      NI_from_B = self.use_B_goal,
                     add_decay_vessel = self.decay_vessel_sensitive,
                      use_diluted = self.use_diluted,
                      SND = self.SND)
        with Pool(self.cores) as pool:
            result = pool.map(run_partial, workloads)
        print('SIMULATION FINISHED')
        all_results = []
        for rr in result:
            resulting_data = rr
            if resulting_data.size == 0: continue
            all_results += [resulting_data]
        if len(all_results) == 0:
            return torch.tensor([[],[],[],[],[],[],[],[]], device=phi.device)
        all_results = np.concatenate(all_results, axis=0).T
        if all_results.dtype != object: # Only convert to tensor if all_results is a numeric array (not array of dicts)
            all_results = torch.as_tensor(all_results, device=phi.device, dtype=torch.get_default_dtype())
        return all_results
    def is_hit(self, px, py, pz, x, y, z, particle, factor=None):
        mask = (torch.abs(x) <= self.x_margin) & (torch.abs(y) <= self.y_margin) 
        mask = mask & (torch.abs(z - self.sensitive_plane[-1]['position']) <= self.sensitive_plane[-1]['dz'])
        mask = mask & (torch.abs(particle).to(torch.int)==self.MUON)
        if self.cut_P is not None: 
            p = torch.sqrt(px**2+py**2+pz**2)
            mask = mask & p.ge(self.cut_P)
        return mask.to(torch.bool)
    def _continuous_loss(self, px, py, pz, x, y, z, particle, weight=None):
        if x.numel() == 0 or x.isnan().all():
            return torch.tensor(0.0, device=x.device)
        charge = -1 * torch.sign(particle)
        mask = self.is_hit(px, py, pz, x, y, z, particle).to(torch.bool)
        loss = torch.zeros_like(x)
        loss[mask] = torch.sqrt(1 + (charge[mask] * x[mask] - self.x_margin) / (2 * self.x_margin))
        return loss
    def _blackbox_loss(self, px, py, pz, x, y, z, particle, weight=None):
        if self.loss_fn == 'continuous':
            loss =  self._continuous_loss(px,py,pz,x,y,z,particle, weight)
        elif self.loss_fn == 'hits':
            loss = self.is_hit(px, py, pz, x, y, z, particle).to(torch.float)
        if (self.reduction != 'none') and (weight is not None):
            loss = weight * loss
        return loss

    def get_total_length(self, phi):
        phi = self.add_fixed_params(phi)
        length = phi[:,1].sum().mul(2) + phi[:,0].sum()
        return length / 100

    def get_electrical_cost(self,phi):
        '''Adapt for multidimensional phi'''
        phi = self.add_fixed_params(phi).detach().cpu().numpy()
        cost = 0
        for idx,params in enumerate(phi):
            Ymgap = 0
            if self.fSC_mag and idx == 1: 
                yoke_type = 'Mag2'
                Ymgap = self.SC_Ymgap
            elif self.use_diluted: yoke_type = 'Mag1'
            else: yoke_type = 'Mag3' if params[14]<0 else 'Mag1'
            cost+= self.estimate_electrical_cost(params,yoke_type,Ymgap,materials_directory = self.materials_directory, NI_from_B = self.use_B_goal)
        return cost

    def get_iron_cost(self, phi):
        #TODO use directly create_magnet from muons_and_matter
        '''Adapt for multidiomensional phi
        make electrical cost esimation with torch'''
        material =  'aisi1010.json'
        with open(os.path.join(self.materials_directory,material)) as f:
            iron_material_data = json.load(f)
        density = iron_material_data['density(g/m3)']*1E-9
        phi = self.add_fixed_params(phi)
        volume = 0
        for idx,params in enumerate(phi):
            Ymgap = 0
            if self.fSC_mag and idx == 1: 
                Ymgap = self.SC_Ymgap
            dZ = params[1]
            dX = params[2]
            dX2 = params[3]
            dY = params[4]
            dY2 = params[5]
            gap = params[6]
            gap2 = params[7]
            ratio_yoke_1 = params[8]
            ratio_yoke_2 = params[9]
            dY_yoke_1 = params[10]
            dY_yoke_2 = params[11]
            X_mgap_1 = params[12]
            X_mgap_2 = params[13]
            corners = torch.tensor([
            [X_mgap_1+dX, 0, 0],
            [X_mgap_1 + dX, dY, 0],
            [0, dY, 0],
            [0, 0, 0],
            [X_mgap_2+dX2,0, 2*dZ],
            [X_mgap_2+dX2, dY2, 2*dZ],
            [0, dY2, 2*dZ],
            [0, 0, 2*dZ]
            ])
            volume += compute_solid_volume(corners)
            corners = torch.tensor([
            [X_mgap_1 + dX + gap, 0, 0],
            [X_mgap_1 + dX + gap + dX * ratio_yoke_1, 0, 0],
            [X_mgap_1 + dX + gap + dX * ratio_yoke_1, dY + Ymgap, 0],
            [X_mgap_1 + dX + gap, dY + Ymgap, 0],
            [X_mgap_2 + dX2 + gap2, 0, 2 * dZ],
            [X_mgap_2 + dX2 + gap2 + dX2 * ratio_yoke_2, 0, 2 * dZ],
            [X_mgap_2 + dX2 + gap2 + dX2 * ratio_yoke_2, dY2 + Ymgap, 2 * dZ],
            [X_mgap_2 + dX2 + gap2, dY2 + Ymgap, 2 * dZ],
            ])
            volume += compute_solid_volume(corners)

            corners = torch.tensor([
            [X_mgap_1, dY, 0],
            [X_mgap_1 + dX + gap + dX * ratio_yoke_1, dY, 0],
            [X_mgap_1 + dX + gap + dX * ratio_yoke_1, dY + dY_yoke_1, 0],
            [X_mgap_1, dY + dY_yoke_1, 0],
            [X_mgap_2, dY2, 2 * dZ],
            [X_mgap_2 + dX2 + gap2 + dX2 * ratio_yoke_2, dY2, 2 * dZ],
            [X_mgap_2 + dX2 + gap2 + dX2 * ratio_yoke_2, dY2 + dY_yoke_2, 2 * dZ],
            [X_mgap_2, dY2 + dY_yoke_2, 2 * dZ],
            ])
            volume += compute_solid_volume(corners)
        M_iron = 4*volume*density    
        C_iron = M_iron*(iron_material_data["material_cost(CHF/kg)"]
                     +  iron_material_data["manufacturing_cost(CHF/kg)"])
        return C_iron.detach()
    def get_total_cost(self,phi):
        try:
            M = self.get_iron_cost(phi) + self.get_electrical_cost(phi)
        except Exception as e:
            print(f"Error in cost estimation of parameters: {phi}")
            print(f"Error: {e}")
            raise
        return M

    def cost_loss(self,W,L = None):
        if self.cost_loss_fn == 'exponential':
            return (1+torch.exp(10*(W-self.W0)/self.W0))
        elif self.cost_loss_fn == 'linear':
            return W/self.W0
        elif self.cost_loss_fn == 'quadratic':
            return (W/self.W0)**2
        elif self.cost_loss_fn == 'linear_length':
            return W/(1-L/self.L0)
        else: return 1
    def __call__(self,phi,muons = None):
        if phi.dim()>1:
            y = []
            for p in phi:
                y.append(self(p))
            return torch.stack(y)
        M = self.get_total_cost(phi)
        jump = self.get_constraints(phi) > 10
        if not self.cost_as_constraint: jump = jump | (M > ((6 * np.log(10) / 10 + 1) * self.W0))
        jump = jump & (self.reduction != 'none')
        if jump: 
            return torch.ones((1,1),device=phi.device)*1E6
        try: loss = self.simulate(phi, muons, return_all=(self.reduction=='none'))
        except Exception as e:
            print(f"Error occurred with input: {self.add_fixed_params(phi)}")
            print(e)
            raise
        loss = self._blackbox_loss(*loss)
        if self.reduction == 'mean':
            return loss.sum()*1e6 / self._sum_weights
        elif self.reduction == 'sum':
            return loss.sum()
        if self.apply_det_loss:
            loss = loss + self._apply_deterministic_loss(phi, loss)
        return loss


    def GetBounds(self,device = torch.device('cpu')):
        z_gap = (10,50)
        magnet_lengths = (100, 350)
        dY_bounds = (5, 250)
        dY_yoke_bounds = (5, 450)
        if self.use_B_goal: NI_bounds = (0.1,1.9)
        else: NI_bounds = (0.0, 70e3)
        if self.use_diluted:
            dX_bounds = (1, 85)
            gap_bounds = (5, 80)
            inner_gap_bounds = (0., 30.)
            yoke_bounds = (1,141)
        else:
            dX_bounds = (5, 250)
            gap_bounds = (2, 150)
            yoke_bounds = (0.99,3)
            inner_gap_bounds = (0., 150.)

        bounds_low = torch.tensor([[z_gap[0],magnet_lengths[0], 
                       dX_bounds[0], dX_bounds[0], 
                       dY_bounds[0], dY_bounds[0],
                       gap_bounds[0], gap_bounds[0],
                       yoke_bounds[0], yoke_bounds[0],
                       dY_yoke_bounds[0], dY_yoke_bounds[0],
                       inner_gap_bounds[0], inner_gap_bounds[0],
                       NI_bounds[0]] for _ in range(self.n_magnets)],device=device,dtype=torch.get_default_dtype())
        bounds_high = torch.tensor([[z_gap[1],magnet_lengths[1], 
                        dX_bounds[1], dX_bounds[1], 
                        dY_bounds[1], dY_bounds[1],
                        gap_bounds[1], gap_bounds[1],
                        yoke_bounds[1], yoke_bounds[1],
                        dY_yoke_bounds[1], dY_yoke_bounds[1],
                        inner_gap_bounds[1], inner_gap_bounds[1],
                        NI_bounds[1]] for _ in range(self.n_magnets)],device=device,dtype=torch.get_default_dtype())
        bounds_low[0,0] = 0

        inverted_polarity = self.DEFAULT_PHI[:, 14] < 0
        if inverted_polarity.any():
            bounds_low[inverted_polarity, 14] = -NI_bounds[1]
            bounds_high[inverted_polarity, 14] = -NI_bounds[0]
            if not self.use_diluted:
                bounds_low[inverted_polarity, 8] = 1.0 / yoke_bounds[1]
                bounds_high[inverted_polarity, 8] = 1.0 / yoke_bounds[0]
                bounds_low[inverted_polarity, 9] = 1.0 / yoke_bounds[1]
                bounds_high[inverted_polarity, 9] = 1.0 / yoke_bounds[0]

        if self.use_diluted:
            if self.key and not self.key.startswith("stella"):
                bounds_low[0,6] = 2.0
                bounds_low[0,7] = 2.0
                bounds_low[0,1] = 120.5
                bounds_low[1,1] = 485.5
                bounds_low[2,1] = 285
                bounds_low[3:,1] = 30
                bounds_high[:,1] = 500
            else:
                ###### MODIFIED BY GUGLIELMO ###################
                # Z LEN All magnets
                bounds_low[1:3,1] = 200 #cm
                bounds_high[1:,1] = 300 #cm
                #M4
                bounds_low[-2,1] = 160 #cm
                bounds_low[-1,-1] = 0 # T or NI
                #M5
                bounds_low[-1,1] = 336/2 #cm
                bounds_high[-1,1] = 950/2 #cm
                bounds_low[-1,-1] = -1.4 if self.use_B_goal else -70e3/1.9*1.4 # T or NI
                bounds_high[-1,-1] = 0 if self.use_B_goal else 0 # T or NI
        
        if self.SND:
            bounds_low[-2,1] = 90
            bounds_high[-2,1] = 350
            bounds_low[-2,[12,13]] = 30
            bounds_high[-2,[12,13]] = 150
            bounds_low[-1,1] = 170
            bounds_high[-1,1] = 350
            bounds_low[-1,2] = 30
            bounds_high[-1,2] = 250
            bounds_low[-1,3] = 40
            bounds_high[-1,3] = 250

        if self.fSC_mag: 
            bounds_low[1,0] = 30
            bounds_high[1,0] = 300
            bounds_low[2,0] = 30
            bounds_high[2,0] = 300
            bounds_low[1,1] = 50
            bounds_high[1,1] = 400
            bounds_low[1,2] = 30
            bounds_high[1,2] = 50
            bounds_low[1,3] = 30
            bounds_high[1,3] = 50
            bounds_low[1,4] = 15
            bounds_high[1,4] = 30
            bounds_low[1,5] = 15
            bounds_high[1,5] = 30
            bounds_low[1,6] = 15
            bounds_high[1,6] = 150
            bounds_low[1,7] = 1.0
            bounds_high[1,7] = 4
            bounds_low[1,8] = 1.0
            bounds_high[1,8] = 4
            bounds_low[1,9] = 1.0
            bounds_high[1,9] = 4
        bounds_low = apply_index(bounds_low, self.params_idx).flatten()
        bounds_high = apply_index(bounds_high, self.params_idx).flatten()
        bounds = torch.stack([bounds_low, bounds_high])
        return bounds

    def add_fixed_params(self, phi: torch.Tensor):
    
        if phi.numel() != (self.n_magnets * self.n_params):
            new_phi = self.DEFAULT_PHI.clone().to(phi.device)
            new_phi = new_phi.index_put(
                (self.params_idx[:,0], self.params_idx[:,1]),
                phi
            )
            # new_phi[0][3] = new_phi[0][2]
            new_phi = new_phi.index_put(
                (torch.tensor([0]), torch.tensor([3])),
                new_phi[0, 2]
            )
            # new_phi[0][5] = new_phi[0][4]
            new_phi = new_phi.index_put(
                (torch.tensor([0]), torch.tensor([5])),
                new_phi[0, 4]
            )
            
            # new_phi[:,13] = new_phi[:,12]
            all_rows = torch.arange(new_phi.size(0))

            # new_phi[:,13] = new_phi[:,12]
            new_phi = new_phi.index_put(
                (all_rows, torch.tensor(13)),
                new_phi[:, 12]
            )

            # new_phi[:,10] = new_phi[:, 2] * new_phi[:, 8]
            values_10 = new_phi[:, 2] * new_phi[:, 8]
            new_phi = new_phi.index_put(
                (all_rows, torch.tensor(10)), 
                values_10
            )

            # new_phi[:,11] = new_phi[:, 3] * new_phi[:, 9]
            values_11 = new_phi[:, 3] * new_phi[:, 9]
            new_phi = new_phi.index_put(
                (all_rows, torch.tensor(11)), 
                values_11
            )
            if self.fSC_mag:
                # new_phi[1][3] = new_phi[1][2]
                new_phi = new_phi.index_put(
                    (torch.tensor([1]), torch.tensor([3])),
                    new_phi[1, 2]
                )
                # new_phi[1][5] = new_phi[1][4]
                new_phi = new_phi.index_put(
                    (torch.tensor([1]), torch.tensor([5])),
                    new_phi[1, 4]
                )
            if self.use_diluted:
                rows = torch.arange(1, new_phi.size(0), device=new_phi.device)
                
                if self.key and not self.key.startswith("stella"):
                    # new_phi[1:,10] = new_phi[1:, 2]
                    new_phi = new_phi.index_put(
                        (rows, torch.tensor([10])),
                        new_phi[1:, 2]
                    )
                    # new_phi[1:,11] = new_phi[1:, 3]
                    new_phi = new_phi.index_put(
                        (rows, torch.tensor([11])),
                        new_phi[1:, 3]
                    )
                    
                    # new_phi[1:,8] = ...
                    piet = torch.tensor(self.params['Piet_solution'], dtype = new_phi.dtype, device=new_phi.device)
                    values_8 = (piet[1:,2] * piet[1:,8] + piet[1:,6] + piet[1:,2] - new_phi[1:,12] - new_phi[1:,6] - new_phi[1:,2]) / new_phi[1:,2]
                    new_phi = new_phi.index_put(
                        (rows, torch.tensor([8])),
                        values_8
                    )
                    
                    # new_phi[1:,9] = ...
                    values_9 = (piet[1:,3] * piet[1:,9] + piet[1:,7] + piet[1:,3] - new_phi[1:,13] - new_phi[1:,7] - new_phi[1:,3]) / new_phi[1:,3]
                    new_phi = new_phi.index_put(
                        (rows, torch.tensor([9])),
                        values_9
                    )   
                else:
                    # X_yoke 1 = X_yoke2 => ∆Xcore,1 · (1 + Ryoke/core,1) = ∆Xcore,2 · (1 + Ryoke/core,2) => ∆Xcore,2 = ∆Xcore,1 · (1 + Ryoke/core,1)/ (1 + Ryoke/core,2)
                    values_3 = new_phi[rows, 2] * (1 + new_phi[rows, 8]) / (1 + new_phi[rows, 9])
                    new_phi.index_put_((rows, torch.tensor([3])), values_3)
                    
                    
                    # ∆Yyoke,1 = ∆Yyoke,2 = 110% · max(∆Xcore,1, ∆Xcore,2)
                    row_max = torch.max(new_phi[rows, 2], new_phi[rows, 3]) * 1.1

                    new_phi.index_put_((rows, torch.tensor([10])), row_max)
                    new_phi.index_put_((rows, torch.tensor([11])), row_max)
                    
                    # Yyoke,1 = Yyoke,2 ⇒ ∆Ycore,1 + ∆Yyoke,1 = ∆Ycore,2 + ∆Yyoke,2 → ∆Ycore,1 = ∆Ycore,2.
                    new_phi.index_put_((rows, torch.tensor([5])), new_phi[rows, 4])
                    
                    #M4 constraint:
                    new_phi[-2,2:14] = new_phi[-3, 2:14]
                
        else:
            new_phi = phi
            
        return new_phi.view(self.n_magnets, self.n_params)
        
    def _apply_deterministic_loss(self,phi,y):
        M = self.get_total_cost(phi)
        loss = self.cost_loss(M)*(y+1)
        loss = loss + self.get_constraints(phi)
        loss = loss.clamp(max=1E6)#soft_clamp(loss,1.E8)
        return loss
    
    def get_constraints(self,phi):
        def fn_pen(x): 
            return torch.nn.functional.relu(x,inplace=False).pow(2)
        phi = self.add_fixed_params(phi)
        constraints = fn_pen((self.get_total_length(phi)-self.L0)*100)
        wall_gap = 1 #cm
        def get_cavern_bounds(z):
            mask = z <= 2051.8 - 214.0
            x_min = torch.where(mask, 356.0, 456.0)
            y_min = torch.where(mask, 170.0, 336.0)
            x_min = x_min - wall_gap
            y_min = y_min - wall_gap
            return x_min, y_min
        Ymgap = torch.zeros(self.n_magnets, device=phi.device)
        if self.fSC_mag: Ymgap[2] = self.SC_Ymgap
        Z_out = torch.cumsum(phi[:,0] + 2*phi[:,1],dim=0)
        Z_in = Z_out - 2*phi[:,1]
        with torch.no_grad(): x_min, y_min = get_cavern_bounds(Z_in)
        constraints = constraints + fn_pen(phi[:,2]+phi[:,8]*phi[:,2]+phi[:,6]+phi[:,12]-x_min)
        constraints = constraints + fn_pen(phi[:,4]+phi[:,10]+Ymgap - y_min)
        with torch.no_grad(): x_min, y_min = get_cavern_bounds(Z_out)
        constraints = constraints + fn_pen(phi[:,3]+phi[:,9]*phi[:,3]+phi[:,7]+phi[:,13] -x_min)
        constraints = constraints + fn_pen(phi[:,5]+phi[:,11]+Ymgap - y_min)
        if self.use_diluted:
            if self.key and not self.key.startswith("stella"):
                constraints = constraints + fn_pen((1-phi[:,8])) 
                constraints = constraints + fn_pen((1-phi[:,9]))
            else:
                # ∆Xcore,i · (1 + Ryoke/core,i) - ∆Xcore,i+1 · (1 + Ryoke/core,i+1) < 0. (here number are the magnets)
                t1 = phi[:-1, 2] * (1 + phi[:-1, 8])
                t2 = phi[1:,  2] * (1 + phi[1:,  8])
                constraints[1:] +=  + fn_pen(t1  - t2 )
        if self.cost_as_constraint:
            M = self.get_total_cost(phi)
            constraints = constraints + fn_pen(M - self.W0)
        constraints = constraints.sum()*self.lambda_constraints
        return constraints.clamp(min=0,max=1E8)

    def get_constraints_func(self, phi):
        """
        This is a class method that calculates all problem constraints.
        It takes a NumPy array (from SciPy) and returns a NumPy array where
        each element represents a constraint in the form `g(x) <= 0`.
        """
        wall_gap = 1
        def get_cavern_bounds(z):
            x_min = torch.zeros_like(z)
            y_min = torch.zeros_like(z)
            mask = z <= 2051.8 - 214.0
            x_min[mask] = 356
            y_min[mask] = 170
            x_min[~mask] = 456
            y_min[~mask] = 336
            x_min -= wall_gap
            y_min -= wall_gap
            return x_min, y_min

        #phi = torch.from_numpy(phi).float()
        
        phi = self.add_fixed_params(phi)
        
        constraint_values = []
        length_constraint = self.L0 - self.get_total_length(phi)
        constraint_values.append(length_constraint)
        z = torch.zeros(phi.size(0))
        z = z + 2 * phi[:,1] + phi[:,0]
        Ymgap = self.SC_Ymgap if (self.fSC_mag and abs(phi[:,-1])>3.0) else 0

        with torch.no_grad():
            x_min_g1, y_min_g1 = get_cavern_bounds(z - 2 * phi[:,1])
        c1 = x_min_g1 - (phi[:,2] + phi[:,8] * phi[:,2] + phi[:,6] + phi[:,12])
        constraint_values.append(c1)
        c2 = y_min_g1 - (phi[:,4] + phi[:,10] + Ymgap)
        constraint_values.append(c2)
        with torch.no_grad():
            x_min_g2, y_min_g2 = get_cavern_bounds(z)
        c3 = x_min_g2 - (phi[:,3] + phi[:,9] * phi[:,3] + phi[:,7] + phi[:,13])
        constraint_values.append(c3)
        c4 = y_min_g2 - (phi[:,5] + phi[:,11] + Ymgap)
        constraint_values.append(c4)
        if self.use_diluted:
            c5 = phi[:,8] - 1
            constraint_values.append(c5)
            c6 = phi[:,9] - 1
            constraint_values.append(c6)
        if self.cost_as_constraint:
            M = self.get_total_cost(phi)
            c_cost = (self.W0 - M)
            constraint_values.append(c_cost)

        all_constraints_tensor = -torch.cat([c.flatten() for c in constraint_values])
        return all_constraints_tensor

       
def save_muons(muons:np.array,tag):
    np.save(os.path.join(PROJECTS_DIR, f'cluster/files/muons_{tag}.npy'), muons)

class ShipMuonShieldCluster(ShipMuonShield):
    def __init__(self,
                 manager_ip=os.getenv('IP_GCLOUD'),
                 port=444,
                 local:bool = False,
                 **kwargs) -> None:
        self.return_files_dir = kwargs.pop('results_dir', None)
        super().__init__(**kwargs)

        self.manager_cert_path = os.getenv('STARCOMPUTE_MANAGER_CERT_PATH')
        self.client_cert_path = os.getenv('STARCOMPUTE_CLIENT_CERT_PATH')
        self.client_key_path = os.getenv('STARCOMPUTE_CLIENT_KEY_PATH')
        self.server_url = 'wss://%s:%s'%(manager_ip, port)
        
        if not local:
            from starcompute.star_client import StarClient
            self.star_client = StarClient(self.server_url, self.manager_cert_path, 
                                    self.client_cert_path, self.client_key_path)
        
    def sample_x_idx(self,phi = None, n_samples = None):
        if n_samples is None: n_samples = self.n_samples
        if phi is not None and phi.dim()==2 and self.parallel:
            cores = int(self.cores/phi.size(0))
        else: cores = self.cores
        cores = min(cores,n_samples)
        return get_split_indices(cores,n_samples) 
    
    def simulate(self,phi:torch.tensor,
                 muons = None,
                 idx = None):
        phi = self.add_fixed_params(phi).flatten()
        if muons is not None:
            n_samples = muons.shape[0] 
        elif idx is not None:
            n_samples = idx[1] - idx[0]
        else: n_samples = self.n_samples
        if n_samples==0: n_samples = self.sample_x().shape[0]
        muons_idx = self.sample_x_idx(n_samples=n_samples)
        if idx is not None:
            muons_idx = [(start + idx[0], stop + idx[0]) for (start, stop) in muons_idx]
        if not self.uniform_fields: 
            print('SIMULATING MAGNETIC FIELDS')
            self.simulate_mag_fields(phi, cores = 9)
        t1 = time.time()
        inputs = split_array_idx(phi.detach().cpu(),muons_idx) 
        result = self.star_client.run(inputs)
        print('SIMULATION FINISHED, took',time.time()-t1)
        if self.return_files_dir is not None:
            results = []
            for filename in result:
                if filename == -1: continue
                m_file = os.path.join(self.return_files_dir,f"outputs_{filename}.pkl")
                results.append(torch.as_tensor(np.load(m_file),dtype=torch.get_default_dtype()))
                os.remove(m_file)
            result = torch.cat(results)
        
        result = torch.as_tensor(result,device = phi.device)
        if self.reduction == 'sum': result = result.sum(-1)
        elif self.reduction == 'mean': result = result.mean(-1)
        return result
    
    def __call__(self,phi,muons = None, file = None):
        if phi.dim()>1 and not self.parallel:
            y = []
            for p in phi:
                y.append(self(p))
            return torch.stack(y)
        phi = self.add_fixed_params(phi)
        M = self.get_total_cost(phi)
        if file is None: file = self.muons_file
        if self.get_constraints(phi) > 10 or M>((6*np.log(10)/10+1)*self.W0): 
            return torch.ones((1,1),device=phi.device)*1E6
        try: loss = self.simulate(phi,muons, file)
        except Exception as e:
            print(f"Error occurred with input: {phi}")
            print(e)
            raise
        n_samples = self.n_samples
        
        
        if self.apply_det_loss: loss = self._apply_deterministic_loss(phi,loss)

        return loss.to(torch.get_default_dtype())   
    
class ShipMuonShieldCuda(ShipMuonShield):
    def __init__(self,
                 n_steps:int = 5000,
                 **kwargs) -> None:

        super().__init__(**kwargs)
        self.n_steps = max(n_steps, np.ceil(self.sensitive_plane[0]['position']/0.02) + 100)
        from cuda_muons import run
        self.run_muonshield = run
        self.muons = super().sample_x()
    def sample_x(self, phi=None, idx=None):
        if 0 < self.n_samples < self.muons.size(0):
            indices = torch.randperm(self.muons.size(0), device=self.muons.device)[: self.n_samples]
            return self.muons[indices].clone()
        return self.muons

    def simulate(self,phi:torch.tensor,muons = None, return_all = False): 
        phi = self.add_fixed_params(phi).detach().cpu()
        if muons is None: muons = self.sample_x()
        self._sum_weights = muons[:, -1].sum()
        assert phi.shape[1] == 15, f"Expected phi to have 15 columns, got {phi.shape}"
        print('SIMULATING MAGNETIC FIELDS')
        #self.simulate_mag_fields(phi)
        try: 
            output = self.run_muonshield(phi.numpy(), 
                                        muons,
                                        sensitive_plane = self.sensitive_plane,
                                        n_steps=self.n_steps,
                                        SmearBeamRadius=self.SmearBeamRadius,
                                        fSC_mag = self.fSC_mag,
                                        field_map_file = self.fields_file,
                                        NI_from_B = self.use_B_goal,
                                        use_diluted = self.use_diluted,
                                        add_cavern = self.cavern,
                                        SND = self.SND,
                                        return_all = return_all,
                                        histogram_dir = os.path.join(PROJECTS_DIR, 'MuonsAndMatter/cuda_muons/data'),
                                        seed = self.seed)
        except Exception as e:
            print(f"Error during CUDA simulation with input: {phi}")
            print(e)
            raise
        px = output['px']   
        py = output['py']
        pz = output['pz']
        x = output['x']
        y = output['y']
        z = output['z']
        particle = output['pdg_id']
        weight = output['weight']
        return torch.stack([px, py, pz, x, y, z, particle, weight])    

class stochastic_RosenbrockProblem(RosenbrockProblem):
    def __init__(self,bounds = (-10,10), 
                 n_samples:int = 1, 
                 average_x = True,
                 std:float = 1.) -> None:
        super().__init__(0)
        self.bounds = bounds
        self.n_samples = n_samples
        self.average_x = average_x
        self.std = std
    def sample_x(self,phi):
        mu = uniform_sample((phi.shape[0],self.n_samples),self.bounds,device = phi.device)
        return torch.randn_like(mu)+mu
    def simulate(self,phi:torch.tensor, x:torch.tensor = None):
        if x is None: x = self.sample_x(phi)
        y = super().__call__(phi) + x
        y += torch.randn(y.size(0),1,device=phi.device)*self.std
        return y
    def loss(self,x):
        return x
    def __call__(self,phi:torch.tensor, x:torch.tensor = None):
        y = self.loss(self.simulate(phi,x))
        if self.average_x: y = y.mean(-1,keepdim = True)
        return y
    @staticmethod
    def GetBounds(device = torch.device('cpu')):
        pass
class stochastic_ThreeHump(ThreeHump):
    def __init__(self, n_samples = 1, 
                 average_x = True,
                 bounds_1 = (-2,0), 
                 bounds_2 = (2,5),
                 std:float = 1.0) -> None:
        super().__init__(0)
        self.bounds_1 = bounds_1
        self.bounds_2 = bounds_2
        self.n_samples = n_samples
        self.average_x = average_x
        self.std = std
    def loss(self,x):
        return torch.sigmoid(x-10)-torch.sigmoid(x)
    def sample_x(self,phi):
        P1 = phi[:,0].div(phi.norm(p=2,dim=-1)).view(-1,1)
        mask = torch.rand((phi.size(0),self.n_samples),device=phi.device).le(P1)
        x1 = uniform_sample((phi.size(0),self.n_samples),self.bounds_1,device=phi.device)
        x2 = uniform_sample((phi.size(0),self.n_samples),self.bounds_2,device=phi.device)
        return torch.where(mask, x1, x2)
    def simulate(self, phi:torch.tensor, x:torch.tensor = None):
        h = super().__call__(phi)
        if x is None: x = self.sample_x(phi)
        mu = x*h + torch.randn_like(h)*self.std
        y =  mu+torch.randn_like(h)*self.std
        return y
    def __call__(self, phi:torch.tensor, x:torch.tensor = None):
        y = self.loss((self.simulate(phi,x)))
        if self.average_x: y = y.mean(-1,keepdim=True)
        return y

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--nodes",type=int,default = 16)
    parser.add_argument("--n_tasks_per_node", type=int, default=32)
    parser.add_argument("--n_tasks", type=int, default=None)
    parser.add_argument("--muons_file", type=str, default=os.path.join(PROJECTS_DIR,'MuonsAndMatter/data/muons/subsample_biased_v4.npy'))
    parser.add_argument("--params_name", type=str, default='tokanut_v5')
    parser.add_argument("--cluster", action='store_true')
    parser.add_argument("--cuda", action='store_true')
    parser.add_argument("--hybrid", action='store_true')
    parser.add_argument("--uniform_fields", action='store_true')
    parser.add_argument("--remove_cavern", dest = "cavern", action='store_false')
    parser.add_argument("--config_file", type=str, default=os.path.join(PROJECTS_DIR, 'BlackBoxOptimization/outputs/config.json'))
    args = parser.parse_args()
    if args.params_name in ShipMuonShield.params:
        phi = ShipMuonShield.params[args.params_name]
    else:
        with open(f'/home/hep/lprate/projects/BlackBoxOptimization/outputs/{args.params_name}/phi_optm.txt', "r") as txt_file:
            phi = [float(line.strip()) for line in txt_file]
    phi = torch.tensor(phi)

    torch.cuda.set_device(1)
    
    d = {}
    t0 = time.time()
    
    seed = 1
    if args.n_tasks is None:
        n_tasks = args.nodes * args.n_tasks_per_node if args.cluster else 45
    else:
        n_tasks = args.n_tasks

    print(f"{'='*60}")
    print(f"Testing configuration: {args.params_name}")
    print(f"{'='*60}")

    config_file = args.config_file
    with open(config_file, 'r') as src:
        CONFIG = json.load(src)
        CONFIG.pop("data_treatment", None)
        CONFIG['W0'] = 13E6
        CONFIG['L0'] = 30.0
        CONFIG['initial_phi'] = phi

    if args.cluster:
        print('Using cluster MaM simulation')
        muon_shield = ShipMuonShieldCluster(**CONFIG)

        t1 = time.time()
        loss_muons = muon_shield.simulate(phi, file=args.muons_file)
        t2 = time.time()
        n_hits = 0
        rate = 0
    elif args.cuda:
        print('Using CUDA simulation')
        muon_shield = ShipMuonShieldCuda(
            **CONFIG,
        )
        
        t1 = time.time()
        px, py, pz, x, y, z, particle, factor = muon_shield.simulate(phi)
        hits = muon_shield.is_hit(px, py, pz, x, y, z, particle)
        t2 = time.time()
        
        loss_muons = muon_shield._blackbox_loss(px, py, pz, x, y, z, particle, factor).sum()*1e6 / muon_shield._sum_weights
        n_hits = hits.sum().item()
        print(f'Number of hits: {n_hits}')

    else:
        # Local simulation
        print('Using local MaM simulation')
        print(f"Superconducting magnets: {args.hybrid}")
        print(f"Tasks: {n_tasks}")
        muon_shield = ShipMuonShield(
            cores=n_tasks,
            fSC_mag=args.hybrid,
            uniform_fields = args.uniform_fields,
            fields_file=os.path.join(PROJECTS_DIR, 'MuonsAndMatter/data/outputs/fields_mm.h5'),
            seed=seed,
            cavern=args.cavern,
            muons_file=args.muons_file
        )
        
        t1 = time.time()
        px, py, pz, x, y, z, particle, factor = muon_shield.simulate(phi)
        print(f'Number of hits: {x.numel()}')
        t2 = time.time()
        
        loss_muons = muon_shield._blackbox_loss(px, py, pz, x, y, z, particle, factor).sum() + 1
        n_hits = factor.sum().item() if hasattr(factor, 'sum') else 0
        
    M_iron = muon_shield.get_iron_cost(phi).item()
    C_electrical = muon_shield.get_electrical_cost(phi).item()
    M_total = muon_shield.get_total_cost(phi).item()
    L = muon_shield.get_total_length(phi).item()
    constraints = muon_shield.get_constraints(phi).item()
    
    # Calculate total loss
    loss_total = muon_shield._apply_deterministic_loss(phi, loss_muons).item()
    
    simulation_time = t2 - t1
    print('Phi:', muon_shield.add_fixed_params(phi).cpu().tolist())
    
    print(f"\n{'='*50}")
    print("SIMULATION RESULTS")
    print(f"{'='*50}")
    print(f"Iron cost:        {M_iron:.2f}")
    print(f"Electrical cost:  {C_electrical:.2f}")
    print(f"Total cost:       {M_total:.2f}")
    print(f"Length:           {L:.2f}")
    print(f"Constraints:      {constraints:.2f}")
    print(f"Muon loss:        {loss_muons:.2f}")
    print(f"Total loss:       {loss_total:.2f}")
    print(f"Number of hits:   {n_hits}")
    print(f"Number of hits WEIGHTED:   {factor[hits].sum().item()}")
    print(f"Simulation time:  {simulation_time:.2f}s")
    print(f"Total time:       {time.time() - t0:.2f}s")



