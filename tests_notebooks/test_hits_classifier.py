import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.abspath(os.path.join('..', 'src')))
sys.path.append(os.path.abspath(os.path.join('..')))
from problems import ThreeHump_stochastic_hits, Rosenbrock_stochastic_hits, HelicalValley_stochastic_hits, ShipMuonShieldCuda, Quadratic_stochastic_hits
from utils import normalize_vector, denormalize_vector, get_freest_gpu
from utils.nets import Classifier, DeepONetClassifier, QuadraticModel
import torch
from tqdm import trange
import json
import numpy as np
import argparse
from scipy.optimize import minimize as scipy_minimize
from scipy.stats.qmc import LatinHypercube


dev = get_freest_gpu()
cpu = torch.device('cpu')
#torch.set_default_dtype(torch.float64)

parser = argparse.ArgumentParser()
parser.add_argument('--second_order', action='store_true', help='Use second order optimization')
parser.add_argument('--problem', type=str, choices=['ThreeHump', 'Rosenbrock', 'HelicalValley', 'MuonShield', 'Quadratic'], default='Rosenbrock', help='Optimization problem to use')
parser.add_argument('--batch_size', type=int, default=32768, help='Batch size for training the surrogate model')
parser.add_argument('--samples_phi', type=int, default=5, help='Number of samples for phi in each iteration')
parser.add_argument('--n_samples', type=int, default=1_000_000, help='Number of samples per phi')
parser.add_argument('--subsamples', type=int, default=200_000, help='Number of subsamples for surrogate training per iteration')
parser.add_argument('--n_test', type=int, default=5, help='Number of test samples to evaluate the surrogate model')
parser.add_argument('--n_epochs', type=int, default=10, help='Number of epochs to train the surrogate model')
parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate for surrogate model training')
parser.add_argument('--step_lr', type=int, default=8, help='Step size for learning rate scheduler')
parser.add_argument('--epsilon', type=float, default=0.1, help='Trust region radius for phi optimization')
parser.add_argument('--dims', type=int, default=4, help='Number of dimensions for Rosenbrock problem')
parser.add_argument('--model', type=str, choices=['mlp', 'deeponet', 'quadratic', 'NW'], default='mlp', help='Type of surrogate model to use')
parser.add_argument('--activation', type=str, default='relu', help='Activation function for surrogate model')
parser.add_argument(
    '--hess_method',
    type=str,
    choices=['hess', 'hvp'],
    default='hess',
    help="How to compute surrogate Hessian for analysis: "
         "'hess' = torch.func.hessian, 'hvp' = build Hessian via (damped) HVP columns")
parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
parser.add_argument('--plot', action='store_true', help='Display plots interactively after saving')
parser.add_argument('--load_model', action='store_true', help='Whether to load a pre-trained surrogate model instead of training from scratch')
parser.add_argument('--model_name', type=str, default = 'classifier_hits', help='Name of the model to load (alternative to --load_model)')

args = parser.parse_args()

torch.manual_seed(args.seed)
np.random.seed(args.seed)

samples_phi = args.samples_phi
epsilon = args.epsilon
n_samples = args.n_samples
outputs_dir = '/home/hep/lprate/projects/BlackBoxOptimization/outputs/test_lcso'

if args.problem == 'ThreeHump':
    dim = 2
    x_dim = 2
    problem = ThreeHump_stochastic_hits(n_samples = n_samples, 
                                    phi_bounds = (((-2.3,-2.0), (2.0,2.0))), 
                                    x_bounds = (-3,3),
                                    reduction = 'none')
    initial_phi = torch.tensor([-1.2, 1.0])
elif args.problem == 'Rosenbrock':
    dim = args.dims
    x_dim = 2
    problem = Rosenbrock_stochastic_hits(dim = dim, 
                                         n_samples = n_samples, 
                                        phi_bounds = ((-2.0, 2.0)), 
                                        x_bounds = (-3.,3.),
                                        reduction = 'none')
    initial_phi = torch.tensor([[-1.2, 1.8]*(dim//2)]).flatten()
elif args.problem == 'HelicalValley':
    dim = 10
    x_dim = 2
    problem = HelicalValley_stochastic_hits(n_samples = n_samples, 
                                    phi_bounds = ((-10.0, 10.0)), 
                                    x_bounds = (-5,5),
                                    reduction = 'none')
    initial_phi = torch.tensor([[-1.2, 1.8]*(dim//2)]).flatten()
elif args.problem == 'MuonShield':
    dim = 24
    x_dim = 8
    config_file = "/home/hep/lprate/projects/BlackBoxOptimization/outputs/config.json"
    with open(config_file, 'r') as f:
        CONFIG = json.load(f)
    CONFIG.pop("data_treatment", None)
    CONFIG.pop('results_dir', None)
    CONFIG['dimensions_phi'] = dim
    CONFIG['initial_phi'] = ShipMuonShieldCuda.params['stellatryon_v3']
    CONFIG['n_samples'] = n_samples
    CONFIG['reduction'] = 'none'
    CONFIG['cost_as_constraint'] = False
    problem = ShipMuonShieldCuda(**CONFIG)
    initial_phi = problem.initial_phi
elif args.problem == 'Quadratic':
    dim = args.dims
    x_dim = 1
    problem = Quadratic_stochastic_hits(dim = dim, 
                                         n_samples = n_samples, 
                                        phi_bounds=((-1.0, 1.0)), x_bounds=(0.0, 1.0),
                                        reduction = 'none')
    initial_phi = torch.tensor([[-0.5, 0.5]*(dim//2)]).flatten()

print(f"Using problem {args.problem} with dim {dim} and x_dim {x_dim}, samples_phi = {samples_phi}")

bounds = problem.GetBounds(device=cpu)


class Sampler():
    def __init__(self,true_model,
                 bounds:tuple,
                 epsilon:float = 0.2,
                 initial_phi:torch.tensor = None,
                 seed:int = 42,
                 second_order:bool = False
                 ):
        
        self.bounds = bounds.to(device=cpu, dtype=torch.get_default_dtype())
        initial_phi = normalize_vector(initial_phi.to(device=cpu, dtype=torch.get_default_dtype()), self.bounds)
        self._current_phi = torch.nn.Parameter(initial_phi.detach().clone())
        self.epsilon = epsilon
        self.lhs_sampler = LatinHypercube(d=initial_phi.size(-1), seed=seed)
        self.true_model = true_model
        self.second_order = second_order
    @property
    def current_phi(self):
        """Returns the current phi in the original bounds."""
        return denormalize_vector(self._current_phi, self.bounds)
    def sample_phi(self, samples_phi:int = 10):
        """Draw samples in a hypercube of side 2*epsilon around current_phi."""
        perturb = self.lhs_sampler.random(samples_phi)
        with torch.no_grad():
            perturb = (2 * torch.from_numpy(perturb).to(device=cpu, dtype=self._current_phi.dtype) - 1.0) * self.epsilon
            phis = (self._current_phi.unsqueeze(0) + perturb).clamp(0.0, 1.0)
        return phis
    def sample_phi_uniform(self, samples_phi:int = 10):
        """Draw samples uniformly within the bounds around current_phi."""
        with torch.no_grad():
            perturb = (torch.rand(samples_phi, self._current_phi.size(-1), device=cpu, dtype=self._current_phi.dtype) * 2 - 1) * self.epsilon
            phis = (self._current_phi.unsqueeze(0) + perturb).clamp(0.0, 1.0)
        return phis
    def sample_x(self, phi, n_samples = None):
        """Sample x from the true model for given phi."""
        if n_samples is not None:
            self.true_model.n_samples = n_samples
        x = self.true_model.sample_x(phi)
        return x.to(torch.get_default_dtype())

SAMPLER = Sampler(true_model=problem,
                  bounds=bounds,
                  epsilon=epsilon,
                  initial_phi=initial_phi,
                  seed=args.seed,
                  second_order=args.second_order)

@torch.no_grad()
def simulate(phi,x = None, n_samples:int = None):
    if phi.dim() == 1:
        phi = phi.view(1,-1)
    phi = phi.cpu()
    if x is None:
        x = SAMPLER.sample_x(phi, n_samples=n_samples)
    else:
        x = x.cpu().to(torch.get_default_dtype())
    phi = denormalize_vector(phi, SAMPLER.bounds).detach().cpu()
    y = problem(phi,x)
    return x,y

class Metrics:
    """Helper to compute surrogate diagnostics for train/validation splits."""

    def __init__(self, bounds, true_model, surrogate_model):
        self.true_model = true_model
        self.surrogate_model = surrogate_model
        self.device = surrogate_model.device
        self.bounds = bounds.to(self.device)
        self.bounds_cpu = bounds.to(cpu)

    def _prepare_true_inputs(self, phi, x):
        if phi.dim() == 1:
            phi = phi.unsqueeze(0)
        phi_cpu = phi.to(cpu, dtype=torch.get_default_dtype())
        x_cpu = x.to(cpu, dtype=torch.get_default_dtype())
        if x_cpu.dim() == 2:
            x_cpu = x_cpu.unsqueeze(0)
        if x_cpu.shape[0] == 1 and phi_cpu.shape[0] > 1:
            x_cpu = x_cpu.expand(phi_cpu.shape[0], -1, -1)
        return phi_cpu, x_cpu

    def true_objective(self, phi, x, reduction='mean'):
        phi_cpu, x_cpu = self._prepare_true_inputs(phi, x)
        phi_denorm = denormalize_vector(phi_cpu, self.bounds_cpu)
        hit_probs = self.true_model.probability_of_hit(phi_denorm, x_cpu)
        if reduction == 'sum':
            return hit_probs.sum(dim=1).mean()
        return hit_probs.mean()

    def compute_batched_metrics(self, phi, x, y, batch_size):
        """
        Compute batched metrics for a given phi, x, y using streaming/batches to save GPU memory.
        Returns: dict with 'loss', 'hits_error', 'prob_gap'.
        """
        phi = phi.to(self.device)
        num_phi = phi.size(0)
        per_phi_batch = max(1, batch_size // max(num_phi, 1))  # ensure step > 0
        num_samples_x = x.size(1)

        loss_sum = 0.0
        total_elements = 0
        pred_hits = torch.zeros(num_phi, dtype=torch.float32)
        diff_sum = 0.0
        phi_denorm = denormalize_vector(phi.detach().cpu(), self.bounds_cpu)

        for start in range(0, num_samples_x, per_phi_batch):
            end = min(start + per_phi_batch, num_samples_x)
            x_batch = x[:, start:end].to(self.device)
            y_batch = y[:, start:end].float().to(self.device)
            logits = self.surrogate_model.forward_logits(phi, x_batch)
            batch_loss = self.surrogate_model.loss_fn(logits, y_batch)
            elems = logits.numel()  # num_phi * batch_len
            loss_sum += batch_loss.item() * elems
            total_elements += elems
            pred_probs_batch = torch.sigmoid(logits).detach().cpu()
            pred_hits += pred_probs_batch.sum(dim=1)
            x_batch = x[:, start:end].detach().cpu()
            real_probs_batch = self.true_model.probability_of_hit(phi_denorm, x_batch)
            diff_sum += torch.abs(pred_probs_batch - real_probs_batch).sum().item()
            del logits, pred_probs_batch, x_batch, y_batch, real_probs_batch

        avg_loss = loss_sum / max(total_elements, 1)
        prob_gap = diff_sum / max(total_elements, 1)
        true_hits = y.detach().cpu().float().sum(dim=1)
        hits_error = (pred_hits - true_hits).abs()

        return {'loss': avg_loss, 'hits_error': hits_error, 'prob_gap': prob_gap}

    def compute_grad_metrics(self, phi, x):
        """
        Compute grad_cosine and newton_cosine for the first phi and x.
        Returns: dict with 'grad_cosine', 'newton_cosine'.
        """
        phi0_sur = phi.detach().clone().to(self.device).requires_grad_(True)
        x0_dev = x[0].to(self.device)
        x0_cpu = x[0].to(cpu)
        grad_sur = torch.func.grad(self.surrogate_objective)(phi0_sur, x0_dev)
        phi0_true = phi.detach().clone().to(cpu).requires_grad_(True)
        true_objective = lambda phi_local: self.true_objective(phi_local, x0_cpu)
        grad_real = torch.func.grad(true_objective)(phi0_true).to(self.device)
        grad_cosine = self.cosine_similarity(grad_sur, grad_real)
        surrogate_hessian = self.surrogate_hessian_from_hvp(phi0_sur, x0_dev, damping=1e-3)
        real_hessian = torch.func.hessian(true_objective)(phi0_true).to(self.device)
        real_hessian = 0.5 * (real_hessian + real_hessian.T)
        step_sur = self._newton_step(grad_sur, surrogate_hessian)
        step_real = self._newton_step(grad_real, real_hessian)
        newton_cosine = self.cosine_similarity(step_sur, step_real)
        return {'grad_cosine': grad_cosine, 'newton_cosine': newton_cosine}
    
    def n_hits(self,y, w = None):
        y = y.flatten()
        #w = torch.ones_like(y) if (w is None) else w.flatten().to(y.device)
        #y = (w*y).sum()
        return y.sum(-1)

    def hits_error(self, y_true, pred_probs):
        pred_hits = pred_probs.sum(dim=1)
        true_hits = y_true.float().sum(dim=1)
        error = (pred_hits - true_hits).abs()
        return error#.mean().item()
    def get_probs(self, phi, x):
        pred_probs_list, real_probs_list = [], []
        num_samples_x = x.size(1)
        num_samples_phi = phi.size(0)
        per_phi_batch = max(1, self.surrogate_model.batch_size // max(num_samples_phi, 1))
        phi = phi.to(self.device)
        phi_denorm = denormalize_vector(phi.detach().cpu(), self.bounds_cpu)
        with torch.no_grad():
            for start in range(0, num_samples_x, per_phi_batch):
                end = min(start + per_phi_batch, num_samples_x)
                x_batch = x[:, start:end].to(self.device)
                p = self.surrogate_model.predict_proba(phi, x_batch).cpu()
                pred_probs_list.append(p)
                p = self.true_model.probability_of_hit(phi_denorm, x_batch.cpu())
                real_probs_list.append(p)
        return torch.cat(pred_probs_list, dim=1), torch.cat(real_probs_list, dim=1)

    @staticmethod 
    def get_calib_stats_vs_real(real_probs, pred_probs, bin_edges):
        bin_midpoints, means, stds = [], [], []
        for i in range(len(bin_edges) - 1):
            lower, upper = bin_edges[i], bin_edges[i+1]
            mask = (real_probs >= lower) & (real_probs <= upper)
            if mask.sum() > 0:
                bin_midpoints.append((lower + upper) / 2)
                preds_in_bin = pred_probs[mask]
                means.append(preds_in_bin.mean())
                stds.append(preds_in_bin.std())
        return torch.tensor(bin_midpoints), torch.tensor(means), torch.tensor(stds)
    
    @staticmethod
    def get_calib_stats_vs_hits(pred_probs, hits, bin_edges):
        mean_preds, frac_positives = [], []
        for i in range(len(bin_edges) - 1):
            lower, upper = bin_edges[i], bin_edges[i+1]
            mask = (pred_probs >= lower) & (pred_probs < upper)
            if mask.sum() > 0:
                mean_preds.append(pred_probs[mask].mean())
                frac_positives.append(hits[mask].float().mean())
        return torch.tensor(mean_preds).numpy(), torch.tensor(frac_positives).numpy()   
    
    def surrogate_objective(self,phi, x):
        if phi.dim() == 1:
            phi = phi.unsqueeze(0)
        if x.dim() == 2:
            x = x.unsqueeze(0)
        if x.shape[0] == 1 and phi.shape[0] > 1:
            x = x.expand(phi.shape[0], -1, -1)
        total = torch.zeros((), device=phi.device, dtype=torch.get_default_dtype())
        for x_batch in x.split(self.surrogate_model.batch_size, dim=1):
            p = self.surrogate_model.predict_proba(phi, x_batch)
            total = total + self.n_hits(p)
        return total / x.size(1)

    def surrogate_hessian_from_hvp(self,phi, x, damping=0.0):
        """
        Builds an explicit Hessian approximation by columns using HVPs:
        H[:, i] = H e_i   (optionally (H + damping I)e_i)

        This is O(d) HVPs; OK for small d (<= ~50).
        """
        phi = phi.detach().requires_grad_(True)
        if x.dim() == 2:
            x = x.unsqueeze(0)
        if x.shape[0] == 1 and phi.dim() == 1:
            x = x.expand(1, -1, -1)
        d = phi.numel()
        H = torch.zeros((d, d), device=phi.device, dtype=phi.dtype)

        # We use the same frozen x_samp and same (global) mean convention as surrogate_objective.
        for i in range(d):
            v = torch.zeros_like(phi)
            v[i] = 1.0

            total_hv = torch.zeros_like(phi)

            for x_batch in x.split(self.surrogate_model.batch_size, dim=1):
                def f_batch(phi_local):
                    p = self.surrogate_model.predict_proba(phi_local.view(1, -1), x_batch).to(phi.device)
                    return self.n_hits(p)
                grad_batch = torch.func.grad(f_batch)
                _, hv = torch.func.jvp(grad_batch, (phi,), (v,))
                total_hv = total_hv + hv.detach()

            total_hv = total_hv / x.size(1)

            if damping != 0.0:
                total_hv = total_hv + damping * v 
            H[:, i] = total_hv
        H = 0.5 * (H + H.T)
        return H

    @torch.no_grad()
    def _probability_gap(self, phi, x, pred_probs):
        phi = denormalize_vector(phi.detach().cpu(), self.bounds_cpu)
        real_probs = self.true_model.probability_of_hit(phi, x)
        diff = torch.abs(pred_probs.detach().cpu() - real_probs)
        diff_per_phi = diff.view(diff.size(0), -1).mean(dim=1)
        return diff_per_phi.mean().item()

    @staticmethod
    def cosine_similarity(vec_a, vec_b, eps=1e-10):
        vec_a = vec_a.reshape(-1)
        vec_b = vec_b.reshape(-1).to(vec_a.device)
        denom = (torch.linalg.norm(vec_a) * torch.linalg.norm(vec_b)).clamp_min(eps)
        return (torch.dot(vec_a, vec_b) / denom).item()

    def gradient_cosine(self, surrogate_objective, phi, real_gradient_fn=None):
        phi = phi.detach().clone().requires_grad_(True)
        grad_sur = torch.func.grad(surrogate_objective)(phi).detach()
        grad_real = real_gradient_fn(phi.unsqueeze(0)).squeeze().detach()
        cosine = self.cosine_similarity(grad_sur.to(self.device), grad_real.to(self.device))
        return {
            'surrogate_grad': grad_sur,
            'real_grad': grad_real,
            'grad_cosine': cosine
        }

    def newton_step_cosine(self, grad_est, hess_est, grad_real, hess_real, eps=1e-6):
        step_est = self._newton_step(grad_est, hess_est, eps)
        step_real = self._newton_step(grad_real, hess_real, eps)
        return {
            'newton_step_cosine': self.cosine_similarity(step_est, step_real)
        }
    def _newton_step(self, grad, hess, eps = 1e-8):
        #sym_hess = 0.5 * (hess + hess.T)
        #eye = torch.eye(sym_hess.size(0), device=sym_hess.device, dtype=sym_hess.dtype)
        #shift = torch.clamp(-torch.linalg.eigvalsh(sym_hess).min() + eps, min=0.0)
        #hpd = sym_hess + shift * eye
        step = -torch.linalg.solve(hess, grad)
        return step

    def gain_newton_true(self, phi0, newton_step, eta, x_samp):
        """
        Computes the gain in total expected hits using a fixed X sample.
        """
        with torch.no_grad():
            phi1 = (phi0 + eta * newton_step).clamp(0.0, 1.0)
            f0 = self.true_objective(phi0, x_samp, reduction='sum')
            f1 = self.true_objective(phi1, x_samp, reduction='sum')
            return (f0 - f1).item()

    def gain_gradient_true(self, phi0, grad, eta, x_samp):
        """
        Computes the gain in total expected hits using a fixed X sample.
        """
        with torch.no_grad():
            phi1 = (phi0 - eta * grad).clamp(0.0, 1.0)
            f0 = self.true_objective(phi0, x_samp, reduction='sum')
            f1 = self.true_objective(phi1, x_samp, reduction='sum')
            return (f0 - f1).item()

    def gain_direction_true(self, phi0, direction, eta, x_samp):
        """
        Computes the gain in total expected hits using a fixed X sample.
        Positive gain means improvement (objective decreased).
        """
        with torch.no_grad():
            phi1 = (phi0 + eta * direction).clamp(0.0, 1.0)
            f0 = self.true_objective(phi0, x_samp, reduction='sum')
            f1 = self.true_objective(phi1, x_samp, reduction='sum')
            return (f0 - f1).item()




class BinaryClassifierModel:
    """
    A wrapper class to handle training and prediction for a given binary classification model.
    It is not an nn.Module itself, but it manages one.
    """
    def __init__(self,
                phi_dim: int,
                 x_dim: int,
                 bounds_phi: torch.tensor,
                 n_epochs: int = 50,
                 batch_size: int = 32,
                step_lr: int = 20,
                 lr: float = 1e-3,
                 model:str = 'mlp',
                 activation:str = 'relu',
                 device: str = 'cuda'):
        self.device = torch.device(device)
        self.model_name = model
        if model == 'mlp':
            self.model = Classifier(phi_dim,x_dim, 128).to(self.device)
        elif model == 'deeponet':
            layers = [[128,128],[128,128]]#[[256,128,128],[128,128]]
            self.model = DeepONetClassifier(phi_dim,x_dim, layers = layers, layer_norm=False, p=128, activation=activation).to(self.device)
        elif model == 'quadratic':
            self.model = QuadraticModel(phi_dim, x_dim,trunk_layers=[128,128], layer_norm=False, p=128).to(self.device)
        elif model == 'NW':
            raise ValueError("NW model is not available in this workspace.")
        else:
            raise ValueError(f"Unsupported model '{model}'")
        
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-5)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=step_lr, gamma=0.1)
        self.bounds_phi = bounds_phi.to(self.device)
        self.step_lr = step_lr
        self.loss_fn = torch.nn.BCEWithLogitsLoss()
        self.metrics = None  # To be set externally
    def forward_logits(self, phi, x):
        phi = phi.to(self.device)
        x = x.to(self.device)
        if self.model_name == 'mlp':
            if x.dim() == 2:
                x = x.unsqueeze(0)
            expanded_phi = phi.unsqueeze(1).expand(-1, x.size(1), -1)
            inputs = torch.cat([expanded_phi, x], dim=-1)
            return self.model(inputs).squeeze(-1)
        return self.model(phi, x)
    def get_model_pred(self, phi, x = None):
        phi = phi.to(self.device)
        n_hits = torch.zeros(phi.size(0), device=self.device)
        if x is None:
            x = SAMPLER.sample_x(phi.detach().cpu())
        num_samples_x = x.size(1)
        num_samples_phi = phi.size(0)
        per_phi_batch = max(1, self.batch_size // max(num_samples_phi, 1))
        for start in range(0, num_samples_x, per_phi_batch):
            x_batch = x[:, start:start + per_phi_batch].detach()
            y_pred = self.predict_proba(phi,x_batch)
            n_hits = n_hits + self.metrics.n_hits(y_pred)
        return n_hits / x.size(1)
    def fit(self, phi, x, y, val_phis=None, val_x=None, val_y=None, **kwargs):
        """
        Trains the classifier model.
        """
        self.model.train()
        phi = phi.to(self.device)
        num_samples_x = x.size(1)
        num_samples_phi = phi.size(0)
        per_phi_batch = max(1, self.batch_size // max(num_samples_phi, 1))
        indices = torch.arange(num_samples_x)

        train_metrics = {'loss': [], 'hits_error': [], 'prob_gap': []}
        val_metrics = {'loss': [], 'hits_error': [], 'prob_gap': []}
        grad_metrics = {'grad_cosine': [], 'newton_cosine': []}

        pbar = trange(self.n_epochs, position=0, leave=True, desc='Classifier Training on ' + str(self.device))
        for epoch in pbar:
            average_loss = 0.0
            shuffled_indices = indices[torch.randperm(num_samples_x)]
            for start in range(0, num_samples_x, per_phi_batch):
                end = min(start + per_phi_batch, num_samples_x)
                batch_idx = shuffled_indices[start:end]
                y_batch = y[:, batch_idx].float().to(self.device)
                x_batch = x[:, batch_idx].to(self.device)
                self.optimizer.zero_grad()
                y_pred_logits = self.forward_logits(phi, x_batch)
                loss = self.loss_fn(y_pred_logits, y_batch)
                assert not torch.isnan(loss), "Loss is NaN, check your model and data."
                loss.backward()
                self.optimizer.step()
                average_loss += loss.item() * (end - start)
            average_loss /= num_samples_x
            self.scheduler.step()
            self.model.eval()
            with torch.no_grad():
                train_m = self.metrics.compute_batched_metrics(phi, x, y, self.batch_size)
                train_metrics['loss'].append(train_m['loss'])
                train_metrics['hits_error'].append(train_m['hits_error'].mean().item())
                train_metrics['prob_gap'].append(train_m['prob_gap'])
                if val_phis is not None:
                    val_m = self.metrics.compute_batched_metrics(val_phis, val_x, val_y, self.batch_size)
                    val_metrics['loss'].append(val_m['loss'])
                    val_metrics['hits_error'].append(val_m['hits_error'].mean().item())
                    val_metrics['prob_gap'].append(val_m['prob_gap'])
            grad_m = self.metrics.compute_grad_metrics(phi[0], x)
            grad_metrics['grad_cosine'].append(grad_m['grad_cosine'])
            grad_metrics['newton_cosine'].append(grad_m['newton_cosine'])
            pbar.set_description(f"Epoch {epoch+1}, Loss: {average_loss:.4f}")
            pbar.set_postfix(loss=f"{average_loss:.4f}")

        return train_metrics, val_metrics, grad_metrics
    def predict_proba(self, phi,x):
        """
        Predicts the probability of Y=1 for each given `phi` and `x`.
        """
        self.model.eval()
        return torch.sigmoid(self.forward_logits(phi, x))
    def normalize_params(self, params):
        return normalize_vector(params, self.bounds_phi)
    def save(self, path):
        torch.save(self.model.state_dict(), path)
    def load(self, path):
        self.model.load_state_dict(torch.load(path, map_location=self.device))

initial_phi = normalize_vector(initial_phi, bounds)
X_initial, Y_initial = simulate(initial_phi.unsqueeze(0), n_samples=args.n_samples)
training_phis = SAMPLER.sample_phi(samples_phi=args.samples_phi)
training_phis = torch.cat([initial_phi.unsqueeze(0), training_phis], dim=0)
validation_phis = SAMPLER.sample_phi_uniform(samples_phi=args.n_test)  


x_training, y_training = simulate(training_phis, n_samples=args.subsamples)
x_validation, y_validation = simulate(validation_phis, n_samples=args.n_samples)

surrogate_model = BinaryClassifierModel(
    phi_dim=dim,
    x_dim=x_dim,
    bounds_phi=bounds,
    n_epochs=args.n_epochs,
    step_lr= args.step_lr,
    batch_size=args.batch_size,
    lr = args.lr,
    model=args.model,
    activation=args.activation,
    device=dev)  

metrics = Metrics(bounds, problem, surrogate_model)
surrogate_model.metrics = metrics 
if args.load_model:
    surrogate_model.load(os.path.join(outputs_dir,args.model_name+'.pt'))
else:
    train_metrics, val_metrics, grad_metrics = surrogate_model.fit(
    training_phis, x_training, y_training,
    val_phis=validation_phis, val_x=x_validation, val_y=y_validation)
    surrogate_model.save(os.path.join(outputs_dir, args.model_name+'.pt'))
    
    # --- Plotting training/validation loss and metrics ---
    epochs = range(1, len(train_metrics['loss']) + 1)
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

    # First subplot: Loss
    ax1.plot(epochs, train_metrics['loss'], label='Train Loss', color='blue')
    ax1.plot(epochs, val_metrics['loss'], label='Validation Loss', color='orange')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.6)

    # Second subplot: hits_error and prob_gap (twin y-axes)
    color1 = 'tab:blue'
    color2 = 'tab:green'
    ax2.plot(epochs, train_metrics['hits_error'], label='Train Hits Error', color=color1, linestyle='-')
    ax2.set_ylabel('Hits Error', color=color1)
    ax2.tick_params(axis='y', labelcolor=color1)
    ax2b = ax2.twinx()
    ax2b.plot(epochs, train_metrics['prob_gap'], label='Train Prob Gap', color=color2, linestyle='-')
    ax2b.set_ylabel('Prob Gap', color=color2)
    ax2b.tick_params(axis='y', labelcolor=color2)
    ax2.plot(epochs, val_metrics['hits_error'], label='Val Hits Error', color=color1, linestyle='--')
    ax2b.plot(epochs, val_metrics['prob_gap'], label='Val Prob Gap', color=color2, linestyle='--')
    ax2.set_xlabel('Epoch')
    ax2.set_title('Hits Error and Probability Gap')
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2b.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

    # Third subplot: Cosine similarities
    ax3.plot(epochs, grad_metrics['grad_cosine'], label='Grad Cosine', color='purple', marker='o')
    ax3.plot(epochs, grad_metrics['newton_cosine'], label='Newton Step Cosine', color='red', marker='x')
    ax3.set_ylabel('Cosine Similarity')
    ax3.set_xlabel('Epoch')
    ax3.set_title('Cosine Similarities (Grad & Newton Step)')
    ax3.legend()
    ax3.grid(True, linestyle='--', alpha=0.6)

    fig.tight_layout()
    plt.savefig('outputs/figs/train_val_metrics.png')
    if args.plot:
        plt.show()
    plt.close(fig)

# Get predicted and real probabilities for training and validation sets
pred_probs_train, real_probs_train = metrics.get_probs(training_phis, x_training)
pred_probs_val, real_probs_val = metrics.get_probs(validation_phis, x_validation)

# True vs surrogate loss (sum of hits) for train/val
true_loss_train = y_training.sum(dim=1).flatten().cpu()
sur_loss_train = pred_probs_train.sum(dim=1).flatten().cpu()

true_loss_val = y_validation.sum(dim=1).flatten().cpu()
sur_loss_val = pred_probs_val.sum(dim=1).flatten().cpu()

# Current / initial phi losses (keep special star marker)
true_loss_current = Y_initial.sum(dim=1).flatten().cpu()
sur_loss_current = metrics.get_probs(initial_phi.unsqueeze(0), X_initial)[0].sum(dim=1).flatten().cpu()

dists_train = torch.norm(training_phis - initial_phi, dim=1).numpy()
dists_val = torch.norm(validation_phis - initial_phi, dim=1).numpy()

plt.figure(figsize=(6, 6))
plt.scatter(true_loss_train, sur_loss_train, s=40, c=dists_train, cmap='viridis', marker='o', label='Train samples')
plt.scatter(true_loss_val, sur_loss_val, s=40, c=dists_val, cmap='plasma', marker='s', label='Val samples')
plt.scatter(true_loss_current, sur_loss_current, color='red', marker='*', s=150, label='Initial sample', zorder=5)
plt.colorbar(label='Distance to initial phi')
mn = min(true_loss_train.min().item(), true_loss_val.min().item(), true_loss_current.min().item())
mx = max(true_loss_train.max().item(), true_loss_val.max().item(), true_loss_current.max().item())
plt.plot([mn, mx], [mn, mx], 'k--', label='y = x')
plt.xlabel('True Loss'); plt.ylabel('Surrogate Loss'); plt.title('Surrogate vs True Loss (Train & Val)')
plt.legend(); plt.grid(True); plt.tight_layout(); plt.savefig('outputs/figs/surrogate_vs_true_loss_lcso.png')
if args.plot:
    plt.show()
plt.close()

# Calibration: predicted vs real probabilities (binned) — plot train & val separately
bin_edges = torch.linspace(0, 1, 12)
mid_t, mean_t, std_t = metrics.get_calib_stats_vs_real(real_probs_train.flatten(), pred_probs_train.flatten(), bin_edges)
mid_v, mean_v, std_v = metrics.get_calib_stats_vs_real(real_probs_val.flatten(), pred_probs_val.flatten(), bin_edges)

plt.figure(figsize=(10, 8))
if len(mid_t) > 0:
    plt.errorbar(mid_t.numpy(), mean_t.numpy(), yerr=std_t.numpy(), fmt="o-", markersize=6, capsize=4, label="Train: Pred vs Real", color='blue')
if len(mid_v) > 0:
    plt.errorbar(mid_v.numpy(), mean_v.numpy(), yerr=std_v.numpy(), fmt="s--", markersize=6, capsize=4, label="Val: Pred vs Real", color='orange')
plt.plot([0, 1], [0, 1], "k--", label="Perfect Agreement")
plt.xlabel('Real Probability (bin midpoint)'); plt.ylabel('Mean Predicted Probability (+/- 1 std)')
plt.title('Calibration: Predicted vs Real Probabilities (Train & Val)')
plt.legend(); plt.grid(True, linestyle='--', alpha=0.6); plt.xlim(0, 1); plt.ylim(0, 1)
plt.gca().set_aspect('equal', adjustable='box'); plt.tight_layout(); plt.savefig('outputs/figs/calibration_plot_real_lcso.png')
if args.plot:
    plt.show()
plt.close()

# Calibration: predicted vs actual hits (sampled outcomes) — train & val
mean_preds_t, frac_pos_t = metrics.get_calib_stats_vs_hits(pred_probs_train.flatten(), y_training.flatten(), bin_edges)
mean_preds_v, frac_pos_v = metrics.get_calib_stats_vs_hits(pred_probs_val.flatten(), y_validation.flatten(), bin_edges)

plt.figure(figsize=(10, 8))
if len(mean_preds_t) > 0:
    plt.plot(mean_preds_t, frac_pos_t, "o-", markersize=6, label="Train: Model Calibration", color='green')
if len(mean_preds_v) > 0:
    plt.plot(mean_preds_v, frac_pos_v, "s--", markersize=6, label="Val: Model Calibration", color='purple')
plt.plot([0, 1], [0, 1], "k--", label="Perfect Calibration")
plt.xlabel("Mean Predicted Probability (per bin)"); plt.ylabel("Fraction of Positives (per bin)")
plt.title("Calibration Plot (vs. Actual Outcomes) - Train & Val")
plt.legend(); plt.grid(True, linestyle='--', alpha=0.6); plt.xlim(0, 1); plt.ylim(0, 1)
plt.gca().set_aspect('equal', adjustable='box'); plt.tight_layout(); plt.savefig('outputs/figs/calibration_plot_sampled_lcso.png')
if args.plot:
    plt.show()
plt.close()

# Histograms of real and predicted probabilities (overlay train & val)
plt.figure(figsize=(10, 7))
bins = np.linspace(0, 1, 50)
log = n_samples > 100_000
rp_t = real_probs_train.flatten().cpu().numpy()
pp_t = pred_probs_train.flatten().cpu().numpy()
rp_v = real_probs_val.flatten().cpu().numpy()
pp_v = pred_probs_val.flatten().cpu().numpy()
plt.hist(rp_t, bins=bins, color='blue', histtype='step', label='Real Probs (Train)', linewidth=2, density=False, log=log)
plt.hist(pp_t, bins=bins, color='green', histtype='step', label='Predicted Probs (Train)', linewidth=2, density=False, log=log)
plt.hist(rp_v, bins=bins, color='cyan', histtype='step', label='Real Probs (Val)', linewidth=2, linestyle='--', density=False, log=log)
plt.hist(pp_v, bins=bins, color='magenta', histtype='step', label='Predicted Probs (Val)', linewidth=2, linestyle='--', density=False, log=log)
plt.xlabel('Probability'); plt.ylabel('Frequency'); plt.title('Histograms of Real and Predicted Probabilities (Train & Val)')
plt.legend(); plt.grid(True, linestyle='--', alpha=0.6); plt.tight_layout(); plt.savefig('outputs/figs/probability_histograms_lcso.png')
if args.plot:
    plt.show()
plt.close()
print("Plots saved.")

eval_x_cpu = SAMPLER.sample_x(initial_phi.unsqueeze(0), n_samples=args.n_samples).squeeze(0).cpu()
eval_x_dev = eval_x_cpu.to(dev)
phi0_cpu = initial_phi.detach().clone().cpu()
phi0_dev = phi0_cpu.to(dev)

surrogate_objective = lambda phi: metrics.surrogate_objective(phi, eval_x_dev)
surrogate_grad = torch.func.grad(surrogate_objective)(phi0_dev)

dev = metrics.surrogate_model.device
surrogate_hessian = metrics.surrogate_hessian_from_hvp(phi0_dev, eval_x_dev, damping=1e-3)

surrogate_step = metrics._newton_step(surrogate_grad.to(surrogate_hessian.device), surrogate_hessian)

phi0 = phi0_dev.detach().cpu().numpy().copy()
search_bounds = [
    (max(0.0, float(phi0[d] - 2.0 * epsilon)), min(1.0, float(phi0[d] + 2.0 * epsilon)))
    for d in range(dim)
]

def surrogate_objective_and_grad(phi_np):
    phi_t = torch.tensor(phi_np, dtype=torch.get_default_dtype(), device=dev)
    value = surrogate_objective(phi_t)
    grad = torch.func.grad(surrogate_objective)(phi_t)
    return value.item(), grad.detach().cpu().numpy().copy()

res_sur = scipy_minimize(
    surrogate_objective_and_grad,
    phi0,
    jac=True,
    bounds=search_bounds,
    method='L-BFGS-B',
    options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-12},
)
surrogate_min_phi = torch.tensor(res_sur.x, dtype=torch.get_default_dtype(), device=dev)
surrogate_min_direction = surrogate_min_phi - phi0_dev

cos_step_surrogate = metrics.cosine_similarity(surrogate_grad.to(surrogate_hessian.device), surrogate_step)
print(f"Cosine between surrogate gradient and surrogate Newton step: {cos_step_surrogate}")
print("Surrogate direct minimum:", surrogate_min_phi.detach().cpu())
print(f"Surrogate direct minimum objective: {surrogate_objective(surrogate_min_phi).item():.6f}")

if args.problem != 'MuonShield':
    true_objective = lambda phi: metrics.true_objective(phi, eval_x_cpu)
    real_gradient = torch.func.grad(true_objective)(phi0_cpu.clone().requires_grad_(True)).to(dev)
    real_hessian = torch.func.hessian(true_objective)(phi0_cpu.clone().requires_grad_(True)).to(dev)
    real_hessian = 0.5 * (real_hessian + real_hessian.T)

    print("Real Gradient:", real_gradient)
    print("Surrogate Gradient:", surrogate_grad)

    eig_real = torch.linalg.eigvalsh(real_hessian)
    eig_sur = torch.linalg.eigvalsh(surrogate_hessian.cpu() if surrogate_hessian.device != real_hessian.device else surrogate_hessian)
    print("Eigenvalues of Real Hessian:", eig_real.detach().cpu().numpy())
    print("Eigenvalues of Surrogate Hessian:", eig_sur.detach().cpu().numpy())

    real_step = metrics._newton_step(real_gradient, real_hessian)
    cos_step_real = metrics.cosine_similarity(real_gradient, real_step)
    cos_between_steps = metrics.cosine_similarity(real_step.to(surrogate_step.device), surrogate_step)

    print("Real Step:", real_step)
    print("Surrogate Step:", surrogate_step)
    print(f"Cosine between real gradient and real Newton step: {cos_step_real}")
    print('=' * 50)
    print("Cosine between real and surrogate gradients:", metrics.cosine_similarity(real_gradient.to(surrogate_grad.device), surrogate_grad))
    print(f"Cosine between real step and surrogate step: {cos_between_steps}")

# --- Gain analysis for gradient, Newton, and direct surrogate-minimum steps ---
print('\n' + '=' * 50)
print("Gain analysis (true objective only)")
print('=' * 50)
x0_true_hits = metrics.true_objective(phi0_dev, eval_x_cpu, reduction='sum').item()
print(f"Objective at x0 (true total hits on fixed X): {x0_true_hits:.6f}")
etas = [1e-4, 1e-3, 0.01, 0.1, 1.0, 10.0]
gains_newton_true = []
gains_grad_true = []
gains_direct_true = []

phi0 = phi0_dev
grad = surrogate_grad.to(dev)
step = surrogate_step.to(dev)

def format_point(point):
    return np.array2string(point.detach().cpu().numpy(), precision=4, floatmode='fixed')

for eta in etas:
    phi1_newton = (phi0 + eta * step).clamp(0.0, 1.0)
    phi1_grad = (phi0 - eta * grad).clamp(0.0, 1.0)
    phi1_direct = (phi0 + eta * surrogate_min_direction).clamp(0.0, 1.0)

    gn_t = metrics.gain_newton_true(phi0, surrogate_step, eta, eval_x_cpu)
    gg_t = metrics.gain_gradient_true(phi0, surrogate_grad, eta, eval_x_cpu)
    gd_t = metrics.gain_direction_true(phi0, surrogate_min_direction, eta, eval_x_cpu)
    gains_newton_true.append(gn_t)
    gains_grad_true.append(gg_t)
    gains_direct_true.append(gd_t)
    print(
        f"eta={eta:.0e}  | Gain Newton={gn_t:.6f}"
        f"  | Gain Grad={gg_t:.6f}"
        f"  | Gain Direct={gd_t:.6f}"
    )
    print(
        f"            x1 Newton={format_point(phi1_newton)}"
        f"  | x1 Grad={format_point(phi1_grad)}"
        f"  | x1 Direct={format_point(phi1_direct)}"
    )

# --- Gain comparison plot ---
fig, ax_true = plt.subplots(1, 1, figsize=(8, 5))
eta_labels = [f"{e:.0e}" for e in etas]
x_pos = np.arange(len(etas))
width = 0.24

ax_true.bar(x_pos - width, gains_newton_true, width, label='Newton step', color='C0')
ax_true.bar(x_pos, gains_grad_true, width, label='Gradient step', color='C1')
ax_true.bar(x_pos + width, gains_direct_true, width, label='Direct surrogate minimum', color='C2')
ax_true.axhline(0.0, color='k', linewidth=1, linestyle='--', alpha=0.6)
ax_true.set_xticks(x_pos)
ax_true.set_xticklabels(eta_labels)
ax_true.set_xlabel('Step size (eta)')
ax_true.set_title('Gain — True Total Hits (Fixed X)')
ax_true.legend()
ax_true.grid(True, axis='y', linestyle='--', alpha=0.4)

fig.suptitle('True-Total-Hits Gain Comparison on a Fixed X Sample')
plt.tight_layout()
plt.savefig('outputs/figs/gain_comparison_lcso.png')
if args.plot:
    plt.show()
plt.close(fig)
print("Gain plot saved to outputs/figs/gain_comparison_lcso.png")

