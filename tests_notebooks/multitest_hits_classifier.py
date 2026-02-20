import torch
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.abspath(os.path.join('..', 'src')))
sys.path.append(os.path.abspath(os.path.join('..')))
from problems import ThreeHump_stochastic_hits, Rosenbrock_stochastic_hits, HelicalValley_stochastic_hits, ShipMuonShieldCuda
from utils import normalize_vector, denormalize_vector, get_freest_gpu
from utils.nets import Classifier, DeepONetClassifier, QuadraticModel, ParametricNWDeepONet
import torch
from tqdm import trange
import json
import numpy as np
import argparse
from time import time
from scipy.stats.qmc import LatinHypercube


dev = get_freest_gpu()
#torch.set_default_dtype(torch.float64)

parser = argparse.ArgumentParser()
parser.add_argument('--second_order', action='store_true', help='Use second order optimization')
parser.add_argument('--problem', type=str, choices=['ThreeHump', 'Rosenbrock', 'HelicalValley', 'MuonShield'], default='Rosenbrock', help='Optimization problem to use')
parser.add_argument('--batch_size', type=int, default=32768, help='Batch size for training the surrogate model')
parser.add_argument('--samples_phi', type=int, default=5, help='Number of samples for phi in each iteration')
parser.add_argument('--n_samples', type=int, default=1_000_000, help='Number of samples per phi')
parser.add_argument('--subsamples', type=int, default=200_000, help='Number of subsamples for surrogate training per iteration')
parser.add_argument('--n_test', type=int, default=5, help='Number of test samples to evaluate the surrogate model')
parser.add_argument('--n_epochs', type=int, default=10, help='Number of epochs to train the surrogate model')
parser.add_argument('--step_lr', type=int, default=8, help='Step size for learning rate scheduler')
parser.add_argument('--epsilon', type=float, default=0.1, help='Trust region radius for phi optimization')
parser.add_argument('--dims', type=int, default=4, help='Number of dimensions for Rosenbrock problem')
parser.add_argument('--model', type=str, choices=['mlp', 'deeponet', 'quadratic', 'NW'], default='mlp', help='Type of surrogate model to use')
parser.add_argument(
    '--hess_method',
    type=str,
    choices=['hess', 'hvp'],
    default='hess',
    help="How to compute surrogate Hessian for analysis: "
         "'hess' = torch.func.hessian, 'hvp' = build Hessian via (damped) HVP columns")
parser.add_argument('--n_experiments', type=int, default=5, help='Number of random experiments / seeds to run')
args = parser.parse_args()


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
    CONFIG['initial_phi'] = ShipMuonShieldCuda.params['stellatryon_v2']
    CONFIG['n_samples'] = n_samples
    CONFIG['reduction'] = 'none'
    CONFIG['cost_as_constraint'] = False
    problem = ShipMuonShieldCuda(**CONFIG)
    initial_phi = problem.initial_phi

print(f"Using problem {args.problem} with dim {dim} and x_dim {x_dim}, samples_phi = {samples_phi}")

bounds = problem.GetBounds(device=torch.device('cpu'))


class Sampler():
    def __init__(self,true_model,
                 bounds:tuple,
                 epsilon:float = 0.2,
                 initial_phi:torch.tensor = None,
                 seed:int = 42,
                second_order:bool = False
                 ):
        
        self.bounds = bounds
        initial_phi = normalize_vector(initial_phi, self.bounds)
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
            perturb = (2*torch.from_numpy(perturb).to(dtype=torch.get_default_dtype()) - 1.0) * self.epsilon
            if self.second_order:
                perturb_small = self.lhs_sampler.random(samples_phi//2)
                perturb_small = (2*torch.from_numpy(perturb_small).to(dtype=torch.get_default_dtype()) - 1.0) * self.epsilon/2
                perturb = torch.cat([perturb[:samples_phi//2], perturb_small], dim=0)
            phis = (self._current_phi.unsqueeze(0).cpu() + perturb).clamp(0.0,1.0)
        return phis
    def sample_phi_uniform(self, samples_phi:int = 10):
        """Draw samples uniformly within the bounds around current_phi."""
        with torch.no_grad():
            perturb = (torch.rand(samples_phi, self._current_phi.size(-1)) * 2 - 1) * self.epsilon
            phis = (self._current_phi.unsqueeze(0).cpu() + perturb).clamp(0.0,1.0)
        return phis
    def sample_x(self, phi, n_samples = None):
        """Sample x from the true model for given phi."""
        if n_samples is not None:
            self.true_model.n_samples = n_samples
        x = self.true_model.sample_x(phi)
        return x.to(torch.get_default_dtype())

SAMPLER = None

class Metrics:
    """Helper to compute surrogate diagnostics for train/validation splits."""

    def __init__(self, bounds, true_model, surrogate_model):
        self.true_model = true_model
        self.surrogate_model = surrogate_model
        self.device = surrogate_model.device
        self.bounds = bounds.to(self.device)
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
        phi_denorm = denormalize_vector(phi.detach(), self.bounds).cpu()

        for start in range(0, num_samples_x, per_phi_batch):
            end = min(start + per_phi_batch, num_samples_x)
            x_batch = x[:, start:end].to(self.device)
            y_batch = y[:, start:end].float().to(self.device)
            logits = self.surrogate_model.model(phi, x_batch)
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
        phi0 = phi.detach().clone().to(self.device).requires_grad_(True)
        phi0_denorm = denormalize_vector(phi0, self.bounds)
        grad_sur = torch.func.grad(self.surrogate_objective)(phi0, x[0])
        grad_real = self.true_model.gradient(phi0_denorm, analytic = False).squeeze()
        grad_cosine = self.cosine_similarity(grad_sur, grad_real)
        surrogate_hessian = self.surrogate_hessian_from_hvp(phi0, x[0].unsqueeze(0), damping=1e-3)
        real_hessian = self.true_model.hessian(phi0_denorm, analytic=False).squeeze()
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
        phi = phi.to(self.device)
        phi_denorm = denormalize_vector(phi.detach(), self.bounds).cpu()
        batch_size = self.surrogate_model.batch_size
        with torch.no_grad():
            for start in range(0, num_samples_x, batch_size//num_samples_phi):
                end = min(start + batch_size//num_samples_phi, num_samples_x)
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
        total = 0.0
        for x_batch in x.split(self.surrogate_model.batch_size, dim=1):
            p = self.surrogate_model.predict_proba(phi.view(1, -1), x_batch.unsqueeze(0))
            total = total + self.n_hits(p.to(phi.device))
        return total / x.size(1)

    def surrogate_hessian_from_hvp(self,phi, x, damping=0.0):
        """
        Builds an explicit Hessian approximation by columns using HVPs:
        H[:, i] = H e_i   (optionally (H + damping I)e_i)

        This is O(d) HVPs; OK for small d (<= ~50).
        """
        phi = phi.detach().requires_grad_(True)
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
        phi = denormalize_vector(phi.detach().cpu(), self.bounds)
        real_probs = self.true_model.probability_of_hit(phi, x)
        diff = torch.abs(pred_probs.detach().cpu() - real_probs)
        diff_per_phi = diff.view(diff.size(0), -1).mean(dim=1)
        return diff_per_phi.mean().item()

    @staticmethod
    def cosine_similarity(vec_a, vec_b, eps=1e-10):
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
        sym_hess = 0.5 * (hess + hess.T)
        eye = torch.eye(sym_hess.size(0), device=sym_hess.device, dtype=sym_hess.dtype)
        shift = torch.clamp(-torch.linalg.eigvalsh(sym_hess).min() + eps, min=0.0)
        hpd = sym_hess + shift * eye
        step = -torch.linalg.solve(hpd, grad)
        return step

@torch.no_grad()
def simulate(phi, n_samples:int = None):
    if phi.dim() == 1:
        phi = phi.view(1,-1)
    x = SAMPLER.sample_x(phi, n_samples=n_samples)
    phi = denormalize_vector(phi, SAMPLER.bounds).detach().cpu()
    y = problem(phi,x)
    return x,y

def get_local_data(n_samples_phi:int = 10, n_samples_x:int = None, sampling = 'lhs'):
    if sampling == 'lhs':
        phis = SAMPLER.sample_phi(samples_phi=n_samples_phi)
    elif sampling == 'uniform':
        phis = SAMPLER.sample_phi_uniform(samples_phi=n_samples_phi)
    xs = []
    ys = []
    for phi in phis:
        x,y = simulate(phi, n_samples=n_samples_x)
        xs.append(x.detach().cpu())
        ys.append(y.detach().cpu())
    xs = torch.stack(xs)
    ys = torch.stack(ys)
    return phis, xs, ys




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
                 device: str = 'cuda'):
        self.device = device
        if model == 'mlp':
            self.model = Classifier(phi_dim,x_dim, 128).to(self.device)
        elif model == 'deeponet':
            layers = [[128,128],[128,128]]#[[256,128,128],[128,128]]
            self.model = DeepONetClassifier(phi_dim,x_dim, layers = layers, layer_norm=False, p=128).to(self.device)
        elif model == 'quadratic':
            self.model = QuadraticModel(phi_dim, x_dim,trunk_layers=[128,128], layer_norm=False, p=128).to(self.device)
        elif model == 'NW':
            self.model = ParametricNWDeepONet(phi_dim,x_dim, layers = [[256,128,128],[128,128]], layer_norm=False, p=128,num_experts=32).to(self.device)
        
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-5)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=step_lr, gamma=0.1)
        self.bounds_phi = bounds_phi.to(self.device)
        self.step_lr = step_lr
        self.loss_fn = torch.nn.BCEWithLogitsLoss()
        self.metrics = None  # To be set externally
    def get_model_pred(self, phi, x = None):
        n_hits = torch.zeros(phi.size(0), device=self.device)
        if x is None: x = SAMPLER.sample_x(phi).to(self.device)
        num_samples_x = x.size(1)
        num_samples_phi = phi.size(0)
        for start in range(0, num_samples_x, self.batch_size//num_samples_phi):
            x_batch = x[:, start:start + self.batch_size//num_samples_phi].detach()
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
        indices = torch.arange(num_samples_x)

        train_metrics = {'loss': [], 'hits_error': [], 'prob_gap': []}
        val_metrics = {'loss': [], 'hits_error': [], 'prob_gap': []}
        grad_metrics = {'grad_cosine': [], 'newton_cosine': []}

        pbar = trange(self.n_epochs, position=0, leave=True, desc='Classifier Training on ' + str(self.device))
        for epoch in pbar:
            average_loss = 0.0
            shuffled_indices = indices[torch.randperm(num_samples_x)]
            for start in range(0, num_samples_x, self.batch_size//num_samples_phi):
                end = min(start + self.batch_size//num_samples_phi, num_samples_x)
                batch_idx = shuffled_indices[start:end]
                y_batch = y[:, batch_idx].float().to(self.device)
                x_batch = x[:, batch_idx].to(self.device)
                self.optimizer.zero_grad()
                y_pred_logits = self.model(phi, x_batch)
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
        phi = phi.to(self.device)
        self.model.eval()
        #phi = self.normalize_params(phi)
        return torch.sigmoid(self.model(phi, x.to(self.device)))
    def normalize_params(self, params):
        return normalize_vector(params, self.bounds_phi)


# Run multiple random experiments (seeds) and aggregate epoch-wise metrics
initial_phi_norm = normalize_vector(initial_phi, bounds)

# Accumulators
train_loss_all = []
val_loss_all = []
train_hits_all = []
val_hits_all = []
train_prob_gap_all = []
val_prob_gap_all = []
grad_cos_all = []
newton_cos_all = []

# Pooled predictions for calibration/scatter/histograms
pred_probs_train_list = []
real_probs_train_list = []
pred_probs_val_list = []
real_probs_val_list = []
true_loss_train_list = []
sur_loss_train_list = []
true_loss_val_list = []
sur_loss_val_list = []
y_train_raw_list = []
y_val_raw_list = []

for seed in range(args.n_experiments):
    print(f"Running experiment seed={seed+1}/{args.n_experiments}")
    torch.manual_seed(seed)
    np.random.seed(seed)

    # recreate sampler for this seed
    SAMPLER = Sampler(true_model=problem,
                      bounds=bounds,
                      epsilon=epsilon,
                      initial_phi=initial_phi,
                      seed=seed,
                      second_order=args.second_order)

    X_initial, Y_initial = simulate(initial_phi_norm.unsqueeze(0), n_samples=args.n_samples)
    training_phis = SAMPLER.sample_phi(samples_phi=args.samples_phi)
    training_phis = torch.cat([initial_phi_norm.unsqueeze(0), training_phis], dim=0)
    validation_phis = SAMPLER.sample_phi_uniform(samples_phi=args.n_test)

    x_training, y_training = simulate(training_phis, n_samples=args.subsamples)
    x_validation, y_validation = simulate(validation_phis, n_samples=args.n_samples)

    surrogate_model = BinaryClassifierModel(
        phi_dim=dim,
        x_dim=x_dim,
        bounds_phi=bounds,
        n_epochs=args.n_epochs,
        step_lr=args.step_lr,
        batch_size=args.batch_size,
        model=args.model,
        device=dev)

    metrics = Metrics(bounds, problem, surrogate_model)
    surrogate_model.metrics = metrics

    train_metrics, val_metrics, grad_metrics = surrogate_model.fit(
        training_phis, x_training, y_training,
        val_phis=validation_phis, val_x=x_validation, val_y=y_validation)

    train_loss_all.append(np.array(train_metrics['loss']))
    val_loss_all.append(np.array(val_metrics['loss']))
    train_hits_all.append(np.array(train_metrics['hits_error']))
    val_hits_all.append(np.array(val_metrics['hits_error']))
    train_prob_gap_all.append(np.array(train_metrics['prob_gap']))
    val_prob_gap_all.append(np.array(val_metrics['prob_gap']))
    grad_cos_all.append(np.array(grad_metrics['grad_cosine']))
    newton_cos_all.append(np.array(grad_metrics['newton_cosine']))

    # store pooled pred/real probs for calibration and scatter
    p_train, r_train = metrics.get_probs(training_phis, x_training)
    p_val, r_val = metrics.get_probs(validation_phis, x_validation)
    pred_probs_train_list.append(p_train.detach().cpu())
    real_probs_train_list.append(r_train.detach().cpu())
    pred_probs_val_list.append(p_val.detach().cpu())
    real_probs_val_list.append(r_val.detach().cpu())

    true_loss_train_list.append(y_training.sum(dim=1).flatten().cpu())
    sur_loss_train_list.append(p_train.sum(dim=1).flatten().cpu())
    y_train_raw_list.append(y_training.flatten().cpu())
    true_loss_val_list.append(y_validation.sum(dim=1).flatten().cpu())
    sur_loss_val_list.append(p_val.sum(dim=1).flatten().cpu())
    y_val_raw_list.append(y_validation.flatten().cpu())

# Convert to arrays and compute mean/std across experiments (axis=0 over experiments)
train_loss_arr = np.stack(train_loss_all)
val_loss_arr = np.stack(val_loss_all)
train_hits_arr = np.stack(train_hits_all)
val_hits_arr = np.stack(val_hits_all)
train_prob_gap_arr = np.stack(train_prob_gap_all)
val_prob_gap_arr = np.stack(val_prob_gap_all)
grad_cos_arr = np.stack(grad_cos_all)
newton_cos_arr = np.stack(newton_cos_all)

epochs = np.arange(1, train_loss_arr.shape[1] + 1)

# Plot mean +/- std for losses and metrics
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

# Loss
mean_train = train_loss_arr.mean(axis=0)
std_train = train_loss_arr.std(axis=0)
mean_val = val_loss_arr.mean(axis=0)
std_val = val_loss_arr.std(axis=0)
ax1.plot(epochs, mean_train, label='Train Loss (mean)', color='blue')
ax1.fill_between(epochs, mean_train - std_train, mean_train + std_train, color='blue', alpha=0.2)
ax1.plot(epochs, mean_val, label='Val Loss (mean)', color='orange')
ax1.fill_between(epochs, mean_val - std_val, mean_val + std_val, color='orange', alpha=0.2)
ax1.set_ylabel('Loss')
ax1.set_title('Training and Validation Loss (mean ± std)')
ax1.legend()
ax1.grid(True, linestyle='--', alpha=0.6)

# Hits error and prob gap
mean_hits = train_hits_arr.mean(axis=0)
std_hits = train_hits_arr.std(axis=0)
mean_prob = train_prob_gap_arr.mean(axis=0)
std_prob = train_prob_gap_arr.std(axis=0)
ax2.plot(epochs, mean_hits, label='Train Hits Error (mean)', color='tab:blue')
ax2.fill_between(epochs, mean_hits - std_hits, mean_hits + std_hits, color='tab:blue', alpha=0.2)
ax2.set_ylabel('Hits Error', color='tab:blue')
ax2.tick_params(axis='y', labelcolor='tab:blue')
ax2b = ax2.twinx()
ax2b.plot(epochs, mean_prob, label='Train Prob Gap (mean)', color='tab:green')
ax2b.fill_between(epochs, mean_prob - std_prob, mean_prob + std_prob, color='tab:green', alpha=0.2)
mean_hits_v = val_hits_arr.mean(axis=0)
std_hits_v = val_hits_arr.std(axis=0)
mean_prob_v = val_prob_gap_arr.mean(axis=0)
std_prob_v = val_prob_gap_arr.std(axis=0)
ax2.plot(epochs, mean_hits_v, label='Val Hits Error (mean)', color='tab:blue', linestyle='--')
ax2b.plot(epochs, mean_prob_v, label='Val Prob Gap (mean)', color='tab:green', linestyle='--')
ax2.set_xlabel('Epoch')
ax2.set_title('Hits Error and Probability Gap (mean ± std)')
lines1, labels1 = ax2.get_legend_handles_labels()
lines2, labels2 = ax2b.get_legend_handles_labels()
ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

# Cosine similarities
mean_grad = grad_cos_arr.mean(axis=0)
std_grad = grad_cos_arr.std(axis=0)
mean_newt = newton_cos_arr.mean(axis=0)
std_newt = newton_cos_arr.std(axis=0)
ax3.plot(epochs, mean_grad, label='Grad Cosine (mean)', color='purple', marker='o')
ax3.fill_between(epochs, mean_grad - std_grad, mean_grad + std_grad, color='purple', alpha=0.2)
ax3.plot(epochs, mean_newt, label='Newton Step Cosine (mean)', color='red', marker='x')
ax3.fill_between(epochs, mean_newt - std_newt, mean_newt + std_newt, color='red', alpha=0.2)
ax3.set_ylabel('Cosine Similarity')
ax3.set_xlabel('Epoch')
ax3.set_title('Cosine Similarities (Grad & Newton Step, mean ± std)')
ax3.legend()
ax3.grid(True, linestyle='--', alpha=0.6)

fig.tight_layout()
plt.savefig('figs/train_val_metrics_avg.png')
plt.close(fig)

# Pool predictions across experiments for calibration/scatter/histograms
pred_probs_train_all = torch.cat(pred_probs_train_list, dim=0)
real_probs_train_all = torch.cat(real_probs_train_list, dim=0)
pred_probs_val_all = torch.cat(pred_probs_val_list, dim=0)
real_probs_val_all = torch.cat(real_probs_val_list, dim=0)

true_loss_train = torch.cat(true_loss_train_list).cpu()
sur_loss_train = torch.cat(sur_loss_train_list).cpu()
true_loss_val = torch.cat(true_loss_val_list).cpu()
sur_loss_val = torch.cat(sur_loss_val_list).cpu()

# Current / initial phi losses (from last experiment's X_initial/Y_initial)
true_loss_current = Y_initial.sum(dim=1).flatten().cpu()
sur_loss_current = metrics.get_probs(initial_phi_norm.unsqueeze(0), X_initial)[0].sum(dim=1).flatten().cpu()

dists_train = torch.norm(torch.vstack([training_phis - initial_phi_norm for _ in range(1)]).squeeze(0) - 0, dim=0) if False else None

plt.figure(figsize=(6, 6))
plt.scatter(true_loss_train.numpy(), sur_loss_train.numpy(), s=20, alpha=0.6, label='Train pooled')
plt.scatter(true_loss_val.numpy(), sur_loss_val.numpy(), s=20, alpha=0.6, label='Val pooled')
plt.scatter(true_loss_current, sur_loss_current, color='red', marker='*', s=150, label='Initial sample', zorder=5)
mn = min(true_loss_train.min().item(), true_loss_val.min().item(), true_loss_current.min().item())
mx = max(true_loss_train.max().item(), true_loss_val.max().item(), true_loss_current.max().item())
plt.plot([mn, mx], [mn, mx], 'k--', label='y = x')
plt.xlabel('True Loss'); plt.ylabel('Surrogate Loss'); plt.title('Surrogate vs True Loss (Pooled)')
plt.legend(); plt.grid(True); plt.tight_layout(); plt.savefig('figs/surrogate_vs_true_loss_pooled.png')
plt.close()

# Calibration: predicted vs real probabilities (binned) — pooled
bin_edges = torch.linspace(0, 1, 12)
mid_t, mean_t, std_t = Metrics.get_calib_stats_vs_real(real_probs_train_all.flatten(), pred_probs_train_all.flatten(), bin_edges)
mid_v, mean_v, std_v = Metrics.get_calib_stats_vs_real(real_probs_val_all.flatten(), pred_probs_val_all.flatten(), bin_edges)

plt.figure(figsize=(10, 8))
if len(mid_t) > 0:
    plt.errorbar(mid_t.numpy(), mean_t.numpy(), yerr=std_t.numpy(), fmt="o-", markersize=6, capsize=4, label="Train: Pred vs Real (pooled)", color='blue')
if len(mid_v) > 0:
    plt.errorbar(mid_v.numpy(), mean_v.numpy(), yerr=std_v.numpy(), fmt="s--", markersize=6, capsize=4, label="Val: Pred vs Real (pooled)", color='orange')
plt.plot([0, 1], [0, 1], "k--", label="Perfect Agreement")
plt.xlabel('Real Probability (bin midpoint)'); plt.ylabel('Mean Predicted Probability (+/- 1 std)')
plt.title('Calibration: Predicted vs Real Probabilities (Pooled)')
plt.legend(); plt.grid(True, linestyle='--', alpha=0.6); plt.xlim(0, 1); plt.ylim(0, 1)
plt.gca().set_aspect('equal', adjustable='box'); plt.tight_layout(); plt.savefig('figs/calibration_plot_real_pooled.png')
plt.close()

# Calibration: predicted vs actual hits (sampled outcomes) — pooled
mean_preds_t, frac_pos_t = Metrics.get_calib_stats_vs_hits(pred_probs_train_all.flatten(), torch.cat(y_train_raw_list), bin_edges)
mean_preds_v, frac_pos_v = Metrics.get_calib_stats_vs_hits(pred_probs_val_all.flatten(), torch.cat(y_val_raw_list), bin_edges)

plt.figure(figsize=(10, 8))
if len(mean_preds_t) > 0:
    plt.plot(mean_preds_t, frac_pos_t, "o-", markersize=6, label="Train: Model Calibration (pooled)", color='green')
if len(mean_preds_v) > 0:
    plt.plot(mean_preds_v, frac_pos_v, "s--", markersize=6, label="Val: Model Calibration (pooled)", color='purple')
plt.plot([0, 1], [0, 1], "k--", label="Perfect Calibration")
plt.xlabel("Mean Predicted Probability (per bin)"); plt.ylabel("Fraction of Positives (per bin)")
plt.title("Calibration Plot (vs. Actual Outcomes) - Pooled")
plt.legend(); plt.grid(True, linestyle='--', alpha=0.6); plt.xlim(0, 1); plt.ylim(0, 1)
plt.gca().set_aspect('equal', adjustable='box'); plt.tight_layout(); plt.savefig('figs/calibration_plot_sampled_pooled.png')
plt.close()

# Histograms of real and predicted probabilities (overlay pooled train & val)
plt.figure(figsize=(10, 7))
bins = np.linspace(0, 1, 50)
log = n_samples > 100_000
rp_t = real_probs_train_all.flatten().numpy()
pp_t = pred_probs_train_all.flatten().numpy()
rp_v = real_probs_val_all.flatten().numpy()
pp_v = pred_probs_val_all.flatten().numpy()
plt.hist(rp_t, bins=bins, color='blue', histtype='step', label='Real Probs (Train pooled)', linewidth=2, density=False, log=log)
plt.hist(pp_t, bins=bins, color='green', histtype='step', label='Predicted Probs (Train pooled)', linewidth=2, density=False, log=log)
plt.hist(rp_v, bins=bins, color='cyan', histtype='step', label='Real Probs (Val pooled)', linewidth=2, linestyle='--', density=False, log=log)
plt.hist(pp_v, bins=bins, color='magenta', histtype='step', label='Predicted Probs (Val pooled)', linewidth=2, linestyle='--', density=False, log=log)
plt.xlabel('Probability'); plt.ylabel('Frequency'); plt.title('Histograms of Real and Predicted Probabilities (Pooled)')
plt.legend(); plt.grid(True, linestyle='--', alpha=0.6); plt.tight_layout(); plt.savefig('figs/probability_histograms_pooled.png')
plt.close()
print("Pooled plots saved.")


# Keep the original single-run Hessian/gradient checks using the last experiment's metrics
x_samp = SAMPLER.sample_x(initial_phi_norm.unsqueeze(0), n_samples=args.n_samples).squeeze(0)
surrogate_objective = lambda phi: metrics.surrogate_objective(phi, x_samp)
surrogate_grad = torch.func.grad(surrogate_objective)(initial_phi_norm)

dev = metrics.surrogate_model.device
surrogate_hessian = metrics.surrogate_hessian_from_hvp(initial_phi_norm.to(dev), x_samp.unsqueeze(0).to(dev), damping=1e-3)

surrogate_step = metrics._newton_step(surrogate_grad.to(surrogate_hessian.device), surrogate_hessian)

cos_step_surrogate = metrics.cosine_similarity(surrogate_grad.to(surrogate_hessian.device), surrogate_step)
print(f"Cosine between surrogate gradient and surrogate Newton step: {cos_step_surrogate}")

if args.problem != 'MuonShield':
    real_gradient = problem.gradient(denormalize_vector(initial_phi_norm.unsqueeze(0), bounds), analytic=False).squeeze()
    real_hessian = problem.hessian(denormalize_vector(initial_phi_norm.unsqueeze(0), bounds), analytic=False).squeeze()

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

