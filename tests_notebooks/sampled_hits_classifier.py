import torch
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.abspath(os.path.join('..', 'src')))
sys.path.append(os.path.abspath(os.path.join('..')))
from models import BinaryClassifierModel
from optimizer import LCSO, denormalize_vector, LCSO_sampled
from problems import ThreeHump_stochastic_hits, Rosenbrock_stochastic_hits, HelicalValley_stochastic_hits, ShipMuonShieldCuda
from utils import normalize_vector
import torch
import json
import numpy as np
import argparse
import time
import pickle

import subprocess
import copy
def get_freest_gpu():
    if not torch.cuda.is_available():
        return torch.device('cpu')
    result = subprocess.run(
        ['nvidia-smi', '--query-gpu=memory.free', '--format=csv,nounits,noheader'],
        stdout=subprocess.PIPE, encoding='utf-8'
    )
    # Parse free memory for each GPU
    mem_free = [int(x) for x in result.stdout.strip().split('\n')]
    max_idx = mem_free.index(max(mem_free))
    return torch.device(f'cuda:{max_idx}')
torch.autograd.set_detect_anomaly(True)
torch.manual_seed(42)
np.random.seed(42)
dev = get_freest_gpu()

parser = argparse.ArgumentParser()
parser.add_argument('--second_order', action='store_true', help='Use second order optimization')
parser.add_argument('--lr', type=float, default=0.1, help='Initial learning rate for phi optimization')
parser.add_argument('--problem', type=str, choices=['ThreeHump', 'Rosenbrock', 'HelicalValley', 'MuonShield'], default='Rosenbrock', help='Optimization problem to use')
parser.add_argument('--batch_size', type=int, default=32768, help='Batch size for training the surrogate model')
parser.add_argument('--samples_phi', type=int, default=5, help='Number of samples for phi in each iteration')
parser.add_argument('--n_samples', type=int, default=5_000_000, help='Number of samples per phi')
parser.add_argument('--subsamples', type=int, default=500_000, help='Number of subsamples for surrogate training per iteration')
parser.add_argument('--n_test', type=int, default=5, help='Number of test samples to evaluate the surrogate model')
parser.add_argument('--n_epochs', type=int, default=30, help='Number of epochs to train the surrogate model')
parser.add_argument('--epsilon', type=float, default=0.05, help='Trust region radius for phi optimization')
parser.add_argument('--dims', type=int, default=4, help='Number of dimensions for Rosenbrock problem')
parser
args = parser.parse_args()

epsilon = args.epsilon
n_samples = args.n_samples
initial_lambda_constraints = 1e-3
initial_lr = args.lr
weight_hits = args.problem == 'MuonShield'
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
    config_file = "/home/hep/lprate/projects/BlackBoxOptimization/outputs/config_tests.json"
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

samples_phi = args.samples_phi
print(f"Using problem {args.problem} with dim {dim} and x_dim {x_dim}, samples_phi = {samples_phi}")

bounds = problem.GetBounds(device=torch.device('cpu'))
diff_bounds = (bounds[1] - bounds[0])
inv_diff_bounds = 1 / diff_bounds
surrogate_model = BinaryClassifierModel(phi_dim=dim,
                            x_dim = x_dim,
                            n_epochs = 10,#args.n_epochs,
                            step_lr = 8,#int(args.n_epochs*0.8),
                            batch_size = args.batch_size,
                            lr = 1e-2,
                            activation = 'silu',
                            data_from_file = False,
                            device = dev)

optimizer = LCSO_sampled(
    true_model=problem,
    surrogate_model=surrogate_model,
    bounds=bounds,
    samples_phi=samples_phi,
    n_epochs = args.n_epochs,
    n_subsamples = args.subsamples,
    epsilon=epsilon,
    initial_phi=initial_phi,
    initial_lambda_constraints=initial_lambda_constraints,  # Initial lambda for constraints
    initial_lr=initial_lr,  # Initial learning rate or trust_radius for phi optimization
    weight_hits=weight_hits,
    device='cpu',
    second_order = args.second_order, 
    outputs_dir=outputs_dir,
    local_file_storage = None,
    resume=False)

local_results = copy.deepcopy(optimizer.local_results)


t1 = time.time()
optimizer._train_local_surrogate()
torch.save(surrogate_model.model.state_dict(), 'outputs/surrogate_model_ship.pth')
print("Surrogate model trained in", time.time() - t1, "seconds")

plt.figure()
plt.plot(surrogate_model.last_train_loss)
plt.xlabel('Epoch')
plt.ylabel('Training Loss')
plt.title('Surrogate Model Training Loss')
plt.grid(True)
plt.savefig('figs/surrogate_training_loss_lcso.png')

def get_probs(conditions):
    pred_probs_list, real_probs_list = [], []
    with torch.no_grad():
        for i in range(0, len(conditions), args.batch_size):
            c = conditions[i:i+args.batch_size]#.reshape(-1, dim+x_dim)
            p = surrogate_model._predict_proba_cond(c).detach().cpu()
            pred_probs_list.append(p)
            phi = denormalize_vector(c[:, :dim], bounds)
            p = problem.probability_of_hit(phi, c[:, dim:]).cpu() if args.problem != 'MuonShield' else p
            real_probs_list.append(p)
    return torch.cat(pred_probs_list, dim=0), torch.cat(real_probs_list, dim=0)

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

def get_calib_stats_vs_hits(pred_probs, hits, bin_edges):
    mean_preds, frac_positives = [], []
    for i in range(len(bin_edges) - 1):
        lower, upper = bin_edges[i], bin_edges[i+1]
        mask = (pred_probs >= lower) & (pred_probs < upper)
        if mask.sum() > 0:
            mean_preds.append(pred_probs[mask].mean())
            frac_positives.append(hits[mask].float().mean())
    return torch.tensor(mean_preds).numpy(), torch.tensor(frac_positives).numpy()

norm_phi = optimizer._current_phi.detach().unsqueeze(0).cpu()
optimizer.samples_phi = args.n_test
test_phis = optimizer.sample_phi_uniform()
optimizer.local_results = local_results
for phi in test_phis:
    optimizer.simulate_and_update(phi, update_history=True)

conditions = torch.stack(optimizer.local_results[0], dim=0).reshape(-1, dim + x_dim)
hits = torch.stack(optimizer.local_results[1], dim=0)
pred_probs, real_probs = get_probs(conditions)
pred_probs = pred_probs.reshape(hits.shape)
real_probs = real_probs.flatten()

true_loss = hits.sum(1).flatten().cpu()
sur_loss = pred_probs.sum(1).flatten().detach()
print('TRUE LOSS', true_loss, 'SUR LOSS', sur_loss)
true_loss_current = optimizer.local_results[1][0].sum().flatten().detach()
sur_loss_current = get_probs(optimizer.local_results[0][0])[0].sum().flatten().cpu()
dists = torch.norm(test_phis - norm_phi, dim=1).numpy()

plt.figure(figsize=(6, 6))
plt.scatter(true_loss_current, sur_loss_current, color='red', marker='*', s=100, label='Initial sample', zorder=5)
sc = plt.scatter(true_loss, sur_loss,s = 40, c = dists, label='Testing samples')
plt.colorbar(sc, label='Distance to current phi')
plt.plot([true_loss.min().item(), true_loss.max().item()], [true_loss.min().item(), true_loss.max().item()], 'k--', label='y = x')
plt.xlabel('True Loss'); plt.ylabel('Surrogate Loss'); plt.title('Surrogate vs True Loss')
plt.legend(); plt.grid(True); plt.savefig('figs/surrogate_vs_true_loss_lcso.png')

pred_probs = pred_probs.flatten()
hits = hits.flatten()

bin_edges = torch.linspace(0, 1, 12)

plt.figure(figsize=(10, 8))
midpoints, means, stds = get_calib_stats_vs_real(real_probs, pred_probs, bin_edges)
if len(midpoints) > 0:
    plt.errorbar(midpoints, means, yerr=stds, fmt="o-", markersize=8, capsize=5, label="Mean Prediction vs. Real Probability")
plt.plot([0, 1], [0, 1], "k--", label="Perfect Agreement")
plt.xlabel('Real Analytical Probability Bins (Midpoint)', fontsize=12)
plt.ylabel('Mean Predicted Probability (+/- 1 std dev)', fontsize=12)
plt.title('Calibration of Predicted vs. Real Probabilities', fontsize=14)
plt.legend(); plt.grid(True, linestyle='--', alpha=0.6); plt.xlim(0, 1); plt.ylim(0, 1)
plt.gca().set_aspect('equal', adjustable='box'); plt.tight_layout(); plt.savefig('figs/calibration_plot_real_lcso.png')
plt.close()

plt.figure(figsize=(10, 8))
mean_preds, frac_pos = get_calib_stats_vs_hits(pred_probs, hits, bin_edges)
if len(mean_preds) > 0:
    plt.plot(mean_preds, frac_pos, "o-", markersize=8, label="Model Calibration")
plt.plot([0, 1], [0, 1], "k--", label="Perfect Calibration")
plt.xlabel("Mean Predicted Probability (per bin)", fontsize=12)
plt.ylabel("Fraction of Positives (per bin)", fontsize=12)
plt.title("Calibration Plot (vs. Actual Outcomes)", fontsize=14)
plt.legend(); plt.grid(True, linestyle='--', alpha=0.6); plt.xlim(0, 1); plt.ylim(0, 1)
plt.gca().set_aspect('equal', adjustable='box'); plt.tight_layout(); plt.savefig('figs/calibration_plot_sampled_lcso.png')
plt.close()

plt.figure(figsize=(10, 7))
bins = np.linspace(0, 1, 50)
density = False
log = n_samples > 100_000
rp = real_probs.cpu().numpy()
pp = pred_probs.cpu().numpy()
plt.hist(rp, bins=bins, color='blue', histtype='step', label='Real Probs', linewidth=2, density=density, log=log)
plt.hist(pp, bins=bins, color='green', histtype='step', label='Predicted Probs', linewidth=2, density=density, log=log)
plt.xlabel('Probability', fontsize=12); plt.ylabel('Frequency', fontsize=12)
plt.title('Histograms of Real and Predicted Probabilities', fontsize=14)
plt.legend(); plt.grid(True, linestyle='--', alpha=0.6); plt.tight_layout()
plt.savefig('figs/probability_histograms_lcso.png')
plt.close()
print("Plots saved.")

x_samp = problem.sample_x()
def surrogate_objective(phi, batch_size=50000):
    n_hits = 0
    phi = normalize_vector(phi, bounds)
    for x in x_samp.split(batch_size):
        n_hits += optimizer.n_hits(surrogate_model.predict_proba(phi,x).to(phi.device), x[:,-1])
    return n_hits
def surrogate_hessian_batched(phi, batch_size=20000):
    print("Computing surrogate Hessian in batches...")
    H = torch.zeros((phi.numel(), phi.numel()), device=phi.device)
    phi = phi.detach().requires_grad_(True)
    for x_batch in x_samp.split(batch_size):
        def f_batch(phi_local):
            phi_n = normalize_vector(phi_local, bounds)
            return optimizer.n_hits(surrogate_model.predict_proba(phi_n, x_batch).to(phi.device), x_batch[:,-1])
        # Get Hessian for the current batch (small enough to fit in memory)
        H_batch = torch.func.hessian(f_batch)(phi)
        H += H_batch.detach()
        torch.cuda.empty_cache()
    return H
if args.problem != 'MuonShield':
    surrogate_grad = torch.func.grad(surrogate_objective)(optimizer.current_phi) * diff_bounds
    surrogate_hessian = surrogate_hessian_batched(optimizer.current_phi) * diff_bounds[:, None] * diff_bounds[None, :]
    surrogate_step = -torch.linalg.solve(surrogate_hessian, surrogate_grad)
    cos_step_surrogate = (torch.dot(surrogate_grad, surrogate_step) / (torch.linalg.norm(surrogate_grad) * torch.linalg.norm(surrogate_step) + 1e-10)).item()
    print(f"Cosine between surrogate gradient and surrogate Newton step: {cos_step_surrogate}")
if args.problem != 'MuonShield':
    real_gradient = problem.gradient(optimizer.current_phi.unsqueeze(0)).squeeze()
    real_hessian = problem.hessian(optimizer.current_phi).squeeze()
    print("Real Gradient:", real_gradient)
    print("Surrogate Gradient:", surrogate_grad)
    print("Real Hessian:", real_hessian) 
    print("Surrogate Hessian:", surrogate_hessian)
    # compute (pseudo-)inverses robustly and print them
    inv_real_hess = torch.linalg.inv(real_hessian)
    inv_sur_hess = torch.linalg.inv(surrogate_hessian)
    print("Inverse Real Hessian:\n", inv_real_hess)
    print("Inverse Surrogate Hessian:\n", inv_sur_hess)
    step = -torch.linalg.solve(real_hessian, real_gradient)
    cos_step_real = (torch.dot(real_gradient, step) / (torch.linalg.norm(real_gradient) * torch.linalg.norm(step) + 1e-10)).item()
    cos_step = (torch.dot(step, surrogate_step) / (torch.linalg.norm(step) * torch.linalg.norm(surrogate_step) + 1e-10)).item()
    print("Real Step:", step)
    print("Surrogate Step:", surrogate_step)
    print(f"Cosine between real gradient and real Newton step: {cos_step_real}")
    print('=' * 50)
    print("cosine between real and surrogate gradients:", (torch.dot(real_gradient, surrogate_grad) / (torch.linalg.norm(real_gradient) * torch.linalg.norm(surrogate_grad) + 1e-10)).item())
    print(f"Cosine between real step and surrogate step: {cos_step}")
