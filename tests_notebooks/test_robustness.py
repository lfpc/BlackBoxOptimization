import torch
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.abspath(os.path.join('..', 'src')))
sys.path.append(os.path.abspath(os.path.join('..')))
from models import BinaryClassifierModel
from optimizer import LCSO, denormalize_vector
from problems import ThreeHump_stochastic_hits, Rosenbrock_stochastic_hits, HelicalValley_stochastic_hits, ShipMuonShieldCuda
from utils import normalize_vector
import torch
import json
import numpy as np
import argparse
import time
import pickle

import subprocess
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
parser.add_argument('--problem', type=str, choices=['ThreeHump', 'Rosenbrock', 'HelicalValley', 'MuonShield'], default='ThreeHump', help='Optimization problem to use')
parser.add_argument('--batch_size', type=int, default=4096, help='Batch size for training the surrogate model')
parser.add_argument('--samples_phi', type=int, default=5, help='Number of samples for phi in each iteration')
parser.add_argument('--n_samples', type=int, default=100_000, help='Number of samples per phi')
parser.add_argument('--n_test', type=int, default=5, help='Number of test samples to evaluate the surrogate model')
parser.add_argument('--n_epochs', type=int, default=30, help='Number of epochs to train the surrogate model')
parser.add_argument('--epsilon', type=float, default=0.05, help='Trust region radius for phi optimization')
parser.add_argument('--dims', type=int, default=4, help='Number of dimensions for Rosenbrock problem')
args = parser.parse_args()

epsilon = args.epsilon
n_samples = args.n_samples
initial_lambda_constraints = 1e-3
initial_lr = args.lr
weight_hits = False
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
    problem = Rosenbrock_stochastic_hits(dim = dim, n_samples = n_samples, 
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
    dim = 42
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
    CONFIG["sensitive_plane"] = [
    {
      "dz": 0.02,
      "dx": 4,
      "dy": 6,
      "position": 82
    }]
    problem = ShipMuonShieldCuda(**CONFIG)
    initial_phi = problem.initial_phi

samples_phi = args.samples_phi
print(f"Using problem {args.problem} with dim {dim} and x_dim {x_dim}, samples_phi = {samples_phi}")

bounds = problem.GetBounds(device=torch.device('cpu'))
#diff_bounds = (bounds[1] - bounds[0])
surrogate_model = BinaryClassifierModel(phi_dim=dim,
                            x_dim = x_dim,
                            n_epochs = args.n_epochs,
                            batch_size = args.batch_size,
                            lr = 1e-2,
                            activation = 'silu',
                            step_lr = 20,
                            data_from_file = False,
                            device = dev)

optimizer = LCSO(
    true_model=problem,
    surrogate_model=surrogate_model,
    bounds=bounds,
    samples_phi=samples_phi,
    epsilon=epsilon,
    initial_phi=initial_phi,
    initial_lambda_constraints=initial_lambda_constraints,  # Initial lambda for constraints
    initial_lr=initial_lr,  # Initial learning rate or trust_radius for phi optimization
    weight_hits=weight_hits,
    device='cpu',
    outputs_dir=outputs_dir,
    second_order = args.second_order,
    local_file_storage = None,
    resume=False)



phis = optimizer.sample_phi()
for phi in phis:
    try: optimizer.simulate_and_update(phi, update_history=False)
    except:
        print("Simulation failed for phi:", phi)
        continue
local_info = optimizer.local_results

with open('outputs/local_info_ship.pkl', "wb") as f:
    pickle.dump(local_info, f)

local_info = (torch.stack(local_info[0]), torch.stack(local_info[1]))

t1 = time.time()
surrogate_model.fit(local_info[0].reshape(-1, dim+x_dim), local_info[1].reshape(-1, 1))
print("Surrogate model trained in", time.time() - t1, "seconds")
surrogate_model.save_weights('outputs/surrogate_model_lcso_ship.pt')

plt.figure()
plt.plot(surrogate_model.last_train_loss)
plt.xlabel('Epoch')
plt.ylabel('Training Loss')
plt.title('Surrogate Model Training Loss')
plt.grid(True)
plt.savefig('figs/surrogate_training_loss_lcso.png')
plt.close()

def get_local_info(phi):
    """
    Computes loss, gradient, and Hessian in a memory-efficient, batched manner.
    """

    phi_norm = normalize_vector(phi, bounds)
    x_all = torch.as_tensor(problem.sample_x(), dtype=phi.dtype, device=phi.device)
    n = x_all.size(0)

    total_loss = torch.tensor(0.0, dtype=phi.dtype, device=phi.device)
    total_grad = torch.zeros_like(phi)
    total_hess = torch.zeros(phi.numel(), phi.numel(), dtype=phi.dtype, device=phi.device)

    def compute_batch_loss(phi_norm, xb):
        cond = torch.cat([phi_norm.repeat(xb.size(0), 1), xb], dim=-1)
        y_pred = optimizer.model.predict_proba(cond).cpu()
        return optimizer.n_hits(y_pred, xb)

    grad_func_batch = torch.func.grad(compute_batch_loss)
    hess_func_batch = torch.func.hessian(compute_batch_loss)

    for start in range(0, n, 5000):
        xb = x_all[start: start + 5000]
        batch_loss = compute_batch_loss(phi_norm, xb).detach()
        batch_grad = grad_func_batch(phi_norm, xb).detach()
        batch_hess = hess_func_batch(phi_norm, xb).detach()
        total_loss += batch_loss
        total_grad += batch_grad
        total_hess += batch_hess
    return total_loss, total_grad, total_hess

def calc_delta_loss_torch(delta_phi: torch.Tensor, 
                          grad: torch.Tensor, 
                          hess: torch.Tensor) -> float:
    """
    Estimates the change in loss for a given parameter perturbation using PyTorch.

    This function uses a second-order Taylor expansion to estimate the change
    in the loss function (delta_loss) based on the local gradient and Hessian.

    Args:
        delta_phi: The expected parameter perturbation (a 1D torch.Tensor).
        grad: The gradient of the loss function at the current point (a 1D torch.Tensor).
        hess: The Hessian matrix of the loss function at the current point (a 2D torch.Tensor).

    Returns:
        The estimated change in loss (a scalar float).
    """
    # Linear term: (∇L)ᵀ * δφ
    def normalize_delta(d, lower_bound, upper_bound):
        return d / (upper_bound - lower_bound)

    delta_phi = normalize_delta(delta_phi, bounds[0], bounds[1])
    linear_term = torch.dot(grad, delta_phi)

    # Quadratic term: 0.5 * δφᵀ * H * δφ
    # The @ operator performs matrix-vector multiplication
    quadratic_term = 0.5 * torch.dot(delta_phi, hess @ delta_phi)

    # The result is a 0-dimensional tensor; .item() extracts the Python scalar
    return (linear_term + quadratic_term).item()


loss = optimizer.history[1][0]

phi = optimizer.current_phi
sur_loss, grad, hess = get_local_info(phi)

results_dict = {
    'loss': loss.item(),
    'sur_loss': sur_loss.item(),
    'grad': grad,
    'hess': hess
}
with open('local_info.pkl', 'wb') as f:
    pickle.dump(results_dict, f)


print(f"Initial True Loss: {optimizer.history[1][0]*1e5}")
print(f"Initial Hessian: {hess}")
eigenvalues, eigenvectors = torch.linalg.eigh(hess)
print(f"Hessian eigenvalues: {eigenvalues}")
for idx, (eigval, eigvec) in enumerate(zip(eigenvalues, eigenvectors.T)):
    print(f"Eigenvalue {idx}: {eigval.item()}")
    print(f"Eigenvector {idx}: {eigvec}")
alphas = np.linspace(-2, 2, 100)
print(f"Initial Surrogate Loss: {loss.item()}")
print(f"Initial Gradient: {grad}")

# --- New plot: all delta_phi directions in one figure with subplots (2 columns) ---
fig, axes = plt.subplots(nrows=(len(eigenvectors.T)+1)//2, ncols=2, figsize=(12, 3*((len(eigenvectors.T)+1)//2)))
axes = axes.flatten()
for idx, eigvec in enumerate(eigenvectors.T):
    delta_losses = []
    for alpha in alphas:
        delta_phi = alpha * eigvec
        delta_loss = calc_delta_loss_torch(delta_phi, grad, hess)
        delta_losses.append(delta_loss+ loss.item())
    axes[idx].plot(alphas, delta_losses)
    axes[idx].set_xlabel('alpha')
    axes[idx].set_ylabel('Delta Loss')
    axes[idx].set_title(f'Eigenvector {idx}')
    axes[idx].grid(True)
# Hide unused subplots if any
for ax in axes[len(eigenvectors.T):]:
    ax.axis('off')
plt.tight_layout()
plt.savefig('delta_loss_vs_alpha_all_eigenvectors.png')

