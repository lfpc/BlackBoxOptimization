import torch
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.abspath(os.path.join('..', 'src')))
sys.path.append(os.path.abspath(os.path.join('..')))
from models import BinaryClassifierModel, GP_RBF
from optimizer import LCSO, denormalize_vector
from problems import ThreeHump_stochastic_hits, Rosenbrock_stochastic_hits, HelicalValley_stochastic_hits, ShipMuonShieldCuda
from utils import normalize_vector
import torch
import json
import numpy as np
import argparse
import time
import pickle
import os
torch.set_default_dtype(torch.float64)

torch.autograd.set_detect_anomaly(True)
torch.manual_seed(42)
np.random.seed(42)
dev = 'cuda:0'#get_freest_gpu()


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
                                    reduction = 'mean')
    initial_phi = torch.tensor([-1.2, 1.0])
elif args.problem == 'Rosenbrock':
    dim = args.dims
    x_dim = 2
    problem = Rosenbrock_stochastic_hits(dim = dim, n_samples = n_samples, 
                                    phi_bounds = ((-2.0, 2.0)), 
                                    x_bounds = (-3.,3.),
                                    reduction = 'mean')
    initial_phi = torch.tensor([[-1.2, 1.8]*(dim//2)]).flatten()
elif args.problem == 'HelicalValley':
    dim = 3
    x_dim = 2
    problem = HelicalValley_stochastic_hits(n_samples = n_samples, 
                                    phi_bounds = ((-10.0, 10.0)), 
                                    x_bounds = (-5,5),
                                    reduction = 'mean')
    initial_phi = torch.tensor([-9, -1.0, -9.0])
elif args.problem == 'MuonShield':
    dim = 63
    x_dim = 8
    config_file = "/home/hep/lprate/projects/BlackBoxOptimization/outputs/config.json"
    with open(config_file, 'r') as f:
        CONFIG = json.load(f)
    CONFIG.pop("data_treatment", None)
    CONFIG.pop('results_dir', None)
    CONFIG['dimensions_phi'] = 63
    CONFIG['initial_phi'] = ShipMuonShieldCuda.params['warm_baseline']
    CONFIG['n_samples'] = n_samples
    CONFIG['reduction'] = 'mean'
    problem = ShipMuonShieldCuda(**CONFIG)
    initial_phi = problem.initial_phi

samples_phi = args.samples_phi
print(f"Using problem {args.problem} with dim {dim} and x_dim {x_dim}, samples_phi = {samples_phi}")

bounds = problem.GetBounds(device=torch.device('cpu'))
diff_bounds = (bounds[1] - bounds[0])
surrogate_model = GP_RBF(bounds=bounds,
                          device=dev)

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
    resume=False)



phis = optimizer.sample_phi()
for phi in phis:
    phi = denormalize_vector(phi, bounds)
    y = problem(phi)
    optimizer.update_history(phi,y)
phis = optimizer.sample_phi_uniform()
for phi in phis:
    phi = denormalize_vector(phi, bounds)
    y = problem(phi)
    optimizer.update_history(phi,y)
history = optimizer.history
t1 = time.time()
surrogate_model.fit(history[0], history[1].to(torch.get_default_dtype()))
print("Surrogate model trained in", time.time() - t1, "seconds")


with open('outputs/history.pkl', 'wb') as f:
    pickle.dump(history, f)

print(f"History saved to {'outputs/history.pkl'}")


true_loss = history[1]
sur_loss = surrogate_model.posterior(history[0].to(dev)).mean.detach().cpu().squeeze()
sur_std = surrogate_model.posterior(history[0].to(dev)).stddev.detach().cpu().squeeze()

orig_count = len(history[0])
test_phis = initial_phi + (torch.rand(args.n_test, dim) * 2.0 - 1.0) * epsilon
for phi in test_phis:
    y = problem(phi)
    optimizer.update_history(phi,y)
dists = torch.norm(test_phis - initial_phi, dim=1).numpy()

plt.figure(figsize=(6, 6))
plt.scatter(true_loss[0], sur_loss[0], color='red', marker='*', s=100, label='Initial sample', zorder=5)
plt.scatter(true_loss[1:orig_count], sur_loss[1:orig_count],s = 40, color='blue', label='Training samples')
sc = plt.scatter(true_loss[orig_count:], sur_loss[orig_count:], c=dists[orig_count:], cmap='viridis', marker='x', s=30, label='Test samples')
plt.errorbar(true_loss[1:orig_count], sur_loss[1:orig_count], yerr=sur_std[1:orig_count], fmt='o', 
             color='blue', alpha=0.7, capsize=3)
plt.errorbar(true_loss[orig_count:], sur_loss[orig_count:], yerr=sur_std[orig_count:], fmt='none', 
             ecolor='gray', alpha=0.5, capsize=2)
plt.colorbar(sc, label='Distance to current phi')
plt.plot([true_loss.min().item(), true_loss.max().item()], [true_loss.min().item(), true_loss.max().item()], 'k--', label='y = x')
plt.xlabel('True Loss'); plt.ylabel('Surrogate Loss'); plt.title('Surrogate vs True Loss')
plt.legend(); plt.grid(True); plt.savefig('figs/surrogate_vs_true_loss_gp.png')
print("Surrogate vs True Loss plot saved.")


def surrogate_objective(phi):
    y_pred_mean = surrogate_model.posterior(phi.to(dev).view(1,-1)).mean.sum()
    return y_pred_mean

surrogate_grad = torch.func.grad(surrogate_objective)(initial_phi).detach()*diff_bounds
surrogate_hessian = torch.func.hessian(surrogate_objective)(initial_phi).detach()* (diff_bounds.unsqueeze(1) * diff_bounds.unsqueeze(0))
surrogate_step = -torch.linalg.solve(surrogate_hessian, surrogate_grad).detach()
cos_step_surrogate = (torch.dot(surrogate_grad, surrogate_step) / (torch.linalg.norm(surrogate_grad) * torch.linalg.norm(surrogate_step) + 1e-10)).item()
if args.problem != 'MuonShield':
    real_gradient = problem.gradient(initial_phi.unsqueeze(0)).squeeze()
    real_hessian = problem.hessian(initial_phi).squeeze()
    print("Real Gradient:", real_gradient)
    print("Surrogate Gradient:", surrogate_grad)
    if dim<6:
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
    print(f"Cosine between surrogate gradient and surrogate Newton step: {cos_step_surrogate}")
    print('=' * 50)
    print("cosine between real and surrogate gradients:", (torch.dot(real_gradient, surrogate_grad) / (torch.linalg.norm(real_gradient) * torch.linalg.norm(surrogate_grad) + 1e-10)).item())
    print(f"Cosine between real step and surrogate step: {cos_step}")
