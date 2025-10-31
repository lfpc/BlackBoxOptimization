import torch
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.abspath(os.path.join('..', 'src')))
from optimizer import LGSO, LCSO, normalize_vector
from problems import ShipMuonShield, make_index, ShipMuonShieldCuda
from models import BinaryClassifierModel, GP_RBF
import torch
import pickle
import numpy as np

torch.autograd.set_detect_anomaly(True)

dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {dev}")
dim = (
    make_index(0, [0,1,2,4,6,8,14]) +
    make_index(1, [0,1,2,3,4,5,6,7,8,9,14]) +
    make_index(2, [0,1,2,3,4,5,6,7,8,9,14]) +
    make_index(3, [0,1,2,3,4,5,6,7,8,9,14]) +
    make_index(4, [0,1,2,3,4,5,6,7,8,9,12,13,14]) +
    make_index(5, [0,1,2,3,4,5,6,7,8,9,12,13,14]) +
    make_index(6, [0,1,2,3,4,5,6,7,8,9,12,13,14]))

initial_phi = torch.tensor(ShipMuonShield.params["tokanut_v5"])

# Minimal ShipMuonShield setup for fast LGSO test
problem = ShipMuonShieldCuda(
            uniform_fields=False,  # No real field simulation
            n_samples=0,       
            apply_det_loss=False,  # Skip detector loss for speed
            loss_fn='hits',
            reduction='none',
            cost_loss_fn=None,     # No cost penalty
            dimensions_phi=dim,
            initial_phi=initial_phi,
            parallel = False,
            fSC_mag = False,
            use_B_goal = True,
            fields_file = "/home/hep/lprate/projects/MuonsAndMatter/data/outputs/fields_rob.h5",
            sensitive_plane = [{'dz': 0.01, 'dx': 4, 'dy': 6,'position': 82}]
        )

initial_phi = problem.initial_phi
bounds = problem.GetBounds(device=torch.device('cpu'))
print(bounds[0])

# Set up a simple generative model (GANModel)
BATCH_SIZE = 2**20
generative_model = GP_RBF(bounds=bounds,
                          device=dev)

# Set up LGSO optimizer
optimizer = LCSO(
    true_model=problem,
    surrogate_model=generative_model,
    bounds=bounds,
    samples_phi=100,    
    epsilon=0.1,       # Local search radius
    initial_phi=initial_phi,
    device='cpu',
    outputs_dir='/home/hep/lprate/projects/BlackBoxOptimization/outputs/test_robustness',
    resume=False)

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
        cond = torch.cat([phi_norm.repeat(xb.size(0), 1), xb[:, :7]], dim=-1)
        y_pred = optimizer.model.predict_proba(cond).cpu()
        return optimizer.n_hits(phi_norm, y_pred, xb)

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
phis = optimizer.sample_phi()
for i,phi in enumerate(phis):
    optimizer.simulate_and_update(phi, update_history=False)
print("Completed simulations")
with open('local_results_test.pkl', 'wb') as f:
    pickle.dump(optimizer.local_results, f)
print("Saved local results")
optimizer.fit_surrogate_model()
plt.figure()
plt.plot(optimizer.model.last_train_loss)
plt.xlabel('Epoch')
plt.ylabel('Training Loss')
plt.title('Surrogate Model Training Loss')
plt.grid(True)
plt.savefig('surrogate_model_train_loss.png')
plt.close()

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