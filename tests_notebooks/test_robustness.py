import torch
import matplotlib.pyplot as plt
import sys
import os
import pickle
sys.path.append(os.path.abspath(os.path.join('..', 'src')))
from optimizer import LGSO, LCSO
from problems import ShipMuonShield
from models import VAEModel, NormalizingFlowModel, BinaryClassifierModel
import torch
import subprocess
import numpy as np

torch.autograd.set_detect_anomaly(True)

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

dev = get_freest_gpu()
dim = ShipMuonShield.parametrization['HA'][1:3] + [ShipMuonShield.parametrization['HA'][-3]]+[ShipMuonShield.parametrization['HA'][-1]]


# Minimal ShipMuonShield setup for fast LGSO test
problem = ShipMuonShield(
    simulate_fields=False,  # No real field simulation
    n_samples=500_000,           # Very low number of samples for speed
    apply_det_loss=False,  # Skip detector loss for speed
    cost_loss_fn=None,     # No cost penalty
    dimensions_phi=dim,
    default_phi=torch.tensor(ShipMuonShield.tokanut_v5), 
    cores = 60,
    parallel = False,
    fSC_mag = False,
    sensitive_plane = [{'dz': 0.01, 'dx': 4, 'dy': 6,'position': 82}]
)

bounds = problem.GetBounds(device=torch.device('cpu'))
initial_phi = problem.initial_phi
dim = len(dim)

# Set up a simple generative model (GANModel)
generative_model = BinaryClassifierModel(phi_dim=dim,
                            x_dim = 7,
                            n_epochs = 15,
                            batch_size = 2048,
                            lr = 1e-3,
                            device = dev)

# Set up LGSO optimizer
optimizer = LCSO(
    true_model=problem,
    surrogate_model=generative_model,
    bounds=bounds,
    samples_phi=30,        
    epsilon=0.5,       # Local search radius
    initial_phi=initial_phi,
    device='cpu',
    outputs_dir='/home/hep/lprate/projects/BlackBoxOptimization/outputs/test_robustness',
    resume=False)

def get_local_info(phi):
    def compute_surrogate_loss(phi):
        """
        This function contains every step that connects 'phi' to the final loss.
        It is a "pure" function for compatibility with torch.func.
        """
        x_samp = torch.as_tensor(problem.sample_x(), dtype=torch.get_default_dtype())
        condition = torch.cat([phi.repeat(x_samp.size(0), 1), x_samp[:,:7]], dim=-1)
        y_pred = optimizer.model.predict_proba(condition)
        loss = optimizer.n_hits(phi, y_pred, x_samp)
        return loss*1e5
    grad_func = torch.func.grad(compute_surrogate_loss)
    hess_func = torch.func.hessian(compute_surrogate_loss)
    return compute_surrogate_loss(phi),grad_func(phi), hess_func(phi)

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
    linear_term = torch.dot(grad, delta_phi)

    # Quadratic term: 0.5 * δφᵀ * H * δφ
    # The @ operator performs matrix-vector multiplication
    quadratic_term = 0.5 * torch.dot(delta_phi, hess @ delta_phi)

    # The result is a 0-dimensional tensor; .item() extracts the Python scalar
    return (linear_term + quadratic_term).item()


            

phis = optimizer.sample_phi()
for phi in phis:
    optimizer.simulate_and_update(phi, update_history=False)
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
loss, grad, hess = get_local_info(phi)


print(f"Initial True Loss: {optimizer.history[1][0]*1e5}")
print(f"Initial Surrogate Loss: {loss.item()}")
print(f"Initial Gradient: {grad}")
print(f"Initial Hessian: {hess}")
eigenvalues, eigenvectors = torch.linalg.eigh(hess)
print(f"Hessian eigenvalues: {eigenvalues}")
for idx, (eigval, eigvec) in enumerate(zip(eigenvalues, eigenvectors.T)):
    print(f"Eigenvalue {idx}: {eigval.item()}")
    print(f"Eigenvector {idx}: {eigvec}")
alphas = np.linspace(-2, 2, 100)

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





    
    



