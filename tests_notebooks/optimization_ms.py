import json
import torch
import matplotlib.pyplot as plt
import sys
import os
from tqdm import trange
import numpy as np

sys.path.append(os.path.abspath(os.path.join('..', 'src')))
sys.path.append(os.path.abspath(os.path.join('..')))
from problems import ShipMuonShieldCuda
from utils.nets import StochasticTaylor, SharedBasisStochasticTaylor
from utils import latin_hypercube_sample, normalize_vector, denormalize_vector
import argparse



def train_hits_classifier(model, phi, x, y, epochs=2000, lr=3e-3, batch_size=8192, device=None, l2_reg=None):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[int(epochs * 0.7), int(epochs * 0.9)], gamma=0.2
    )
    loss_fn = torch.nn.BCEWithLogitsLoss()

    n_phi = phi.shape[0]
    n_samples = x.shape[1]
    per_phi_batch = max(1, batch_size // max(n_phi, 1))
    indices = torch.arange(n_samples)
    phi_dev = phi.to(device)

    losses = []
    for _ in trange(epochs):
        model.train()
        total = 0.0
        total_elements = 0
        shuffled = indices[torch.randperm(n_samples)]
        for start in range(0, n_samples, per_phi_batch):
            end = min(start + per_phi_batch, n_samples)
            batch_idx = shuffled[start:end]
            x_batch = x[:, batch_idx].to(device, non_blocking=True)
            y_batch = y[:, batch_idx].float().to(device, non_blocking=True)

            optimizer.zero_grad()
            logits = model(phi_dev, x_batch)

            loss = loss_fn(logits, y_batch)
            if l2_reg is not None:
                l2_penalty = sum((p ** 2).sum() for p in model.parameters())
                loss = loss + l2_reg * l2_penalty
            loss.backward()
            optimizer.step()

            elems = logits.numel()
            total += loss.item() * elems
            total_elements += elems
        scheduler.step()
        losses.append(total / max(total_elements, 1))
    return losses



parser = argparse.ArgumentParser(description="Muon Shield Optimization")
parser.add_argument("--dim", type=int, default=3, help="Dimension of phi")
parser.add_argument("--x_dim", type=int, default=7, help="Dimension of x")
parser.add_argument("--total_n_samples", type=int, default=0, help="Total number of samples")
parser.add_argument("--n_samples_train", type=int, default=20_000_000, help="Number of training samples")
parser.add_argument("--epsilon", type=float, default=0.2, help="Epsilon for LHS sampling")
parser.add_argument("--N", type=int, default=30, help="Number of LHS samples")
parser.add_argument("--n_test", type=int, default=20, help="Number of test samples")
parser.add_argument("--FIGS_DIR", type=str, default="outputs/figs", help="Directory for figures")
parser.add_argument("--load_model", action='store_true', help="Whether to load a pre-trained model")
parser.add_argument("--normalize", action='store_true', help="Whether to normalize phi")
parser.add_argument("--residual", action='store_true', help="Whether to use a residual architecture in the surrogate model")
parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
args = parser.parse_args()

device = torch.device("cuda:3")
cpu = torch.device("cpu")
torch.manual_seed(args.seed)
np.random.seed(args.seed)

dim = args.dim
x_dim = args.x_dim
total_n_samples = args.total_n_samples
n_samples_train = args.n_samples_train
epsilon = args.epsilon
N = args.N
n_test = args.n_test
FIGS_DIR = args.FIGS_DIR

os.makedirs(FIGS_DIR, exist_ok=True)

def random_sample(muons, n_samples):
    """Return a random sample of n_samples from the muons tensor along the first dimension."""
    total_samples = muons.shape[0]
    if n_samples >= total_samples:
        return muons
    idx = torch.randperm(total_samples)[:n_samples]
    return muons[idx]

config_file = "/home/hep/lprate/projects/BlackBoxOptimization/outputs/config_tests.json"
with open(config_file, 'r') as f:
    CONFIG = json.load(f)
CONFIG.pop("data_treatment", None)
CONFIG.pop('results_dir', None)
CONFIG['dimensions_phi'] = dim
CONFIG['initial_phi'] = ShipMuonShieldCuda.params['warm_baseline']
CONFIG['n_samples'] = total_n_samples
CONFIG['reduction'] = 'none'
CONFIG['cost_as_constraint'] = False
problem = ShipMuonShieldCuda(**CONFIG)
total_n_samples = problem.muons.shape[0]

bounds = problem.GetBounds(device=cpu).to(dtype=torch.get_default_dtype())

def phi_phys_to_model(phi_phys: torch.Tensor) -> torch.Tensor:
    return normalize_vector(phi_phys, bounds.to(phi_phys.device)) if args.normalize else phi_phys

def phi_model_to_phys(phi_model: torch.Tensor) -> torch.Tensor:
    return denormalize_vector(phi_model, bounds.to(phi_model.device)) if args.normalize else phi_model

initial_phi_phys = problem.initial_phi.to(device, dtype=torch.get_default_dtype())
initial_phi = phi_phys_to_model(initial_phi_phys).cpu()
'''model = StochasticTaylor(
        phi_dim=dim,
        x_dim=x_dim,
    phi0=initial_phi,
        hidden_dim=256,
        p=128,
        use_residual=args.residual,
    ).to(device)'''


model = SharedBasisStochasticTaylor(
        phi_dim=dim,
        x_dim=x_dim,
        phi0=initial_phi,
        hidden_dim=128,
        residual_width=64,
        rank=3,
        use_diagonal=True,
        use_residual=True,
        eps_norm=1e-12,
    ).to(device)


        

if not args.load_model:    
    # LHS sampling should happen in model-space; if normalize=True, this means [0, 1] coordinates.
    phis = latin_hypercube_sample(N, initial_phi.detach().cpu(), epsilon=epsilon)
    if args.normalize:
        phis = phis.clamp(0.0, 1.0)
    # Include the initial phi explicitly (useful for diagnostics/plotting).
    phis = torch.cat([initial_phi.detach().cpu().view(1, -1), phis], dim=0)
    muons = []
    y = []
    for phi in phis:
        m = random_sample(problem.muons, n_samples_train).cpu()
        muons.append(m[...,:x_dim])
        phi_phys = phi_model_to_phys(phi.to(device)).detach().cpu()
        print(f"Simulating for phi (model-space): {phi.cpu().numpy()}, phi (physical): {phi_phys.cpu().numpy()}")
        y.append(problem(phi_phys, m))
    muons = torch.stack(muons, dim=0)
    y = torch.stack(y, dim=0)
    losses = train_hits_classifier(model, phis, muons, y, epochs=50, lr=3e-3, batch_size=4*16384, device=device)
    torch.save(model.state_dict(), "hits_classifier.pth")
    plt.plot(losses)
    plt.yscale('log')
    plt.xlabel('Epoch')
    plt.ylabel('BCE Loss')
    plt.title('Hits-classifier training loss')
    plt.savefig(os.path.join(FIGS_DIR, 'Hits-classifier_training_loss.png'))
    plt.show()
else: 
    model.load_state_dict(torch.load("hits_classifier.pth", map_location=device))


phi0 = initial_phi.view(1, -1).to(device)
phi0_phys = initial_phi_phys.view(1, -1).cpu()
x0_full = random_sample(problem.muons, total_n_samples).view(1, total_n_samples, -1).cpu()
weights_0 = x0_full[...,-1]
x0 = x0_full[...,:x_dim]

model.eval()

def predict_hits_batched_ms(model, phi, x, batch_size=16384, device=device):
    """Batched surrogate E[hits] prediction."""
    model.eval()
    phi_dev = phi.to(device)
    n_phi = phi_dev.shape[0]
    n_samples = x.shape[1]
    per_phi_batch = max(1, batch_size // max(n_phi, 1))
    out = torch.zeros(n_phi, device=device)
    with torch.no_grad():
        for start in range(0, n_samples, per_phi_batch):
            end = min(start + per_phi_batch, n_samples)
            xb = x[:, start:end].to(device, non_blocking=True)
            out = out + model.predict_hits(phi_dev, xb)
    return out.cpu()


def plot_predicted_vs_true_hits(
    model,
    problem,
    epsilon: float,
    N: int,
    x_dim: int,
    total_n_samples: int,
    figs_dir: str,
    device: torch.device,
    phis_plot: torch.Tensor | None = None,
    x_plot: torch.Tensor | None = None,
    y_plot: torch.Tensor | None = None,
):
    """Scatter plot: surrogate predicted hits vs true hits.

    If (phis_plot, x_plot, y_plot) are provided, uses them directly (no new simulations).
    Otherwise, runs a small simulation-based evaluation set.
    """
    model.eval()

    if phis_plot is None or x_plot is None or y_plot is None:
        # Keep this lightweight: this script can already be very expensive.
        n_plot_phi = max(2, min(int(N), 10))
        n_plot_muons = int(min(50_000, max(1, total_n_samples)))

        phis_plot = latin_hypercube_sample(n_plot_phi, phi0.view(-1).detach().cpu(), epsilon=epsilon)
        if args.normalize:
            phis_plot = phis_plot.clamp(0.0, 1.0)

        # Include the initial point explicitly (star marker in the plot).
        phis_plot = torch.cat([phi0.view(-1).detach().cpu().view(1, -1), phis_plot], dim=0)

        muons_list = []
        y_list = []
        for phi_model_cpu in phis_plot:
            mu = random_sample(problem.muons, n_plot_muons).cpu()
            mu_x = mu[..., :x_dim]
            phi_phys_cpu = phi_model_to_phys(phi_model_cpu.to(device)).detach().cpu()
            y = problem(phi_phys_cpu, mu).detach().cpu()
            muons_list.append(mu_x)
            y_list.append(y)

        x_plot = torch.stack(muons_list, dim=0)  # (n_phi, S, x_dim)
        y_plot = torch.stack(y_list, dim=0)      # (n_phi, S)
    else:
        # Ensure consistent shapes.
        if phis_plot.dim() == 1:
            phis_plot = phis_plot.view(1, -1)

    true_hits = y_plot.sum(dim=1).flatten().cpu().numpy()
    pred_hits = predict_hits_batched_ms(model, phis_plot, x_plot, batch_size=16384, device=device).flatten().cpu().numpy()

    plt.figure(figsize=(6, 6))
    # First point is initial phi
    if len(true_hits) >= 1:
        plt.scatter(true_hits[1:], pred_hits[1:], s=40, marker='o', label='LHS samples')
        plt.scatter(true_hits[0], pred_hits[0], color='red', marker='*', s=150, label='Initial phi', zorder=5)
    else:
        plt.scatter(true_hits, pred_hits, s=40, marker='o', label='Samples')

    mn = float(min(true_hits.min(), pred_hits.min()))
    mx = float(max(true_hits.max(), pred_hits.max()))
    plt.plot([mn, mx], [mn, mx], 'k--', label='y = x')
    plt.xlabel('True hits')
    plt.ylabel('Predicted hits (surrogate)')
    plt.title('Surrogate vs True Hits')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(figs_dir, 'surrogate_vs_true_hits.png'))
    plt.show()
    plt.close()

def apply_step(phi,step_size,step):
    """Perform a step"""
    new_phi = phi + step_size * step
    if args.normalize:
        new_phi = new_phi.clamp(0.0, 1.0)
    return new_phi
def get_grad(model, phi, x, batch_size=16384):
    """Compute the surrogate gradient."""
    n_samples = x.shape[1]
    grad = torch.zeros_like(phi)
    phi = phi.to(device)
    for start in range(0, n_samples, batch_size):
        end = min(start + batch_size, n_samples)
        xb = x[:, start:end].to(device, non_blocking=True)
        grad += model.grad_phi(phi, xb)
    return grad / n_samples
def gradient_step(model, phi, x, step_size, batch_size=16384, grad = None):
    """Perform a gradient step using the surrogate gradient."""
    if grad is None:
        grad = get_grad(model, phi, x, batch_size)
    # We minimize expected hits => step in the negative gradient direction.
    return apply_step(phi, step_size, -grad)
def get_hess(model, phi, x, batch_size=16384):
    """Compute the surrogate Hessian."""
    n_samples = x.shape[1]
    hess = torch.zeros(phi.shape[0], phi.shape[1], phi.shape[1], device=device)
    phi = phi.to(device)
    for start in range(0, n_samples, batch_size):
        end = min(start + batch_size, n_samples)
        xb = x[:, start:end].to(device, non_blocking=True)
        hess += model.hess_phi(phi, xb)
    return hess / n_samples
def newton_step(model, phi, x, step_size, batch_size=16384, grad = None, hess = None, damping=1e-4):
    """Perform a Newton step using the surrogate gradient and Hessian."""
    if grad is None:
        grad = get_grad(model, phi, x, batch_size)
    if hess is None:
        H = get_hess(model, phi, x, batch_size)
    else:
        H = hess
    if H.dim() == 2:
        H = H.unsqueeze(0)
    # Add a small damping term to ensure invertibility
    damping = damping * torch.eye(H.shape[-1], device=H.device).unsqueeze(0)
    H_damped = H + damping
    H_inv = torch.linalg.inv(H_damped)
    step = -H_inv @ grad.unsqueeze(-1)
    return apply_step(phi, step_size, step.squeeze(-1))


hits0 = problem(phi0_phys, x0_full).cpu().sum()
hits_sur = predict_hits_batched_ms(model, phi0, x0).item()

# Plot predicted hits vs true hits (similar to test_hits_classifier.py)
if not args.load_model:
    # Reuse the training dataset to avoid extra expensive simulator calls.
    plot_predicted_vs_true_hits(
        model=model,
        problem=problem,
        epsilon=epsilon,
        N=N,
        x_dim=x_dim,
        total_n_samples=total_n_samples,
        figs_dir=FIGS_DIR,
        device=device,
        phis_plot=phis.detach().cpu(),
        x_plot=muons.detach().cpu(),
        y_plot=y.detach().cpu(),
    )

steps_grad = []
steps_newton = []
hits_grad = []
hits_newton = []
etas = [0.001,0.01, 0.1, 1.0, 10.0]
for eta in etas:
    print("#################################")
    print(f"Step size: {eta}")
    print("#################################")
    phi1_grad = gradient_step(model, phi0, x0, step_size=eta)
    phi1_grad_phys = phi_model_to_phys(phi1_grad).detach().cpu()
    steps_grad.append((phi1_grad_phys - phi0_phys).cpu().numpy())
    hits_grad.append(problem(phi1_grad_phys, x0_full).cpu().sum().item())
    phi1_newton = newton_step(model, phi0, x0, step_size=eta)
    phi1_newton_phys = phi_model_to_phys(phi1_newton).detach().cpu()
    hits_newton.append(problem(phi1_newton_phys, x0_full).cpu().sum().item())
    steps_newton.append((phi1_newton_phys - phi0_phys).cpu().numpy())

print("#################################")
print("Initial phi (physical):", phi0_phys.cpu().numpy())
if args.normalize:
    print("Initial phi (model-space):", phi0.cpu().numpy())
print(f"Initial hits: {hits0.item():.4f}")
print(f"Surrogate hits: {hits_sur:.4f}")
grad0 = get_grad(model, phi0, x0).cpu().numpy()
print(f"Gradient vector: {grad0}, norm: {np.linalg.norm(grad0):.4f}")
for eta, hg, hn, sg, sn in zip(etas, hits_grad, hits_newton, steps_grad, steps_newton):
    print('---------------------------------')
    print(f"Step size: {eta}")
    print(f"Gradient step: {sg}, norm: {np.linalg.norm(sg):.4f}")
    print(f"Newton step: {sn}, norm: {np.linalg.norm(sn):.4f}")
    print(f"Hits after gradient step: {hg:.4f}, reduction: {hits0 - hg:.4f}")
    print(f"Hits after Newton step: {hn:.4f}, reduction: {hits0 - hn:.4f}")
