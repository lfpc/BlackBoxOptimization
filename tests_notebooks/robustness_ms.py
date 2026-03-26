import json
import torch
import matplotlib.pyplot as plt
import sys
import os
from tqdm import trange
import numpy as np
import argparse

device = torch.device("cuda:3")
sys.path.append(os.path.abspath(os.path.join('..', 'src')))
sys.path.append(os.path.abspath(os.path.join('..')))
from problems import ShipMuonShieldCuda
from utils.nets import StochasticTaylor
from utils import latin_hypercube_sample, normalize_vector, denormalize_vector


def train_hits_classifier(model, phi, x, y, epochs=2000, lr=3e-3, batch_size=8192, device=None, l2_reg=None):
    device = torch.device(device) if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    x = x[:,:,:7]
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



parser = argparse.ArgumentParser(description="Muon Shield robustness analysis (StochasticTaylor)")
parser.add_argument('--normalize', action='store_true', help='Normalize phi to [0,1] using bounds (model-space); denormalize only for simulator calls')
args = parser.parse_args()

dim = 3
x_dim = 7
total_n_samples = 500_000_000
n_samples_train = 20_000_000
epsilon = 0.2
N = 30
n_test = 20
FIGS_DIR = "outputs/figs"

os.makedirs(FIGS_DIR, exist_ok=True)

def random_sample(muons, n_samples):
    idx = torch.randint(0, muons.shape[0], (n_samples,))
    return muons[idx]

config_file = "/home/hep/lprate/projects/BlackBoxOptimization/outputs/config_tests.json"
with open(config_file, 'r') as f:
    CONFIG = json.load(f)
CONFIG.pop("data_treatment", None)
CONFIG.pop('results_dir', None)
CONFIG['dimensions_phi'] = dim
CONFIG['initial_phi'] = ShipMuonShieldCuda.params['tokanut_v6']
CONFIG['n_samples'] = total_n_samples
CONFIG['reduction'] = 'none'
CONFIG['cost_as_constraint'] = False
problem = ShipMuonShieldCuda(**CONFIG)
cpu = torch.device('cpu')
initial_phi_phys = problem.initial_phi.to(cpu, dtype=torch.get_default_dtype())
initial_phi_phys_dev = initial_phi_phys.to(device)

bounds = problem.GetBounds(device=cpu).to(dtype=torch.get_default_dtype())

def phi_phys_to_model(phi_phys: torch.Tensor) -> torch.Tensor:
    return normalize_vector(phi_phys, bounds.to(phi_phys.device)) if args.normalize else phi_phys

def phi_model_to_phys(phi_model: torch.Tensor) -> torch.Tensor:
    return denormalize_vector(phi_model, bounds.to(phi_model.device)) if args.normalize else phi_model

initial_phi = phi_phys_to_model(initial_phi_phys_dev)

model = StochasticTaylor(
        phi_dim=dim,
        x_dim=x_dim,
        phi0=initial_phi,
        hidden_dim=128,
        p=64,
        use_residual=False,
    ).to(device)

if True:    
    phis = latin_hypercube_sample(N, initial_phi.detach().cpu(), epsilon=epsilon)
    if args.normalize:
        phis = phis.clamp(0.0, 1.0)
    # Include the initial phi explicitly.
    phis = torch.cat([initial_phi.detach().cpu().view(1, -1), phis], dim=0)
    m = []
    y = []
    for phi in phis:
        m.append(random_sample(problem.muons, n_samples_train).cpu())
        phi_phys = phi_model_to_phys(phi.to(device)).detach().cpu()
        y.append(problem(phi_phys, m[-1]))
    m = torch.stack(m, dim=0)
    y = torch.stack(y, dim=0)
    losses = train_hits_classifier(model, phis, m, y, epochs=50, lr=3e-3, batch_size=4*16384, device=device)
    torch.save(model.state_dict(), "hits_classifier.pth")
    plt.plot(losses)
    plt.yscale('log')
    plt.xlabel('Epoch')
    plt.ylabel('BCE Loss')
    plt.title('Hits-classifier training loss')
    plt.savefig(os.path.join(FIGS_DIR, 'Hits-classifier_training_loss_nd.png'))
    plt.show()
else: 
    model.load_state_dict(torch.load("hits_classifier.pth", map_location=device))

os.makedirs(FIGS_DIR, exist_ok=True)


# ============================================================
# Gradient, Hessian, GN decomposition & eigenanalysis
# ============================================================
phi0 = initial_phi.view(1, -1)
x0 = random_sample(problem.muons, n_samples_train).unsqueeze(0).to(device)[:, :, :x_dim]

model.eval()
sur_grad = model.grad_phi(phi0, x0).cpu()        # (1, D)
H_sur = model.hess_phi(phi0, x0).cpu()            # (1, D, D) or (D, D)
if H_sur.dim() == 2:
    H_sur = H_sur.unsqueeze(0)


print('\n====== Surrogate analysis at phi0 ======')
print('phi0:', phi0.view(-1).cpu().tolist())
print('surrogate grad E[hits]:', sur_grad.view(-1).tolist())
print('surrogate Hessian:\n', H_sur[0].numpy())

eigenvalues, eigenvectors = torch.linalg.eigh(H_sur[0])
print('\nHessian eigenvalues:', eigenvalues.tolist())
for idx, (eigval, eigvec) in enumerate(zip(eigenvalues, eigenvectors.T)):
    print(f'  Eigenvalue {idx}: {eigval.item():.6e}')
    print(f'  Eigenvector {idx}: {eigvec.tolist()}')


# ============================================================
# Error vs distance plot (adapted from stochastic_quadratic_nd)
# ============================================================

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


def true_expected_hits(phi, muons, n_eval):
    """Evaluate true E[hits] for each phi using problem().

    Note: `phi` is in model-space if args.normalize is True.
    """
    if phi.dim() == 1:
        phi = phi.unsqueeze(0)
    results = []
    for i in range(phi.shape[0]):
        m = random_sample(muons, n_eval)
        phi_phys = phi_model_to_phys(phi[i].to(device)).detach().cpu()
        hits = problem(phi_phys, m)  # returns (n_eval,)
        results.append(hits.float().sum().item())
    return torch.tensor(results)


n_dist_bins = 15
n_per_bin = 40
max_dist = 2.0 * epsilon
n_eval_true = n_samples_train  # samples for true E[hits]

dist_edges = torch.linspace(0, max_dist, n_dist_bins + 1)
dist_centres = 0.5 * (dist_edges[:-1] + dist_edges[1:])

# Surrogate value at phi0
f0_sur = predict_hits_batched_ms(model, phi0, x0).item()
g_quad = sur_grad.view(-1)
H_quad = H_sur[0]


# Collect statistics per distance bin
keys = ['sur', 'nw', 'gn1']
err_stats = {k: {'mean': [], 'std': [], 'min': [], 'max': []} for k in keys}
hits_stats = {k: {'mean': [], 'std': []} for k in ['true'] + keys}

print('\nComputing error vs distance ...')
for b in trange(n_dist_bins):
    r_lo, r_hi = dist_edges[b].item(), dist_edges[b + 1].item()

    # Spherical-shell sampling (works in any dimension)
    dirs = torch.randn(n_per_bin, dim)
    dirs = dirs / dirs.norm(dim=1, keepdim=True).clamp_min(1e-12)
    u = torch.rand(n_per_bin)
    r = (u * (r_hi ** dim - r_lo ** dim) + r_lo ** dim) ** (1.0 / dim)
    phis_shell = initial_phi.detach().cpu().view(1, -1) + dirs * r.unsqueeze(1)
    if args.normalize:
        phis_shell = phis_shell.clamp(0.0, 1.0)

    # True E[hits]
    true_Ehits = true_expected_hits(phis_shell, problem.muons, n_eval_true)

    # Surrogate prediction (use same x0 — shared muon set)
    pred_sur = predict_hits_batched_ms(model, phis_shell, x0)

    # Quadratic approximation (full Hessian)
    dphi = phis_shell - initial_phi.detach().cpu().view(1, -1)
    quad_nw = f0_sur + dphi @ g_quad + 0.5 * (dphi @ H_quad * dphi).sum(dim=1)


    denom = true_Ehits.abs().clamp_min(1e-8)
    preds = {'sur': pred_sur, 'nw': quad_nw}
    for k, pred in preds.items():
        e = (pred - true_Ehits).abs() / denom
        err_stats[k]['mean'].append(e.mean().item())
        err_stats[k]['std'].append(e.std().item())
        err_stats[k]['min'].append(e.min().item())
        err_stats[k]['max'].append(e.max().item())

    hits_stats['true']['mean'].append(true_Ehits.mean().item())
    hits_stats['true']['std'].append(true_Ehits.std().item())
    for k, pred in preds.items():
        hits_stats[k]['mean'].append(pred.mean().item())
        hits_stats[k]['std'].append(pred.std().item())

# Convert to numpy
dist_np = dist_centres.numpy()
for k in keys:
    for s in ['mean', 'std', 'min', 'max']:
        err_stats[k][s] = np.array(err_stats[k][s])
for k in ['true'] + keys:
    for s in ['mean', 'std']:
        hits_stats[k][s] = np.array(hits_stats[k][s])

# ---- Plot ----
colors = {'sur': 'tab:blue', 'nw': 'tab:orange'}
labels = {'sur': 'Surrogate', 'nw': 'Quadratic (full H)'}
markers = {'sur': 'o', 'nw': 's', 'gn1': 'd'}

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 9), sharex=True)

# Top: relative error
for k in keys:
    m_ = err_stats[k]['mean']
    s_ = err_stats[k]['std']
    ax1.plot(dist_np, m_, f'{markers[k]}-', color=colors[k], label=labels[k])
    ax1.fill_between(dist_np, m_ - s_, m_ + s_, alpha=0.1, color=colors[k])
    ax1.fill_between(dist_np, err_stats[k]['min'], err_stats[k]['max'],
                     alpha=0.05, color=colors[k])
ax1.axvline(epsilon, ls='--', color='gray', lw=1)
ax1.set_ylabel('|pred - true| / |true|')
ax1.set_title('Relative prediction error vs distance from phi_init')
ax1.legend(fontsize=8)
ax1.set_xlim(0, max_dist)

# Bottom: expected hits
ax2.plot(dist_np, hits_stats['true']['mean'], 'k-', lw=2, label='True E[hits]')
ax2.fill_between(dist_np,
                 hits_stats['true']['mean'] - hits_stats['true']['std'],
                 hits_stats['true']['mean'] + hits_stats['true']['std'],
                 alpha=0.12, color='k')
for k in keys:
    m_ = hits_stats[k]['mean']
    s_ = hits_stats[k]['std']
    ax2.plot(dist_np, m_, f'{markers[k]}-', color=colors[k], label=labels[k])
    ax2.fill_between(dist_np, m_ - s_, m_ + s_, alpha=0.10, color=colors[k])
ax2.axvline(epsilon, ls='--', color='gray', lw=1)
ax2.set_xlabel(r'$||\phi - \phi_{\mathrm{init}}||$')
ax2.set_ylabel('Expected hits')
ax2.set_title('Expected number of hits vs distance from phi_init')
ax2.legend(fontsize=8)

plt.tight_layout()
plt.savefig(os.path.join(FIGS_DIR, 'error_vs_distance_ms.png'))
plt.show()

# ============================================================
# Eigenvector-direction analysis (parabola along each eigenvector)
# ============================================================
print('\n====== Eigenvector parabola analysis ======')
alphas = np.linspace(-2 * epsilon, 2 * epsilon, 80)
n_eig = len(eigenvalues)
ncols = 2
nrows = (n_eig + ncols - 1) // ncols

fig_eig, axes_eig = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12, 3.5 * nrows))
axes_eig = axes_eig.flatten()

for idx, (eigval, eigvec) in enumerate(zip(eigenvalues, eigenvectors.T)):
    # Quadratic prediction along eigenvector
    quad_vals = []
    true_vals = []
    sur_vals = []
    for a in alphas:
        dphi = a * eigvec
        # quadratic: f0 + g^T δφ + 0.5 δφ^T H δφ
        q = f0_sur + float(g_quad @ dphi) + 0.5 * float(dphi @ H_quad @ dphi)
        quad_vals.append(q)

        # surrogate
        phi_t = (initial_phi.cpu().view(-1) + dphi).view(1, -1)
        s = predict_hits_batched_ms(model, phi_t, x0).item()
        sur_vals.append(s)

    # true (subsample for speed)
    n_alpha_true = min(20, len(alphas))
    alpha_true_idx = np.linspace(0, len(alphas) - 1, n_alpha_true, dtype=int)
    true_alphas = alphas[alpha_true_idx]
    for a in true_alphas:
        dphi = a * eigvec
        phi_t = (initial_phi.detach().cpu().view(-1) + dphi).view(1, -1)
        true_vals.append(true_expected_hits(phi_t, problem.muons, n_eval_true).item())

    ax = axes_eig[idx]
    ax.plot(alphas, quad_vals, '-', color='tab:orange', label='Quadratic')
    ax.plot(alphas, sur_vals, '-', color='tab:blue', label='Surrogate')
    ax.plot(true_alphas, true_vals, 'ko', ms=4, label='True')
    ax.axvline(0, ls=':', color='gray', lw=0.8)
    ax.set_xlabel(r'$\alpha$')
    ax.set_ylabel('E[hits]')
    ax.set_title(f'Eigvec {idx}  (λ={eigval.item():.3e})')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

for ax in axes_eig[n_eig:]:
    ax.axis('off')
plt.tight_layout()
plt.savefig(os.path.join(FIGS_DIR, 'eigenvector_parabolas_ms.png'))
plt.show()


print('\nAnalysis complete. Figures (error_vs_distance_ms.png, eigenvector_parabolas_ms.png) saved to:', FIGS_DIR)










