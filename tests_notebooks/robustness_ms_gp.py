#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Gaussian-process robustness analysis for ShipMuonShieldCuda.

What this script does
---------------------
1. Samples parameter vectors phi in a local ball around the initial configuration.
2. Estimates the true expected hit count for each phi by repeated Monte Carlo simulation.
3. Fits a noisy Gaussian Process surrogate with BoTorch / GPyTorch.
4. Computes local gradient and Hessian of the GP posterior mean at phi0.
5. Identifies:
   - most sensitive individual parameters,
   - worst directions (directions that increase hit count the most),
   - predicted damaging directions from Hessian eigenvectors and gradient.
6. Validates those directions with the real simulator by plotting true damage curves.

Outputs
-------
Saved under outputs/gp_robustness/ :
- gp_training_fit.png
- gp_param_sensitivity.png
- gp_direction_damage.png
- gp_oat_damage.png
- gp_robustness_summary.json

Notes
-----
- This script models expected TOTAL hits directly as a function of phi.
- It uses repeated simulator calls to estimate both mean and observation noise.
- It avoids distorted clamping in local sampling; points outside bounds are rejected.
- The "worst direction" test is real: for a chosen direction, it simulates along the ray
  and shows how much the hit count increases.

Requirements
------------
- torch
- numpy
- matplotlib
- gpytorch
- botorch

Example
-------
python gp_robustness_analysis.py --normalize --device cuda:0
"""

import os
import sys
import json
import math
import time
import argparse
from typing import Dict, List, Tuple

import numpy as np
import torch
import matplotlib.pyplot as plt

import gpytorch
from botorch.models import SingleTaskGP
from botorch.models.transforms import Standardize
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.fit import fit_gpytorch_mll

# -----------------------------------------------------------------------------
# Project imports
# -----------------------------------------------------------------------------
sys.path.append(os.path.abspath(os.path.join("..", "src")))
sys.path.append(os.path.abspath(os.path.join("..")))

from problems import ShipMuonShieldCuda
from utils import normalize_vector, denormalize_vector


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------
def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def safe_mkdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def symmetrize(H: torch.Tensor) -> torch.Tensor:
    return 0.5 * (H + H.T)


def unit(v: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    n = torch.norm(v)
    if n.item() < eps:
        return torch.zeros_like(v)
    return v / n


def random_sample(muons: torch.Tensor, n_samples: int) -> torch.Tensor:
    idx = torch.randint(0, muons.shape[0], (n_samples,))
    return muons[idx]


def finite_diff_grad(
    f,
    x: torch.Tensor,
    step: float,
) -> torch.Tensor:
    """
    Central finite-difference gradient of scalar function f at x.
    f must take shape (D,) and return Python float or scalar tensor.
    """
    x = x.detach().clone()
    d = x.numel()
    g = torch.zeros(d, dtype=x.dtype)
    for i in range(d):
        xp = x.clone()
        xm = x.clone()
        xp[i] += step
        xm[i] -= step
        fp = float(f(xp))
        fm = float(f(xm))
        g[i] = (fp - fm) / (2.0 * step)
    return g


def finite_diff_hessian(
    f,
    x: torch.Tensor,
    step: float,
) -> torch.Tensor:
    """
    Central finite-difference Hessian of scalar function f at x.
    """
    x = x.detach().clone()
    d = x.numel()
    H = torch.zeros(d, d, dtype=x.dtype)

    f0 = float(f(x))
    for i in range(d):
        ei = torch.zeros_like(x)
        ei[i] = step

        f_pp = float(f(x + ei))
        f_mm = float(f(x - ei))
        H[i, i] = (f_pp - 2.0 * f0 + f_mm) / (step ** 2)

        for j in range(i + 1, d):
            ej = torch.zeros_like(x)
            ej[j] = step
            f_pp = float(f(x + ei + ej))
            f_pm = float(f(x + ei - ej))
            f_mp = float(f(x - ei + ej))
            f_mm = float(f(x - ei - ej))
            Hij = (f_pp - f_pm - f_mp + f_mm) / (4.0 * step ** 2)
            H[i, j] = Hij
            H[j, i] = Hij
    return H


# -----------------------------------------------------------------------------
# Problem wrapper
# -----------------------------------------------------------------------------
class MuonShieldProblemWrapper:
    def __init__(self, config_file: str, dim: int, normalize: bool, device: torch.device):
        with open(config_file, "r") as f:
            config = json.load(f)

        config.pop("data_treatment", None)
        config.pop("results_dir", None)
        config["dimensions_phi"] = dim
        config["initial_phi"] = ShipMuonShieldCuda.params["tokanut_v6"]
        config["reduction"] = "none"
        config["cost_as_constraint"] = False

        self.problem = ShipMuonShieldCuda(**config)
        self.dim = dim
        self.normalize = normalize
        self.device = device
        self.cpu = torch.device("cpu")

        self.bounds_phys = self.problem.GetBounds(device=self.cpu).to(dtype=torch.get_default_dtype())
        self.initial_phi_phys = self.problem.initial_phi.to(self.cpu, dtype=torch.get_default_dtype())
        self.initial_phi_model = self.phys_to_model(self.initial_phi_phys)

    def phys_to_model(self, phi_phys: torch.Tensor) -> torch.Tensor:
        if self.normalize:
            return normalize_vector(phi_phys, self.bounds_phys.to(phi_phys.device))
        return phi_phys

    def model_to_phys(self, phi_model: torch.Tensor) -> torch.Tensor:
        if self.normalize:
            return denormalize_vector(phi_model, self.bounds_phys.to(phi_model.device))
        return phi_model

    def in_bounds_model(self, phi_model: torch.Tensor) -> bool:
        if self.normalize:
            return bool(torch.all(phi_model >= 0.0) and torch.all(phi_model <= 1.0))
        phi_phys = self.model_to_phys(phi_model)
        lb = self.bounds_phys[:, 0]
        ub = self.bounds_phys[:, 1]
        return bool(torch.all(phi_phys >= lb) and torch.all(phi_phys <= ub))

    def sample_local_points(
        self,
        n_points: int,
        radius: float,
        seed: int = 0,
        include_center: bool = True,
        oversample_factor: int = 10,
    ) -> torch.Tensor:
        """
        Uniform sample in a D-ball around initial_phi_model with rejection at bounds.
        """
        g = torch.Generator(device="cpu")
        g.manual_seed(seed)

        center = self.initial_phi_model.detach().cpu().view(1, -1)
        pts = []

        if include_center:
            pts.append(center[0].clone())

        needed = n_points - len(pts)
        if needed <= 0:
            return torch.stack(pts[:n_points], dim=0)

        d = self.dim
        tries = 0
        max_tries = max(500, n_points * oversample_factor)

        while len(pts) < n_points and tries < max_tries:
            batch = min(5 * needed, 256)
            dirs = torch.randn(batch, d, generator=g)
            dirs = dirs / dirs.norm(dim=1, keepdim=True).clamp_min(1e-12)
            u = torch.rand(batch, generator=g)
            r = radius * (u ** (1.0 / d))
            cand = center + dirs * r.unsqueeze(1)

            for k in range(cand.shape[0]):
                if self.in_bounds_model(cand[k]):
                    pts.append(cand[k].clone())
                    if len(pts) >= n_points:
                        break
            needed = n_points - len(pts)
            tries += 1

        if len(pts) < n_points:
            raise RuntimeError(
                f"Could not sample enough in-bounds local points: requested={n_points}, got={len(pts)}"
            )
        return torch.stack(pts, dim=0)

    def simulate_expected_hits(
        self,
        phi_model: torch.Tensor,
        muons_per_rep: int,
        n_reps: int,
    ) -> Dict[str, float]:
        """
        Estimate expected TOTAL hits at phi by repeated Monte Carlo.
        """
        phi_model = phi_model.detach().cpu().view(-1)
        phi_phys = self.model_to_phys(phi_model).detach().cpu()

        vals = []
        for _ in range(n_reps):
            mu = random_sample(self.problem.muons, muons_per_rep)
            hits = self.problem(phi_phys, mu)
            vals.append(float(hits.float().sum().item()))

        vals_np = np.array(vals, dtype=np.float64)
        mean = float(vals_np.mean())
        std = float(vals_np.std(ddof=1)) if len(vals_np) > 1 else 0.0
        var = float(std ** 2)
        sem = float(std / math.sqrt(max(len(vals_np), 1)))
        return {
            "mean": mean,
            "std": std,
            "var": max(var, 1e-10),
            "sem": sem,
            "n_reps": int(n_reps),
        }

    def simulate_expected_hits_shared_muons(
        self,
        phi_model: torch.Tensor,
        muons: torch.Tensor,
    ) -> float:
        """
        Deterministic comparison on a fixed muon set.
        """
        phi_model = phi_model.detach().cpu().view(-1)
        phi_phys = self.model_to_phys(phi_model).detach().cpu()
        hits = self.problem(phi_phys, muons)
        return float(hits.float().sum().item())


# -----------------------------------------------------------------------------
# GP model / training
# -----------------------------------------------------------------------------
def build_and_fit_gp(
    train_X: torch.Tensor,
    train_Y: torch.Tensor,
    train_Yvar: torch.Tensor,
) -> SingleTaskGP:
    """
    train_X: (N, D)
    train_Y: (N, 1)
    train_Yvar: (N, 1)
    """
    gp = SingleTaskGP(
        train_X=train_X,
        train_Y=train_Y,
        train_Yvar=train_Yvar,
        outcome_transform=Standardize(m=1),
    )
    mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
    fit_gpytorch_mll(mll)
    gp.eval()
    gp.likelihood.eval()
    return gp


def posterior_mean_fn(gp: SingleTaskGP, x: torch.Tensor) -> torch.Tensor:
    """
    x: (..., D)
    returns: (...,)
    """
    post = gp.posterior(x)
    return post.mean.squeeze(-1)


def gp_mean_grad_hess(
    gp: SingleTaskGP,
    x0: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, float]:
    """
    Gradient and Hessian of GP posterior mean at x0 using autograd.
    x0: (D,)
    """
    x = x0.detach().clone().view(1, -1).requires_grad_(True)
    y = posterior_mean_fn(gp, x)[0]
    g = torch.autograd.grad(y, x, create_graph=True)[0].view(-1)

    d = g.numel()
    H_rows = []
    for i in range(d):
        gi = g[i]
        row = torch.autograd.grad(gi, x, retain_graph=True)[0].view(-1)
        H_rows.append(row)
    H = torch.stack(H_rows, dim=0)
    H = symmetrize(H.detach())
    return g.detach(), H.detach(), float(y.detach().item())


# -----------------------------------------------------------------------------
# Direction search
# -----------------------------------------------------------------------------
def max_feasible_alpha(
    center: torch.Tensor,
    direction: torch.Tensor,
    problem: MuonShieldProblemWrapper,
    alpha_cap: float,
) -> float:
    """
    Largest alpha >= 0 such that center + alpha * direction remains in bounds.
    """
    direction = direction.detach().cpu().view(-1)
    center = center.detach().cpu().view(-1)

    if torch.norm(direction).item() < 1e-14:
        return 0.0

    alpha_max = alpha_cap
    for _ in range(60):
        mid = 0.5 * alpha_max
        test = center + mid * direction
        if problem.in_bounds_model(test):
            alpha_max = min(alpha_cap, alpha_max * 2.0) if alpha_max < alpha_cap else alpha_max
            break

    lo, hi = 0.0, alpha_cap
    if not problem.in_bounds_model(center + hi * direction):
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            if problem.in_bounds_model(center + mid * direction):
                lo = mid
            else:
                hi = mid
        return lo
    return hi


def evaluate_direction_damage(
    phi0: torch.Tensor,
    direction: torch.Tensor,
    alphas: np.ndarray,
    gp: SingleTaskGP,
    problem: MuonShieldProblemWrapper,
    muons_shared: torch.Tensor,
    n_reps_true: int,
    muons_per_rep_true: int,
) -> Dict[str, List[float]]:
    """
    Evaluate GP mean and true expected hits along phi0 + alpha * direction.
    Only keeps feasible points.
    """
    d = unit(direction.detach().cpu().view(-1))
    gp_vals = []
    true_means = []
    true_sems = []
    kept_alphas = []

    for a in alphas:
        phi = phi0 + float(a) * d
        if not problem.in_bounds_model(phi):
            continue

        kept_alphas.append(float(a))
        with torch.no_grad():
            gp_mean = float(posterior_mean_fn(gp, phi.view(1, -1)).item())
        gp_vals.append(gp_mean)

        stats = problem.simulate_expected_hits(
            phi_model=phi,
            muons_per_rep=muons_per_rep_true,
            n_reps=n_reps_true,
        )
        true_means.append(stats["mean"])
        true_sems.append(stats["sem"])

    return {
        "alpha": kept_alphas,
        "gp_mean": gp_vals,
        "true_mean": true_means,
        "true_sem": true_sems,
    }


def predicted_damage_score(
    g: torch.Tensor,
    H: torch.Tensor,
    direction: torch.Tensor,
    radius: float,
) -> float:
    """
    Second-order predicted increase:
        max_{0<=a<=radius} a g^T d + 0.5 a^2 d^T H d
    on the positive side only.
    """
    d = unit(direction.view(-1))
    lin = float(torch.dot(g.view(-1), d).item())
    curv = float((d @ H @ d).item())

    # Check endpoints and stationary point if relevant
    candidates = [0.0, radius]
    if abs(curv) > 1e-14:
        a_star = -lin / curv
        if 0.0 <= a_star <= radius:
            candidates.append(a_star)

    vals = [a * lin + 0.5 * (a ** 2) * curv for a in candidates]
    return max(vals)


# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------
def plot_training_fit(train_y: np.ndarray, pred_y: np.ndarray, outpath: str) -> None:
    plt.figure(figsize=(6, 5))
    plt.scatter(train_y, pred_y, s=30, alpha=0.8)
    lo = min(float(np.min(train_y)), float(np.min(pred_y)))
    hi = max(float(np.max(train_y)), float(np.max(pred_y)))
    plt.plot([lo, hi], [lo, hi], "k--", lw=1)
    plt.xlabel("True MC estimated expected hits")
    plt.ylabel("GP posterior mean")
    plt.title("GP fit on training data")
    plt.tight_layout()
    plt.savefig(outpath, dpi=160)
    plt.close()


def plot_param_sensitivity(
    names: List[str],
    true_slopes: np.ndarray,
    gp_grad: np.ndarray,
    gp_diag_hess: np.ndarray,
    outpath: str,
) -> None:
    x = np.arange(len(names))
    width = 0.25

    plt.figure(figsize=(10, 5))
    plt.bar(x - width, np.abs(true_slopes), width=width, label="|True FD slope|")
    plt.bar(x, np.abs(gp_grad), width=width, label="|GP gradient|")
    plt.bar(x + width, np.abs(gp_diag_hess), width=width, label="|GP Hessian diag|")

    plt.xticks(x, names)
    plt.ylabel("Sensitivity magnitude")
    plt.title("Parameter sensitivity ranking")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=160)
    plt.close()


def plot_direction_damage(
    results: List[Dict],
    outpath: str,
) -> None:
    plt.figure(figsize=(9, 6))
    for res in results:
        a = np.array(res["curve"]["alpha"])
        gp_mean = np.array(res["curve"]["gp_mean"])
        true_mean = np.array(res["curve"]["true_mean"])
        true_sem = np.array(res["curve"]["true_sem"])

        if len(a) == 0:
            continue

        baseline = true_mean[np.argmin(np.abs(a))] if np.any(np.isclose(a, 0.0)) else true_mean[0]
        gp_baseline = gp_mean[np.argmin(np.abs(a))] if np.any(np.isclose(a, 0.0)) else gp_mean[0]

        plt.plot(a, gp_mean - gp_baseline, "-", label=f"{res['name']} (GP)")
        plt.errorbar(a, true_mean - baseline, yerr=true_sem, fmt="o", ms=4, capsize=3, label=f"{res['name']} (true)")

    plt.axvline(0.0, color="k", ls=":", lw=1)
    plt.xlabel("alpha along direction")
    plt.ylabel("Increase in expected hits")
    plt.title("Damage along candidate worst directions")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(outpath, dpi=160)
    plt.close()


def plot_oat_damage(
    oat_results: List[Dict],
    outpath: str,
) -> None:
    n = len(oat_results)
    ncols = 2
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 3.8 * nrows))
    axes = np.array(axes).reshape(-1)

    for i, res in enumerate(oat_results):
        ax = axes[i]
        a = np.array(res["alpha"])
        gp_mean = np.array(res["gp_mean"])
        true_mean = np.array(res["true_mean"])
        true_sem = np.array(res["true_sem"])

        baseline = true_mean[np.argmin(np.abs(a))]
        gp_baseline = gp_mean[np.argmin(np.abs(a))]

        ax.plot(a, gp_mean - gp_baseline, "-", label="GP")
        ax.errorbar(a, true_mean - baseline, yerr=true_sem, fmt="o", ms=4, capsize=3, label="True")
        ax.axvline(0.0, color="k", ls=":", lw=1)
        ax.set_title(res["name"])
        ax.set_xlabel("delta")
        ax.set_ylabel("Increase in expected hits")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)

    for j in range(n, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    plt.savefig(outpath, dpi=160)
    plt.close()


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="GP robustness analysis for Muon Shield")
    parser.add_argument("--normalize", action="store_true", help="Work in normalized phi-space.")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", type=str, default="float64", choices=["float32", "float64"])
    parser.add_argument("--seed", type=int, default=123)

    # Problem / data
    parser.add_argument("--config_file", type=str,
                        default="/home/hep/lprate/projects/BlackBoxOptimization/outputs/config_tests.json")
    parser.add_argument("--dim", type=int, default=3)
    parser.add_argument("--radius", type=float, default=0.20, help="Local ball radius in model space.")
    parser.add_argument("--n_train", type=int, default=40, help="Number of local phi points for GP fit.")
    parser.add_argument("--muons_per_rep_train", type=int, default=10_000_000, help="Muons per replication at each train phi.")
    parser.add_argument("--n_reps_train", type=int, default=3, help="Replications per train phi for noise estimate.")

    # Validation / testing
    parser.add_argument("--muons_per_rep_test", type=int, default=50_000_000)
    parser.add_argument("--n_reps_test", type=int, default=5)
    parser.add_argument("--fd_step", type=float, default=0.03, help="Finite-difference step for parameter slopes.")
    parser.add_argument("--n_alpha_dir", type=int, default=11)
    parser.add_argument("--n_alpha_oat", type=int, default=9)
    parser.add_argument("--direction_alpha_cap", type=float, default=0.30)

    parser.add_argument("--outdir", type=str, default="outputs/gp_robustness")
    args = parser.parse_args()

    # dtype
    if args.dtype == "float64":
        torch.set_default_dtype(torch.float64)
    else:
        torch.set_default_dtype(torch.float32)

    device = torch.device(args.device)
    safe_mkdir(args.outdir)
    set_seed(args.seed)

    t0 = time.time()

    # -------------------------------------------------------------------------
    # Problem
    # -------------------------------------------------------------------------
    problem = MuonShieldProblemWrapper(
        config_file=args.config_file,
        dim=args.dim,
        normalize=args.normalize,
        device=device,
    )

    phi0 = problem.initial_phi_model.detach().cpu().view(-1)
    param_names = [f"phi_{i}" for i in range(args.dim)]

    print("\n=== Initial point ===")
    print("phi0 (model space):", phi0.tolist())
    print("phi0 (physical):", problem.model_to_phys(phi0).tolist())

    # -------------------------------------------------------------------------
    # Sample local training points
    # -------------------------------------------------------------------------
    print("\nSampling local training points...")
    train_X = problem.sample_local_points(
        n_points=args.n_train,
        radius=args.radius,
        seed=args.seed,
        include_center=True,
    )

    # -------------------------------------------------------------------------
    # Simulate true expected hits with replicated MC for noise estimates
    # -------------------------------------------------------------------------
    print("Evaluating simulator on training points...")
    train_Y_list = []
    train_Yvar_list = []

    for i in range(train_X.shape[0]):
        stats = problem.simulate_expected_hits(
            phi_model=train_X[i],
            muons_per_rep=args.muons_per_rep_train,
            n_reps=args.n_reps_train,
        )
        train_Y_list.append(stats["mean"])
        train_Yvar_list.append(stats["var"])
        print(
            f"[{i+1:03d}/{train_X.shape[0]:03d}] "
            f"mean={stats['mean']:.6e}, std={stats['std']:.6e}, sem={stats['sem']:.6e}"
        )

    train_Y = torch.tensor(train_Y_list, dtype=torch.get_default_dtype()).view(-1, 1)
    train_Yvar = torch.tensor(train_Yvar_list, dtype=torch.get_default_dtype()).view(-1, 1)

    # Small nugget for safety
    train_Yvar = train_Yvar.clamp_min(1e-8)

    # -------------------------------------------------------------------------
    # Fit GP
    # -------------------------------------------------------------------------
    print("\nFitting GP...")
    gp = build_and_fit_gp(train_X, train_Y, train_Yvar)

    with torch.no_grad():
        pred_train = posterior_mean_fn(gp, train_X).cpu().numpy()
    plot_training_fit(
        train_y=train_Y.view(-1).cpu().numpy(),
        pred_y=pred_train,
        outpath=os.path.join(args.outdir, "gp_training_fit.png"),
    )

    # -------------------------------------------------------------------------
    # Local GP derivatives at phi0
    # -------------------------------------------------------------------------
    print("\nComputing GP local derivatives at phi0...")
    gp_grad, gp_hess, gp_f0 = gp_mean_grad_hess(gp, phi0)
    eigvals, eigvecs = torch.linalg.eigh(gp_hess)

    print("GP posterior mean at phi0:", gp_f0)
    print("GP gradient:", gp_grad.tolist())
    print("GP Hessian:\n", gp_hess.numpy())
    print("GP Hessian eigenvalues:", eigvals.tolist())

    # -------------------------------------------------------------------------
    # True one-at-a-time finite-difference slopes
    # -------------------------------------------------------------------------
    print("\nEstimating true one-at-a-time slopes...")
    true_fd_slopes = torch.zeros(args.dim, dtype=torch.get_default_dtype())
    oat_results = []

    for i in range(args.dim):
        ei = torch.zeros(args.dim, dtype=torch.get_default_dtype())
        ei[i] = 1.0

        # Bound-respecting step
        step_pos = args.fd_step
        step_neg = args.fd_step

        while step_pos > 1e-6 and not problem.in_bounds_model(phi0 + step_pos * ei):
            step_pos *= 0.5
        while step_neg > 1e-6 and not problem.in_bounds_model(phi0 - step_neg * ei):
            step_neg *= 0.5

        if step_pos <= 1e-6 or step_neg <= 1e-6:
            true_fd_slopes[i] = float("nan")
        else:
            stats_p = problem.simulate_expected_hits(
                phi_model=phi0 + step_pos * ei,
                muons_per_rep=args.muons_per_rep_test,
                n_reps=args.n_reps_test,
            )
            stats_m = problem.simulate_expected_hits(
                phi_model=phi0 - step_neg * ei,
                muons_per_rep=args.muons_per_rep_test,
                n_reps=args.n_reps_test,
            )
            true_fd_slopes[i] = (stats_p["mean"] - stats_m["mean"]) / (step_pos + step_neg)

        # OAT curve
        alpha_max_pos = max_feasible_alpha(phi0, ei, problem, args.direction_alpha_cap)
        alpha_max_neg = max_feasible_alpha(phi0, -ei, problem, args.direction_alpha_cap)
        alpha_cap = min(alpha_max_pos, alpha_max_neg)
        alphas = np.linspace(-alpha_cap, alpha_cap, args.n_alpha_oat)

        curve = evaluate_direction_damage(
            phi0=phi0,
            direction=ei,
            alphas=alphas,
            gp=gp,
            problem=problem,
            muons_shared=None,
            n_reps_true=args.n_reps_test,
            muons_per_rep_true=args.muons_per_rep_test,
        )
        curve["name"] = param_names[i]
        oat_results.append(curve)

        slope_val = true_fd_slopes[i].item()
        print(f"{param_names[i]}: true FD slope = {slope_val:.6e}")

    # -------------------------------------------------------------------------
    # Parameter sensitivity ranking
    # -------------------------------------------------------------------------
    gp_grad_np = gp_grad.cpu().numpy()
    gp_diag_hess_np = torch.diag(gp_hess).cpu().numpy()
    true_fd_np = true_fd_slopes.cpu().numpy()

    plot_param_sensitivity(
        names=param_names,
        true_slopes=true_fd_np,
        gp_grad=gp_grad_np,
        gp_diag_hess=gp_diag_hess_np,
        outpath=os.path.join(args.outdir, "gp_param_sensitivity.png"),
    )

    sensitivity_rows = []
    for i in range(args.dim):
        oat_true = np.array(oat_results[i]["true_mean"], dtype=float)
        oat_alpha = np.array(oat_results[i]["alpha"], dtype=float)
        base_idx = int(np.argmin(np.abs(oat_alpha)))
        max_damage = float(np.max(oat_true - oat_true[base_idx]))

        sensitivity_rows.append({
            "name": param_names[i],
            "true_fd_slope": float(true_fd_np[i]),
            "gp_grad_component": float(gp_grad_np[i]),
            "gp_hessian_diag": float(gp_diag_hess_np[i]),
            "max_true_oat_damage": max_damage,
        })

    sensitivity_rows = sorted(
        sensitivity_rows,
        key=lambda z: abs(z["max_true_oat_damage"]),
        reverse=True,
    )

    print("\n=== Most sensitive parameters (by max true OAT damage) ===")
    for row in sensitivity_rows:
        print(
            f"{row['name']}: "
            f"max_true_oat_damage={row['max_true_oat_damage']:.6e}, "
            f"true_fd_slope={row['true_fd_slope']:.6e}, "
            f"gp_grad={row['gp_grad_component']:.6e}, "
            f"gp_hdiag={row['gp_hessian_diag']:.6e}"
        )

    # -------------------------------------------------------------------------
    # Candidate worst directions
    # -------------------------------------------------------------------------
    print("\nSearching candidate worst directions...")
    candidate_dirs = []

    # Gradient direction
    if torch.norm(gp_grad).item() > 1e-12:
        candidate_dirs.append(("grad", unit(gp_grad)))

    # Hessian eigenvectors: positive curvature directions are particularly dangerous
    for i in range(args.dim):
        candidate_dirs.append((f"eig_{i}", unit(eigvecs[:, i])))

    # Coordinate directions too
    for i in range(args.dim):
        e = torch.zeros(args.dim, dtype=torch.get_default_dtype())
        e[i] = 1.0
        candidate_dirs.append((f"coord_{i}", e))

    # Random directions
    g = torch.Generator(device="cpu")
    g.manual_seed(args.seed + 999)
    for i in range(12):
        d = torch.randn(args.dim, generator=g, dtype=torch.get_default_dtype())
        candidate_dirs.append((f"rand_{i}", unit(d)))

    # Score each direction in both signs
    direction_scores = []
    for name, d in candidate_dirs:
        for sign in [+1.0, -1.0]:
            ds = sign * d
            alpha_cap = max_feasible_alpha(phi0, ds, problem, args.direction_alpha_cap)
            score = predicted_damage_score(gp_grad, gp_hess, ds, alpha_cap)
            direction_scores.append({
                "name": f"{name}_{'plus' if sign > 0 else 'minus'}",
                "direction": ds.clone(),
                "alpha_cap": float(alpha_cap),
                "pred_score": float(score),
            })

    direction_scores = sorted(direction_scores, key=lambda z: z["pred_score"], reverse=True)

    print("\nTop predicted damaging directions:")
    for row in direction_scores[:8]:
        print(
            f"{row['name']}: pred_score={row['pred_score']:.6e}, "
            f"alpha_cap={row['alpha_cap']:.6e}, dir={row['direction'].tolist()}"
        )

    # Take top few unique directions for true validation
    tested_directions = []
    used_prefixes = set()
    for row in direction_scores:
        prefix = row["name"].rsplit("_", 1)[0]
        if prefix in used_prefixes:
            continue
        tested_directions.append(row)
        used_prefixes.add(prefix)
        if len(tested_directions) >= min(5, len(direction_scores)):
            break

    direction_curve_results = []
    for row in tested_directions:
        alpha_cap = row["alpha_cap"]
        alphas = np.linspace(0.0, alpha_cap, args.n_alpha_dir)
        curve = evaluate_direction_damage(
            phi0=phi0,
            direction=row["direction"],
            alphas=alphas,
            gp=gp,
            problem=problem,
            muons_shared=None,
            n_reps_true=args.n_reps_test,
            muons_per_rep_true=args.muons_per_rep_test,
        )
        direction_curve_results.append({
            "name": row["name"],
            "pred_score": row["pred_score"],
            "direction": row["direction"].tolist(),
            "curve": curve,
        })

    plot_direction_damage(
        results=direction_curve_results,
        outpath=os.path.join(args.outdir, "gp_direction_damage.png"),
    )

    plot_oat_damage(
        oat_results=oat_results,
        outpath=os.path.join(args.outdir, "gp_oat_damage.png"),
    )

    # -------------------------------------------------------------------------
    # Summary JSON
    # -------------------------------------------------------------------------
    summary = {
        "phi0_model": phi0.tolist(),
        "phi0_phys": problem.model_to_phys(phi0).tolist(),
        "gp_posterior_mean_at_phi0": gp_f0,
        "gp_gradient": gp_grad.tolist(),
        "gp_hessian": gp_hess.tolist(),
        "gp_hessian_eigenvalues": eigvals.tolist(),
        "gp_hessian_eigenvectors": eigvecs.tolist(),
        "parameter_sensitivity": sensitivity_rows,
        "top_direction_tests": [
            {
                "name": row["name"],
                "pred_score": row["pred_score"],
                "direction": row["direction"],
                "curve": row["curve"],
            }
            for row in direction_curve_results
        ],
        "args": vars(args),
        "elapsed_seconds": time.time() - t0,
    }

    with open(os.path.join(args.outdir, "gp_robustness_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("\nSaved outputs to:", args.outdir)
    print("Done.")


if __name__ == "__main__":
    main()