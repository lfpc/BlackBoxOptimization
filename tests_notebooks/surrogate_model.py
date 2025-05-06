import torch
import os
import sys
PROJECTS_DIR = os.getenv('PROJECTS_DIR')
sys.path.append('/home/hep/lprate/projects/BlackBoxOptimization/src')
import problems
import gzip
import pickle
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np


def histograms(inputs, outputs, fontsize=12):

    px, py, pz, x, y, z, charge, W = inputs
    pt = np.sqrt(px**2 + py**2)
    p = np.sqrt(px**2 + py**2 + pz**2)
    px_fs, py_fs, pz_fs, x_fs, y_fs, z_fs, charge_fs, W_fs = outputs
    pt_fs = np.sqrt(px_fs**2 + py_fs**2)
    p_fs = np.sqrt(px_fs**2 + py_fs**2 + pz_fs**2)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True, sharex=True)
    fig.tight_layout()

    # First histogram (Enriched Sample)
    h1, xedges, yedges, im1 = axes[0].hist2d(p,pt, bins=50, cmap='viridis', norm=LogNorm(vmax=3e3), density=False, weights=W)
    axes[0].set_xlabel('$|P|$ [GeV]', fontsize=fontsize)
    axes[0].set_ylabel('$P_t$ [GeV]', fontsize=fontsize)
    axes[0].set_title('Input Sample', fontsize=fontsize)
    cbar1 = fig.colorbar(im1, ax=axes[0], label='Density')

    # Second histogram (Full Sample)
    h2, _, _, im2 = axes[1].hist2d(p_fs,pt_fs, bins=[xedges, yedges], cmap='viridis', norm=LogNorm(), density=False, weights=W_fs)
    axes[1].set_xlabel('$|P|$ [GeV]', fontsize=fontsize)
    axes[1].set_title('Output Sample', fontsize=fontsize)
    cbar2 = fig.colorbar(im2, ax=axes[1], label='Density')

    # Adjust tick parameters
    for ax in axes:
        ax.tick_params(axis='both', which='major', labelsize=fontsize)
        ax.tick_params(axis='both', which='minor', labelsize=fontsize)
    # Save the figure
    plt.savefig('histogram_momentum.png', dpi=300, bbox_inches='tight')
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True, sharex=True)
    fig.tight_layout()

    # Second histogram (Full Sample)
    h2, xedges, yedges, im2 = axes[1].hist2d(x_fs,y_fs, bins=50, cmap='viridis', norm = LogNorm(),  density=False, weights=W_fs)
    axes[1].set_xlabel('x [m]', fontsize=fontsize)
    axes[1].set_title('Output', fontsize=fontsize)
    cbar2 = fig.colorbar(im2, ax=axes[1], label='Density')

    # First histogram (Enriched Sample)
    h1, _,_, im1 = axes[0].hist2d(x,y, bins=[xedges, yedges], cmap='viridis', norm = LogNorm(),  density=False, weights=W)
    axes[0].set_xlabel('x [m]', fontsize=fontsize)
    axes[0].set_ylabel('y [m]', fontsize=fontsize)
    axes[0].set_title('Input', fontsize=fontsize)
    cbar1 = fig.colorbar(im1, ax=axes[0], label='Density')

    

    # Adjust tick parameters
    for ax in axes:
        ax.tick_params(axis='both', which='major', labelsize=fontsize)
        ax.tick_params(axis='both', which='minor', labelsize=fontsize)
    # Save the figure
    plt.savefig('histogram_position.png', dpi=300, bbox_inches='tight')
    plt.close(fig)



class OneMagnet(problems.ShipMuonShield):
    DEFAULT_PHI = torch.tensor([120.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                   50.00,  50.00, 119.00, 119.00,   2.00,   2.00, 1.00,1.0,50.00,  50.00,0.0, 0.00, 0.,
                   0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                   0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                   0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                   0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                   0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                   0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    def __init__(self, n_samples = 0,
                 cores = 50,
                 sens_plane = 5,
                 muons_file = os.path.join(PROJECTS_DIR,'MuonsAndMatter/data/muons/subsample_4M.pkl')
                 ):
        sensitive_plane = {'dz': 0.01, 'dx': 10, 'dy': 10,'position': sens_plane}
        self.params_idx = self.parametrization['HA'][:9] + self.parametrization['HA'][12:13]
        super().__init__(n_samples = n_samples,
                         muons_file = muons_file,
                         sensitive_plane = sensitive_plane,
                         fSC_mag = False, 
                         simulate_fields=False,
                         cavern = False, 
                         apply_det_loss = False,
                         dimensions_phi = -1,
                         SmearBeamRadius = 0,
                         cores = cores)
        



if __name__ == "__main__":
    import argparse
    import time
    parser = argparse.ArgumentParser(description="Run OneMagnet simulation.")
    parser.add_argument("--n_samples", type=int, default=10000, help="Number of samples to simulate.")
    parser.add_argument("--cores", type=int, default=50, help="Number of cores to use.")
    parser.add_argument("--sens_plane", type=float, default=5, help="Sensitive plane position.")
    parser.add_argument("--muons_file", type=str, default=os.path.join(PROJECTS_DIR, 'MuonsAndMatter/data/muons/subsample_4M.pkl'), help="Path to the muons file.")
    args = parser.parse_args()

    Magnet = OneMagnet(n_samples=args.n_samples, 
                    cores=args.cores,
                    sens_plane=args.sens_plane,
                    muons_file=args.muons_file)
    phi = torch.tensor([20, 50.00,  50.00, 119.00, 119.00, 2.00, 2.00, 1.00,1.0,0.0])
    z_sens = (phi[0].item()*2-10+5)/100
    Magnet.sensitive_plane = {'dz': 0.01, 'dx': 10, 'dy': 10,'position': z_sens}
    with gzip.open(Magnet.muons_file, 'rb') as f:
        muons = pickle.load(f)[:args.n_samples]
    start_time = time.time()
    outputs = Magnet.simulate(phi, muons = muons, return_nan = True)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Simulation took {elapsed_time:.2f} seconds.")
    print(outputs.shape)
    hits = outputs[-2].abs().int()==13
    print(hits.sum())

    histograms(muons.T, outputs)

