import torch
import os
import tqdm
import matplotlib.pyplot as plt
import pickle
PROJECTS_DIR = os.getenv('PROJECTS_DIR')
import sys
sys.path.insert(1, os.path.join(PROJECTS_DIR,'BlackBoxOptimization/src'))
sys.path.insert(1, os.path.join(PROJECTS_DIR,'MuonsAndMatter/python/bin'))
from problems import ShipMuonShieldCluster,ShipMuonShield
from run_full_sample import get_total_hits
from get_NI import get_NI
from optimizer import LCSO

class RobustnessAnalysis:
    def __init__(self,n_files:int = 67, inputs_dir = 'data/full_sample', outputs_dir= 'outputs' , config:dict = {}):
        self.inputs_dir = inputs_dir
        self.outputs_dir = outputs_dir
        self.n_files = n_files
        self.config = config
    def run_zero_order(self, initial_param:torch.tensor, idx, error:torch.tensor, steps:int = 11):
        range_p = torch.linspace(error[0], error[1], steps=steps)
        losses = []
        params = []
        for p in tqdm.tqdm(range_p):
            print(f"Testing parameter {idx} with perturbation {p}")
            full_param = initial_param.clone()
            full_param[idx] = full_param[idx] + p
            loss = get_total_hits(full_param, n_files=self.n_files, inputs_dir=self.inputs_dir, outputs_dir=self.outputs_dir, config=self.config)[1]
            losses.append(loss)
            params.append(full_param[idx])
            with open('results_robustness.pkl', 'wb') as f:
                pickle.dump({'params': params, 'losses': losses}, f)
        return params, losses
    

if __name__ == '__main__':
    INPUTS_DIR = '/home/hep/lprate/projects/MuonsAndMatter/data/full_sample'
    OUTPUTS_DIR = '/home/hep/lprate/projects/MuonsAndMatter/data/outputs'
    import argparse
    import json
    CONFIG_PATH = os.path.join(PROJECTS_DIR, 'cluster', 'config.json')
    with open(CONFIG_PATH, 'r') as f:
        CONFIG = json.load(f)
    CONFIG.pop("data_treatment", None)

    parser = argparse.ArgumentParser(description="Run robustness analysis.")
    parser.add_argument('--n_files', type=int, default=67, help='Number of files to process')
    parser.add_argument('--inputs_dir', type=str, default=INPUTS_DIR, help='Directory with input files')
    parser.add_argument('--outputs_dir', type=str, default=OUTPUTS_DIR, help='Directory to save outputs')
    parser.add_argument('--magnet', type=str, default='HA', help='Magnet configuration')
    parser.add_argument('--idx', type=int, default=0, help='Index of the parameter to vary')
    parser.add_argument('--error', type=float, nargs=2, default=[-0.4, 0.4], help='Range of errors to apply to the parameter')
    parser.add_argument('--steps', type=int, default=11, help='Number of steps in the perturbation range')
    args = parser.parse_args()

    initial_param = ShipMuonShieldCluster.tokanut_v5
    initial_param = get_NI(initial_param, fSC_mag=False, use_diluted=False)
    initial_param = torch.from_numpy(initial_param).float()

    idx = ShipMuonShieldCluster.parametrization[str(args.magnet)][args.idx]
    error = torch.tensor(args.error)
    robustness_analysis = RobustnessAnalysis(n_files=args.n_files, inputs_dir=INPUTS_DIR, outputs_dir=OUTPUTS_DIR, config=CONFIG)
    params, losses = robustness_analysis.run_zero_order(initial_param, idx, error, steps = args.steps)

    print("Robustness Analysis Results:")
    for p, l in zip(params, losses):
        print(f"Parameter: {p}, Loss: {l}")
    plt.plot(params, losses)
    plt.axvline(initial_param[idx].item(), color='r', linestyle='--')
    plt.axhline(0, color='g', linestyle='--')
    plt.xlabel('Parameter Value')
    plt.ylabel('Loss')
    plt.grid()
    plt.title('Robustness Analysis')
    plt.savefig(os.path.join(robustness_analysis.outputs_dir, 'robustness_analysis.png'))
    plt.close()
    




