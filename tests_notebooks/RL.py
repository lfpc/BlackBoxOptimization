import torch
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.abspath(os.path.join('..', 'src')))
sys.path.append(os.path.abspath(os.path.join('..')))
from problems import ShipMuonShieldCuda, make_index
import json

n_samples = 0
dim = 15
x_dim = 8
config_file = "/home/hep/lprate/projects/BlackBoxOptimization/outputs/config_tests.json"
with open(config_file, 'r') as f:
    CONFIG = json.load(f)
CONFIG.pop("data_treatment", None)
CONFIG.pop('results_dir', None)
CONFIG['dimensions_phi'] = make_index(0, list(range(dim)))
CONFIG['initial_phi'] = ShipMuonShieldCuda.params['only_HA']
CONFIG['n_samples'] = n_samples
CONFIG['reduction'] = 'sum'
CONFIG['cost_as_constraint'] = False
problem = ShipMuonShieldCuda(**CONFIG)
hits_0 = problem(problem.initial_phi)
phi = torch.stack([problem.initial_phi, problem.initial_phi]).flatten()
problem.n_magnets += 1
hits = problem(phi)
print("Initial parameters:", problem.initial_phi)
print("Initial hits:", hits_0)
print("Parameters:", phi)
print("Hits:", hits)
