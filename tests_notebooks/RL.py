import torch
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.abspath(os.path.join('..', 'src')))
sys.path.append(os.path.abspath(os.path.join('..')))
from muons_and_matter.cuda_muons_ship import run_from_params as simulate_muon_shield
import json

class MuonShieldEnvironment:
    def __init__(self):
        self.kwargs = {'NI_from_B': True,
                       'simulate_fields': False,
                       "add_cavern": True,
                       "use_diluted": False}
    def reset(self):
        self.muons = sample_x()
    def step(self, phi):
        Z = 0.02
        muons_output = simulate_muon_shield(phi, self.muons, 
                                            sensitive_plane = {'dz': 0.02, 'dx': 20, 'dy': 20, 'position': Z}, 
                                            **self.kwargs)
        self.muons = torch.cat(muons_output['px'], muons_output['py'], muons_output['pz'], 
                               muons_output['x'], muons_output['y'], muons_output['z'], 
                               muons_output['pdg_id'], muons_output['weight'], dim=1)
        
        return 

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
