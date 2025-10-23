#Script must be executed inside the container

import pickle
with open("hall_of_fame_3.pkl", "rb") as f:
    hall_of_fame = pickle.load(f)
for individual_index in range(len(hall_of_fame)):
    individual=hall_of_fame[individual_index]
    print(f"Individual: {individual}, fitness value: {individual.fitness}, genes: {individual.genes}")

import os
import argparse
import json
import torch
from src import problems

parser = argparse.ArgumentParser()
parser.add_argument("--cpu",dest = 'cuda', action = 'store_false')
parser.add_argument("--seed", type=int, default=13)
parser.add_argument("--maxiter", type=int, default=1000)
parser.add_argument('--problem', type=str, default='ship_cuda')
parser.add_argument('--optimization', type=str, default='bayesian')
parser.add_argument('--model', type=str, default='gp_rbf')
parser.add_argument('--name', type=str, default='optimizationtest')
parser.add_argument('--group', type=str, default='BayesianOptimization')
parser.add_argument("--dont_save_history", action='store_false', dest='save_history')
parser.add_argument("--resume", action='store_true')
parser.add_argument("--reduce_bounds", type=int, default=-1)
parser.add_argument("--multi_fidelity", type=int, nargs='?', const=-1, default=None)
parser.add_argument("--parallel", type=int, default = 1)
parser.add_argument("--model_switch", type=int,default = -1)
parser.add_argument('--n_samples', type=int, default=0)
parser.add_argument("--n_initial", type=int, default=-1)
parser.add_argument('--float64', action = 'store_true')
parser.add_argument('--config_file', type=str, default='outputs/config.json')
args = parser.parse_args()

PROJECTS_DIR = os.getenv('PROJECTS_DIR', default = '~')
OUTPUTS_DIR = os.path.join(PROJECTS_DIR,'BlackBoxOptimization/outputs',args.name)
config_file = os.path.join(OUTPUTS_DIR,'config.json')

run_in_background=True#False

if args.resume:
    with open(config_file, 'r') as f:
        CONFIG = json.load(f)
else: 
    if not os.path.exists(OUTPUTS_DIR):
        os.makedirs(OUTPUTS_DIR)
    with open(args.config_file, 'r') as src, open(config_file, 'w') as dst:
        CONFIG = json.load(src)
        if run_in_background:
            CONFIG['W0'] = float(11E6)
            CONFIG['L0'] = float(29.7)
            CONFIG['dimensions_phi'] = int(63)
            default_phi_name = str('')
        else:
            CONFIG['W0'] = float(input("Enter Reference Cost (W0) [default: 11E6]: ") or 11E6)
            CONFIG['L0'] = float(input("Enter Maximum Length (L0) [default: 29.7]: ") or 29.7)
            CONFIG['dimensions_phi'] = int(input("Enter number of dimensions [default: 63]: ") or 63)
            default_phi_name = str(input("Enter name of initial phi [default: see DEFAULT_PHI of Ship class]: ") or '')
        print('default_phi_name', default_phi_name)
        if default_phi_name == '': CONFIG['initial_phi'] = problems.ShipMuonShield.DEFAULT_PHI.tolist()
        else: CONFIG['initial_phi'] = problems.ShipMuonShield.params[default_phi_name]
        if args.multi_fidelity is not None:
            if args.multi_fidelity > 0:
                CONFIG['n_samples'] = args.multi_fidelity
            elif args.multi_fidelity == -1 and CONFIG.get('n_samples', 0) == 0:
                CONFIG['n_samples'] = int(float(input("Enter number of samples for low_fidelity [default: 5E5]: ") or 5E5) )
        json.dump(CONFIG, dst, indent=4)
CONFIG.pop("data_treatment", None)
CONFIG.pop('results_dir', None)
problem_fn = problems.ShipMuonShieldCuda(parallel=args.parallel, **CONFIG)
for individual_index in range(len(hall_of_fame)):
    individual=hall_of_fame[individual_index]
    phi=torch.tensor(individual.genes, dtype=torch.float32).unsqueeze(0)
    constraints=problem_fn.get_constraints(phi)
    loss = problem_fn(phi)
    print("hola")
    print(f"Individual: {individual}, fitness value: {individual.fitness}, genes: {individual.genes}")
    print(constraints)
    print(loss)
    print(hola)