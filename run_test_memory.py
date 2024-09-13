import torch
from src import problems
import argparse
import os

PROJECTS_DIR = os.getenv('PROJECTS_DIR', default = '~')

from warnings import filterwarnings
filterwarnings("ignore")

parser = argparse.ArgumentParser()

parser.add_argument("--maxiter", type=int, default=5000)
parser.add_argument("--nodes",type=int,default = 4)
parser.add_argument("--n_tasks_per_node", type=int, default=96)
parser.add_argument("--n_tasks", type=int, default=None)
args = parser.parse_args()

OUTPUTS_DIR = os.path.join(PROJECTS_DIR,'BlackBoxOptimization/outputs','test_memory')
if not os.path.exists(OUTPUTS_DIR):
    os.makedirs(OUTPUTS_DIR)


from os import getenv
class ShipMuonShieldMemory(problems.ShipMuonShieldCluster):
    DEF_N_SAMPLES = 484449
    def __init__(self,
                **kwargs) -> None:
        super().__init__(**kwargs)
    def simulate(self,phi:torch.tensor,muons = None):
        phi = phi.flatten() #Can we make it paralell on phi also?
        if len(phi) ==42: phi = self.add_fixed_params(phi)
        if muons is None: muons = self.sample_x(phi)
        #star_client = self.StarClient(self.server_url, self.manager_cert_path, 
        #                         self.client_cert_path, self.client_key_path) #redefine it every iteration?
        muons = problems.split_array_idx(phi.cpu(),muons) #If we can pass phi previously (??), no need to do this 
        memory = self.star_client.run(muons)
        return memory
    def __call__(self,phi,muons = None):
        memory = self.simulate(phi,muons)
        return memory    

import gzip
import pickle
if __name__ == "__main__":

    if args.n_tasks is None: n_tasks = args.nodes*args.n_tasks_per_node
    else: n_tasks = args.n_tasks



    problem_fn = ShipMuonShieldMemory(cores = n_tasks)
    for i in range(args.maxiter):
        print("ITERATION: ", i)
        phi = problem_fn.DEFAULT_PHI[0]+torch.rand(42)
        memory = problem_fn(phi)
        print('memory_usage = ', memory)
        with gzip.open(os.path.join(OUTPUTS_DIR,'memory_usage.pkl.gz'), 'wb') as f:
            pickle.dump(memory, f)

    


    








