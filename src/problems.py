import torch
import gzip
import pickle
import sys
import numpy as np
from os import getenv
from os.path import join
from multiprocessing import Pool
from subprocess import call 

PROJECTS_DIR = getenv('PROJECTS_DIR')
sys.path.insert(1, join(PROJECTS_DIR,'BlackBoxOptimization'))
from utils import split_array, split_array_idx, get_split_indices
import logging
logging.basicConfig(level=logging.WARNING)

def uniform_sample(shape,bounds,device = 'cpu'):
    return (bounds[0]-bounds[1])*torch.rand(shape,device=device)+bounds[1]


class oliver_fn():
    def __init__(self,omega:float = 5,gamma:float = 1,noise:float = 0):
        self.omega = omega
        self.gamma = gamma
        self.noise = noise
    def __call__(self,phi:torch.tensor):
        y = torch.prod(torch.sin(self.omega*phi)*(1-torch.tanh(self.gamma*phi.pow(2))),-1,keepdims=True)
        if self.noise:
            y += torch.randn_like(y)*self.noise
        return y
    
class RosenbrockProblem():
    def __init__(self,noise:float = 0) -> None:
        self.noise = noise
    def __call__(self,phi:torch.tensor,x:torch.tensor = None):
        y =  (phi[:, 1:] - phi[:, :-1].pow(2)).pow(2).sum(dim=1,
                                                        keepdim=True) + (1 - phi[:, :-1]).pow(2).sum(dim=1, keepdim=True)
        if self.noise:
            y += x
        return y
    def sample_x(self,phi,*args):
        return torch.randn((phi.shape[0],1),device = phi.device)*self.noise
    
    
class stochastic_RosenbrockProblem(RosenbrockProblem):
    def __init__(self,bounds = (-10,10), 
                 n_samples:int = 1, 
                 average_x = True,
                 std:float = 1.) -> None:
        super().__init__(0)
        self.bounds = bounds
        self.n_samples = n_samples
        self.average_x = average_x
        self.std = std
    def sample_x(self,phi):
        mu = uniform_sample((phi.shape[0],self.n_samples),self.bounds,device = phi.device)
        return torch.randn_like(mu)+mu
    def simulate(self,phi:torch.tensor, x:torch.tensor = None):
        if x is None: x = self.sample_x(phi)
        y = super().__call__(phi) + x
        y += torch.randn(y.size(0),1,device=phi.device)*self.std
        return y
    def loss(self,x):
        return x
    def __call__(self,phi:torch.tensor, x:torch.tensor = None):
        y = self.loss(self.simulate(phi,x))
        if self.average_x: y = y.mean(-1,keepdim = True)
        return y
    @staticmethod
    def GetBounds(device = torch.device('cpu')):
        pass

    
class ThreeHump():
    def __init__(self,noise:float = 0) -> None:
        self.noise = noise
    def __call__(self,phi:torch.tensor, x:torch.tensor = None):
        y = 2*(phi[:, 0].pow(2)) - 1.05*(phi[:, 0]**4) + phi[:, 0] ** 6 / 6 + phi[:,0] * phi[:,1] + phi[:, 1] ** 2
        y = y.view(-1,1)
        if self.noise:
            y += x
        return y
    def sample_x(self,phi,*args):
        return torch.randn((phi.shape[0],1),device = phi.device)*self.noise

class stochastic_ThreeHump(ThreeHump):
    def __init__(self, n_samples = 1, 
                 average_x = True,
                 bounds_1 = (-2,0), 
                 bounds_2 = (2,5),
                 std:float = 1.0) -> None:
        super().__init__(0)
        self.bounds_1 = bounds_1
        self.bounds_2 = bounds_2
        self.n_samples = n_samples
        self.average_x = average_x
        self.std = std
    def loss(self,x):
        return torch.sigmoid(x-10)-torch.sigmoid(x)
    def sample_x(self,phi):
        P1 = phi[:,0].div(phi.norm(p=2,dim=-1)).view(-1,1)
        mask = torch.rand((phi.size(0),self.n_samples),device=phi.device).le(P1)
        x1 = uniform_sample((phi.size(0),self.n_samples),self.bounds_1,device=phi.device)
        x2 = uniform_sample((phi.size(0),self.n_samples),self.bounds_2,device=phi.device)
        return torch.where(mask, x1, x2)
    def simulate(self, phi:torch.tensor, x:torch.tensor = None):
        h = super().__call__(phi)
        if x is None: x = self.sample_x(phi)
        mu = x*h + torch.randn_like(h)*self.std
        y =  mu+torch.randn_like(h)*self.std
        return y
    def __call__(self, phi:torch.tensor, x:torch.tensor = None):
        y = self.loss((self.simulate(phi,x)))
        if self.average_x: y = y.mean(-1,keepdim=True)
        return y

class ShipMuonShield():
    baseline_1 = torch.tensor([205.,205.,280.,245.,305.,240.,87.,65.,
    35.,121,11.,2.,65.,43.,121.,207.,11.,2.,6.,33.,32.,13.,70.,11.,5.,16.,112.,5.,4.,2.,15.,34.,235.,32.,5.,
    8.,31.,90.,186.,310.,2.,55.])
    combi = torch.as_tensor([ 208.  , 207.  , 281.  , 172.82, 212.54, 168.64,
      72.  ,  51.  ,  29.  ,  46.  ,
    10.  ,   7.  ,  54.  ,  38.  ,  46.  , 192.  ,  14.  ,   9.  ,
    10.  ,  31.  ,  35.  ,  31.  ,  51.  ,  11.  ,   3.  ,  32.  ,
    54.  ,  24.  ,   8.  ,   8.  ,  22.  ,  32.  , 209.  ,  35.  ,
    8.  ,  13.  ,  33.  ,  77.  ,  85.  , 241.  ,   9.  ,  26.  ])

    sc_v6 = torch.as_tensor([0,353.078,125.083,184.834,150.193,186.812,72,51,29,46,10,7,45.6888,
         45.6888,22.1839,22.1839,27.0063,16.2448,10,31,35,31,51,11,24.7961,48.7639,8,104.732,15.7991,16.7793,3,100,192,192,2,
         4.8004,3,100,8,172.729,46.8285,2])

    opt_oliver =  torch.tensor([208.0, 207.0, 281.0, 248.0, 305.0, 242.0, 72.0, 51.0, 29.0, 46.0, 10.0, 7.0, 54.0,
                         38.0, 46.0, 192.0, 14.0, 9.0, 10.0, 31.0, 35.0, 31.0, 51.0, 11.0, 3.0, 32.0, 54.0, 
                         24.0, 8.0, 8.0, 22.0, 32.0, 209.0, 35.0, 8.0, 13.0, 33.0, 77.0, 85.0, 241.0, 9.0, 26.0])
    DEFAULT_PHI = torch.stack([baseline_1,combi,sc_v6])

    def __init__(self,
                 W0:float = 1915820.,
                 cores:int = 45,
                 n_samples:int = 0,
                 input_dist:float = 0.1,
                 sensitive_plane:float = 57,#distance between end of shield and sensplane
                 average_x:bool = True,
                 loss_with_weight:bool = True,
                 fSC_mag:bool = True,
                 ) -> None:
        
        self.left_margin = 2.6
        self.MUON = 13
        self.right_margin = 3.
        self.y_margin = 5.
        self.z_bias = 50
        self.W0 = W0
        self.cores = cores
        self.muons_file = join(PROJECTS_DIR,'MuonsAndMatter/data/inputs.pkl')
        self.n_samples = n_samples
        self.input_dist = input_dist
        self.average_x = average_x
        self.loss_with_weight = loss_with_weight
        self.sensitive_plane = sensitive_plane
        self.sensitive_film_params = {'dz': 0.01, 'dx': 20, 'dy': 30,'position': 0} #the center is in end of muon shield + position
        self.fSC_mag = fSC_mag

        sys.path.insert(1, join(PROJECTS_DIR,'MuonsAndMatter/python/bin'))
        from run_simulation import run
        self.run_muonshield = run

    def sample_x(self,phi=None):
        with gzip.open(self.muons_file, 'rb') as f:
            x = pickle.load(f)
        if 0<self.n_samples<=x.shape[0]: x = x[:self.n_samples]
        return x
    def simulate(self,phi:torch.tensor,muons = None): 
        phi = phi.flatten() #Can we make it paralell on phi also?
        if len(phi) ==42: phi = self.add_fixed_params(phi)
        if muons is None: muons = self.sample_x()

        workloads = split_array(muons,self.cores)

        with Pool(self.cores) as pool:
            result = pool.starmap(self.run_muonshield, 
                                  [(workload,phi.cpu().numpy(),self.z_bias,self.input_dist,True,self.fSC_mag,self.sensitive_film_params) for workload in workloads])

        all_results = []
        for rr in result:
            resulting_data,weight = rr
            if resulting_data.size == 0: continue
            all_results += [resulting_data]
            
        all_results = torch.as_tensor(np.concatenate(all_results, axis=0).T,device = phi.device)
        return *all_results, torch.as_tensor(weight,device = phi.device)
    
    def muon_loss(self,x,y,particle):
        charge = -1*torch.sign(particle)
        mask = (-charge*x <= self.left_margin) & (-self.right_margin <= -charge*x) & (torch.abs(y) <= self.y_margin) & ((torch.abs(particle).to(torch.int))==self.MUON)
        x = x[mask]
        charge = charge[mask]
        return torch.sqrt(1 + (charge*x-self.right_margin)/(self.left_margin+self.right_margin)) #plus or minus x????? #1+ is oliver, sergey does not use it
    def weight_loss(self,W,beta = 10):
        return 1+torch.exp(beta*(W-self.W0)/self.W0) #oliver uses beta =1, sergey =10
    def __call__(self,phi,muons = None):
        if phi.dim()>1:
            y = []
            for p in phi:
                y.append(self(p))
            return torch.stack(y)
        px,py,pz,x,y,z,particle,W = self.simulate(phi,muons)
        x,y,z = self.propagate_to_sensitive_plane(px,py,pz,x,y,z)
        loss = self.muon_loss(x,y,particle).sum()+1
        if self.loss_with_weight:
            loss *= self.weight_loss(W)
            loss = torch.where(W>3E6,1e8,loss)
        return loss.to(torch.get_default_dtype())
    
    def propagate_to_sensitive_plane(self,px,py,pz,x,y,z):
        z += self.sensitive_plane
        x += self.sensitive_plane*px/pz
        y += self.sensitive_plane*py/pz
        return x,y,z

    @staticmethod
    def GetBounds(zGap:float = 1.,device = torch.device('cpu')):
        magnet_lengths = [(170 + zGap, 300 + zGap)] * 6  
        dX_bounds = [(10, 100)] * 2
        dY_bounds = [(20, 200)] * 2 
        gap_bounds = [(2, 70)] * 2 
        bounds = magnet_lengths + 6*(dX_bounds + dY_bounds + gap_bounds)
        bounds = torch.tensor(bounds,device=device,dtype=torch.get_default_dtype()).T
        bounds[0] = torch.minimum(bounds[0],ShipMuonShield.DEFAULT_PHI.to(device)).min(0).values
        bounds[1] = torch.maximum(bounds[1],ShipMuonShield.DEFAULT_PHI.to(device)).max(0).values
        return bounds
    @staticmethod
    def add_fixed_params(phi:torch.tensor):
        return torch.cat((torch.tensor([70.0, 170.0],device = phi.device), 
                         phi[:6],
                         torch.tensor([40.0, 40.0, 150.0, 150.0, 2.0, 2.0, 80.0, 80.0, 150.0, 150.0, 2.0, 2.0],device = phi.device), 
                         phi[6:]))
    
    def add_fixed_params(phi:torch.Tensor):
        # Fixed parameters, create them as tensors
        fixed_params_start = torch.tensor([70.0, 170.0], device=phi.device)
        fixed_params_middle = torch.tensor([40.0, 40.0, 150.0, 150.0, 2.0, 2.0, 80.0, 80.0, 150.0, 150.0, 2.0, 2.0], device=phi.device)
        if phi.dim() == 1:
            phi_start = phi[:6]
            phi_rest = phi[6:]
        else: 
            phi_start = phi[:, :6] 
            phi_rest = phi[:, 6:]
            fixed_params_start = fixed_params_start.unsqueeze(0).expand(phi.size(0), -1)
            fixed_params_middle = fixed_params_middle.unsqueeze(0).expand(phi.size(0), -1)
        return torch.cat((fixed_params_start, phi_start, fixed_params_middle, phi_rest), dim=-1)

class ShipMuonShieldCluster(ShipMuonShield):
    DEF_N_SAMPLES = 484449
    def __init__(self,
                 W0:float = 1558731.375,#1915820.,
                 cores:int = 96,
                 n_samples:int = 484449,
                 loss_with_weight:bool = True,
                 manager_ip='34.65.198.159',
                 port=444,
                 local:bool = False,
                 **kwargs) -> None:
        #super().__init__(W0 = W0,
        #                 n_samples=n_samples,
        #                cores = cores,
        #                loss_with_weight = loss_with_weight)

        self.W0 = W0
        self.cores = cores
        self.loss_with_weight = loss_with_weight
        if 0<n_samples<self.DEF_N_SAMPLES: self.n_samples = n_samples
        else: self.n_samples = self.DEF_N_SAMPLES

        self.manager_cert_path = getenv('STARCOMPUTE_MANAGER_CERT_PATH')
        self.client_cert_path = getenv('STARCOMPUTE_CLIENT_CERT_PATH')
        self.client_key_path = getenv('STARCOMPUTE_CLIENT_KEY_PATH')
        self.server_url = 'wss://%s:%s'%(manager_ip, port)

        self.local = local
        if not local:
            from starcompute.star_client import StarClient
            self.star_client = StarClient(self.server_url, self.manager_cert_path, 
                                    self.client_cert_path, self.client_key_path)

    def sample_x(self,phi = None):
        return get_split_indices(self.cores,self.n_samples)
    def simulate(self,phi:torch.tensor,muons = None):
        phi = phi.flatten() #Can we make it paralell on phi also?
        if len(phi) ==42: phi = self.add_fixed_params(phi)
        if muons is None: muons = self.sample_x(phi)
        #star_client = self.StarClient(self.server_url, self.manager_cert_path, 
        #                         self.client_cert_path, self.client_key_path) #redefine it every iteration?
        muons = split_array_idx(phi.cpu(),muons) #If we can pass phi previously (??), no need to do this 
        if self.local: result = call(f'srun python3 {PROJECTS_DIR}/cluster/muon_shield_worker_https.py', shell=True) #TODO
        else:  result = self.star_client.run(muons)
        result,W = torch.as_tensor(result,device = phi.device).T
        W = W.mean()
        result = result.sum()+1
        return result,W
    def __call__(self,phi,muons = None):
        if phi.dim()>1:
            y = []
            for p in phi:
                y.append(self(p))
            return torch.stack(y)
        loss,W = self.simulate(phi,muons)
        if self.loss_with_weight:
            loss *= self.weight_loss(W)
            loss = torch.where(W>3E6,1e8,loss)
        return loss.to(torch.get_default_dtype())    
    







import time
import argparse
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--nodes",type=int,default = 3)
    parser.add_argument("--n_tasks_per_node", type=int, default=96)
    parser.add_argument("--n_tasks", type=int, default=None)
    parser.add_argument("--warm", dest = 'SC', action='store_false')
    args = parser.parse_args()
    with open('/home/hep/lprate/projects/BlackBoxOptimization/outputs/complete_57_SC/phi_optm.txt', "r") as txt_file:
        data = [float(line.strip()) for line in txt_file]
    phi_sc = np.array(data)
    with open('/home/hep/lprate/projects/BlackBoxOptimization/outputs/gprbf_57_warm_5000/phi_1150.txt', "r") as txt_file:
        data = [float(line.strip()) for line in txt_file]
    phi_warm = np.array(data)
    
    d = {}
    for name,phi in {'optimal':ShipMuonShield.opt_oliver, 'reoptimized_warm':phi_warm,
                     'reoptimized_sc':phi_sc,'sc_v6':ShipMuonShield.sc_v6,
                     'combi':ShipMuonShield.combi, 'baseline':ShipMuonShield.baseline_1,}.items(): #
        t1 = time.time()
        
        if name == 'sc_v6' and not args.SC: continue
        if args.n_tasks is None: n_tasks = args.nodes*args.n_tasks_per_node
        else: n_tasks = args.n_tasks
        #muon_shield = ShipMuonShieldCluster(cores = n_tasks)
        muon_shield = ShipMuonShield(cores = n_tasks,fSC_mag=args.SC,sensitive_plane=57, input_dist = 0.1)

        #loss_muons, W = muon_shield.simulate(torch.as_tensor(phi))
        px,py,pz,x,y,z,particle,W = muon_shield.simulate(torch.as_tensor(phi))
        x,y,z = muon_shield.propagate_to_sensitive_plane(px,py,pz,x,y,z)
        loss_muons = muon_shield.muon_loss(x,y,particle).sum()+1
        loss = loss_muons*muon_shield.weight_loss(W)
        loss = torch.where(W>3E6,1e8,loss)
        d[name] = (W,loss_muons,loss)
        t2 = time.time()
    for name,(W,loss_muons,loss) in d.items():
        print(f"{name}:")
        print(f"Weight: {W.item()}")
        print(f"Muon Loss: {loss_muons.item()}")
        print(f"Total Loss: {loss.item()}")
    
    