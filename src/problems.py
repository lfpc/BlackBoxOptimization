import torch
import gzip
import h5py
import pickle
import sys
import numpy as np
import os
from multiprocessing import Pool, cpu_count
PROJECTS_DIR = os.getenv('PROJECTS_DIR', '~/projects')
sys.path.insert(1, os.path.join(PROJECTS_DIR,'BlackBoxOptimization'))
from utils import split_array, split_array_idx, get_split_indices, compute_solid_volume
import logging
import json
from functools import partial
logging.basicConfig(level=logging.WARNING)
import time
#torch.set_default_dtype(torch.float64)


def uniform_sample(shape,bounds,device = 'cpu'):
    return (bounds[0]-bounds[1])*torch.rand(shape,device=device)+bounds[1]



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

    parametrization = {'HA': [0, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
                       'M1': [1, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32],
                       'M2': [2, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45], #SC
                       'M3': [3, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58],
                       'M4': [4, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71],
                       'M5': [5, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84],
                       'M6': [6, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97]}
    idx_mag = {0:'Z_len[cm]',
                  1:'dXIn[cm]', 2:'dXOut[cm]', 
                  3:'dYIn[cm]', 4:'dYOut[cm]', 
                  5:'gapIn[cm]', 6:'gapOut[cm]', 
                  7:'ratio_yokesIn', 8:'ratio_yokesOut',
                  9:'dY_yokeIn[cm]', 10: 'dY_yokeOut[cm]',
                  11:'XmgapIn[cm]', 12:'XmgapOut[cm]',
                  13:'NI[A]'}
    
    SC_Ymgap = 15
    
    warm_baseline = [120.50, 250, 250., 250., 150., 150., 251, 
                  50.00,  50.00, 119.00, 119.00,   2.00,   2.00, 1.00,1.0,50.00,  50.00,0.0, 0.00, 1.9,
                  72.00, 51.00, 29.00, 46.00, 10.00, 7.00, 1.00,1.0,72.00, 51.00,0.0, 0.00, 1.9,
                  54.00, 38.00, 46.00, 130.00, 14.00, 9.00, 1.00,1.0,54.00, 38.00,0.0, 0.00, 1.9,
                  10.00, 31.00, 35.00, 31.00, 51.00, 11.00, 1.00,1.0,10.00, 31.00,0.0, 0.00, 1.9,
                  5.00, 32.00, 54.00, 24.00, 8.00, 8.00, 1.00,1.0,5.00, 32.00,0.0, 0.00, -1.9,
                  22.00, 32.00, 130.00, 35.00, 8.00, 13.00, 1.00,1.0,22.00, 32.00,30.0, 30.00, -1.9,
                  33.00, 77.00, 85.00, 90.00, 9.00, 26.00, 1.00,1.0,33.00, 77.00,5.0, 5.00, -1.9]
    
    hybrid_baseline = [120.50, 35, 350., 125., 200., 200., 195., 
                    50.00,  50.00, 119.00, 119.00,   2.00,   2.00, 1.00,1.0,50.00,  50.00,0.0, 0.00, 1.9,
                    0.,  0.,  0.,  0.,  0.,   0., 1.,1.0,0.,0.,0.0, 0., 0.0,
                    45.,  45.,  25.,  25.,  35.,  35., 2.67,2.67,120.15,120.15,0.0, 0.00, 5.7,
                    0.,  0.,  0.,  0.,  0.,  0., 1.,1.0,0.,0.,0.0, 0., 0.,
                    5.263,55.632, 39.860, 5.278, 2.000,2.000, 1.000, 0.900, 5.263,50.069, 0.100, 0.100, -1.9,
                    33.016, 12.833, 123.270, 78.523,22.414, 2.001, 0.959, 0.953,31.663, 12.233, 30., 30.,-1.9, 
                    30., 40., 69.398,152.828, 2.000, 35.402, 1.000,0.900, 15.755, 69.564, 0.0,0.0, -1.9]

    tokanut_v5 = [
        120.50, 230.38, 279.43, 289.85, 119.58, 170.53, 249.28,
        50.00, 50.00, 119.00, 119.00, 2.00, 2.00, 1.00, 1.00, 50.00, 50.00, 0.00, 0.00, 1.9,
        52.25, 78.36, 6.54, 9.37, 2.04, 40.36, 1.01, 1.05, 52.88, 82.20, 0.00, 0.00, 1.9,
        52.47, 11.70, 41.59, 79.05, 2.19, 2.01, 1.03, 1.00, 54.15, 11.70, 0.00, 0.00, 1.9,
        33.05, 24.10, 55.13, 30.61, 90.36, 2.00, 1.00, 1.00, 33.05, 24.10, 0.11, 0.11, 1.9,
        5.62, 60.38, 31.37, 5.00, 2.00, 2.86, 1.00, 0.74, 5.59, 44.82, 0.00, 0.00, -1.9,
        90.85, 7.44, 137.45, 66.69, 10.09, 2.00, 0.88, 1.00, 80.16, 7.41, 0.00, 0.00, -1.9,
        9.81, 47.60, 19.59, 164.19, 2.00, 2.05, 1.00, 0.89, 9.81, 42.16, 0.10, 0.10, -1.9
    ]

    tokanut_v5_SND = [
        120.50, 230.38, 279.43, 289.85, 119.58, 170.53, 249.28,
        50.00, 50.00, 119.00, 119.00, 2.00, 2.00, 1.00, 1.00, 50.00, 50.00, 0.00, 0.00, 1.9,
        52.25, 78.36, 6.54, 9.37, 2.04, 40.36, 1.01, 1.05, 52.88, 82.20, 0.00, 0.00, 1.9,
        52.47, 11.70, 41.59, 79.05, 2.19, 2.01, 1.03, 1.00, 54.15, 11.70, 0.00, 0.00, 1.9,
        33.05, 24.10, 55.13, 30.61, 90.36, 2.00, 1.00, 1.00, 33.05, 24.10, 0.11, 0.11, 1.9,
        5.62, 60.38, 31.37, 5.00, 2.00, 2.86, 1.00, 0.74, 5.59, 44.82, 0.00, 0.00, -1.9,
        90.85, 7.44, 137.45, 66.69, 10.09, 2.00, 0.88, 1.00, 80.16, 7.41, 30.00, 30.00, -1.9,
        30., 47.60, 19.59, 164.19, 2.00, 2.05, 1.00, 0.89, 9.81, 42.16, 0.10, 0.10, -1.9
    ]

    sc_v6 = [231.00,  0., 353.08, 125.08, 184.83, 150.19, 186.81, 
         50.00,  50.00, 119.00, 119.00,   2.00,   2.00, 1.00,1.0,50.00,  50.00,0.0, 0.00, 0,
        0.,  0.,  0.,  0.,  0.,   0., 1.,1.0,0.,0.,0.0, 0.,0.,
        45.69,  45.69,  22.18,  22.18,  27.01,  16.24, 3.00,3.0,137.1,137.1,0.0, 0.00, 3200000.0,
        0.,  0.,  0.,  0.,  0.,  0., 1.,1.0,0.,0.,0.0, 0., 0.,
        24.80,  48.76,   8.00, 104.73,  15.80,  16.78, 1.00,1.0,24.80,  48.76,0.0, 0.00,0,
        3.00, 100.00, 192.00, 192.00,   2.00,   4.80, 1.00,1.0,3.00, 100.00,0.0, 0.00, 0,
        3.00, 100.00,   8.00, 172.73,  46.83,   2.00, 1.00,1.0,3.00, 100.00,0.0, 0.00, 0]
    
    supernut_v2 =  [
    120.5, 31.6, 263.52, 273.98, 249.02, 177.17, 172.25,
    50.0, 50.0, 119.0, 119.0, 2.0, 2.0, 1.0, 1.0, 50.0, 50.0, 0.0, 0.0, 1.9, 
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0,1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
    45.0, 45.0, 25.0, 25.0, 65.8, 50.65, 2.44, 3.29,107.83, 120.28, 0.0, 0.0, 5.7, 
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0,0.0, 0.0, 0.0, 0.0, 
    5.39, 21.56, 29.39, 17.26, 68.87, 65.04,0.94, 0.84, 5.26, 50.07, 77.41, 77.41, -1.9,
    34.2, 8.87, 100.69, 32.75, 148.15, 38.3,0.99, 0.65, 31.66, 12.23, 51.04, 51.04, -1.9,
    21.44, 62.75, 197.1, 143.48, 6.63, 22.1,0.94, 0.94, 15.76, 69.56, 0.06, 0.06, -1.9]

    Piet_solution = [120.5, 485.5, 245.0,  0., 255.0, 197.0, 197.0, 
    50.000, 50.000, 119.000, 119.000, 2.000, 2.000, 1.000, 1.000, 50.000, 50.000, 0.000, 0.000, 1.9,
    30.0, 31.0, 27.0, 43.0, 6.0, 6.0, 4.29, 4.290322580645161, 34.0,34.0, 0.0, 0.0, 1.9, 
    31.0, 35.0, 43.0, 56.0, 6.0, 6.0, 4.290322580645161, 3.7857142857142856, 44.0,44.0, 0.0, 0.0, 1.9, 
    0., 0., 0., 0.0, 0.0, 0.0, 1., 1., 1.0,1.0, 0., 0., 0.,
    3.0, 18.6, 56.0, 56.0, 6.0, 6.0,  54.8, 8.2, 44.0,44.0, 0.0, 0.0, -1.7, 
    18.6, 31.8, 56.0, 56.0, 6.0, 5.9999999999999964, 8.204301075268816, 4.471698113207546, 44.0,44.0, 30.0, 30.0, -1.7, 
    31.8, 45.0, 56.0, 56.0, 5.9999999999999964, 6.0, 4.471698113207546, 2.8666666666666667, 44.0,44.0, 0.0, 0.0, -1.7 ]

    

    
    hybrid_idx = (np.array(parametrization['M2'])[[0,1,3, 5, 6, 7,8,9,10]]).tolist() + \
                      [parametrization['M1'][0]]+ [parametrization['M3'][0]]+\
                      parametrization['M4'][:9] + parametrization['M4'][12:13] + \
                      parametrization['M5'][:9] + parametrization['M5'][12:13] + \
                      parametrization['M6'][:9] + parametrization['M6'][12:13]
    
    warm_idx =        parametrization['M1'][:9] + parametrization['M1'][12:13] + \
                      parametrization['M2'][:9] + parametrization['M2'][12:13] + \
                      parametrization['M3'][:9] + parametrization['M3'][12:13] + \
                      parametrization['M4'][:9] + parametrization['M4'][12:] + \
                      parametrization['M5'][:9] + parametrization['M5'][12:] + \
                      parametrization['M6'][:9] + parametrization['M6'][12:]

    piet_idx = parametrization['M1'][:3] + parametrization['M1'][5:8] + \
                      parametrization['M2'][:3] + parametrization['M2'][5:8] + \
                      parametrization['M4'][:3]  + parametrization['M4'][5:8] + parametrization['M4'][12:] + \
                      parametrization['M5'][:3]  + parametrization['M5'][5:8] + parametrization['M5'][12:] + \
                      parametrization['M6'][:3]  + parametrization['M6'][5:8] + parametrization['M6'][12:]
    
    
    DEFAULT_PHI = torch.tensor(warm_baseline)
    initial_phi = DEFAULT_PHI.clone()
    full_dim = 98
    MUON = 13

    def __init__(self,
                 W0:float = 11E6,
                 L0:float = 29.7,
                 cores:int = 45,
                 n_samples:int = 0,
                 input_dist:float = None,
                 sensitive_plane:float = [{'dz': 0.01, 'dx': 4, 'dy': 6,'position': 82}],
                 apply_det_loss:bool = True,
                 cost_loss_fn:bool = 'exponential',
                 fSC_mag:bool = False,
                 simulate_fields:bool = False,
                 cavern:bool = True,
                 seed:int = None,
                 left_margin = 2.1,
                 right_margin = 2.1,
                 y_margin = 3,
                 SmearBeamRadius = 5,
                 dimensions_phi = 98,
                muons_file = os.path.join(PROJECTS_DIR,'MuonsAndMatter/data/muons/subsample_biased_v4.npy'),
                fields_file = None,
                extra_magnet = False,
                cut_P:float = None,
                default_phi:torch.tensor = None,
                multi_fidelity:float = False,
                SND:bool = False,
                decay_vessel_sensitive:bool = False,
                use_diluted:bool = False,
                parallel:bool = False,
                use_B_goal:bool = True,
                 ) -> None:
        
        self.left_margin = left_margin
        self.right_margin = right_margin
        self.y_margin = y_margin
        self.W0 = W0
        self.L0 = L0
        self.cores = cores
        self.muons_file = muons_file
        self.n_samples = n_samples
        if n_samples==0: self.n_samples = self.sample_x(muons_file = muons_file).shape[0]
        self.input_dist = input_dist
        self.cost_loss_fn = cost_loss_fn
        self.sensitive_plane = sensitive_plane
        self.fSC_mag = fSC_mag
        self.simulate_fields = simulate_fields
        self.seed = seed  
        self.cavern = cavern
        self.apply_det_loss = apply_det_loss
        self.extra_magnet = extra_magnet    
        self.SmearBeamRadius = SmearBeamRadius
        self.lambda_constraints = 50
        self.cut_P = cut_P
        self.use_B_goal = use_B_goal
        self.SND = SND
        self.decay_vessel_sensitive = decay_vessel_sensitive
        self.use_diluted = use_diluted
        self.parallel = parallel

        if default_phi is not None:
            self.DEFAULT_PHI = torch.as_tensor(default_phi)
        if isinstance(dimensions_phi,list):
            self.params_idx = dimensions_phi
        elif dimensions_phi == len(self.hybrid_idx): self.params_idx = self.hybrid_idx
        elif dimensions_phi == len(self.warm_idx): self.params_idx = self.warm_idx
        elif dimensions_phi == len(self.piet_idx): self.params_idx = self.piet_idx
        elif dimensions_phi == self.full_dim: self.params_idx = slice(None)
        self.initial_phi = self.DEFAULT_PHI[self.params_idx]
        self.dimensions_phi = len(self.initial_phi)

        self.materials_directory = os.path.join(PROJECTS_DIR,'MuonsAndMatter/data/materials')
        sys.path.insert(1, os.path.join(PROJECTS_DIR,'MuonsAndMatter'))
        sys.path.insert(1, os.path.join(PROJECTS_DIR,'MuonsAndMatter/python/lib'))
        from run_simulation import run, get_field
        from ship_muon_shield_customfield import estimate_electrical_cost, RESOL_DEF
        self.estimate_electrical_cost = estimate_electrical_cost
        self.run_muonshield = run
        self.run_magnet = get_field
        self.resol = RESOL_DEF
        self.fields_file = fields_file
        self.multi_fidelity = multi_fidelity

    def sample_x(self,phi=None, muons_file = None, idx:tuple = None):
        muons_file = muons_file if muons_file is not None else self.muons_file
        idx = slice(idx[0], idx[1]) if idx is not None else slice(None)
        if muons_file.endswith('.npy'):
            x = np.load(muons_file)[idx]
        elif muons_file.endswith('.h5'):
            with h5py.File(muons_file, 'r') as f:
                px = f['px'][idx]
                py = f['py'][idx]
                pz = f['pz'][idx]
                x_ = f['x'][idx]
                y_ = f['y'][idx]
                z_ = f['z'][idx]
                pdg = f['pdg'][idx]
                weight = f['weight'][idx]
                x = np.stack([px, py, pz, x_, y_, z_, pdg, weight], axis=1)
        else:
            with gzip.open(muons_file, 'rb') as f:
                x = pickle.load(f)[idx]
        if 0<self.n_samples<x.shape[0]: 
            indices = np.random.choice(x.shape[0], self.n_samples, replace=False)
            x = x[indices]
        return x
    def simulate_mag_fields(self,phi:torch.tensor, cores:int = 7):
        phi = self.add_fixed_params(phi)
        Z = phi[0:7].sum().item()*2/100 + 0.1
        max_x = 0
        max_y = 0  
        for m,idx in self.parametrization.items():
            params = phi[idx]
            dXIn, dXOut, dYIn, dYOut, gapIn, gapOut, ratio_yokesIn, ratio_yokesOut,dY_yokeIn, dY_yokeOut, XmgapIn,XmgapOut,NI = params[1:]
            dX = max(torch.max(dXIn + dXIn * ratio_yokesIn + gapIn+XmgapIn).item(), torch.max(dXOut + dXOut*ratio_yokesOut+gapOut+XmgapOut).item())/100
            if dX > max_x:
                max_x = dX
            dY = max(torch.max(dYIn + dY_yokeIn).item(), torch.max(dYOut + dY_yokeOut).item())/100
            if dY > max_y:
                max_y = dY
        max_x = float(np.round(max_x,decimals=1).item())
        max_y = float(np.round(max_y,decimals=1).item())
        d_space = ((0,max_x+0.3, float(self.resol[0])), (0,max_y+0.3, float(self.resol[1])), (-0.5, int(np.ceil(Z+0.5).item()), float(self.resol[2])))
        self.run_magnet(True,phi.cpu().numpy(),file_name = self.fields_file,d_space = d_space, cores = cores, fSC_mag = self.fSC_mag, use_diluted = self.use_diluted,NI_from_B_goal = self.use_B_goal)

    def simulate(self,phi:torch.tensor,muons = None, return_nan = False): 
        phi = self.add_fixed_params(phi)
        if muons is None: muons = self.sample_x()
        workloads = split_array(muons,self.cores)
        assert phi.numel()==98, f"{len(workloads), workloads[0].shape}"
        if self.simulate_fields: 
            print('SIMULATING MAGNETIC FIELDS')
            self.simulate_mag_fields(phi)
        run_partial = partial(self.run_muonshield, 
                      params=phi.cpu().numpy(), 
                      input_dist=self.input_dist, 
                      return_cost=False, 
                      fSC_mag=self.fSC_mag, 
                      sensitive_film_params=self.sensitive_plane, 
                      add_cavern=self.cavern, 
                      simulate_fields=self.simulate_fields, 
                      field_map_file=self.fields_file, 
                      return_nan=return_nan, 
                      seed=self.seed, 
                      draw_magnet=False, 
                      SmearBeamRadius=self.SmearBeamRadius, 
                      add_target=True, 
                      keep_tracks_of_hits=False, 
                      extra_magnet=self.extra_magnet,
                      NI_from_B = self.use_B_goal,
                     add_decay_vessel = self.decay_vessel_sensitive,
                      use_diluted = self.use_diluted)
        with Pool(self.cores) as pool:
            result = pool.map(run_partial, workloads)
        print('SIMULATION FINISHED')
        all_results = []
        for rr in result:
            resulting_data = rr
            if resulting_data.size == 0: continue
            all_results += [resulting_data]
        if len(all_results) == 0:
            all_results = [[np.nan]*8]
        all_results = np.concatenate(all_results, axis=0).T
        if all_results.dtype != object: # Only convert to tensor if all_results is a numeric array (not array of dicts)
            all_results = torch.as_tensor(all_results, device=phi.device, dtype=torch.get_default_dtype())
        return all_results
    def is_hit(self,px,py,pz,x,y,z,particle,factor = None):
        p = torch.sqrt(px**2+py**2+pz**2)
        mask = (torch.abs(x) <= self.left_margin) & (torch.abs(y) <= self.y_margin)
        mask = mask & (torch.abs(particle).to(torch.int)==self.MUON)
        if self.cut_P is not None: mask = mask & p.ge(self.cut_P)
        return mask.to(torch.bool)
    def muon_loss(self,px,py,pz,x,y,z,particle, weight = None):
        charge = -1*torch.sign(particle)
        mask = self.is_hit(px,py,pz,x,y,z,particle).to(torch.bool) 
        assert mask.shape == x.shape, f"MASK SHAPE: {mask.shape}, X SHAPE: {x.shape}"
        x = x[mask]
        charge = charge[mask]
        loss = torch.sqrt(1 + (charge*x-self.right_margin)/(self.left_margin+self.right_margin)) 
        if weight is not None:
            weight = weight[mask]
            loss = loss * weight
        return loss
    def get_total_length(self, phi):
        phi = self.add_fixed_params(phi)
        phi = phi.view(-1,self.full_dim)
        length = torch.zeros(phi.size(0), device=phi.device)
        for m, idx in self.parametrization.items():
            params = phi[:, idx]
            length = length + params[:, 0]
        return 2 * length / 100

    def get_electrical_cost(self,phi):
        '''Adapt for multidimensional phi'''
        phi = self.add_fixed_params(phi).detach().cpu()
        cost = 0
        for m,idx in self.parametrization.items():
            params = phi[idx].numpy()
            if params[0]<1 or params[1]<1: continue
            if self.fSC_mag and m  == 'M2':
                Ymgap = self.SC_Ymgap; yoke_type = 'Mag2'
            elif self.use_diluted: Ymgap = 0.; yoke_type = 'Mag1'
            else: 
                Ymgap = 0.; 
                yoke_type = 'Mag3' if params[13]<0 else 'Mag1'
            cost+= self.estimate_electrical_cost(params,yoke_type,Ymgap,materials_directory = self.materials_directory, NI_from_B = self.use_B_goal)
        return cost

    def get_iron_cost(self, phi,  zGap= 10):
        '''Adapt for multidiomensional phi
        make electrical cost esimation with torch'''
        material =  'aisi1010.json'#magnet_simulations.get_fixed_params()['material']
        with open(os.path.join(self.materials_directory,material)) as f:
            iron_material_data = json.load(f)
        density = iron_material_data['density(g/m3)']*1E-9
        phi = self.add_fixed_params(phi).view(-1) # change this for multi-dim phi
        volume = 0#torch.zeros(phi.size(0), device=phi.device)
        for m,idx in self.parametrization.items():
            Ymgap = self.SC_Ymgap if self.fSC_mag and m == 'M2' else 0
            if self.fSC_mag and m in ['M1', 'M3']: continue
            params = phi[idx]
            dZ = params[0] - zGap/2
            dX = params[1]
            dX2 = params[2]
            dY = params[3]
            dY2 = params[4]
            gap = params[5]
            gap2 = params[6]
            ratio_yoke_1 = params[7]
            ratio_yoke_2 = params[8]
            dY_yoke_1 = params[9]
            dY_yoke_2 = params[10]
            X_mgap_1 = params[11]
            X_mgap_2 = params[12]
            corners = torch.tensor([
            [X_mgap_1+dX, 0, 0],
            [X_mgap_1 + dX, dY, 0],
            [0, dY, 0],
            [0, 0, 0],
            [X_mgap_2+dX2,0, 2*dZ],
            [X_mgap_2+dX2, dY2, 2*dZ],
            [0, dY2, 2*dZ],
            [0, 0, 2*dZ]
            ])
            volume += compute_solid_volume(corners)
            corners = torch.tensor([
            [X_mgap_1 + dX + gap, 0, 0],
            [X_mgap_1 + dX + gap + dX * ratio_yoke_1, 0, 0],
            [X_mgap_1 + dX + gap + dX * ratio_yoke_1, dY + Ymgap, 0],
            [X_mgap_1 + dX + gap, dY + Ymgap, 0],
            [X_mgap_2 + dX2 + gap2, 0, 2 * dZ],
            [X_mgap_2 + dX2 + gap2 + dX2 * ratio_yoke_2, 0, 2 * dZ],
            [X_mgap_2 + dX2 + gap2 + dX2 * ratio_yoke_2, dY2 + Ymgap, 2 * dZ],
            [X_mgap_2 + dX2 + gap2, dY2 + Ymgap, 2 * dZ],
            ])
            volume += compute_solid_volume(corners)

            corners = torch.tensor([
            [X_mgap_1, dY, 0],
            [X_mgap_1 + dX + gap + dX * ratio_yoke_1, dY, 0],
            [X_mgap_1 + dX + gap + dX * ratio_yoke_1, dY + dY_yoke_1, 0],
            [X_mgap_1, dY + dY_yoke_1, 0],
            [X_mgap_2, dY2, 2 * dZ],
            [X_mgap_2 + dX2 + gap2 + dX2 * ratio_yoke_2, dY2, 2 * dZ],
            [X_mgap_2 + dX2 + gap2 + dX2 * ratio_yoke_2, dY2 + dY_yoke_2, 2 * dZ],
            [X_mgap_2, dY2 + dY_yoke_2, 2 * dZ],
            ])
            volume += compute_solid_volume(corners)
        M_iron = 4*volume*density    
        C_iron = M_iron*(iron_material_data["material_cost(CHF/kg)"]
                     +  iron_material_data["manufacturing_cost(CHF/kg)"])
        return C_iron.detach()
    def get_total_cost(self,phi):
        try:
            M = self.get_iron_cost(phi)
            if self.simulate_fields or self.fields_file is not None: M += self.get_electrical_cost(phi)
        except Exception as e:
            print(f"Error in cost estimation of parameters: {phi}")
            print(f"Error: {e}")
            raise
        return M

    def cost_loss(self,W,L = None):
        if self.cost_loss_fn == 'exponential':
            return (1+torch.exp(10*(W-self.W0)/self.W0))
        elif self.cost_loss_fn == 'linear':
            return W/self.W0
        elif self.cost_loss_fn == 'quadratic':
            return (W/self.W0)**2
        elif self.cost_loss_fn == 'linear_length':
            return W/(1-L/self.L0)
        else: return 1
    def calc_loss(self,px,py,pz,x,y,z,particle,factor,W = None, L = None):
        loss = self.muon_loss(px,py,pz,x,y,z,particle,factor).sum()
        if self.apply_det_loss: loss = self.deterministic_loss(phi,y)
        return loss.to(torch.get_default_dtype()).view(-1,1)
    def __call__(self,phi,muons = None):
        L = self.get_total_length(phi)
        M = self.get_total_cost(phi)
        #if L>=self.L0: return torch.tensor([1e8])
        #if W>=3E6: return torch.tensor([1e8])
        if phi.dim()>1:
            y = []
            for p in phi:
                y.append(self(p))
            return torch.stack(y)
        
        return self.calc_loss(*self.simulate(phi,muons),M,L)
    
    def propagate_to_sensitive_plane(self,px,py,pz,x,y,z, epsilon = 1e-12):
        sensitive_plane = self.sensitive_plane['position']
        z += sensitive_plane
        x += sensitive_plane*px/(pz+epsilon)
        y += sensitive_plane*py/(pz+epsilon)
        return x,y,z

    def GetBounds(self,device = torch.device('cpu'), correct_bounds = True):
        dY_bounds = [(4, 160)] * 2 
        if self.use_diluted:
            magnet_lengths = [(self.Piet_solution[0], 500), (self.Piet_solution[1], 500), (self.Piet_solution[2], 600), (30, 400), (30, 400), (30, 400), (30, 400)]
            dX_bounds = [(1, 85)] * 2
            gap_bounds = [(5, 80)] * 2 
            inner_gap_bounds = [(0., 30.)]*2
            yoke_bounds = [(1,60)]*2
        else:
            magnet_lengths = [(50, 350)] * 7  #previously 170-300
            dX_bounds = [(5, 250)] * 2
            gap_bounds = [(2, 150)] * 2
            yoke_bounds = [(0.99,3)]*2#[(0.25, 4)]
            inner_gap_bounds = [(0., 150.)]*2
        dY_yoke_bounds = [(5, 160)]*2
        NI_bounds = [(0.1,1.91)]
        bounds = magnet_lengths + 2*(dX_bounds + dY_bounds + gap_bounds + yoke_bounds + dY_yoke_bounds + inner_gap_bounds + NI_bounds)
        dY_yoke_bounds = [(4, 130)]*2 if self.fSC_mag else [(4, 300)]*2
        dY_bounds = [(5, 250)] * 2 
        bounds += 2*(dX_bounds + dY_bounds + gap_bounds + yoke_bounds + dY_yoke_bounds + inner_gap_bounds + NI_bounds)
        if not self.use_diluted: yoke_bounds = [(0.3,1.01)]*2
        NI_bounds = [(-1.91, -0.1)]
        bounds += 3*(dX_bounds + dY_bounds + gap_bounds + yoke_bounds + dY_yoke_bounds + inner_gap_bounds + NI_bounds)

        if self.SND:
            bounds[self.parametrization['M6'][0]] = (170, 350)
            bounds[self.parametrization['M6'][1]] = (30, 250)
            bounds[self.parametrization['M6'][2]] = (40, 250)
            bounds[self.parametrization['M5'][12]] = (30, 150)
            bounds[self.parametrization['M5'][11]] = (30, 150)
            bounds[self.parametrization['M5'][0]] = (90, 350)
        if self.fSC_mag: 
            bounds[self.parametrization['M1'][0]] = (30,100)
            bounds[self.parametrization['M3'][0]] = (30,300)
            bounds[self.parametrization['M2'][0]] = (50,400)
            bounds[self.parametrization['M2'][1]] = (30,50)
            bounds[self.parametrization['M2'][2]] = (30,50)
            bounds[self.parametrization['M2'][3]] = (15,30)
            bounds[self.parametrization['M2'][4]] = (15,30)
            bounds[self.parametrization['M2'][5]] = (15,150)
            bounds[self.parametrization['M2'][6]] = (15,150)
            bounds[self.parametrization['M2'][7]] = (1.0,4)
            bounds[self.parametrization['M2'][8]] = (1.0,4)
        bounds = torch.tensor(bounds,device=device,dtype=torch.get_default_dtype()).T
        bounds = bounds[:,self.params_idx]
        return bounds

    def add_fixed_params(self, phi: torch.Tensor):
        if phi.dim() == 1:
            phi = phi.unsqueeze(0)
        
        def place(target_idx, source_idx_or_value):
            """Internal helper to scatter values while preserving gradients"""
            nonlocal new_phi
            if isinstance(source_idx_or_value, int):
                values = new_phi[:, source_idx_or_value].unsqueeze(1)
            else:
                values = source_idx_or_value.unsqueeze(1) if source_idx_or_value.dim() == 1 else source_idx_or_value
            idx_tensor = torch.tensor([target_idx], device=phi.device).unsqueeze(0).expand(phi.size(0), -1)
            new_phi = new_phi.scatter(1, idx_tensor, values)
        
        if phi.size(-1) != self.full_dim:
            assert phi.squeeze().size(-1) == len(self.params_idx), f"INPUT SHAPE: {phi.shape}"
            new_phi = self.DEFAULT_PHI.clone().to(phi.device).repeat(phi.size(0), 1)
            new_phi[:, torch.as_tensor(self.params_idx, device=phi.device)] = phi
            
            # HA parametrization constraints
            place(self.parametrization['HA'][2], self.parametrization['HA'][1])
            place(self.parametrization['HA'][4], self.parametrization['HA'][3])
            
            if self.fSC_mag:
                place(self.parametrization['M2'][2], self.parametrization['M2'][1])
                place(self.parametrization['M2'][4], self.parametrization['M2'][3])
            
            if self.dimensions_phi == len(self.warm_idx):
                for m, idx in self.parametrization.items():
                    # Fix dY_yoke = dX_core*ratio_yoke
                    place(idx[9], new_phi[:, idx[1]] * new_phi[:, idx[7]])
                    place(idx[10], new_phi[:, idx[2]] * new_phi[:, idx[8]])
                    # fix Xmgap_1 = Xmgap_2
                    place(idx[11], idx[12])
                    
            elif self.dimensions_phi == len(self.hybrid_idx):
                for m, idx in self.parametrization.items():
                    place(idx[11], idx[12])
                    
            if self.dimensions_phi == len(self.piet_idx):
                for m, idx in self.parametrization.items():
                    if m == 'HA' or m == 'M3': 
                        continue
                    
                    place(idx[11], idx[12])
                    place(idx[9], idx[1])
                    place(idx[10], idx[2])
                    
                    values_7 = ((self.Piet_solution[idx[1]] * self.Piet_solution[idx[7]] + self.Piet_solution[idx[5]] + self.Piet_solution[idx[1]]) - (new_phi[:, idx[11]] + new_phi[:, idx[5]] + new_phi[:, idx[1]])) / new_phi[:, idx[1]]
                    place(idx[7], values_7)
                    
                    values_8 = (self.Piet_solution[idx[2]] * self.Piet_solution[idx[8]] + self.Piet_solution[idx[6]] + self.Piet_solution[idx[2]] - new_phi[:, idx[12]] - new_phi[:, idx[6]] - new_phi[:, idx[2]]) / new_phi[:, idx[2]]
                    place(idx[8], values_8)
        else:
            new_phi = phi
        
        assert new_phi.size(-1) == self.full_dim, f"FINAL SHAPE: {new_phi.shape}"
        return new_phi.squeeze(0) if phi.size(0) == 1 else new_phi
        
    
    def deterministic_loss(self,phi,y):
        y = y.view(-1,1)
        M = self.get_total_cost(phi)
        loss = self.cost_loss(M)*(y+1)
        loss = loss + self.get_constraints(phi)
        loss = loss.clamp(max=1E6)#soft_clamp(loss,1.E8)
        return loss
    
    def get_constraints(self,phi):
        def fn_pen(x): return torch.nn.functional.relu(x,inplace=False).pow(2)
        phi = self.add_fixed_params(phi)
        phi = phi.view(-1,self.full_dim)
        constraints = fn_pen((self.get_total_length(phi)-self.L0)*100)
        #return (constraints.reshape(-1,1)*self.lambda_constraints)#.clamp(max=1E8)
        #cavern constraints
        wall_gap = 1
        def get_cavern_bounds(z):
            x_min = torch.zeros_like(z)
            y_min = torch.zeros_like(z)
            mask = z <= 2051.8-214.0#234.5
            x_min[mask] = 356
            y_min[mask] = 170
            x_min[~mask] = 456
            y_min[~mask] = 336
            x_min -= wall_gap
            y_min -= wall_gap
            return x_min, y_min
        z = torch.zeros(phi.size(0),device=phi.device)
        for m,idx in self.parametrization.items():
            p = phi[:,idx]
            z = z + 2*p[:,0]
            Ymgap = self.SC_Ymgap if (self.fSC_mag and m =='M2') else 0
            with torch.no_grad(): x_min, y_min = get_cavern_bounds(z-2*p[:,0])
            constraints = constraints + fn_pen(p[:,1]+p[:,7]*p[:,1]+p[:,5]+p[:,11]-x_min)
            constraints = constraints + fn_pen(p[:,3]+p[:,9]+Ymgap - y_min)
            with torch.no_grad(): x_min, y_min = get_cavern_bounds(z)
            constraints = constraints + fn_pen(p[:,2]+p[:,8]*p[:,2]+p[:,6]+p[:,12] -x_min)
            constraints = constraints + fn_pen(p[:,4]+p[:,10]+Ymgap - y_min)
            if self.use_diluted:
                constraints = constraints + fn_pen((1-p[:,7])) 
                constraints = constraints + fn_pen((1-p[:,8]))
            #assert False, constraints
        return (constraints.reshape(-1,1)*self.lambda_constraints).clamp(min=0,max=1E8)

    def get_constraints_func(self, phi):
        """
        This is a class method that calculates all problem constraints.
        It takes a NumPy array (from SciPy) and returns a NumPy array where
        each element represents a constraint in the form `g(x) >= 0`.
        """
        
        def get_cavern_bounds(z):
            x_min = torch.zeros_like(z)
            y_min = torch.zeros_like(z)
            wall_gap = 1
            
            mask = z <= 2051.8 - 214.0
            x_min[mask] = 356
            y_min[mask] = 170
            x_min[~mask] = 456
            y_min[~mask] = 336
            x_min -= wall_gap
            y_min -= wall_gap
            return x_min, y_min

        #phi = torch.from_numpy(phi).float()
        
        phi = self.add_fixed_params(phi)
        phi = phi.view(1, -1) 
        
        constraint_values = []
        length_constraint = self.L0 - self.get_total_length(phi)
        constraint_values.append(length_constraint)

        z = torch.zeros(phi.size(0))
        for m, idx in self.parametrization.items():
            p = phi[:, idx]
            z = z + 2 * p[:, 0]
            Ymgap = self.SC_Ymgap if (self.fSC_mag and m == 'M2') else 0

            with torch.no_grad():
                x_min_g1, y_min_g1 = get_cavern_bounds(z - 2 * p[:, 0])
            c1 = x_min_g1 - (p[:, 1] + p[:, 7] * p[:, 1] + p[:, 5] + p[:, 11])
            constraint_values.append(c1)
            c2 = y_min_g1 - (p[:, 3] + p[:, 9] + Ymgap)
            constraint_values.append(c2)
            with torch.no_grad():
                x_min_g2, y_min_g2 = get_cavern_bounds(z)
            c3 = x_min_g2 - (p[:, 2] + p[:, 8] * p[:, 2] + p[:, 6] + p[:, 12])
            constraint_values.append(c3)
            c4 = y_min_g2 - (p[:, 4] + p[:, 10] + Ymgap)
            constraint_values.append(c4)
            if self.use_diluted:
                c5 = p[:, 7] - 1
                constraint_values.append(c5)
                c6 = p[:, 8] - 1
                constraint_values.append(c6)

        all_constraints_tensor = torch.cat([c.flatten() for c in constraint_values])
        return all_constraints_tensor

        

def save_muons(muons:np.array,tag):
    np.save(os.path.join(PROJECTS_DIR, f'cluster/files/muons_{tag}.npy'), muons)

class ShipMuonShieldCluster(ShipMuonShield):
    def __init__(self,
                 manager_ip='34.65.198.159',
                 port=444,
                 local:bool = False,
                 return_files = None,
                 **kwargs) -> None:
        super().__init__(**kwargs)

        self.manager_cert_path = os.getenv('STARCOMPUTE_MANAGER_CERT_PATH')
        self.client_cert_path = os.getenv('STARCOMPUTE_CLIENT_CERT_PATH')
        self.client_key_path = os.getenv('STARCOMPUTE_CLIENT_KEY_PATH')
        self.server_url = 'wss://%s:%s'%(manager_ip, port)
        self.return_files = return_files
        if not local:
            from starcompute.star_client import StarClient
            self.star_client = StarClient(self.server_url, self.manager_cert_path, 
                                    self.client_cert_path, self.client_key_path)
        
    def sample_x_idx(self,phi = None, n_samples = None):
        if n_samples is None: n_samples = self.n_samples
        if phi is not None and phi.dim()==2 and self.parallel:
            cores = int(self.cores/phi.size(0))
        else: cores = self.cores
        cores = min(cores,n_samples)
        return get_split_indices(cores,n_samples) 
    
    def simulate(self,phi:torch.tensor,
                 muons = None, 
                 file = None,
                 reduction = 'sum'):
        phi = self.add_fixed_params(phi)
        n_samples = muons.shape[0] if muons is not None else self.n_samples
        if n_samples==0: n_samples = self.sample_x(muons_file = file).shape[0]
        muons_idx = self.sample_x_idx(n_samples=n_samples)
        if muons is not None:
            t1 = time.time()
            with Pool(cpu_count()) as pool:
                pool.starmap(save_muons, [(muons[idx[0]:idx[1]], idx[0]) for idx in muons_idx])
        if self.simulate_fields: 
            print('SIMULATING MAGNETIC FIELDS')
            self.simulate_mag_fields(phi, cores = 9)
        t1 = time.time()
        inputs = split_array_idx(phi.detach().cpu(),muons_idx, file = file) 
        result = self.star_client.run(inputs)
        assert len(result) == self.cores, f"RESULT LENGTH: {len(result)}"
        print('SIMULATION FINISHED, took',time.time()-t1)
        t1 = time.time()
        if self.return_files is not None:
            results = []
            for filename in result:
                if filename == -1: continue
                m_file = os.path.join(self.return_files,str(filename)+'.npy')
                results.append(torch.as_tensor(np.load(m_file),dtype=torch.get_default_dtype()))
                #os.remove(m_file)
            result = torch.cat(results, dim=1)
        result = torch.as_tensor(result,device = phi.device)
        if not (phi.dim()==1 or phi.size(0)==1):
            result = result.view(phi.size(0),-1)
        if reduction == 'sum': result = result.sum(-1)
        if self.multi_fidelity: result *= 1E6/n_samples
        return result
    
    def __call__(self,phi,muons = None, file = None):
        if phi.dim()>1 and not self.parallel:
            y = []
            for p in phi:
                y.append(self(p))
            return torch.stack(y)
        phi = self.add_fixed_params(phi)
        M = self.get_total_cost(phi)
        if file is None: file = self.muons_file
        if self.get_constraints(phi) > 10 or M>((6*np.log(10)/10+1)*self.W0): 
            return torch.ones((1,1),device=phi.device)*1E6
        try: loss = self.simulate(phi,muons, file)
        except Exception as e:
            print(f"Error occurred with input: {phi}")
            print(e)
            raise
        n_samples = self.n_samples
        if self.multi_fidelity and loss < 3000:
            self.n_samples = 0
            loss = self.simulate(phi, file=file)
            self.n_samples = n_samples
        
        
        if self.apply_det_loss: loss = self.deterministic_loss(phi,loss)

        return loss.to(torch.get_default_dtype())   
    



import argparse

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--nodes",type=int,default = 16)
    parser.add_argument("--n_tasks_per_node", type=int, default=32)
    parser.add_argument("--n_tasks", type=int, default=None)
    parser.add_argument("--warm", dest = 'SC', action='store_false')
    parser.add_argument("--muons_file", type=str, default=os.path.join(PROJECTS_DIR,'MuonsAndMatter/data/muons/subsample_mid_biased.pkl'))
    parser.add_argument("--params_name", type=str, default=None)
    parser.add_argument("--cluster", action='store_true')
    parser.add_argument("--field_map", action='store_true')
    parser.add_argument("--multi_fidelity", action='store_true')
    parser.add_argument("--remove_cavern", dest = "cavern", action='store_false')
    args = parser.parse_args()
    params_dict = {'old_warm_opt':torch.tensor(ShipMuonShield.old_warm_opt),
                    'sc_v6':torch.tensor(ShipMuonShield.sc_v6),
                    'baseline':torch.tensor(ShipMuonShield.warm_scaled_baseline),
                    'warm_optm_scaled':torch.tensor(ShipMuonShield.warm_opt_scaled),
                    'warm_optm_scaled_2':torch.tensor(ShipMuonShield.warm_opt_scaled_2),
                     }
    dim = ShipMuonShield.full_dim
    if args.params_name is not None:
        with open(f'/home/hep/lprate/projects/BlackBoxOptimization/outputs/{args.params_name}/phi_optm.txt', "r") as txt_file:
            data = [float(line.strip()) for line in txt_file]
        phi_newparameters = torch.tensor(data)
        params_dict[args.params_name] = phi_newparameters
        dim = len(phi_newparameters.flatten())
    d = {}
    t0 = time.time()
    print(dim)
    
    
    
    seed = 1
    for name,phi in params_dict.items(): #
        print(name)
        
        if name == 'sc_v6' and not args.cluster: fSC_mag = True
        else: fSC_mag = args.SC
        if name == 'sc_v6' and not fSC_mag: continue

        if args.n_tasks is None and args.cluster: n_tasks = args.nodes*args.n_tasks_per_node
        elif args.n_tasks is None: n_tasks = 45
        else: n_tasks = args.n_tasks
        if args.cluster:
            muon_shield = ShipMuonShieldCluster(cores = n_tasks,dimensions_phi=dim,simulate_fields=args.field_map, fSC_mag=fSC_mag, seed=seed, fields_file = os.path.join(PROJECTS_DIR,'MuonsAndMatter/data/outputs/fields.npy'), multi_fidelity=args.multi_fidelity)
            t1 = time.time()
            loss_muons = muon_shield.simulate(torch.as_tensor(phi), file = args.muons_file)
            if args.multi_fidelity and loss_muons < 5000:
                file = os.path.join(PROJECTS_DIR,'MuonsAndMatter/data/muons/subsample.pkl')
                loss_muons = (loss_muons + 2*muon_shield.simulate(phi, file=file))/3
                if loss_muons < 500:
                    file = os.path.join(PROJECTS_DIR,'MuonsAndMatter/data/muons/subsample_4M.pkl')
                    loss_muons = (8*loss_muons + 3*muon_shield.simulate(phi, file=file))/11
            print(muon_shield.add_fixed_params(phi))
            t2 = time.time()
            loss_muons += 1
            n_hits = 0
            rate = 0
        else:
            muon_shield = ShipMuonShield(cores = n_tasks,fSC_mag=fSC_mag, dimensions_phi=dim,
                                         sensitive_plane=82,simulate_fields=False, seed=seed, cavern=args.cavern, muons_file = args.muons_file)
            if args.field_map: 
                muon_shield.fields_file = os.path.join(PROJECTS_DIR,'MuonsAndMatter/data/outputs/fields_mm.npy')
                muon_shield.simulate_mag_fields(torch.as_tensor(phi))
            t1 = time.time()
            px,py,pz,x,y,z,particle,factor = muon_shield.simulate(torch.as_tensor(phi))
            print('n_hits: ',x.numel())
            t2 = time.time()
            loss_muons = muon_shield.muon_loss(px,py,pz,x,y,z,particle,factor).sum()+1
            n_hits = factor.sum()
            n_inputs = muon_shield.sample_x()[:,7].sum()
            rate = n_hits/n_inputs
        M_i = muon_shield.get_iron_cost(torch.as_tensor(phi))
        C_e = muon_shield.get_electrical_cost(torch.as_tensor(phi))
        M = muon_shield.get_total_cost(torch.as_tensor(phi))
        L = muon_shield.get_total_length(torch.as_tensor(phi))
        print('loss_muons',loss_muons)
        C = muon_shield.get_constraints(torch.as_tensor(phi))
        print('constraints', C)
        print('length', L)     
        loss = muon_shield.deterministic_loss(phi,loss_muons)
        print('loss',loss)
        #print('n_hits',n_hits)
        #print('n_inputs',n_inputs)
        #print(f'rate: {rate:.1e}')
        d[name] = (loss_muons,loss, M_i, M,C_e, L, C, n_hits, rate)
        
        
        seed += 5
        print(f"took {t2-t1} sec")
    print('Results:')
    for name,(loss_muons,loss,M_i, M,C_e, L, C, n_hits, rate) in d.items():
        print("\n" + "***************" + name.upper()+ "***************" + "\n")
        print(f"Iron cost: {M_i}")
        print(f"Elet Cost: {C_e}")
        print(f"Cost: {M}")
        print(f"Length: {L}")
        print(f"Constraints Loss: {C}")
        print(f"Muon Loss: {loss_muons}")
        print(f"Total Loss: {loss}")
        print(f"n_hits: {n_hits}")
        print(f"rate: {rate:.1e}")
    print(f"Total Time: {time.time()-t0}")
