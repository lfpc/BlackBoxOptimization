import torch
import gzip
import pickle
import sys
import numpy as np
import os
from multiprocessing import Pool
PROJECTS_DIR = os.getenv('PROJECTS_DIR')
sys.path.insert(1, os.path.join(PROJECTS_DIR,'BlackBoxOptimization'))
from utils import split_array, split_array_idx, get_split_indices, soft_clamp
import logging
logging.basicConfig(level=logging.WARNING)
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

    parametrization = {'HA': [0, 7, 8, 9, 10, 11, 12, 13, 14],
                       'M1': [1, 15, 16, 17, 18, 19, 20, 21, 22],
                       'M2': [2, 23, 24, 25, 26, 27, 28, 29, 30], #SC
                       'M3': [3, 31, 32, 33, 34, 35, 36, 37, 38],
                       'M4': [4, 39, 40, 41, 42, 43, 44, 45, 46],
                       'M5': [5, 47, 48, 49, 50, 51, 52, 53, 54],
                       'M6': [6, 55, 56, 57, 58, 59, 60, 61, 62]}
    
    old_warm_opt = torch.tensor([[231.00, 208.0, 207.0, 281.0, 248.0, 305.0, 242.0,
                  50.00,  50.00, 119.00, 119.00,   2.00,   2.00, 1.00, 0.00,
                  72.0, 51.0, 29.0, 46.0, 10.0, 7.0, 1.0, 0.,
                  54.0, 38.0, 46.0, 192.0, 14.0, 9.0, 1.0, 0.,
                  10.0, 31.0, 35.0, 31.0, 51.0, 11.0, 1.0, 0.,
                  3.0, 32.0, 54.0, 24.0, 8.0, 8.0, 1.0, 0.,
                  22.0, 32.0, 209.0, 35.0, 8.0, 13.0, 1.0, 0.,
                  33.0, 77.0, 85.0, 241.0, 9.0, 26.0, 1.0, 0.]])
    
    warm_opt = torch.tensor([231.00, 170.89, 170.07, 230.87, 203.76, 250.59, 198.83,
                  50.00,  50.00, 118.00, 118.00,   2.00,   2.00, 1.00, 0.00,
                  72.00, 51.00, 29.00, 46.00, 10.00, 7.00, 1.00, 0.00,
                  54.00, 38.00, 46.00, 130.00, 14.00, 9.00, 1.00, 0.00,
                  10.00, 31.00, 35.00, 31.00, 51.00, 11.00, 1.00, 0.00,
                  5.00, 32.00, 54.00, 24.00, 8.00, 8.00, 1.00, 0.00,
                  22.00, 32.00, 130.00, 35.00, 8.00, 13.00, 1.00, 0.00,
                  33.00, 77.00, 85.00, 90.00, 9.00, 26.00, 1.00, 0.00])

    sc_v6 = torch.tensor([231.00,  0., 353.08, 125.08, 184.83, 150.19, 186.81, 
         50.00,  50.00, 119.00, 119.00,   2.00,   2.00, 1.00, 0.00,
        0.,  0.,  0.,  0.,  0.,   0., 1., 0.,
        45.69,  45.69,  22.18,  22.18,  27.01,  16.24, 3.00, 0.00,
        0.,  0.,  0.,  0.,  0.,  0., 1., 0., 
        24.80,  48.76,   8.00, 104.73,  15.80,  16.78, 1.00, 0.00,
        3.00, 100.00, 192.00, 192.00,   2.00,   4.80, 1.00, 0.00,
        3.00, 100.00,   8.00, 172.73,  46.83,   2.00, 1.00, 0.00])
    
    hybrid_idx = (np.array(parametrization['M2'])[[0, 1, 3, 5, 6, 7]]).tolist() + [parametrization['M3'][0]]+\
           parametrization['M4'] + parametrization['M5'] + parametrization['M6']
    
    warm_idx = parametrization['M1'] + parametrization['M2'] + parametrization['M3'] + parametrization['M4'] + parametrization['M5'] + parametrization['M6']
    
    hybrid_no_xmgap_idx = (np.array(parametrization['M2'])[[0, 1, 3, 5, 6, 7]]).tolist() + [parametrization['M3'][0]]+\
           parametrization['M4'][:-1] + parametrization['M5'][:-1] + parametrization['M6'][:-1]
    
    fixed_sc = [parametrization['HA'][0]] + [parametrization['M3'][0]]+\
           parametrization['M4'] + parametrization['M5'] + parametrization['M6']
    
    DEFAULT_PHI = warm_opt#sc_v6
    DEF_N_SAMPLES = 643118
    full_dim = 63

    def __init__(self,
                 W0:float = 1700000.,
                 L0:float = 29.7,
                 cores:int = 45,
                 n_samples:int = 0,
                 input_dist:float = None,
                 sensitive_plane:float = 82,
                 average_x:bool = True,
                 apply_det_loss:bool = True,
                 weight_loss_fn:bool = 'exponential',
                 fSC_mag:bool = True,
                 simulate_fields:bool = False,
                 cavern:bool = True,
                 seed:int = None,
                 left_margin = 2,
                 right_margin = 2,
                 y_margin = 3,
                 dimensions_phi = 34,
                muons_file = os.path.join(PROJECTS_DIR,'MuonsAndMatter/data/enriched_input.pkl'),
                fields_file = None,
                extra_magnet = False,
                cut_P:float = None,
                 ) -> None:
        
        self.left_margin = left_margin
        self.MUON = 13
        self.right_margin = right_margin
        self.y_margin = y_margin
        self.W0 = W0
        self.L0 = L0
        self.cores = cores
        self.muons_file = muons_file
        self.n_samples = n_samples
        self.input_dist = input_dist
        self.average_x = average_x
        self.weight_loss_fn = weight_loss_fn
        self.sensitive_plane = sensitive_plane
        self.sensitive_film_params = {'dz': 0.01, 'dx': 4, 'dy': 6,'position': sensitive_plane}
        self.fSC_mag = fSC_mag
        self.simulate_fields = simulate_fields
        self.seed = seed
        self.dimensions_phi = dimensions_phi   
        self.cavern = cavern
        self.apply_det_loss = apply_det_loss
        self.extra_magnet = extra_magnet    
        self.lambda_constraints = 50
        self.cut_P = cut_P

        if dimensions_phi == 29: self.params_idx = self.fixed_sc
        elif dimensions_phi == 31: self.params_idx = self.hybrid_no_xmgap_idx   
        elif dimensions_phi == 34: self.params_idx = self.hybrid_idx
        elif dimensions_phi == 54: self.params_idx = self.warm_idx
        elif dimensions_phi == self.full_dim: self.params_idx = slice(None)
        self.DEFAULT_PHI = self.DEFAULT_PHI[self.params_idx]


        sys.path.insert(1, os.path.join(PROJECTS_DIR,'MuonsAndMatter/python/bin'))
        from run_simulation import run, get_field
        self.run_muonshield = run
        self.run_magnet = get_field
        self.fields_file = fields_file#os.path.join(PROJECTS_DIR,'MuonsAndMatter/data/outputs/fields.pkl')

    def sample_x(self,phi=None):
        try: 
            with open(self.muons_file, 'rb') as f:
                x = pickle.load(f)
        except:
            with gzip.open(self.muons_file, 'rb') as f:
                x = pickle.load(f)
        if 0<self.n_samples<=x.shape[0]: 
            indices = np.random.choice(x.shape[0], self.n_samples, replace=False)
            x = x[indices]
        return torch.as_tensor(x, dtype=torch.get_default_dtype())
    def simulate_mag_fields(self,phi:torch.tensor, cores:int = 1):
        phi = self.add_fixed_params(phi)
        Z = phi[0:7].sum().item()*2/100 + 0.1
        max_x = 0
        max_y = 0  
        for m,idx in self.parametrization.items():
            if self.fSC_mag and (m in ['M1', 'M3']): continue
            params = phi[idx]
            dXIn, dXOut, dYIn, dYOut, gapIn, gapOut, ratio_yokes, Xmgap = params[1:]
            dX = max(torch.max(dXIn + dXIn * ratio_yokes + gapIn+Xmgap).item(), torch.max(dXOut + dXOut*ratio_yokes+gapOut+Xmgap).item())/100
            if dX > max_x:
                max_x = dX
            dY = max(torch.max(dYIn + dXIn * ratio_yokes).item(), torch.max(dYOut + dXOut * ratio_yokes).item())/100
            if dY > max_y:
                max_y = dY
        d_space = (max_x+0.5, max_y+0.5, (-1, np.ceil(Z+0.5)))
        resol = (0.02,0.02,0.05)
        self.run_magnet(True,phi.cpu().numpy(),file_name = self.fields_file,d_space = d_space,resol = resol, cores = cores, fSC_mag = self.fSC_mag)

    def simulate(self,phi:torch.tensor,muons = None, return_nan = False): 
        phi = self.add_fixed_params(phi)
        if muons is None: muons = self.sample_x()

        workloads = split_array(muons,self.cores)
        if self.simulate_fields: 
            print('SIMULATING MAGNETIC FIELDS')
            self.simulate_mag_fields(phi)
        with Pool(self.cores) as pool:
            result = pool.starmap(self.run_muonshield, 
                                  [(workload,phi.cpu().numpy(),self.input_dist,True,self.fSC_mag,self.sensitive_film_params,self.cavern,
                                    self.simulate_fields,self.fields_file,return_nan,self.seed, False, 5., True, False, self.extra_magnet) for workload in workloads])
        print('SIMULATION FINISHED')
        all_results = []
        for rr in result:
            resulting_data,weight = rr
            if resulting_data.size == 0: continue
            all_results += [resulting_data]
        if len(all_results) == 0:
            all_results = [[np.nan]*8]
        all_results = torch.as_tensor(np.concatenate(all_results, axis=0).T,device = phi.device,dtype=torch.get_default_dtype())
        return all_results
    def is_hit(self,px,py,pz,x,y,z,particle,factor = None):
        p = torch.sqrt(px**2+py**2+pz**2)
        charge = -1*torch.sign(particle)
        mask = (-charge*x <= self.left_margin) & (-self.right_margin <= -charge*x) & (torch.abs(y) <= self.y_margin)
        mask = mask & (torch.abs(particle).to(torch.int)==self.MUON)
        if self.cut_P is not None: mask = mask & p.ge(self.cut_P)
        return mask.to(torch.int)
    def muon_loss(self,px,py,pz,x,y,z,particle, weight = None):
        charge = -1*torch.sign(particle)
        mask = self.is_hit(px,py,pz,x,y,z,particle)
        x = x[mask]
        charge = charge[mask]
        loss = torch.sqrt(1 + (charge*x-self.right_margin)/(self.left_margin+self.right_margin)) 
        if weight is not None:
            weight = weight[mask]
            loss *= weight
        return loss
    def get_total_length(self, phi):
        phi = self.add_fixed_params(phi)
        phi = phi.view(-1,self.full_dim)
        length = torch.zeros(phi.size(0), device=phi.device)
        for m, idx in self.parametrization.items():
            params = phi[:, idx]
            length = length + 2*params[:, 0]
        return length / 100
    def get_weight(self,phi,z_gap = 10,density = 7.874E-3):
        '''Get the weight of the muon shield.
        phi: torch.tensor with the parameters of the muon shield.
        density: density of iron in kg/cm^3'''
        def volume_block(dz,dx1,dx2,dy1,dy2):
            return (dz/3)*(dx1*dy1+dx2*dy2+torch.sqrt(dx1*dy1*dx2*dy2))
        phi = self.add_fixed_params(phi)
        phi = phi.view(-1,self.full_dim)
        volume = torch.zeros(phi.size(0), device=phi.device)
        for m,idx in self.parametrization.items():
            if self.fSC_mag and (m in ['M1', 'M3']): continue
            params = phi[:,idx]
            Ymgap = 5 if self.fSC_mag and m == 'M2' else 0
            dz = 2*params[:,0]-z_gap
            volume = volume + volume_block(dz, params[:, 1], params[:, 2], params[:, 3], params[:, 4])  # core
            volume = volume + volume_block(dz, params[:, 1] * params[:, 7], params[:, 2] * params[:, 7], params[:, 3] + Ymgap, params[:, 4] + Ymgap)  # lateral yoke
            volume = volume + volume_block(dz, params[:, 1] + params[:, 5] + params[:, 1] * params[:, 7] + params[:, 8], params[:, 2] + params[:, 6] + params[:, 2] * params[:, 7] + params[:, 8], params[:, 1] * params[:, 7], params[:, 2] * params[:, 7])  # upper yoke
        return 4*volume.view(-1,1)*density 
    
    def weight_loss(self,W,L = None):
        if self.weight_loss_fn == 'exponential':
            return (1+torch.exp(10*(W-self.W0)/self.W0))
        elif self.weight_loss_fn == 'linear':
            return W/self.W0
        elif self.weight_loss_fn == 'quadratic':
            return (W/self.W0)**2
        elif self.weight_loss_fn == 'linear_length':
            return W/(1-L/self.L0)
        else: return 1
    def calc_loss(self,px,py,pz,x,y,z,particle,factor,W = None, L = None):
        loss = self.muon_loss(px,py,pz,x,y,z,particle,factor).sum()+1
        if self.apply_det_loss: loss *= self.weight_loss(W,L)+self.get_constraints(phi)
        return loss.to(torch.get_default_dtype()).view(-1,1)
    def __call__(self,phi,muons = None):
        L = self.get_total_length(phi)
        W = self.get_weight(phi)
        #if L>=self.L0: return torch.tensor([1e8])
        #if W>=3E6: return torch.tensor([1e8])
        if phi.dim()>1:
            y = []
            for p in phi:
                y.append(self(p))
            return torch.stack(y)
        
        return self.calc_loss(*self.simulate(phi,muons),W,L)
    
    def propagate_to_sensitive_plane(self,px,py,pz,x,y,z, epsilon = 1e-12):
        z += self.sensitive_plane
        x += self.sensitive_plane*px/(pz+epsilon)
        y += self.sensitive_plane*py/(pz+epsilon)
        return x,y,z

    def GetBounds(self,zGap:float = 1.,device = torch.device('cpu'), correct_bounds = True):
        magnet_lengths = [(50 + zGap, 300 + zGap)] * 7  #previously 170-300
        dX_bounds = [(5, 250)] * 2
        dY_bounds = [(5, 160)] * 2 
        gap_bounds = [(2, 150)] * 2 
        yoke_bounds = [(0.75,1.25)]#[(0.25, 4)]
        inner_gap_bounds = [(0., 150.)]
        bounds = magnet_lengths + 2*(dX_bounds + dY_bounds + gap_bounds + yoke_bounds + inner_gap_bounds)
        dY_bounds = [(5, 300)] * 2 
        bounds += 5*(dX_bounds + dY_bounds + gap_bounds + yoke_bounds + inner_gap_bounds)
        if self.fSC_mag: 
            bounds[self.parametrization['M2'][0]] = (50,400)
            bounds[self.parametrization['M2'][5]] = (15,70)
            bounds[self.parametrization['M2'][6]] = (15,70)
            bounds[self.parametrization['M2'][7]] = (1.0,4)
        bounds = torch.tensor(bounds,device=device,dtype=torch.get_default_dtype()).T
        return bounds[:,self.params_idx]
    def add_fixed_params(self, phi:torch.Tensor):
        if phi.dim() == 1:
            phi = phi.unsqueeze(0)
        if phi.size(-1) != self.full_dim:
            assert phi.squeeze().size(-1) == len(self.params_idx), f"INPUT SHAPE: {phi.shape}"
            new_phi = self.warm_opt.clone().to(phi.device).repeat(phi.size(0), 1)
            new_phi[:, torch.as_tensor(self.params_idx, device=phi.device)] = phi
            if self.fSC_mag:
                new_phi[:, self.parametrization['M2'][2]] = new_phi[:, self.parametrization['M2'][1]]
                new_phi[:, self.parametrization['M2'][4]] = new_phi[:, self.parametrization['M2'][3]]
        else:
            new_phi = phi
        assert new_phi.size(-1) == self.full_dim, f"FINAL SHAPE: {new_phi.shape}"
        return new_phi.squeeze(0) if phi.size(0) == 1 else new_phi
    
    def deterministic_loss(self,phi,y):
        y = y.view(-1,1)
        W = self.get_weight(phi)
        loss = self.weight_loss(W)*y
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
        wall_gap = 2
        def get_cavern_bounds(z):
            x_min = torch.zeros_like(z)
            y_min = torch.zeros_like(z)
            mask = z <= 2200-234.5
            x_min[mask] = 356#min(np.abs(-3.56), 6.43)
            y_min[mask] = 170#min(np.abs(-1.7), 5.8)
            x_min[~mask] = 456#min(np.abs(-4.56), 11.43)
            y_min[~mask] = 336#min(np.abs(-3.36), 10)
            x_min -= wall_gap
            y_min -= wall_gap
            return x_min, y_min

        z = torch.zeros(phi.size(0),device=phi.device)
        
        for m,idx in self.parametrization.items():
            p = phi[:,idx]
            z = z + 2*p[:,0]
            Ymgap = 5 if (self.fSC_mag and m =='M2') else 0
            x_min, y_min = get_cavern_bounds(z-2*p[:,0])
            constraints = constraints + fn_pen(p[:,1]+p[:,7]*p[:,1]+p[:,5]+p[:,8]-x_min)
            constraints = constraints + fn_pen(p[:,3]+p[:,7]*p[:,1]+Ymgap - y_min)
            x_min, y_min = get_cavern_bounds(z)
            constraints = constraints + fn_pen(p[:,2]+p[:,7]*p[:,2]+p[:,6]+p[:,8] -x_min)
            constraints = constraints + fn_pen(p[:,4]+p[:,7]*p[:,2]+Ymgap - y_min)
        return (constraints.reshape(-1,1)*self.lambda_constraints)#.clamp(1E8)
    
    def get_constraints_fn(self):
        
        return self.L0 - self.get_total_length(phi)
        constraints = []
        constraints.append((lambda phi: self.get_total_length(phi) - self.L0,1))
        wall_gap = 0.1
        def get_cavern_bounds(z):
            x_min = torch.zeros_like(z)
            y_min = torch.zeros_like(z)
            mask = z < 22-2.345
            x_min[mask] = min(np.abs(-3.56), 6.43)
            y_min[mask] = min(np.abs(-1.7), 5.8)
            x_min[~mask] = min(np.abs(-4.56), 11.43)
            y_min[~mask] = min(np.abs(-3.36), 10)
            x_min += wall_gap
            y_min += wall_gap
            return x_min, y_min
        
        z = torch.zeros(phi.size(0),device=phi.device)
        
        for m,idx in self.parametrization.items():
            p = phi[:,idx]/100
            z += p[:,0]
            Ymgap = 0.05 if (self.fSC_mag and m =='M2') else 0
            x_min, y_min = get_cavern_bounds(z-p[:,0])
            constraints.append(lambda phi: x_min - p[:, 1] + p[:, 7] * p[:, 1] + p[:, 5] + p[:, 8])
            constraints.append(lambda phi: y_min - p[:, 3] + p[:, 7] * p[:, 1] + Ymgap)
            x_min, y_min = get_cavern_bounds(z+p[:,0])
            constraints.append(lambda phi: x_min - p[:, 2] + p[:, 7] * p[:, 2] + p[:, 6] + p[:, 8])
            constraints.append(lambda phi: y_min - p[:, 4] + p[:, 7] * p[:, 2] + Ymgap)
        return constraints

        


class ShipMuonShieldCluster(ShipMuonShield):
    def __init__(self,
                 W0:float = 1700000.,
                 L0:float = 29.7,
                 cores:int = 512,
                 n_samples:int = 0,
                 weight_loss_fn:bool = 'exponential',
                 manager_ip='34.65.198.159',
                 port=444,
                 local:bool = False,
                 parallel:bool = False,
                 seed = None,
                 dimensions_phi = 54,
                 fSC_mag:bool = True, 
                 simulate_fields:bool = False,
                 return_files = None,
                 apply_det_loss:bool = False,
                 **kwargs) -> None:
        super().__init__(W0 = W0, L0 = L0, cores = cores, n_samples = n_samples, weight_loss_fn = weight_loss_fn,
                         fSC_mag = fSC_mag, simulate_fields = simulate_fields,
                         dimensions_phi = dimensions_phi, apply_det_loss = apply_det_loss, seed = seed, **kwargs)

        self.parallel = parallel
        self.manager_cert_path = os.getenv('STARCOMPUTE_MANAGER_CERT_PATH')
        self.client_cert_path = os.getenv('STARCOMPUTE_CLIENT_CERT_PATH')
        self.client_key_path = os.getenv('STARCOMPUTE_CLIENT_KEY_PATH')
        self.server_url = 'wss://%s:%s'%(manager_ip, port)
        self.return_files = return_files
        if not local:
            from starcompute.star_client import StarClient
            self.star_client = StarClient(self.server_url, self.manager_cert_path, 
                                    self.client_cert_path, self.client_key_path)
        if simulate_fields:
            self.fields_file = os.path.join(PROJECTS_DIR,'MuonsAndMatter/data/outputs/fields.pkl')
        
    def sample_x_idx(self,phi = None, slice = True):
        if not slice: return super().sample_x_idx(phi)
        if phi is not None and phi.dim()==2:
            cores = int(self.cores/phi.size(0))
        else: cores = self.cores
        return get_split_indices(cores,self.n_samples) 
    
    def simulate(self,phi:torch.tensor,
                 muons = None, 
                 file = None,
                 reduction = 'sum'):
        phi = self.add_fixed_params(phi)
        if muons is not None:
            with open(os.path.join(PROJECTS_DIR,'cluster/muons.pkl'),'wb') as f: pickle.dump(muons.cpu(),f)
            self.n_samples = muons.size(0)
        muons = self.sample_x_idx(phi)
        if self.simulate_fields: 
            print('SIMULATING MAGNETIC FIELDS')
            self.simulate_mag_fields(phi, cores = 9)
        t1 = time.time()
        inputs = split_array_idx(phi.cpu(),muons, file = file) 
        result = self.star_client.run(inputs)
        print('SIMULATION FINISHED, took',time.time()-t1)
        t1 = time.time()
        if self.return_files is not None:
            results = []
            for filename in result:
                if filename == -1: continue
                m_file = os.path.join(self.return_files,str(filename)+'.pkl')
                with open(m_file, "rb") as f:
                    results.append(pickle.load(f))
                #os.remove(m_file)
            result = torch.cat(results, dim=1)
            del results
        result = torch.as_tensor(result,device = phi.device)
        print('result shape',result.shape, 'took',time.time()-t1)
        if not (phi.dim()==1 or phi.size(0)==1):
            result = result.view(phi.size(0),-1)
        if reduction == 'sum': result = result.sum(-1)
        return result
    
    def __call__(self,phi,muons = None, file = None):
        if phi.dim()>1 and not self.parallel:
            y = []
            for p in phi:
                y.append(self(p))
            return torch.stack(y)
        W = self.get_weight(phi)
        #if self.get_constraints(phi) > 100 or W>3E6: return torch.ones((phi.size(0),1),device=phi.device)*1E6
        loss = self.simulate(phi,muons, file)
        loss += 1
        if self.apply_det_loss: loss *= self.weight_loss(W).item()+self.get_constraints(phi).item()
        return loss.to(torch.get_default_dtype())    
    


import time
import argparse
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--nodes",type=int,default = 16)
    parser.add_argument("--n_tasks_per_node", type=int, default=32)
    parser.add_argument("--n_tasks", type=int, default=None)
    parser.add_argument("--warm", dest = 'SC', action='store_false')
    parser.add_argument("--muons_file", type=str, default=None)
    parser.add_argument("--params_name", type=str, default=None)
    parser.add_argument("--cluster", action='store_true')
    parser.add_argument("--field_map", action='store_true')
    args = parser.parse_args()
    params_dict = {'Adjusted oliver':ShipMuonShield.warm_opt,
                    'sc_v6':ShipMuonShield.sc_v6, 
                     #'combi':ShipMuonShield.combi, 
                     #'baseline':ShipMuonShield.baseline_1,
                     }
    if args.params_name is not None:
        with open(f'/home/hep/lprate/projects/BlackBoxOptimization/outputs/{args.params_name}/phi_optm.txt', "r") as txt_file:
            data = [float(line.strip()) for line in txt_file]
        phi_newparameters = torch.tensor(data)
        params_dict[args.params_name] = phi_newparameters
    d = {}
    t0 = time.time()
    
    
    
    seed = 1
    for name,phi in params_dict.items(): #
        print(name)
        
        if name == 'sc_v6' and not args.SC: continue
        if args.n_tasks is None: n_tasks = args.nodes*args.n_tasks_per_node
        else: n_tasks = args.n_tasks
        if args.cluster:
            muon_shield = ShipMuonShieldCluster(cores = n_tasks,dimensions_phi=54,sensitive_plane=0,simulate_fields=args.field_map)
            t1 = time.time()
            loss_muons, W = muon_shield.simulate(torch.as_tensor(phi), file = args.muons_file)
            t2 = time.time()
            loss_muons += 1
        else:
            muon_shield = ShipMuonShield(cores = n_tasks,fSC_mag=args.SC, dimensions_phi=54,
                                         sensitive_plane=82,input_dist = None,simulate_fields=args.field_map, seed=seed
                                         )
            muon_shield.fields_file = os.path.join(PROJECTS_DIR,'MuonsAndMatter/data/outputs/fields_mm.pkl')
            if args.field_map: 
                muon_shield.simulate_mag_fields(phi)
            t1 = time.time()
            px,py,pz,x,y,z,particle,factor = muon_shield.simulate(torch.as_tensor(phi,dtype = torch.float32))
            print('n_hits: ',x.numel())
            t2 = time.time()
            loss_muons = muon_shield.muon_loss(x,y,particle, factor).sum()+1
            n_hits = factor.sum()
            n_inputs = muon_shield.sample_x()[:,7].sum()
            rate = n_hits/n_inputs
        W_new = muon_shield.get_weight(torch.as_tensor(phi,dtype = torch.float32))
        L = muon_shield.get_total_length(torch.as_tensor(phi,dtype = torch.float))
        print('loss_muons',loss_muons)
        C = muon_shield.get_constraints(torch.as_tensor(phi,dtype = torch.float))
        print('constraints', C)
        print('length', L)     
        loss = muon_shield.deterministic_loss(phi,loss_muons)
        print('loss',loss)
        print('n_hits',n_hits)
        print('n_inputs',n_inputs)
        print(f'rate: {rate:.1e}')
        d[name] = (loss_muons,loss, W_new, L, C, n_hits, rate)
        
        seed += 5
        print(f"took {t2-t1} sec")
    print('Results:')
    for name,(loss_muons,loss,W_new, L, C, n_hits, rate) in d.items():
        print(name)
        print(f"Weight: {W_new}")
        print(f"Length: {L}")
        print(f"Constraints Loss: {C}")
        print(f"Muon Loss: {loss_muons}")
        print(f"Total Loss: {loss}")
        print(f"n_hits: {n_hits}")
        print(f"rate: {rate:.1e}")
    print(f"Total Time: {time.time()-t0}")
