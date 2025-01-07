import torch
import gzip
import pickle
import sys
import numpy as np
from os import getenv
from os.path import join
from multiprocessing import Pool
PROJECTS_DIR = getenv('PROJECTS_DIR')
sys.path.insert(1, join(PROJECTS_DIR,'BlackBoxOptimization'))
from utils import split_array, split_array_idx, get_split_indices
import logging
logging.basicConfig(level=logging.WARNING)

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
    combi = torch.as_tensor([ 208.  , 207.  , 281.  , 172.82, 212.54, 168.64,
      72.  ,  51.  ,  29.  ,  46.  ,
    10.  ,   7.  ,  54.  ,  38.  ,  46.  , 192.  ,  14.  ,   9. ,
    10.  ,  31.  ,  35.  ,  31.  ,  51.  ,  11.  ,   3.  ,  32. ,
    54.  ,  24.  ,   8.  ,   8.  ,  22.  ,  32.  , 209.  ,  35. ,
    8.  ,  13.  ,  33.  ,  77.  ,  85.  , 241.  ,   9.  ,  26.  ])

    parametrization = {'?':  [0,8, 9, 10, 11, 12, 13, 14, 15],
                       'HA': [1,16, 17, 18, 19, 20, 21, 22, 23],
                       'M1': [2,24, 25, 26, 27, 28, 29, 30, 31],
                       'M2': [3,32, 33, 34, 35, 36, 37, 38, 39], 
                       'M3': [4,40, 41, 42, 43, 44, 45, 46, 47],
                       'M4': [5,48, 49, 50, 51, 52, 53, 54, 55],
                       'M5': [6,56, 57, 58, 59, 60, 61, 62, 63],
                       'M6': [7,64, 65, 66, 67, 68, 69, 70, 71]}

    sc_v6 = torch.as_tensor([ 40.00, 231.00,   0.00, 353.08, 125.08, 184.83, 150.19, 186.81,  
                                0.00,  0.00, 0.00, 0.00,   0.00,   0.00, 1.00, 0.00,
                                50.00,  50.00, 130.00, 130.00,   2.00,   2.00, 1.00, 0.00,
                                72.00,  51.00,  29.00,  46.00,  10.00,   7.00, 1.00, 0.00,
                                45.69,  45.69,  22.18,  22.18,  27.01,  16.24, 3.00, 0.00,
                                0.00,  0.00,  0.00,  0.00,  0.00,  0.00, 1.00, 0.00, 
                                24.80,  48.76,   8.00, 104.73,  15.80,  16.78, 1.00, 0.00,
                                3.00, 100.00, 192.00, 192.00,   2.00,   4.80, 1.00, 0.00,
                                3.00, 100.00,   8.00, 172.73,  46.83,   2.00, 1.00, 0.00])
    
    hybrid_idx = (np.array(parametrization['M2'])[[0, 1, 3, 5, 6, 7]]).tolist() + [parametrization['M3'][0]]+\
           parametrization['M4'] + parametrization['M5'] + parametrization['M6']
    
    fixed_sc = [parametrization['HA'][0]] + [parametrization['M3'][0]]+\
           parametrization['M4'] + parametrization['M5'] + parametrization['M6']
    
    DEFAULT_PHI = sc_v6

    def __init__(self,
                 W0:float = 1558731.375,
                 L0:float = 35,
                 cores:int = 45,
                 n_samples:int = 0,
                 input_dist:float = 0.85,
                 sensitive_plane:float = 83.2,#distance between end of shield and sensplane
                 average_x:bool = True,
                 weight_loss_fn:bool = 'exponential',
                 fSC_mag:bool = True,
                 simulate_fields:bool = False,
                 cavern:bool = False,
                 seed:int = None,
                 left_margin = 2,
                 right_margin = 2,
                 y_margin = 3,
                 dimensions_phi = 34,
                 ) -> None:
        
        self.left_margin = left_margin
        self.MUON = 13
        self.right_margin = right_margin
        self.y_margin = y_margin
        self.W0 = W0
        self.L0 = L0
        self.cores = cores
        self.muons_file = join(PROJECTS_DIR,'MuonsAndMatter/data/inputs.pkl')
        self.n_samples = n_samples
        self.input_dist = input_dist
        self.average_x = average_x
        self.weight_loss_fn = weight_loss_fn
        self.sensitive_plane = sensitive_plane
        self.sensitive_film_params = {'dz': 0.01, 'dx': 4, 'dy': 6,'position': sensitive_plane} #the center is in end of muon shield + position
        self.fSC_mag = fSC_mag
        self.simulate_fields = simulate_fields
        self.seed = seed
        self.dimensions_phi = dimensions_phi   
        self.cavern = cavern

        if dimensions_phi == 29: self.params_idx = self.fixed_sc
        elif dimensions_phi == 34: self.params_idx = self.hybrid_idx
        elif dimensions_phi == 72: self.params_idx = slice(None)
        self.DEFAULT_PHI = self.DEFAULT_PHI[self.params_idx]

        
        sys.path.insert(1, join(PROJECTS_DIR,'MuonsAndMatter/python/bin'))
        from run_simulation import run, get_field
        self.run_muonshield = run
        self.run_magnet = get_field
        self.fields_file = join(PROJECTS_DIR,'MuonsAndMatter/data/outputs/fields.pkl')

    def sample_x(self,phi=None):
        with gzip.open(self.muons_file, 'rb') as f:
            x = pickle.load(f)
        if 0<self.n_samples<=x.shape[0]: x = x[:self.n_samples]
        return x
    def simulate_mag_fields(self,phi:torch.tensor):
        zgap = 0.1
        Z = phi[1:8].sum().item()*2/100 + (7 * zgap / 2) + 0.1
        d_space = (3., 3., (-1, np.ceil(Z+0.5)))
        resol = (0.05,0.05,0.05)
        self.run_magnet(True,phi.cpu().numpy(),file_name = self.fields_file,d_space = d_space,resol = resol)

    def simulate(self,phi:torch.tensor,muons = None, return_nan = False): 
        phi = phi.flatten() 
        phi = self.add_fixed_params(phi)
        if muons is None: muons = self.sample_x()

        workloads = split_array(muons,self.cores)
        #if self.simulate_fields: 
        #    self.simulate_mag_fields(phi)

        with Pool(self.cores) as pool:
            result = pool.starmap(self.run_muonshield, 
                                  [(workload,phi.cpu().numpy(),self.input_dist,True,self.fSC_mag,self.sensitive_film_params,self.cavern,
                                    self.simulate_fields,self.fields_file,return_nan,self.seed, False) for workload in workloads])
        all_results = []
        for rr in result:
            resulting_data,weight = rr
            if resulting_data.size == 0: continue
            all_results += [resulting_data]
        if len(all_results) == 0:
            all_results = [[np.nan]*8]
        all_results = torch.as_tensor(np.concatenate(all_results, axis=0).T,device = phi.device,dtype=torch.get_default_dtype())
        all_results[3],all_results[4],all_results[5] = self.propagate_to_sensitive_plane(*all_results[:6])
        return *all_results, torch.as_tensor(weight,device = phi.device)
    
    def muon_loss(self,x,y,particle, weight = None):
        charge = -1*torch.sign(particle)
        mask = (-charge*x <= self.left_margin) & (-self.right_margin <= -charge*x) & (torch.abs(y) <= self.y_margin) & ((torch.abs(particle).to(torch.int))==self.MUON)
        x = x[mask]
        charge = charge[mask]
        #loss = torch.sigmoid(((1.5-charge*x)*3))
        loss = torch.sqrt(1 + (charge*x-self.right_margin)/(self.left_margin+self.right_margin)) 
        if weight is not None:
            weight = weight[mask]
            loss *= weight
        return loss
    def get_total_length(self,phi):
        phi = self.add_fixed_params(phi)
        length = 0
        for m,idx in self.parametrization.items():
            if m == '?': continue
            params = phi[idx]
            length += params[0]
        return length/100
    def get_weight(self,phi,z_gap = 10,density = 7.874E-3):
        '''Get the weight of the muon shield.
        phi: torch.tensor with the parameters of the muon shield.
        density: density of iron in kg/cm^3'''
        def volume_block(dz,dx1,dx2,dy1,dy2):
            return (dz/3)*(dx1*dy1+dx2*dy2+np.sqrt(dx1*dy1*dx2*dy2))#dz*(dx1*dy1+dx2*dy2)/2
        phi = self.add_fixed_params(phi)
        volume = 0
        for m,idx in self.parametrization.items():
            if m == '?': continue
            if self.fSC_mag and m in ['M1', 'M3']: continue
            params = phi[idx]
            Ymgap = 5 if self.fSC_mag and m == 'M2' else 0
            dz = 2*params[0]-z_gap
            volume += volume_block(dz,params[1],params[2],params[3],params[4]) #core
            volume += volume_block(dz,params[1]*params[7],params[2]*params[7],params[3]+Ymgap,params[4]+Ymgap) #lateral yoke
            volume += volume_block(dz,params[1]+params[5]+params[1]*params[7]+params[8],params[2]+params[6]+params[2]*params[7]+params[8],params[1]*params[7], params[2]*params[7]) #upper yoke
        volume += volume_block(20,params[2]+params[6]+params[2]*params[7]+params[8],params[2]+params[6]+params[2]*params[7]+params[8],params[2]*params[7], params[2]*params[7]) #lower yoke
        return 4*volume*density #4 due to symmetry
    def weight_loss(self,W,L = None):
        if self.weight_loss_fn == 'exponential':
            return 1+torch.exp(10*(W-self.W0)/self.W0)
        elif self.weight_loss_fn == 'linear':
            return W/self.W0
        elif self.weight_loss_fn == 'quadratic':
            return (W/self.W0)**2
        elif self.weight_loss_fn == 'linear_length':
            return W/(1-L/self.L0)
        else: return 1
    def calc_loss(self,px,py,pz,x,y,z,particle,W = None, L = None):
        loss = self.muon_loss(x,y,particle).sum()+1
        if self.weight_loss_fn is not None:
            loss *= self.weight_loss(W,L)
            loss = torch.where(W>3E6,1e8,loss)
        return loss.to(torch.get_default_dtype())
    def __call__(self,phi,muons = None):
        L = self.get_total_length(phi)
        #W = self.get_weight(phi)
        if L>=self.L0: return 1e8
        if phi.dim()>1:
            y = []
            for p in phi:
                y.append(self(p))
            return torch.stack(y)
        px,py,pz,x,y,z,particle,W = self.simulate(phi,muons)
        return self.calc_loss(None,None,None,x,y,None,particle,W,L)
    
    def propagate_to_sensitive_plane(self,px,py,pz,x,y,z, epsilon = 1e-12):
        '''Deprecated'''
        return x,y,z
        z += self.sensitive_plane
        x += self.sensitive_plane*px/(pz+epsilon)
        y += self.sensitive_plane*py/(pz+epsilon)
        return x,y,z

    def GetBounds(self,zGap:float = 1.,device = torch.device('cpu'), correct_bounds = True):
        magnet_lengths = [(50 + zGap, 400 + zGap)] * 8  #previously 170-300
        dX_bounds = [(1, 100)] * 2
        dY_bounds = [(1, 200)] * 2 
        gap_bounds = [(2, 70)] * 2 
        yoke_bounds = [(0.25, 4)]
        inner_gap_bounds = [(0., 20.)]
        bounds = magnet_lengths + 8*(dX_bounds + dY_bounds + gap_bounds + yoke_bounds + inner_gap_bounds)
        if self.fSC_mag: 
            bounds[self.parametrization['M2'][4]] = (15,70)
            bounds[self.parametrization['M2'][5]] = (15,70)
        bounds = torch.tensor(bounds,device=device,dtype=torch.get_default_dtype()).T
        return bounds[:,self.params_idx]
    
    def add_fixed_params(self,phi:torch.Tensor):
        if len(phi) != 72:
            new_phi = self.sc_v6.clone().to(phi.device)
            new_phi[torch.as_tensor(self.params_idx,device = phi.device)] = phi
            if self.fSC_mag:
                new_phi[self.parametrization['M2'][2]] = new_phi[self.parametrization['M2'][1]]
                new_phi[self.parametrization['M2'][4]] = new_phi[self.parametrization['M2'][3]]
        else: new_phi = phi
        return new_phi


class ShipMuonShieldCluster(ShipMuonShield):
    DEF_N_SAMPLES = 484449
    def __init__(self,
                 W0:float = 1558731.375,
                 cores:int = 512,
                 n_samples:int = 0,
                 weight_loss_fn:bool = 'exponential',
                 manager_ip='34.65.198.159',
                 port=444,
                 local:bool = False,
                 parallel:bool = False,
                 seed = None,
                 dimensions_phi = 34,
                 fSC_mag:bool = True, 
                 simulate_fields:bool = False,
                 **kwargs) -> None:

        self.W0 = W0
        self.cores = cores# if not parallel else 1
        self.weight_loss_fn = weight_loss_fn
        if 0<n_samples<self.DEF_N_SAMPLES: self.n_samples = n_samples
        else: self.n_samples = self.DEF_N_SAMPLES

        self.dimensions_phi = dimensions_phi
        self.simulate_fields = simulate_fields

        self.manager_cert_path = getenv('STARCOMPUTE_MANAGER_CERT_PATH')
        self.client_cert_path = getenv('STARCOMPUTE_CLIENT_CERT_PATH')
        self.client_key_path = getenv('STARCOMPUTE_CLIENT_KEY_PATH')
        self.server_url = 'wss://%s:%s'%(manager_ip, port)
        self.fSC_mag = fSC_mag
        self.local = local
        self.seed = seed
        if not local:
            from starcompute.star_client import StarClient
            self.star_client = StarClient(self.server_url, self.manager_cert_path, 
                                    self.client_cert_path, self.client_key_path)
        self.parallel = parallel
        if dimensions_phi == 29: self.params_idx = self.fixed_sc
        elif dimensions_phi == 34: self.params_idx = self.hybrid_idx
        elif dimensions_phi == 72: self.params_idx = slice(None)
        self.DEFAULT_PHI = self.DEFAULT_PHI[self.params_idx]

        sys.path.insert(1, join(PROJECTS_DIR,'MuonsAndMatter/python/bin'))
        from run_simulation import get_field
        self.run_magnet = get_field
        self.fields_file = join(PROJECTS_DIR,'MuonsAndMatter/data/outputs/fields.pkl')
        
    def sample_x(self,phi = None):
        if phi is not None and phi.dim()==2:
            cores = int(self.cores/phi.size(0)) #a ser otimizado. como usar todos os cores?
        else: cores = self.cores
        return get_split_indices(cores,self.n_samples) 
    def simulate_mag_fields(self, phi):
        return super().simulate_mag_fields(phi)
    
    def simulate(self,phi:torch.tensor,
                 muons = None, 
                 file = None,
                 sum_outputs = True):
        phi = self.add_fixed_params(phi)
        if muons is None: muons = self.sample_x(phi)
        if self.simulate_fields: 
            self.simulate_mag_fields(phi)

        inputs = split_array_idx(phi.cpu(),muons, file = file) 
        result = self.star_client.run(inputs)
        
        result,W = torch.as_tensor(result,device = phi.device).T

        if not (phi.dim()==1 or phi.size(0)==1):
            W = W.view(phi.size(0),-1)
            result = result.view(phi.size(0),-1)
        W = W.mean(-1)
        if sum_outputs: result = result.sum(-1) #use reduction?
        return result,W
    
    def __call__(self,phi,muons = None, file = None):
        if phi.dim()>1 and not self.parallel:
            y = []
            for p in phi:
                y.append(self(p))
            return torch.stack(y)
        loss,W = self.simulate(phi,muons, file)
        loss += 1
        if self.weight_loss_fn is not None:
            loss *= self.weight_loss(W)
            loss = torch.where(W>3E6,1e8,loss)
        return loss.to(torch.get_default_dtype())    
    







import time
import argparse
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--nodes",type=int,default = 16)
    parser.add_argument("--n_tasks_per_node", type=int, default=32)
    parser.add_argument("--n_tasks", type=int, default=None)
    parser.add_argument("--warm", dest = 'SC', action='store_false')
    parser.add_argument("--file", type=str, default=None)
    parser.add_argument("--cluster", action='store_true')
    parser.add_argument("--field_map", action='store_true')
    args = parser.parse_args()
    
    with open('/home/hep/lprate/projects/BlackBoxOptimization/outputs/optimization_new_parameters_uniform/phi_optm.txt', "r") as txt_file:
        data = [float(line.strip()) for line in txt_file]
    phi_sc_newparameters = np.array(data)
    d = {}
    t0 = time.time()
    params_dict = {#'optimal oliver':ShipMuonShield.opt_oliver,
                    'sc_v6':ShipMuonShield.sc_v6, 
                     #'combi':ShipMuonShield.combi, 
                     #'baseline':ShipMuonShield.baseline_1,
                     }
    
    seed = 0
    N = 100
    for name,phi in params_dict.items(): #
        print(name)
        
        if name == 'sc_v6' and not args.SC: continue
        if args.n_tasks is None: n_tasks = args.nodes*args.n_tasks_per_node
        else: n_tasks = args.n_tasks
        if args.cluster:
            muon_shield = ShipMuonShieldCluster(cores = n_tasks,dimensions_phi=34,sensitive_plane=0,simulate_fields=args.field_map)
            t1 = time.time()
            loss_muons, W = muon_shield.simulate(torch.as_tensor(phi), file = args.file)
            t2 = time.time()
            loss_muons += 1
        else:
            muon_shield = ShipMuonShield(cores = n_tasks,fSC_mag=args.SC, dimensions_phi=34,
                                         sensitive_plane=83.2,input_dist = 0.9,simulate_fields=args.field_map, seed=seed)
            t1 = time.time()
            px,py,pz,x,y,z,particle,W = muon_shield.simulate(torch.as_tensor(phi,dtype = torch.float32))
            t2 = time.time()
            loss_muons = muon_shield.muon_loss(x,y,particle).sum()+1
        print('loss_muons',loss_muons)
        loss = loss_muons*muon_shield.weight_loss(W)
        loss = torch.where(W>3E6,1e8,loss)
        print('loss',loss)
        W_new = muon_shield.get_weight(torch.as_tensor(phi,dtype = torch.float32))
        d[name] = (W,loss_muons,loss, W_new)
        
        seed += 5
        print(f"took {t2-t1} sec")
    print('Results:')
    for name,(W,loss_muons,loss,W_new) in d.items():
        print(name)
        print(f"Weight GEANT: {W}")
        print(f"Weight PYTHON: {W_new}")
        print(f"Muon Loss: {loss_muons}")
        print(f"Total Loss: {loss}")
    
    print(f"Total Time: {time.time()-t0}")
    '''print('Muon loss (N iterations) = ', list(d.values()))
    print('Mean Loss = ', np.mean(list(d.values())))
    print('Mean STD = ', np.std(list(d.values())))
    print('Error = ', np.std(list(d.values()))/np.sqrt(N))'''