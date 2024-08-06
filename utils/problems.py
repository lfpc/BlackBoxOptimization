import torch
import gzip
import pickle
import sys
sys.path.insert(1, '/home/hep/lprate/projects/MuonsAndMatter/python/bin')
from run_simulation import run as run_muonshield
import numpy as np
from multiprocessing import Pool

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
    def __init__(self,reduction = 'sum', 
                 epsilon:float = 1e-5,
                 W0:float = 1915820.,
                 cores:int = 45,
                 n_samples:int = 0,
                 z_dist:float = 0.1,
                 average_x:bool = True) -> None:
        
        self.left_margin = 3#2.6
        self.right_margin = 3
        self.y_margin = 5
        self.z_bias = 50
        self.reduction = reduction #to implement
        self.epsilon = epsilon
        self.W0 = W0
        self.cores = cores
        self.muons_file = '/home/hep/lprate/projects/MuonsAndMatter/'+'data/inputs.pkl'#'data/oliver_data_enriched.pkl'
        self.n_samples = n_samples
        self.z_dist = z_dist
        self.average_x = average_x

    def sample_x(self,phi=None):
        with gzip.open(self.muons_file, 'rb') as f:
            x = pickle.load(f)
        if 0<self.n_samples<=x.shape[0]: x = x[:self.n_samples]
        return x

    def simulate(self,phi:torch.tensor,muons = None): #make it to not remove muons due to not being divisible by cores?
        if len(phi) ==42: phi = self.add_fixed_params(phi)
        if muons is None: muons = self.sample_x()
        
        division = int(len(muons) / self.cores)
        workloads = []
        for i in range(self.cores):
            workloads.append(muons[i * division:(i + 1) * division, :])

        with Pool(self.cores) as pool:
            result = pool.starmap(run_muonshield, [(workload,phi.cpu().numpy(),self.z_bias,self.z_dist,True) for workload in workloads])

        all_results = []
        for rr in result:
            resulting_data,weight = rr
            all_results += [resulting_data]
        all_results = torch.as_tensor(np.concatenate(all_results, axis=0).T,device = phi.device)
        return *all_results, torch.as_tensor(weight,device = phi.device)
    
    def muon_loss(self,x,y,charge):
        mask = (x <= self.left_margin) & (-self.right_margin <= x) & (torch.abs(y) < self.y_margin) 
        x = x[mask]
        charge = torch.sign(charge)[mask]
        return torch.sqrt(1 + (charge*x-self.right_margin)/(self.left_margin+self.right_margin)+self.epsilon).sum()
    def weight_loss(self,W):
        return 1+torch.exp(10*(W-self.W0)/self.W0)
    def __call__(self,phi):
        print(self.simulate(phi))
        px,py,pz,x,y,z,charge,W = self.simulate(phi)
        loss = self.muon_loss(x,y,charge)*self.weight_loss(W)
        loss = torch.where(W>3E8,1e8,loss)
        print('peso', W)
        if self.average_x: loss = loss.mean(-1)
        if self.reduction == 'sum': loss = loss.sum()
        return loss
    
    @staticmethod
    def add_fixed_params(phi:torch.tensor):
        return torch.cat((torch.tensor([70.0, 170.0],device = phi.device), 
                         phi[:6],
                         torch.tensor([40.0, 40.0, 150.0, 150.0, 2.0, 2.0, 80.0, 80.0, 150.0, 150.0, 2.0, 2.0],device = phi.device), 
                         phi[6:]))

    
if __name__ == '__main__':
    phi = torch.tensor([208.0, 207.0, 281.0, 248.0, 305.0, 242.0, 72.0, 51.0, 29.0, 46.0, 10.0, 7.0, 54.0,
                         38.0, 46.0, 192.0, 14.0, 9.0, 10.0, 31.0, 35.0, 31.0, 51.0, 11.0, 3.0, 32.0, 54.0, 
                         24.0, 8.0, 8.0, 22.0, 32.0, 209.0, 35.0, 8.0, 13.0, 33.0, 77.0, 85.0, 241.0, 9.0, 26.0])
    problem = ShipMuonShield()
    x = problem.sample_x()
    loss = problem(phi)
    print(f'LOSS = {loss}')
    