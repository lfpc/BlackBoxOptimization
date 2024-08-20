import torch
import gzip
import pickle
import sys
import numpy as np
from multiprocessing import Pool
sys.path.insert(1, '/home/hep/lprate/projects/MuonsAndMatter/python/bin')
from run_simulation import run as run_muonshield

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
    DEFAULT_PHI = torch.tensor([[205.,205.,280.,245.,305.,240.,87.,65.,
    35.,121,11.,2.,65.,43.,121.,207.,11.,2.,6.,33.,32.,13.,70.,11.,5.,16.,112.,5.,4.,2.,15.,34.,235.,32.,5.,
    8.,31.,90.,186.,310.,2.,55.]],device = torch.device('cuda'))

    MUON = 13

    def __init__(self,
                 W0:float = 1915820.,
                 cores:int = 45,
                 n_samples:int = 0,
                 input_dist:float = 0.1,
                 sensitive_plane:float = 50,#distance between end of shield and sensplane
                 average_x:bool = True,
                 loss_with_weight:bool = True) -> None:
        
        self.left_margin = 2.6
        self.right_margin = 3
        self.y_margin = 5
        self.z_bias = 50
        self.W0 = W0
        self.cores = cores
        self.muons_file = '/home/hep/lprate/projects/MuonsAndMatter/'+'data/inputs.pkl'#'data/oliver_data_enriched.pkl'
        self.n_samples = n_samples
        self.input_dist = input_dist
        self.average_x = average_x
        self.loss_with_weight = loss_with_weight
        self.sentitive_plane = sensitive_plane

    def sample_x(self,phi=None):
        with gzip.open(self.muons_file, 'rb') as f:
            x = pickle.load(f)
        if 0<self.n_samples<=x.shape[0]: x = x[:self.n_samples]
        return x
    def simulate(self,phi:torch.tensor,muons = None): #make it to not remove muons due to not being divisible by cores?
        phi = phi.flatten() #Can we make it paralell on phi also?
        if len(phi) ==42: phi = self.add_fixed_params(phi)
        if muons is None: muons = self.sample_x()
        division = int(len(muons) / (self.cores))
        workloads = []
        for i in range(self.cores):
            workloads.append(muons[i * division:(i + 1) * division, :])
        for j,w in enumerate(muons[(i + 1) * division:, :]):    
            workloads[j] = np.append(workloads[j],w.reshape(1,-1),axis=0)

        sensitive_film_params = {'dz': 0.01, 'dx': 20, 'dy': 30,'position': 0} #the center is in end of muon shield + position
        with Pool(self.cores) as pool:
            result = pool.starmap(run_muonshield, 
                                  [(workload,phi.cpu().numpy(),self.z_bias,self.input_dist,True,sensitive_film_params) for workload in workloads])

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
        px,py,pz,x,y,z,particle,W = self.simulate(phi,muons)
        x,y,z = self.propagate_to_sensitive_plane(px,py,pz,x,y,z)
        loss = self.muon_loss(x,y,particle).sum()+1
        if self.loss_with_weight:
            loss *= self.weight_loss(W)
            loss = torch.where(W>3E6,1e8,loss)
        return loss.to(torch.get_default_dtype())
    
    def propagate_to_sensitive_plane(self,px,py,pz,x,y,z):
        z += self.sentitive_plane
        x += self.sentitive_plane*px/pz
        y += self.sentitive_plane*py/pz
        return x,y,z

    @staticmethod
    def GetBounds(zGap:float = 1.,device = torch.device('cpu')):
        magnet_lengths = [(170 + zGap, 300 + zGap)] * 6  
        dX_bounds = [(10, 100)] * 2
        dY_bounds = [(20, 200)] * 2 
        gap_bounds = [(2, 70)] * 2 
        bounds = magnet_lengths + 6*(dX_bounds + dY_bounds + gap_bounds)
        bounds = torch.tensor(bounds,device=device,dtype=torch.get_default_dtype()).T
        bounds[0] = torch.minimum(bounds[0],ShipMuonShield.DEFAULT_PHI[0].to(device))
        bounds[1] = torch.maximum(bounds[1],ShipMuonShield.DEFAULT_PHI[0].to(device))
        return bounds
    @staticmethod
    def add_fixed_params(phi:torch.tensor):
        return torch.cat((torch.tensor([70.0, 170.0],device = phi.device), 
                         phi[:6],
                         torch.tensor([40.0, 40.0, 150.0, 150.0, 2.0, 2.0, 80.0, 80.0, 150.0, 150.0, 2.0, 2.0],device = phi.device), 
                         phi[6:]))

    
if __name__ == '__main__':
    phi_baseline = torch.tensor([208.0, 207.0, 281.0, 248.0, 305.0, 242.0, 72.0, 51.0, 29.0, 46.0, 10.0, 7.0, 54.0,
                         38.0, 46.0, 192.0, 14.0, 9.0, 10.0, 31.0, 35.0, 31.0, 51.0, 11.0, 3.0, 32.0, 54.0, 
                         24.0, 8.0, 8.0, 22.0, 32.0, 209.0, 35.0, 8.0, 13.0, 33.0, 77.0, 85.0, 241.0, 9.0, 26.0])
    default_oliver = torch.tensor([205.,205.,280.,245.,305.,240.,87.,65.,
    35.,121,11.,2.,65.,43.,121.,207.,11.,2.,6.,33.,32.,13.,70.,11.,5.,16.,112.,5.,4.,2.,15.,34.,235.,32.,5.,
    8.,31.,90.,186.,310.,2.,55.])
    
    array_values = []
    with open('/home/hep/lprate/projects/BlackBoxOptimization/outputs/optimizationsensitive32/phi_optm.txt', "r") as txt_file:
        for line in txt_file:
            array_values.append(float(line.strip()))
    phi_new_opt = torch.tensor(array_values)
    phi = phi_new_opt
    d = {'Weight':{}, 'MuonLoss':{}, 'WeightLoss':{},'TotalLoss':{}}
    for name,phi in {'Oliver opt':phi_baseline,
                     'Def (oliver)':default_oliver,'Opt (new)':phi_new_opt, 
                     'Smaller magnet': ShipMuonShield.GetBounds()[0],
                     'Biggest magnet': ShipMuonShield.GetBounds()[1]}.items():
        problem = ShipMuonShield(cores = 45)
        px,py,pz,x,y,z,particle,W = problem.simulate(phi)
        x,y,z = problem.propagate_to_sensitive_plane(px,py,pz,x,y,z)
        charge = -1*torch.sign(particle)
        mask = (-charge*x <= problem.left_margin) & (-problem.right_margin <= -charge*x) & (torch.abs(y) <= problem.y_margin) & ((torch.abs(particle).to(torch.int))==13)
        muon_loss = problem.muon_loss(x,y,particle).sum()
        weight_loss = problem.weight_loss(W)

        print('N muons:', len(x), '|  filtered:', mask.sum().item())
        print('peso', W.item())
        print(f'MUON LOSS = {muon_loss.item()}')
        print(f'WEIGHT LOSS = {weight_loss.item()}')
        print(f'TOTAL_LOSS = {torch.where(W>3E6,1e8,(muon_loss+1)*weight_loss).item()}')
        

        d['Weight'][name] = W.item()
        d['MuonLoss'][name] = muon_loss.item()
        d['TotalLoss'][name] = torch.where(W>3E6,1e8,muon_loss*weight_loss).item()
    for i,v in d.items():
        print(i)
        print(v)
    #with open('loss_results.pickle', 'wb') as handle:
    #    pickle.dump(d, handle, protocol=pickle.HIGHEST_PROTOCOL)
    from matplotlib import pyplot as plt
    plt.hist(problem.muon_loss(x,y,particle),bins = 'auto')
    plt.savefig('/home/hep/lprate/projects/BlackBoxOptimization/outputs/testloss.png')
    plt.close()
    plt.hist(x[mask][particle[mask]==13],bins = 'auto',label = r'$\mu^-$')
    plt.hist(x[mask][particle[mask]==-13],bins = 'auto',label = r'$\mu^+$')
    plt.legend()
    plt.savefig('/home/hep/lprate/projects/BlackBoxOptimization/outputs/x.png')
    plt.close()
    plt.hist(px[particle==13],bins = 'auto',label = r'$\mu^-$')
    plt.hist(px[particle==-13],bins = 'auto',label = r'$\mu^+$')
    plt.legend()
    plt.savefig('/home/hep/lprate/projects/BlackBoxOptimization/outputs/px.png')
    plt.close()
    plt.hist(py[particle==13],bins = 'auto',label = r'$\mu^-$')
    plt.hist(py[particle==-13],bins = 'auto',label = r'$\mu^+$')
    plt.legend()
    plt.savefig('/home/hep/lprate/projects/BlackBoxOptimization/outputs/py.png')
    plt.close()
    plt.hist(pz[particle==13],bins = 'auto',label = r'$\mu^-$')
    plt.hist(pz[particle==-13],bins = 'auto',label = r'$\mu^+$')
    plt.legend()
    plt.savefig('/home/hep/lprate/projects/BlackBoxOptimization/outputs/pz.png')
    plt.close()
    plt.hist(y[mask],bins = 'auto')
    plt.ylim(-5,5)
    plt.savefig('/home/hep/lprate/projects/BlackBoxOptimization/outputs/y.png')