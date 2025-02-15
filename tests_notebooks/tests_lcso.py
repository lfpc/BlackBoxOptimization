import sys
import os
PROJECTS_DIR = '/home/hep/lprate/projects'   
os.environ['PROJECTS_DIR'] = PROJECTS_DIR
sys.path.append('/home/hep/lprate/projects/BlackBoxOptimization/src')
from optimizer import OptimizerClass
from problems import ShipMuonShield, ShipMuonShieldCluster
import torch
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import pickle
from scipy.stats.qmc import LatinHypercube
dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_float32_matmul_precision('high')
def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
MUONS_FILE = '/home/hep/lprate/projects/MuonsAndMatter/data/full_sample/full_sample_0.pkl'
from tests_hits_classifier import Classifier, plot_calibration_curve, plot_hist_errors 

class HitsClassifier():
    def __init__(self,
                 device = 'cpu',
                 **classifier_kargs) -> None:
        self.model = Classifier(**classifier_kargs).to(device)
        self.loss_fn = torch.nn.BCEWithLogitsLoss()
    def fit(self, phi,y,x, n_epochs: int = 10, batch_size: int = 2048):
        if phi.dim() == 1:
            phi = phi.unsqueeze(0)
            x = x.unsqueeze(0)
            y = y.unsqueeze(0)
        phi = phi.repeat_interleave(x.shape[1],dim=0)
        x = x.view(-1,x.size(-1))[:,:3]
        y = y.view(-1,1).float()
        inputs = torch.cat([phi,x],dim=-1)
        self.model.train()
        losses = []
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)#torch.optim.SGD(self.model.parameters(), lr=0.01)#
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
        for e in tqdm(range(n_epochs), desc="Training Progress"):
            epoch_losses = []
            permutation = torch.randperm(inputs.size(0))
            for i in range(0, inputs.size(0), batch_size):
                indices = permutation[i:i + batch_size]
                batch_inputs, batch_y = inputs[indices].to(dev), y[indices].to(dev)
                optimizer.zero_grad()
                p_hits = self.model(batch_inputs)
                loss = self.loss_fn(p_hits, batch_y)
                loss.backward()
                optimizer.step()
                epoch_losses.append(loss.item())
            scheduler.step()
            losses.append(sum(epoch_losses) / len(epoch_losses))
        plt.plot(losses, color = 'b', label = 'Train')
        plt.grid()
        plt.legend()
        plt.savefig('losses.png')  
        plt.close()
        return self
    def get_predictions(self, phi,x, batch_size=200000):
        if phi.dim() == 1:
            phi = phi.unsqueeze(0)
            x = x.unsqueeze(0)
        phi = phi.repeat_interleave(x.shape[1],dim=0)
        x = x.view(-1,x.size(-1))[:,:3]
        inputs = torch.cat([phi,x],dim=-1)
        self.model.eval()
        predictions = []
        for i in range(0, inputs.size(0), batch_size):
            batch_inputs = inputs[i:i + batch_size].to(dev)
            batch_predictions = self.model(batch_inputs).sigmoid()
            predictions.append(batch_predictions)
        return torch.cat(predictions, dim=0)
    def __call__(self,phi,x):
        predictions = self.get_predictions(phi,x)
        return predictions  


class LCSO(OptimizerClass):
    def __init__(self,true_model,
                 surrogate_model:torch.nn.Module,
                 bounds:tuple,
                 samples_phi:int,
                 epsilon:float = 0.2,
                 initial_phi:torch.tensor = None,
                 history:tuple = (),
                 WandB:dict = {'name': 'LCSOptimization'},
                 device = torch.device('cuda'),
                 outputs_dir = 'outputs',
                 resume:bool = False
                 ):
        super().__init__(true_model,
                 surrogate_model,
                 bounds = bounds,
                 device = device,
                 history = history,
                 WandB = WandB,
                 outputs_dir = outputs_dir,
                 resume = resume)
        if resume: 
            with open(os.path.join(outputs_dir,'history.pkl'), "rb") as f:
                self.history = pickle.load(f)
        else:
            x = self.true_model.sample_x(initial_phi)
            y = self.calc_y(initial_phi,x)
            self.update_history(initial_phi,y,x)
        self.current_phi = initial_phi
        self.epsilon = epsilon
        self.lhs_sampler = LatinHypercube(d=initial_phi.size(-1))
        self.samples_phi = samples_phi
        self.phi_optimizer = torch.optim.SGD([self.current_phi],lr=0.1, momentum=0.9)
        self.n_epochs = 10
        self.batch_size = 500
    def active_sample_x(self,phi):
        x = self.true_model.sample_x(phi)
        x = x.repeat(phi.size(0), 1,1)
        idx = torch.randperm(x.shape[1])[:500000]
        return x[:,idx]
        
    def sample_phi(self,current_phi):
        sample = (2*self.lhs_sampler.random(n=self.samples_phi)-1)*self.epsilon
        sample = torch.as_tensor(sample,device=current_phi.device,dtype=torch.get_default_dtype())
        return sample+current_phi
    
    def clean_training_data(self):
        dist = (self.history[0]-self.current_phi).abs()
        is_local = dist.le(self.epsilon).all(-1)
        return self.history[0][is_local], self.history[1][is_local], self.history[2][is_local]
    
    def optimization_iteration(self):
        sampled_phi = self.sample_phi(self.current_phi)
        sampled_x = self.active_sample_x(sampled_phi)
        for phi,x in zip(sampled_phi,sampled_x):
            y = self.calc_y(phi,x)
            self.update_history(phi,y,x)
        self.fit_surrogate_model(n_epochs = self.n_epochs, batch_size = self.batch_size)
        self.get_new_phi()
        return self.get_optimal()

    def run_optimization(self, save_optimal_phi: bool = True, save_history: bool = False, **convergence_params):
        super().run_optimization(save_optimal_phi, save_history, **convergence_params)
        x = self.true_model.sample_x(self.current_phi)
        y = self.calc_y(self.current_phi,x)
        self.update_history(self.current_phi,y,x)
        return self.get_optimal()
    def get_new_phi(self):
        self.phi_optimizer.zero_grad()
        x = self.true_model.sample_x(self.current_phi)
        l = self.loss(self.current_phi,self.model(self.current_phi,x).flatten())
        l.backward()
        self.phi_optimizer.step()
        return self.current_phi
    def update_history(self,phi,y,x):
        phi,y,x = phi.cpu(),y.cpu(),x.cpu()
        phi = phi.view(-1,phi.size(-1))
        y = y.reshape(phi.shape[0],-1)
        x = x.reshape(phi.shape[0],-1, x.size(-1))
        if len(self.history) ==0: 
            self.history = phi,y,x
        else:
            self.history = (torch.cat([self.history[0], phi]),torch.cat([self.history[1], y]),torch.cat([self.history[2], x]))
    def calc_y(self,phi,x):
        phi = torch.clamp(phi, *self.true_model.GetBounds())
        return self.true_model.is_hit(*self.true_model.simulate(phi,x, reduction = None))
    def loss(self,phi,y,x = None):
        return y.sum(-1)
        return (x[:,y,-1]).sum(-1)
        
if __name__ == '__main__':
    MUONS_FILE = '/home/hep/lprate/projects/MuonsAndMatter/data/full_sample/full_sample_0.pkl'
    model = HitsClassifier(phi_dim=54, x_dim=3, device=dev)
    ship_problem = ShipMuonShieldCluster(cores=512, fSC_mag=False, dimensions_phi=54,simulate_fields=False, 
                                                apply_det_loss=False,n_samples = 0,muons_file = MUONS_FILE, 
                                                return_files = "/home/hep/lprate/projects/MuonsAndMatter/data/outputs/results")
    optimizer = LCSO(true_model = ship_problem,
                     surrogate_model=model,
                    bounds = ship_problem.GetBounds(),
                    samples_phi = 50,
                    epsilon=0.2,
                    initial_phi=ship_problem.DEFAULT_PHI.cpu())
    optimizer.run_optimization(save_optimal_phi = True, save_history = True, max_iter = 3)

                     
