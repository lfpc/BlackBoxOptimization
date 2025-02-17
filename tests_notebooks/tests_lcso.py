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
import argparse
dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_float32_matmul_precision('high')

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

class Classifier(torch.nn.Module):
    def __init__(self, phi_dim,x_dim = 3, hidden_dim=256):
        super().__init__()
        self.fc1 = torch.nn.Linear(x_dim + phi_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        #self.fc3 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = torch.nn.Linear(hidden_dim, 1)
        self.activation = torch.nn.ReLU()

    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        #x = self.activation(self.fc3(x))
        x = self.fc4(x)
        return x


class HitsClassifier():
    def __init__(self,
                 device = 'cpu',
                 **classifier_kargs) -> None:
        self.model = Classifier(**classifier_kargs).to(device)
        self.loss_fn = torch.nn.BCEWithLogitsLoss()
        self.device = device    
    def concatenate_inputs(self,phi,x):

        if isinstance(phi,list): #if x has irregular shape, phi and x are lists of tensors
            inputs = []
            for phi,x in zip(phi,x):
                inputs.append(self.concatenate_inputs(phi,x))
            return torch.cat(inputs,dim=0)
        if phi.dim() == 1:
            phi = phi.unsqueeze(0)
            x = x.unsqueeze(0)
        phi = phi.repeat_interleave(x.shape[1],dim=0)
        x = x.view(-1,x.size(-1))[:,:3]
        inputs = torch.cat([phi,x],dim=-1)
        return inputs
    
    def fit(self, phi,y:torch.tensor,x, n_epochs: int = 10, batch_size: int = 2048):
        inputs = self.concatenate_inputs(phi,x)
        y = y.view(-1,1).float()
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
    def get_predictions(self,phi,x):
        self.model.eval()
        inputs = self.concatenate_inputs(phi,x)
        inputs = inputs.to(self.device)
        return self.model(inputs).sigmoid()
    def __call__(self,phi,x):
        return self.get_predictions(phi,x).sum()
    def print_metrics(self,phi,x, hits):
        inputs = self.concatenate_inputs(phi,x)
        predictions = self.get_predictions(inputs)
        hits_sum = hits.sum().item()
        expected_hits = predictions.sum().item()
        error = expected_hits - hits_sum

        accuracy = ((predictions > 0.5) == hits).float().mean().item() * 100
        perc_error = (error / hits_sum) * 100
        print(f"Accuracy: {accuracy:.2f}%")
        print(f"Hits: {hits_sum}")
        print(f"Expected number of hits: {expected_hits}")
        print(f"Error: {error}, Percentage Error: {perc_error:.2f}%")


class LCSO(OptimizerClass):
    def __init__(self,true_model,
                 surrogate_model:torch.nn.Module,
                 bounds:tuple,
                 samples_phi:int,
                 epsilon:float = 0.2,
                 subsamples:int = 500000,
                 batch_size:int = 100000,
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
            self.current_phi = self.history[0][-1]
        else:
            x = self.true_model.sample_x(initial_phi)
            y = self.calc_y(initial_phi,x)
            self.update_history(initial_phi,y,x)
            self.current_phi = initial_phi
        self.local_history = ()
        self.epsilon = epsilon
        self.lhs_sampler = LatinHypercube(d=initial_phi.size(-1))
        self.samples_phi = samples_phi
        self.phi_optimizer = torch.optim.Adam([self.current_phi],lr=0.1)#torch.optim.SGD([self.current_phi],lr=0.1, momentum=0.9)
        self.n_epochs = 5
        self.batch_size = batch_size
        self.subsamples = subsamples
    def active_sample_x(self, phi):
        # Get initial samples
        x = torch.as_tensor(self.true_model.sample_x(phi), dtype=torch.get_default_dtype())
        x = x.repeat(phi.size(0), 1, 1)
        
        # Get predictions for all samples
        y_pred = self.model.get_predictions(phi, x)
        
        # Define confidence thresholds
        low_conf = 0.25
        high_conf = 0.75
        
        # Create masks for different regions
        low_conf_mask = (y_pred < low_conf).squeeze()
        high_conf_mask = (y_pred > high_conf).squeeze()
        mid_conf_mask = (~low_conf_mask & ~high_conf_mask)
        
        # Calculate number of samples to take from each region (20% each for high and low)
        total_samples = self.subsamples
        low_samples = total_samples // 5
        high_samples = total_samples // 5
        mid_samples = total_samples - low_samples - high_samples  # Remaining 60% for middle region
        
        def sample_from_mask(mask, n_samples):
            indices = torch.where(mask)[0]
            if len(indices) == 0:
                return torch.tensor([], dtype=torch.long)
            n_samples = min(n_samples, len(indices))
            perm = torch.randperm(len(indices))[:n_samples]
            return indices[perm]
        
        # Sample from each region
        low_indices = sample_from_mask(low_conf_mask, low_samples)
        high_indices = sample_from_mask(high_conf_mask, high_samples)
        mid_indices = sample_from_mask(mid_conf_mask, mid_samples)
        
        # If we didn't get enough samples from the middle region, redistribute to low and high
        remaining = total_samples - (len(low_indices) + len(high_indices) + len(mid_indices))
        if remaining > 0:
            # First try to get all remaining samples from middle region
            extra_mid_indices = sample_from_mask(mid_conf_mask, remaining)
            
            if len(extra_mid_indices) == remaining:
                # If we got all needed samples from middle region, use them
                selected_indices = torch.cat([low_indices, high_indices, mid_indices, extra_mid_indices])
            else:
                # If middle region is exhausted, distribute remaining between low and high
                remaining_after_mid = remaining - len(extra_mid_indices)
                extra_low = remaining_after_mid // 2
                extra_high = remaining_after_mid - extra_low
                
                extra_low_indices = sample_from_mask(low_conf_mask, extra_low)
                extra_high_indices = sample_from_mask(high_conf_mask, extra_high)
                
                selected_indices = torch.cat([
                    low_indices, high_indices, mid_indices, 
                    extra_mid_indices, extra_low_indices, extra_high_indices
                ])
                
                # If we still don't have enough samples, take any remaining available
                final_remaining = total_samples - len(selected_indices)
                if final_remaining > 0:
                    used_indices = selected_indices
                    all_indices = torch.arange(x.shape[1])
                    available_indices = torch.tensor([i for i in all_indices if i not in used_indices])
                    if len(available_indices) > 0:
                        final_indices = available_indices[torch.randperm(len(available_indices))[:final_remaining]]
                        selected_indices = torch.cat([selected_indices, final_indices])
        else:
            selected_indices = torch.cat([low_indices, high_indices, mid_indices])
        
        return x[:, selected_indices]
    def active_sample_x(self,phi):
        x = torch.as_tensor(self.true_model.sample_x(phi),dtype=torch.get_default_dtype())
        x = x.repeat(phi.size(0), 1,1)
        idx = torch.randperm(x.shape[1])[:self.subsamples]
        return x[:,idx]
        
    def sample_phi(self,current_phi):
        sample = (2*self.lhs_sampler.random(n=self.samples_phi)-1)*self.epsilon
        sample = torch.as_tensor(sample,device=current_phi.device,dtype=torch.get_default_dtype())
        return sample+current_phi
    
    def clean_training_data(self):
        dist = (self.local_history[0]-self.current_phi).abs()
        is_local = dist.le(self.epsilon).all(-1)
        return [self.local_history[0][is_local], self.history[0]],torch.cat([self.local_history[1][is_local].view(-1,1),self.history[1].view(-1,1)],dim=0),[self.local_history[2][is_local],self.history[2]]
    
    def optimization_iteration(self):
        with torch.no_grad(): sampled_phi = self.sample_phi(self.current_phi)
        sampled_x = self.active_sample_x(sampled_phi)
        i = 0
        for phi,x in zip(sampled_phi,sampled_x):
            print(i)
            i+=1
            y = self.calc_y(phi,x)
            self.update_local_history(phi,y,x)
        self.fit_surrogate_model(n_epochs = self.n_epochs, batch_size = self.batch_size)

        self.get_new_phi()
        x = self.true_model.sample_x(self.current_phi)
        y = self.calc_y(self.current_phi,x).flatten()
        self.update_history(self.current_phi,y,x)
        return self.current_phi,self.loss(self.current_phi,y)
    
    def get_new_phi(self):
        self.phi_optimizer.zero_grad()
        x = torch.as_tensor(self.true_model.sample_x(self.current_phi),dtype=torch.get_default_dtype())
        x.requires_grad = False
        for i in range(0, x.size(0), self.batch_size):
            x = x[i:i + self.batch_size]#.to(dev)
            batch_loss = self.model(self.current_phi,x)
            batch_loss.backward()

        self.phi_optimizer.step()
        self.current_phi = torch.clamp(self.current_phi, *self.true_model.GetBounds()).cpu()
        return self.current_phi
    
    def update_history(self,phi,y,x):
        x = torch.as_tensor(x,dtype=torch.get_default_dtype(), device = 'cpu')
        phi,y = phi.cpu(),y.cpu()
        phi = phi.view(-1,phi.size(-1))
        y = y.reshape(phi.shape[0],-1)
        x = x.reshape(phi.shape[0],-1, x.size(-1))
        if len(self.history) ==0: 
            self.history = phi,y,x
        else:
            self.history = (torch.cat([self.history[0], phi]),torch.cat([self.history[1], y],dim=0),torch.cat([self.history[2], x]))

    def update_local_history(self,phi,y,x):
        x = torch.as_tensor(x,dtype=torch.get_default_dtype(), device = 'cpu')
        phi,y = phi.cpu(),y.cpu()
        phi = phi.view(-1,phi.size(-1))
        y = y.reshape(phi.shape[0],-1)
        x = x.reshape(phi.shape[0],-1, x.size(-1))
        if len(self.local_history) ==0: 
            self.local_history = phi,y,x
        else:
            self.local_history = (torch.cat([self.local_history[0], phi]),torch.cat([self.local_history[1], y]),torch.cat([self.local_history[2], x]))
    def calc_y(self,phi,x):
        with torch.no_grad():
            phi = torch.clamp(phi, *self.true_model.GetBounds())
            return self.true_model.is_hit(*self.true_model.simulate(phi,x, reduction = None))
    def loss(self,phi,y,x = None):
        return y.sum(-1)
        return (x[:,y,-1]).sum(-1)
        
if __name__ == '__main__':
    MUONS_FILE = '/home/hep/lprate/projects/MuonsAndMatter/data/full_sample/full_sample_0.pkl'
    model = HitsClassifier(phi_dim=54, x_dim=3, device=dev)
    parser = argparse.ArgumentParser(description='LCSO Optimization')
    parser.add_argument('--resume', action='store_true', help='Resume optimization from previous state')
    parser.add_argument('--n_samples_main', type=int, default=0, help='Number of main samples')
    parser.add_argument('--samples_phi', type=int, default=90, help='Number of phi samples')
    parser.add_argument('--subsamples', type=int, default=500_000, help='Number of subsamples')
    parser.add_argument('--max_iter', type=int, default=500, help='Maximum number of iterations')
    parser.add_argument('--batch_size', type=int, default=1_000_000, help='Batch size')
    parser.add_argument('--epsilon', type=float, default=0.2, help='Epsilon value')
    parser.add_argument('--muons_file', type=str, default='/home/hep/lprate/projects/MuonsAndMatter/data/full_sample/full_sample_0.pkl', help='Path to muons file')

    args = parser.parse_args()

    ship_problem = ShipMuonShieldCluster(cores=512, fSC_mag=False, dimensions_phi=54, simulate_fields=False, 
                                         apply_det_loss=False, n_samples=args.n_samples_main, muons_file=args.muons_file, 
                                         return_files="/home/hep/lprate/projects/MuonsAndMatter/data/outputs/results")
    optimizer = LCSO(true_model=ship_problem,
                     surrogate_model=model,
                     bounds=ship_problem.GetBounds(),
                     samples_phi=args.samples_phi,
                     epsilon=args.epsilon,
                     subsamples=args.subsamples,
                     initial_phi=ship_problem.DEFAULT_PHI.cpu())
    optimizer.run_optimization(save_optimal_phi=True, save_history=True, max_iter=args.max_iter)
