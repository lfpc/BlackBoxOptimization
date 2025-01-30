import sys
import os
os.environ['PROJECTS_DIR'] = '/home/hep/lprate/projects'
sys.path.append('/home/hep/lprate/projects/BlackBoxOptimization/src')
from optimizer import LGSO
from problems import ShipMuonShield
from models import GANModel, Classifier
import torch
import wandb

dev = 'cpu'#torch.device('cuda' if torch.cuda.is_available() else 'cpu')

wandb.login()
WANDB = {'project': 'Tests', 'group': "testLGSO"}

class Classifier(torch.nn.Module):
    def __init__(self, phi_dim,x_dim = 7, hidden_dim=256):
        super().__init__()
        self.fc1 = torch.nn.Linear(x_dim + phi_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = torch.nn.Linear(hidden_dim, 1)
        self.activation = torch.nn.ReLU()

    def forward(self, x):
        #x = torch.cat((phi,muons),axis = 1)
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.activation(self.fc3(x))
        x = torch.sigmoid(self.fc4(x))
        return x
    
class HitsClassifier():
    def __init__(self,
                 n_models:int = 1,
                 **classifier_kargs) -> None:
        self.models = [Classifier(**classifier_kargs) for i in range(n_models)]
        self.loss_fn = torch.nn.BCELoss()
    def fit(self,phi,y,x,n_epochs:int = 1000):
        inputs = torch.cat([phi.repeat(x.size(0),1),x.repeat(phi.size(0),1)],1)
        for model in self.models:
            optimizer = torch.optim.SGD(model.parameters(),lr = 0.1,momentum=0.9)
            for e in range(n_epochs):
                optimizer.zero_grad()
                p_hits = model(inputs)
                loss = self.loss_fn(p_hits,y)
                loss.backward()
                optimizer.step()
    def get_predictions(self,phi,x):
        inputs = torch.cat([phi.repeat(x.size(0),1),x.repeat(phi.size(0),1)],1)
        return torch.tensor([model(inputs) for model in self.models])
    def loss(self,phi,y,x = None):
        return y.eq(0).all(-1).logical_not().sum()
    
    def __call__(self,phi,x, return_unc = False):
        predictions = self.get_predictions(phi,x)
        if return_unc: return torch.mean(predictions,axis=0), torch.var(predictions,axis=0)
        else: return torch.mean(predictions,axis=0)

class LCSO(LGSO):
    def __init__(self,problem,surrogate_model,bounds,epsilon,samples_phi,initial_phi,device):
        super().__init__(problem,surrogate_model,bounds,epsilon,samples_phi,initial_phi,device)

problem = ShipMuonShield(n_samples = 20000,cores = 45,fSC_mag=False, dimensions_phi=54,
                                        sensitive_plane=82,simulate_fields=False, apply_det_loss = False)
phi_range = problem.GetBounds(device=dev)
optimizer = LGSO(problem, surrogate_model=GANModel(54,643118,64,device = dev), bounds = phi_range,
                    epsilon = 0.2, samples_phi = 3, initial_phi = problem.DEFAULT_PHI,
                    device = dev)


optimizer.run_optimization(save_optimal_phi=True,save_history=True,
                               max_iter = 100,use_scipy=False)
