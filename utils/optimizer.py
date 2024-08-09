import botorch
import torch
from tqdm import tqdm
from scipy.stats.qmc import LatinHypercube
from matplotlib import pyplot as plt
from pickle import dump, HIGHEST_PROTOCOL
import wandb


class LGSO():
    def __init__(self,true_model,
                 surrogate_model:torch.nn.Module,
                 initial_phi:torch.tensor,
                 epsilon:float,
                 samples_phi:int,
                 loss_fn,
                 D:tuple = ()) -> None:
        self.model = surrogate_model
        self.true_model = true_model
        self.D = D
        self._i = 0
        self.epsilon = epsilon
        self.lhs_sampler = LatinHypercube(d=initial_phi.size(-1))
        self.samples_phi = samples_phi
        self.current_phi = initial_phi
        self.phi_optimizer = torch.optim.SGD([self.current_phi],lr=0.1)
        self.loss_fn = loss_fn

    def sample_phi(self,current_phi):
        return torch.as_tensor((2*self.lhs_sampler.random(n=self.samples_phi)-1)*self.epsilon,device=current_phi.device,dtype=torch.get_default_dtype())+current_phi
    def filter_D(self,phi):
        dist = (self.D[0]-phi).norm(phi.size(-1),-1)
        is_local = dist.le(self.epsilon)
        return self.D[0][is_local], self.D[1][is_local], self.D[2][is_local]
    def run_optimization(self,**convergence_params):
        with tqdm(total=convergence_params['max_iter'],position=0, leave=True, desc = 'Optimization Loop') as pbar:
            while not self.stopping_criterion(**convergence_params):
                sampled_phi = torch.cat((self.sample_phi(self.current_phi),self.current_phi))
                x = self.true_model.sample_x(sampled_phi)
                y = self.true_model(sampled_phi,x)
                self.update_D(sampled_phi,y,x)
                self.fit_surrogate_model(*self.filter_D(self.current_phi))
                self.update_phi()
                self._i += 1
                pbar.update()
                idx = self.D[1].argmin()
        return self.D[0][idx],self.D[1][idx]
    def fit_surrogate_model(self,*args,**kwargs):
        self.model = self.model.fit(*args,**kwargs)
        self.model.eval()
    def update_phi(self):
        self.phi_optimizer.zero_grad()
        x = self.true_model.sample_x(self.current_phi)
        l = self.loss_fn(self.model.generate(condition = torch.cat([self.current_phi,x],dim=-1)))
        l.backward()
        self.phi_optimizer.step()
        return self.current_phi
    def update_D(self,phi,y,x):
        if len(self.D) ==0: self.D = phi,y,x
        else:
            phi,y,x = phi.reshape(-1,self.D[0].shape[1]).to(phi.device), y.reshape(-1,self.D[1].shape[1]).to(phi.device),x.reshape(-1,self.D[2].shape[1]).to(phi.device)
            self.D = (torch.cat([self.D[0], phi]),torch.cat([self.D[1], y]),torch.cat([self.D[2], x]))
    def n_iterations(self):
        return self._i
    def n_calls_fn(self):
        return self._i*self.samples_phi*self.true_model.n_samples
    def stopping_criterion(self,**convergence_params):
        return self._i >= convergence_params['max_iter']
    
    

class BayesianOptimizer():
    
    def __init__(self,true_model,
                 surrogate_model,
                 bounds,
                 initial_phi,
                 device = torch.device('cuda'),
                 acquisition_fn = botorch.acquisition.ExpectedImprovement,
                 acquisition_params = {'num_restarts': 20, 'raw_samples':2000},
                 D:tuple = (),
                 WandB:dict = {'name': 'BayesianOptimization'}):
        
        self.device = device
        self.true_model = true_model
        #self.surrogate_model_class = surrogate_model_class
        self.acquisition_fn = acquisition_fn
        if len(D)==0 or D[0].size ==0: 
            self.D = (initial_phi.view(-1,initial_phi.size(-1)),
                      true_model(initial_phi).view(-1,1))
        else: self.D = D
        #self.model = self.surrogate_model_class(*self.D).to(self.device)
        self._i = 0
        self.acquisition_params = acquisition_params
        self.model = surrogate_model
        self.bounds = bounds
        self.wandb = WandB
        
    def fit_surrogate_model(self,**kwargs):
        self.model = self.model.fit(*self.D,**kwargs)
    
    def get_new_phi(self):
        acquisition = self.acquisition_fn(self.model, self.D[1].min(), maximize=False)
        return botorch.optim.optimize.optimize_acqf(acquisition, self.bounds, q=1,**self.acquisition_params)[0]
    def update_D(self,phi,y):
        phi,y = phi.reshape(-1,self.D[0].shape[1]).to(self.device), y.reshape(-1,self.D[1].shape[1]).to(self.device)
        self.D = (torch.cat([self.D[0], phi]),torch.cat([self.D[1], y]))
    def run_optimization(self, use_scipy = True,**convergence_params):
        with wandb.init(reinit = True,**self.wandb) as wb, tqdm(total=convergence_params['max_iter']) as pbar:
            for phi,y in zip(*self.D):
                print(phi)
                print(y)
                log = {'loss':y.item(), 
                        'min_loss':self.D[1].min().item()}
                for i,p in enumerate(phi.flatten()):
                    log['phi_%d'%i] = p
                wb.log(log)
            print('START')
            while not self.stopping_criterion(**convergence_params):
                options = {'lr': 1e-2, 'maxiter': 100} if not use_scipy else None
                # Create GP model
                self.fit_surrogate_model(use_scipy = use_scipy,options = options)
                phi = self.get_new_phi()
                y = self.true_model(phi)
                self.update_D(phi,y)
                self._i += 1
                pbar.update()
                log = {'loss':y.item(), 
                        'min_loss':self.D[1].min().item()}
                for i,p in enumerate(phi.flatten()):
                    log['phi_%d'%i] = p
                wb.log(log)
        idx = self.D[1].argmin()
        wb.finish()
        return self.D[0][idx],self.D[1][idx]
    
    def n_iterations(self):
        return self._i
    
    def stopping_criterion(self,**convergence_params):
        return self._i >= convergence_params['max_iter']
    
    
if __name__ == '__main__':
    from problems import stochastic_RosenbrockProblem
    from models import GANModel,GP_RBF
    dev = torch.device('cuda')
    n_samples_x = 21
    dimensions_phi = 2
    bounds = [-10.,10.]
    bounds = torch.as_tensor(bounds,device=dev).view(2,-1)  
    if bounds.size(0) != dimensions_phi: bounds = bounds.repeat(1,dimensions_phi)
    problem = stochastic_RosenbrockProblem(n_samples=n_samples_x,average_x=True)
    model = GP_RBF(bounds)#GANModel(problem,10,1,16,epochs = 20,iters_discriminator=25,iters_generator=5,device=dev)
    phi = 2*torch.ones(5,dimensions_phi,device=dev)
    optimizer = BayesianOptimizer(problem,
                 model,
                 bounds,
                 initial_phi = phi)#LGSO(problem,model,phi, loss_fn = torch.mean, samples_phi= 11, epsilon=0.2)
    print(optimizer.D)
    optimizer.run_optimization(max_iter = 5000)

    plt.grid()
    plt.plot(optimizer.D[1].cpu().numpy())
    plt.savefig('optimizer_test.png')
    plt.close()
    with open('D_lgso', 'wb') as handle:
        dump(optimizer.D, handle, protocol=HIGHEST_PROTOCOL)
    with torch.no_grad():
        phi = torch.rand(50,10,device=dev)
        x = problem.sample_x(phi,n_samples_x).view(-1,1)
        phi = phi.repeat(n_samples_x,1)
        y = problem(phi,x).cpu().numpy()
        y_gen = model.generate(torch.cat((phi,x),dim=-1)).cpu().numpy()
    plt.scatter(y_gen,y)
    plt.grid()
    plt.plot([y_gen.min(),y_gen.max()],[y_gen.min(),y_gen.max()],'k--')
    plt.savefig('testgan.png')
    plt.close()
    
