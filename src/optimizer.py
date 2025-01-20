import botorch
import torch
from tqdm import tqdm
from scipy.stats.qmc import LatinHypercube
from matplotlib import pyplot as plt
from pickle import dump,  load
import wandb
from os.path import join
from gzip import open as gzip_open
from utils import standardize


class OptimizerClass():
    '''Mother class for optimizers'''
    def __init__(self,true_model,
                 surrogate_model,
                 bounds:tuple,
                 initial_phi:torch.tensor = None,
                 device = torch.device('cuda'),
                 history:tuple = (),
                 WandB:dict = {'name': 'Optimization'},
                 outputs_dir = 'outputs',
                 resume:bool = False):
        
        self.device = device
        self.true_model = true_model
        if len(history)==0:
            if resume: 
                with gzip_open(join(outputs_dir,'history.pkl')) as f:
                    self.history = load(f)
            else: self.history = (initial_phi.cpu().view(-1,initial_phi.size(0)),
                      true_model(initial_phi).cpu().view(-1,1))
        else: self.history = history
        #self.model = self.surrogate_model_class(*self.history).to(self.device)
        self._i = len(self.history[0]) if resume else 0
        print('STARTING FROM i = ', self._i)
        self.model = surrogate_model
        self.bounds = bounds.cpu()
        self.wandb = WandB
        self.outputs_dir = outputs_dir
    def loss(self,x = None, y = None):
        return y
    def fit_surrogate_model(self,**kwargs):
        D = self.clean_training_data() #Should we do this here, in every iteration?
        self.model = self.model.fit(*D,**kwargs)
    def update_history(self,phi,y):
        '''Append phi and y to D'''
        phi,y = phi.reshape(-1,self.history[0].shape[1]).cpu(), y.reshape(-1,self.history[1].shape[1]).cpu()
        self.history = (torch.cat([self.history[0], phi]),torch.cat([self.history[1], y]))
    def n_iterations(self):
        return self._i
    def n_calls(self):
        return self.history[1].size(0)
    def stopping_criterion(self,**convergence_params):
        return self._i >= convergence_params['max_iter']
    def get_optimal(self, return_idx = False):
        '''Get the current optimal'''
        idx = self.loss(*self.history).flatten().argmin()
        if return_idx: return self.history[0][idx],self.loss(self.history[0][idx],self.history[1][idx]),idx
        else: return self.history[0][idx],self.loss(self.history[0][idx],self.history[1][idx])
    def clean_training_data(self):
        '''Get samples on history for training'''
        return self.history
    def optimization_iteration(self):
        return torch.empty(1),self.loss(torch.empty(1))
    def run_optimization(self,
                         save_optimal_phi:bool = True,
                         save_history:bool = False,
                         **convergence_params):
        with wandb.init(reinit = True,**self.wandb) as wb, tqdm(initial = self._i,total=convergence_params['max_iter']) as pbar:
            for min_loss,phi,y in zip(self.loss(*self.history).cummin(0).values,*self.history[:2]):
                log = {'loss':self.loss(phi,y).item(), 
                        'min_loss':min_loss}
                #for i,p in enumerate(phi.flatten()):
                #    log['phi_%d'%i] = p
                wb.log(log)
            while not self.stopping_criterion(**convergence_params):
                phi,loss = self.optimization_iteration()
                pbar.update()
                #for i,p in enumerate(phi.flatten()): #Send phi to wandb
                #    log['phi_%d'%i] = p
                if (loss<min_loss.to(self.device)):
                    min_loss = loss
                    if save_optimal_phi:
                        with open(join(self.outputs_dir,f'phi_optm.txt'), "w") as txt_file:
                            for p in phi.view(-1):
                                txt_file.write(str(p.item()) + "\n")
                log = {'loss':loss.item(), 
                        'min_loss':min_loss.item()}
                if save_history:
                    with gzip_open(join(self.outputs_dir,f'history.pkl'), "wb") as f:
                        dump(self.history, f)
                wb.log(log)
                self._i += 1
        wb.finish()
        return phi,loss


    
class LGSO(OptimizerClass):
    def __init__(self,true_model,
                 surrogate_model:torch.nn.Module,
                 bounds:tuple,
                 samples_phi:int,
                 epsilon:float = 0.2,
                 initial_phi:torch.tensor = None,
                 history:tuple = (),
                 WandB:dict = {'name': 'LGSOptimization'},
                 device = torch.device('cuda'),
                 outputs_dir = 'outputs',
                 resume:bool = False
                 ):
        super().__init__(true_model,
                 surrogate_model,
                 bounds = bounds,
                 initial_phi=initial_phi,
                 device = device,
                 history = history,
                 WandB = WandB,
                 outputs_dir = outputs_dir,
                 resume = resume)
        
        self.epsilon = epsilon
        self.lhs_sampler = LatinHypercube(d=initial_phi.size(-1))
        self.samples_phi = samples_phi
        self.current_phi = initial_phi if not resume else history[0][-1]
        self.phi_optimizer = torch.optim.SGD([self.current_phi],lr=0.1)

    def sample_phi(self,current_phi):
        sample = (2*self.lhs_sampler.random(n=self.samples_phi)-1)*self.epsilon
        sample = torch.as_tensor(sample,device=current_phi.device,dtype=torch.get_default_dtype())
        return sample+current_phi
    def loss_fn(self,y):
        return self.true_model.calc_loss(*y)
    def clean_training_data(self):
        dist = (self.history[0]-self.current_phi).abs()#(self.history[0]-phi).norm(2,-1)
        is_local = dist.le(self.epsilon).all(-1)
        return self.history[0][is_local], self.history[1][is_local], self.history[2][is_local]
    def optimization_iteration(self):
        sampled_phi = self.sample_phi(self.current_phi)
        x = self.true_model.sample_x(sampled_phi)
        y = self.true_model.simulate(sampled_phi,x) #aqui já tem que transformar em concatenaçao...
        mask = y.eq(0).all(-1)
        self.update_history(sampled_phi[mask],y[mask],x[mask])
        self.fit_surrogate_model()
        self.get_new_phi()
        return self.get_optimal()
    def run_optimization(self, save_optimal_phi: bool = True, save_history: bool = False, **convergence_params):
        super().run_optimization(save_optimal_phi, save_history, **convergence_params)
        x = self.true_model.sample_x(self.current_phi)
        y = self.true_model.simulate(self.current_phi,x)
        self.update_history(self.current_phi,y,x)
        return self.get_optimal()
    def get_new_phi(self):
        self.phi_optimizer.zero_grad()
        x = self.true_model.sample_x(self.current_phi)
        l = self.loss_fn(self.model(self.current_phi,x))
        l.backward()
        self.phi_optimizer.step()
        return self.current_phi
    def update_history(self,phi,y,x):
        if len(self.history) ==0: self.history = phi,y,x
        else:
            phi,y,x = phi.reshape(-1,self.history[0].shape[1]).to(phi.device), y.reshape(-1,self.history[1].shape[1]).to(phi.device),x.reshape(-1,self.history[2].shape[1]).to(phi.device)
            self.history = (torch.cat([self.history[0], phi]),torch.cat([self.history[1], y]),torch.cat([self.history[2], x]))
    #def n_calls(self):
    #    return self._i*self.samples_phi*self.true_model.n_samples


    
    

class BayesianOptimizer(OptimizerClass):
    
    def __init__(self,true_model,
                 surrogate_model,
                 bounds:tuple,
                 initial_phi:torch.tensor = None,
                 device = torch.device('cuda'),
                 acquisition_fn = botorch.acquisition.LogExpectedImprovement,
                 acquisition_params = {'q':1,'num_restarts': 30, 'raw_samples':5000},
                 history:tuple = (),
                 model_scheduler:dict = {},
                 WandB:dict = {'name': 'BayesianOptimization'},
                 reduce_bounds:int = 4000,
                 outputs_dir = 'outputs',
                 resume:bool = False):
        super().__init__(true_model,
                 surrogate_model,
                 bounds,
                 initial_phi=initial_phi,
                 device = device,
                 history = history,
                 WandB = WandB,
                 outputs_dir = outputs_dir,
                 resume = resume)
        
        self.acquisition_fn = acquisition_fn
        self.acquisition_params = acquisition_params
        self.model_scheduler = model_scheduler
        self._iter_reduce_bounds = reduce_bounds
        if resume: #current model from model_scheduler
            for i in model_scheduler:
                if self._i > i and i>0:
                    self.model = model_scheduler[i]
            if self._i > reduce_bounds and reduce_bounds>0:
                self.reduce_bounds() 
        self.model.deterministic_loss = true_model.deterministic_loss if hasattr(true_model,'deterministic_loss') else None
        self.true_model.apply_det_loss = False
        
    def get_new_phi(self):
        '''Minimize acquisition function, returning the next phi to evaluate'''
        
        loss_best = self.get_optimal()[1].flatten()*(-1)
        acquisition = self.acquisition_fn(self.model, loss_best.to(self.device)) 
        #constraints = [(self.true_model.get_constraints_fn,1)]
        return botorch.optim.optimize.optimize_acqf(acquisition, self.bounds.to(self.device),**self.acquisition_params)[0]
    
    def optimization_iteration(self):
        if self._i in self.model_scheduler:
            self.model = self.model_scheduler[self._i](self.bounds,self.device)
        if self._i == self._iter_reduce_bounds:
            self.reduce_bounds()
        self.fit_surrogate_model()
        phi = self.get_new_phi().cpu()
        y = self.true_model(phi)
        self.update_history(phi,y)
        print('weight', self.true_model.get_weight(phi))
        print('weight loss', self.true_model.weight_loss(self.true_model.get_weight(phi)))
        print('LENGTH', self.true_model.get_total_length(phi))
        print('constraints', self.true_model.get_constraints(phi))
        print('det_loss', self.true_model.deterministic_loss(phi,y))
        print('y:',y)
        print('loss:', self.loss(phi,y))
        y,idx = self.loss(phi,y).flatten().min(0)
        return phi[idx],y
    
    def clean_training_data(self):
        '''Remove samples in D that are not contained in the bounds.'''
        idx = self.bounds[0].le(self.history[0]).logical_and(self.bounds[1].ge(self.history[0])).all(-1)
        return (self.history[0][idx],(-1)*self.history[1][idx])
    
    def reduce_bounds(self,local_bounds:float = 0.1):
        '''Reduce the bounds to the region (+-local_bounds) of the current optimal, respecting also the previous bounds.'''
        phi = self.get_optimal()[0]
        new_bounds = torch.stack([phi*(1-local_bounds),phi*(1+local_bounds)])
        new_bounds[0] = torch.maximum(self.bounds[0],new_bounds[0])
        new_bounds[1] = torch.minimum(self.bounds[1],new_bounds[1])
        new_bounds[1] = torch.maximum(new_bounds[1],0.15*torch.ones_like(new_bounds[1]))
        self.bounds = new_bounds
        self.model.bounds = new_bounds.to(self.device)
    def loss(self,x,y):
        return self.true_model.deterministic_loss(x,y)
    
    
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
        dump(optimizer.D, handle)
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
    
