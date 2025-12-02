import os
import botorch
import torch
import torch.multiprocessing as mp
from tqdm import tqdm
from scipy.stats.qmc import LatinHypercube
from matplotlib import pyplot as plt
from pickle import dump,  load
import wandb
from os.path import join, exists
from scipy.optimize import minimize, Bounds as ScipyBounds, NonlinearConstraint
from scipy.spatial.distance import pdist
import numpy as np
import sys
sys.path.append('..')
from utils.acquisition_functions import Custom_LogEI
from utils import normalize_vector, denormalize_vector
#torch.set_default_dtype(torch.float64)
from time import time
import random
import csv
#import gymnasium as gym
#import d3rlpy
#from d3rlpy.models import IQNQFunctionFactory
#from d3rlpy.metrics import EnvironmentEvaluator
import h5py
import cma


class OptimizerClass():
    '''Mother class for optimizers'''
    def __init__(self,true_model,
                 surrogate_model,
                 bounds:tuple,
                 device = torch.device('cuda'),
                 history:tuple = (),
                 WandB:dict = {'name': 'Optimization'},
                 outputs_dir = 'outputs',
                 resume:bool = False):
        
        self.device = device
        self.true_model = true_model
        self.history = history
        #self.model = self.surrogate_model_class(*self.history).to(self.device)
        if len(history)>0: self._i = len(self.history[0]) 
        else: self._i =  0
        print('STARTING FROM i = ', self._i)
        self.model = surrogate_model
        self.bounds = bounds.cpu()
        self.wandb = WandB
        self.outputs_dir = outputs_dir
    def loss(self,x = None, y = None):
        return y
    def fit_surrogate_model(self,**kwargs):
        D = self.clean_training_data()
        self.model = self.model.fit(*D,**kwargs)
    def update_history(self,phi,y):
        '''Append phi and y to D'''
        if y.dim() == 0:
            y = y.reshape(-1, 1)
        phi,y = phi.reshape(-1,phi.shape[-1]).cpu(), y.reshape(-1,y.shape[-1]).cpu()
        if len(self.history) == 0:
            self.history = (phi,y)
        else: self.history = (torch.cat([self.history[0], phi]),torch.cat([self.history[1], y]))
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
                if (loss<min_loss.to(self.device)):
                    min_loss = loss
                    if save_optimal_phi:
                        with open(join(self.outputs_dir,f'phi_optm.txt'), "w") as txt_file:
                            for p in self.true_model.add_fixed_params(phi).view(-1):
                                txt_file.write(str(p.item()) + "\n")
                    pbar.set_description(f"Opt: {min_loss.item()} (it. {self._i})")
                pbar.update()
                log = {'loss':loss.item(), 
                        'min_loss':min_loss.item()}
                if save_history:
                    with open(join(self.outputs_dir,f'history.pkl'), "wb") as f:
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
                 device = device,
                 history = history,
                 WandB = WandB,
                 outputs_dir = outputs_dir,
                 resume = resume)
        if resume: 
            with open(join(outputs_dir,'history.pkl'), "rb") as f:
                self.history = load(f)
        else: self.simulate_and_update(initial_phi)
            
            
        initial_phi = initial_phi if not resume else history[0][-1]
        self._current_phi = torch.nn.Parameter(initial_phi.detach().clone().to(device))
        assert torch.all(self._current_phi >= self.bounds[0].to(self.device)) and torch.all(self._current_phi <= self.bounds[1].to(self.device)), "current_phi is out of bounds"
        self.epsilon = epsilon
        self.lhs_sampler = LatinHypercube(d=initial_phi.size(-1))
        self.samples_phi = samples_phi
        self.phi_optimizer = torch.optim.SGD([self._current_phi],lr=0.01)

    def sample_phi(self):
        """Draw samples in a hypercube of side 2*epsilon around current_phi."""
        H = self.lhs_sampler.random(self.samples_phi)
        with torch.no_grad():
            perturb = (2*torch.from_numpy(H).to(dtype=torch.get_default_dtype()) - 1.0) * self.epsilon
            phis = (self._current_phi.unsqueeze(0).cpu() + perturb).clamp(self.bounds[0], self.bounds[1])
        return phis
    def loss(self,phi,y,x = None):
        return self.true_model.calc_loss(*y.T)
    def clean_training_data(self):
        dist = (self.history[0]-self._current_phi.cpu()).abs()#(self.history[0]-phi).norm(2,-1)
        is_local = dist.le(self.epsilon).all(-1)
        return self.history[0][is_local], self.history[1][is_local], self.history[2][is_local]
    def simulate_and_update(self, phi):
        x = torch.as_tensor(self.true_model.sample_x(phi),device='cpu', dtype=torch.get_default_dtype())
        y = self.true_model.simulate(phi.detach().cpu(),x, return_nan = True).T
        self.update_history(phi,y,x)
        with torch.no_grad():
            print('real Y:', y)
            print('number of non-zeros', y.ne(0).all(dim=1).sum().item(), 'out of', y.size(0))

    def optimization_iteration(self):
        """
        1) Sample locally around current_phi
        2) Evaluate true_model
        3) Fit local surrogate
        4) Update current_phi by minimizing surrogate
        5) Return (phi, loss)
        """
        # 1) local sampling
        phis = self.sample_phi()
        # 2) true evaluations
        x_samp = torch.as_tensor(self.true_model.sample_x(phis))
        for phi in phis:
            self.simulate_and_update(phi)
        self.fit_surrogate_model()

        # 4) move current_phi by gradient descent on surrogate
        # prepare surrogate inputs
        #phi_rep = self._current_phi.unsqueeze(0).repeat(x_samp.size(0), 1)
        x_samp = torch.as_tensor(self.true_model.sample_x(self._current_phi), device=self.device, dtype=torch.get_default_dtype())
        self.phi_optimizer.zero_grad()
        y_pred = self.model.generate(self._current_phi, x_samp)
        loss_sur = self.loss(self._current_phi, y_pred).mean()
        loss_sur.backward()
        print("Gradients w.r.t. current_phi:", self._current_phi.grad)
        print("Loss on surrogate:", loss_sur.item())
        self.phi_optimizer.step()
        with torch.no_grad():
            y_pred2 = self.model.generate(self._current_phi, x_samp)
            print("Loss on surrogate after step:", self.loss(self._current_phi, y_pred2).item())
            
        # clip to bounds
        with torch.no_grad():
            self._current_phi.data = self._current_phi.data.clamp(self.bounds[0], self.bounds[1])

        # 5) return new candidate and its true loss
        x_new = torch.as_tensor(self.true_model.sample_x(self._current_phi), device=self.device)
        y_new = self.true_model.simulate(self._current_phi.unsqueeze(0).detach(), x_new, return_nan=True).T
        with torch.no_grad():
            loss_new = self.loss(self._current_phi.unsqueeze(0), y_new)
        return self._current_phi.clone(), loss_new.flatten().item()
    def run_optimization(self, save_optimal_phi: bool = True, save_history: bool = False, **convergence_params):
        super().run_optimization(save_optimal_phi, save_history, **convergence_params)
        x = self.true_model.sample_x(self._current_phi)
        y = self.true_model.simulate(self._current_phi,x)
        self.update_history(self._current_phi,y,x)
        return self.get_optimal()
    def get_new_phi(self):
        self.phi_optimizer.zero_grad()
        x = torch.as_tensor(self.true_model.sample_x(self._current_phi), device=self.device, dtype=torch.get_default_dtype())
        phi = self._current_phi.repeat(x.size(0), 1)
        l = self.loss(phi,self.model(phi,x))
        l.backward()
        self.phi_optimizer.step()
        return self._current_phi
    def update_history(self,phi,y,x):
        phi,y,x = phi.cpu(),y.cpu(),x.cpu()
        phi = phi.view(-1,phi.size(-1))
        phi = phi.repeat(y.size(0), 1)
        if len(self.history) ==0: 
            self.history = phi,y.view(-1,y.size(-1)),x.view(-1,x.size(-1))
        else:
            phi,y,x = phi, y.reshape(-1,self.history[1].shape[1]).to(phi.device),x.reshape(-1,self.history[2].shape[1]).to(phi.device)
            self.history = (torch.cat([self.history[0], phi]),torch.cat([self.history[1], y]),torch.cat([self.history[2], x]))

class LCSO(OptimizerClass):
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
                 resume:bool = False,
                 second_order:bool = False,
                 weight_hits:bool = False,
                 initial_lambda_constraints:float = 1e2,
                 initial_lr:float = 1e-3,
                 local_file_storage:str = 'local_results.h5'
                 ):
        super().__init__(true_model,
                 surrogate_model,
                 bounds = bounds,
                 device = device,
                 history = history,
                 WandB = WandB,
                 outputs_dir = outputs_dir,
                 resume = resume)
        
        self.weight_hits = weight_hits
        self.local_results = [[],[]]
        within_lower = initial_phi.ge(self.bounds[0])
        within_upper = initial_phi.le(self.bounds[1])
        within_bounds = within_lower.logical_and(within_upper)
        if not within_bounds.all():
            violating_indices = (~within_bounds).nonzero(as_tuple=True)
            raise ValueError(f"Some initial_phi values are out of bounds at indices: {violating_indices}\nViolating values: {initial_phi[violating_indices]}")
        initial_phi = normalize_vector(initial_phi, self.bounds)
            
            
        initial_phi = initial_phi if not resume else history[0][-1]
        self._current_phi = torch.nn.Parameter(initial_phi.detach().clone().to(device))
        
        self.epsilon = epsilon
        self.lhs_sampler = LatinHypercube(d=initial_phi.size(-1))
        self.samples_phi = samples_phi
        self.phi_optimizer = torch.optim.SGD([self._current_phi],lr=initial_lr)
        self.second_order = second_order
        self.lambda_constraints = initial_lambda_constraints
        self.trust_radius = initial_lr
        if local_file_storage is not None: 
            self.local_file_storage = join(self.outputs_dir, local_file_storage)
            if os.path.exists(self.local_file_storage):
                os.remove(self.local_file_storage)
        else: self.local_file_storage = None
        
        self.rhos = []
        if resume: 
            with open(join(outputs_dir,'history.pkl'), "rb") as f:
                self.history = load(f)
        else: self.simulate_and_update(initial_phi)

    @property
    def current_phi(self):
        """Returns the current phi in the original bounds."""
        return denormalize_vector(self._current_phi, self.bounds)
    def stopping_criterion(self,**convergence_params):
        return self._i >= convergence_params['max_iter']
    
    def sample_phi(self):
        """Draw samples in a hypercube of side 2*epsilon around current_phi."""
        perturb = self.lhs_sampler.random(self.samples_phi//2)
        with torch.no_grad():
            perturb = (2*torch.from_numpy(perturb).to(dtype=torch.get_default_dtype()) - 1.0) * self.epsilon
            if self.second_order:
                perturb_small = self.lhs_sampler.random(self.samples_phi//2)
                perturb_small = (2*torch.from_numpy(perturb_small).to(dtype=torch.get_default_dtype()) - 1.0) * self.epsilon/2
                perturb = torch.cat([perturb, perturb_small], dim=0)
            phis = (self._current_phi.unsqueeze(0).cpu() + perturb).clamp(0.0,1.0)#(self.bounds[0], self.bounds[1])
        return phis
    def sample_phi_uniform(self):
        """Draw samples uniformly within the bounds around current_phi."""
        with torch.no_grad():
            perturb = (torch.rand(self.samples_phi, self._current_phi.size(-1), device=self.device) * 2 - 1) * self.epsilon
            phis = (self._current_phi.unsqueeze(0).cpu() + perturb).clamp(0.0,1.0)
        return phis
    def n_hits(self,y,x = None):
        y = y.flatten()
        w = torch.ones_like(y) if (x is None or not self.weight_hits) else x.flatten().to(y.device)
        return  (w*y).sum()/w.sum()
    def get_model_pred(self, phi):
        phi = normalize_vector(phi, self.bounds)
        x_samp = torch.as_tensor(self.true_model.sample_x(),device=self.device, dtype=torch.get_default_dtype())
        condition = torch.cat([phi.repeat(x_samp.size(0), 1), x_samp], dim=-1)
        y_pred = self.model.predict_proba(condition)
        return self.n_hits(y_pred, x_samp)

    def clean_training_data(self):
        if self.local_file_storage is not None and os.path.exists(self.local_file_storage):
            return [self.local_file_storage]
        return torch.cat(self.local_results[0], dim=0), torch.cat(self.local_results[1], dim=0)

    @torch.no_grad()
    def simulate_and_update(self, phi, update_history:bool = True):
        x = torch.as_tensor(self.true_model.sample_x(phi),device=self.device, dtype=torch.get_default_dtype())
        condition = torch.cat([phi.repeat(x.size(0), 1), x], dim=-1).to(self.device)
        phi = denormalize_vector(phi, self.bounds).clone().detach()
        y = self.true_model(phi.cpu(),x).view(-1,1)

        if self.local_file_storage is not None:
            with h5py.File(self.local_file_storage, 'a') as f:
                if 'condition' not in f:
                    f.create_dataset('condition', data=condition.cpu().numpy(), 
                                maxshape=(None, condition.shape[1]), chunks=True)
                    f.create_dataset('y', data=y.cpu().numpy(), 
                                maxshape=(None, y.shape[1]), chunks=True)
                else:
                    old_size = f['condition'].shape[0]
                    f['condition'].resize((old_size + condition.shape[0], condition.shape[1]))
                    f['condition'][old_size:] = condition.cpu().numpy()
                    
                    f['y'].resize((old_size + y.shape[0], y.shape[1]))
                    f['y'][old_size:] = y.cpu().numpy()
        else: 
            self.local_results[0].append(condition)
            self.local_results[1].append(y)
        
        loss = self.n_hits(y, x).reshape(1)
        if update_history:
            self.update_history(phi, loss.detach())
        return loss


    def optimization_iteration(self):
        """
        1) Sample locally around current_phi
        2) Evaluate true_model
        3) Fit local surrogate
        4) Update current_phi by minimizing surrogate
        5) Return (phi, loss)
        """

        # 1) local sampling
        phis = self.sample_phi()
        for phi in phis:
            self.simulate_and_update(phi, update_history=False)
        
        print('Iteration {} : Finished simulations for local samples.'.format(self._i))
        self.fit_surrogate_model()
        
        x_samp = torch.as_tensor(self.true_model.sample_x(),device=self.device, dtype=torch.get_default_dtype())

        condition = torch.cat([self._current_phi.repeat(x_samp.size(0), 1), x_samp], dim=-1)
        y_pred = self.model.predict_proba(condition)
        
        self.phi_optimizer.zero_grad()
        loss_sur = self.n_hits(y_pred, x_samp)
        constraints = self.true_model.get_constraints(self.current_phi).to(loss_sur.device)
        constraints = -self.lambda_constraints * torch.log(constraints + 1e-4).sum()
        loss_sur = loss_sur + constraints
        loss_sur.backward()
        self.phi_optimizer.step()
        with torch.no_grad():
            self._current_phi.data = self._current_phi.data.clamp(0.0,1.0)#(self.bounds[0], self.bounds[1])
        with torch.no_grad():
            self.local_results = [[],[]]
            if self.local_file_storage is not None and os.path.exists(self.local_file_storage):
                os.remove(self.local_file_storage)
            self.simulate_and_update(self._current_phi)

        return self.history[0][-1], self.history[1][-1]

    def _compute_merit_function(self, phi, objective_loss):
        """
        Computes the L1 penalty merit function value.
        phi_merit = f(x) + mu * ||max(0, -g(x))||_1
        """
        # Constraints g(x) >= 0 are converted to -g(x) <= 0 for the penalty
        constraints = self.true_model.get_constraints(denormalize_vector(phi, self.bounds))
        # L1 norm of constraint violations
        violation = torch.sum(torch.relu(-constraints))
        return objective_loss + self.lambda_constraints * violation
    def surrogate_objective(self,phi, x):
            condition = torch.cat([phi.repeat(x.size(0), 1), x], dim=-1)
            y_pred = self.model.predict_proba(condition).to(phi.device)
            return self.n_hits(y_pred, x)
    def _get_avg_grad_fn(self, x_samp, batch_size):
        num_samples = x_samp.size(0)

        
        def avg_grad_fn(phi):
            if num_samples == 0:
                return torch.zeros_like(phi)

            total_grad = torch.zeros_like(phi)
            num_batches = (num_samples + batch_size - 1) // batch_size
            assert num_batches > 0, "Number of batches must be greater than zero."
            for i in range(num_batches):
                start_idx = i * batch_size
                end_idx = min(start_idx + batch_size, num_samples)
                x_batch = x_samp[start_idx:end_idx].detach()
                def obj_fn(p):
                    return self.surrogate_objective(p, x_batch).float()
                total_grad += torch.func.grad(obj_fn)(phi)
            return total_grad
        
        return avg_grad_fn
    
    def optimization_iteration_second_order(self, debug = False):
        """
        Performs ONE optimization step using a trust-region method,
        implemented entirely in PyTorch.
        """

        phis = self.sample_phi()
        for phi in phis:
            self.simulate_and_update(phi, update_history=False)
        print('Iteration {} : Finished simulations for local samples.'.format(self._i))
        self.fit_surrogate_model()

        x_samp = torch.as_tensor(self.true_model.sample_x(), device=self.device, dtype=torch.get_default_dtype())
        current_phi = self._current_phi.detach().clone()

        constraints_current = self.true_model.get_constraints(denormalize_vector(current_phi, self.bounds))
        constraint_jac = torch.func.jacrev(
            lambda phi: self.true_model.get_constraints(denormalize_vector(phi, self.bounds))
        )(current_phi)


        def grad_fn(phi):
            return torch.func.grad(lambda p: self.surrogate_objective(p, x_samp))(phi)
        surrogate_grad = grad_fn(current_phi)
        def hvp_func(v):
            _, hvp_val = torch.func.jvp(grad_fn, (current_phi,), (v,))
            return hvp_val

        p = self._solve_trust_region_subproblem(surrogate_grad, hvp_func, self.trust_radius)
        if not p.any(): print("Trust-region step is zero, no improvement possible with current model.")
        with torch.no_grad(): 
            cos_step = (torch.dot(surrogate_grad, p) / (torch.linalg.norm(surrogate_grad) * torch.linalg.norm(p) + 1e-10)).item()


        proposed_phi = (current_phi + p).clamp(0.0, 1.0)
        #p = proposed_phi - current_phi
        @torch.no_grad()
        def get_rho():
            current_loss_true = self.history[1][-1]
            merit_current = self._compute_merit_function(current_phi, current_loss_true)

            proposed_loss_true = self.simulate_and_update(proposed_phi, update_history=False)
            merit_proposed = self._compute_merit_function(proposed_phi, proposed_loss_true)
            actual_reduction = merit_current - merit_proposed

            predicted_objective_reduction = -(torch.dot(surrogate_grad, p) + 0.5 * torch.dot(p, hvp_func(p)))
            predicted_penalty_value = self.lambda_constraints * torch.sum(torch.relu(-(constraints_current + constraint_jac @ p)))
            current_penalty_value = self.lambda_constraints * torch.sum(torch.relu(-constraints_current))
            predicted_penalty_reduction = current_penalty_value - predicted_penalty_value
            pred_merit_reduction = predicted_objective_reduction + predicted_penalty_reduction

            if pred_merit_reduction.item() <= 1e-10: 
                 return -1.0 if actual_reduction < 0 else 1.0, proposed_loss_true

            rho = actual_reduction / pred_merit_reduction
            return rho.item(), proposed_loss_true
        
        rho, proposed_loss = get_rho()
        self.rhos.append(rho)

        if rho < 0.25:
            # Poor model fit. Shrink radius. Step is not taken unless rho is positive.
            self.trust_radius = max(self.trust_radius * 0.25, 0.001)
            print(f"Step has poor agreement (rho={rho}). Shrinking trust radius to {self.trust_radius:.3f}")
        elif rho > 0.75 and torch.linalg.norm(p).item() >= 0.8*self.trust_radius:
            self.trust_radius = min(1.25 * self.trust_radius, np.sqrt(self.true_model.dim))
            print(f"Excellent step (rho={rho}). Expanding trust radius to {self.trust_radius:.3f}")

        if rho > 0.1:
            with torch.no_grad():
                self._current_phi.data  = proposed_phi
                self.local_results = [[self.local_results[0][-1]], [self.local_results[1][-1]]]
                self.update_history(self.current_phi.detach(), proposed_loss.detach())
            print("Step accepted.")
        else:
            self.local_results = [[self.local_results[0][0]], [self.local_results[1][0]]]
            print("Step rejected. Proposed loss:", proposed_loss.item(), "Current loss:", self.history[1][-1].item())
        if debug:
            return self.history[0][-1].detach().clone(), self.history[1][-1].detach().clone(), cos_step
        return self.history[0][-1].detach().clone(), self.history[1][-1].detach().clone()


    def _solve_trust_region_subproblem(self, g, hvp_func, trust_radius):
        """
        Solves the trust-region subproblem using the Steihaug CG method.
        """
        p = torch.zeros_like(g)
        r = g.clone()
        d = -r.clone()
        current_phi = self._current_phi.detach().clone()    

        if torch.linalg.norm(r) < 1e-10:
            return p

        for _ in range(len(g)): # Max CG iterations
            Bd = hvp_func(d)
            d_dot_Bd = torch.dot(d, Bd)

            # Safety Check 1: Negative Curvature Detected
            if d_dot_Bd <= 0:
                # Find intersection with trust-region boundary and exit
                p_dot_d = torch.dot(p, d)
                d_dot_d = torch.dot(d, d)
                p_dot_p = torch.dot(p, p)
                rad = torch.sqrt(p_dot_d**2 + d_dot_d * (trust_radius**2 - p_dot_p))
                tau = (rad - p_dot_d) / d_dot_d
                return p + tau * d

            alpha = torch.dot(r, r) / d_dot_Bd

            d_masked = d.clone()
            d_masked[d.abs() < 1e-8] = 1e-8 
            t_lower = (0.0 - (current_phi + p)) / d_masked
            t_upper = (1.0 - (current_phi + p)) / d_masked
            t = torch.where(d > 0, t_upper, t_lower)
            t[t <= 0] = float('inf') 
            alpha_bound, _ = torch.min(t, dim=0)
            alpha = min(alpha, alpha_bound.item())

            p_new = p + alpha * d

            # Safety Check 2: Step Exceeds Trust Radius
            if torch.linalg.norm(p_new) >= trust_radius:
                # Find intersection with trust-region boundary and exit
                p_dot_d = torch.dot(p, d)
                d_dot_d = torch.dot(d, d)
                p_dot_p = torch.dot(p, p)
                rad = torch.sqrt(p_dot_d**2 + d_dot_d * (trust_radius**2 - p_dot_p))
                tau = (rad - p_dot_d) / d_dot_d
                return p + tau * d

            p = p_new
            r_new = r + alpha * Bd
            
            if torch.linalg.norm(r_new) < 1e-10:
                break

            beta = torch.dot(r_new, r_new) / torch.dot(r, r)
            r = r_new
            d = -r + beta * d
            
        return p
    
    def get_new_phi(self):
        self.phi_optimizer.zero_grad()
        x = torch.as_tensor(self.true_model.sample_x(self._current_phi), device=self.device, dtype=torch.get_default_dtype())
        phi = self._current_phi.repeat(x.size(0), 1)
        l = self.loss(phi,self.model(phi,x))
        l.backward()
        self.phi_optimizer.step()
        return self._current_phi
    

class AnalyticalOptimizer(OptimizerClass):
    """
    Optimizer that uses analytical gradients and Hessians from the true model.
    """
    def __init__(self,true_model,
                 bounds:tuple,
                 initial_phi:torch.tensor = None,
                 history:tuple = (),
                 WandB:dict = {'name': 'AnalyticalOptimization'},
                 device = torch.device('cuda'),
                 outputs_dir = 'outputs',
                 resume:bool = False,
                 initial_lambda_constraints:float = 1e2,
                 initial_lr:float = 0.1
                 ):
        super().__init__(true_model,
                 surrogate_model = None,
                 bounds = bounds,
                 device = device,
                 history = history,
                 WandB = WandB,
                 outputs_dir = outputs_dir,
                 resume = resume)
        initial_phi = initial_phi if not resume else history[0][-1]
        initial_phi = normalize_vector(initial_phi, bounds)
        self._current_phi = torch.nn.Parameter(initial_phi.detach().clone().to(device))
        assert torch.all(self._current_phi >= self.bounds[0].to(self.device)) and torch.all(self._current_phi <= self.bounds[1].to(self.device)), "current_phi is out of bounds"
        self.lambda_constraints = initial_lambda_constraints
        self.trust_radius = initial_lr
        self.phi_optimizer = torch.optim.SGD([self._current_phi],lr=initial_lr)
        self.rhos = []
        if resume: 
            with open(join(outputs_dir,'history.pkl'), "rb") as f:
                self.history = load(f)
        else: self.simulate_and_update(initial_phi)
    @property
    def current_phi(self):
        """Returns the current phi in the original bounds."""
        return denormalize_vector(self._current_phi, self.bounds)
    def _compute_merit_function(self, phi, objective_loss):
        """
        Computes the L1 penalty merit function value.
        phi_merit = f(x) + mu * ||max(0, -g(x))||_1
        """
        # Constraints g(x) >= 0 are converted to -g(x) <= 0 for the penalty
        constraints = self.true_model.get_constraints(denormalize_vector(phi, self.bounds))
        # L1 norm of constraint violations
        violation = torch.sum(torch.relu(-constraints))
        return objective_loss + self.lambda_constraints * violation
    def simulate_and_update(self, phi):
        with torch.no_grad():
            phi = denormalize_vector(phi, self.bounds).clone().detach().cpu()
            y = self.true_model(phi).view(-1,1)
            self.update_history(phi,y.detach())
    def optimization_iteration(self):
        """
        1) Sample locally around current_phi
        2) Evaluate true_model
        3) Fit local surrogate
        4) Update current_phi by minimizing surrogate
        5) Return (phi, loss)
        """
        y_pred = self.true_model(self.current_phi)
        self.phi_optimizer.zero_grad()
        constraints = self.true_model.get_constraints(self.current_phi).to(y_pred.device)
        #constraints = -self.lambda_constraints * torch.log(constraints + 1e-4).sum()
        y_pred = y_pred + constraints
        y_pred.backward()
        grad = self._current_phi.grad
        print("Gradients w.r.t. current_phi:", grad)
        self.phi_optimizer.step()
        with torch.no_grad():
            self._current_phi.data = self._current_phi.data.clamp(0.0,1.0)#(self.bounds[0], self.bounds[1])
        self.simulate_and_update(self._current_phi)
        return self.history[0][-1], self.history[1][-1]
    
    def optimization_iteration_second_order(self, debug = False):
        """
        Performs ONE optimization step using a trust-region method
        with ANALYTICAL gradients and Hessian from self.true_model.
        """
        current_phi = self._current_phi.detach().clone()
        denorm_phi = denormalize_vector(current_phi, self.bounds)
        denorm_bounds_diff = self.bounds[1] - self.bounds[0] 
        
        # Get analytical objective derivatives
        grad_denorm = self.true_model.gradient(denorm_phi).flatten()
        grad_norm = grad_denorm * denorm_bounds_diff # Apply chain rule
        
        # Define the Hessian-vector-product (HVP) function using the analytical Hessian
        # HVP_norm(v) = (J^T * H_denorm * J) * v
        # where J is the diagonal matrix diag(denorm_bounds_diff)
        surrogate_objective = lambda phi: self.true_model(phi).sum()
        def hvp_func(p_norm):
            v_denorm = p_norm * denorm_bounds_diff
            _, hvp_denorm_val = torch.func.jvp(
                torch.func.grad(surrogate_objective), 
                (denorm_phi,), 
                (v_denorm,)
            )
            hvp_norm_val = hvp_denorm_val * denorm_bounds_diff
            
            return hvp_norm_val

        # --- Get Analytical Constraint Derivatives (same as original) ---
        constraints_current = self.true_model.get_constraints(denorm_phi)
        constraint_jac = torch.func.jacrev(
            lambda phi: self.true_model.get_constraints(denormalize_vector(phi, self.bounds))
        )(current_phi)

        # 2. Solve the subproblem
        p = self._solve_trust_region_subproblem(grad_norm, hvp_func, self.trust_radius)
        if not p.any(): print("Trust-region step is zero.")
        print("Cosine between grad and step:", (torch.dot(grad_norm, p) / (torch.linalg.norm(grad_norm) * torch.linalg.norm(p) + 1e-10)).item())
        cos_step = (torch.dot(grad_norm, p) / (torch.linalg.norm(grad_norm) * torch.linalg.norm(p) + 1e-10)).item()

        proposed_phi = (current_phi + p).clamp(0.0, 1.0)
        @torch.no_grad()
        def get_rho_analytical():
            current_loss_true = self.history[1][-1]
            merit_current = self._compute_merit_function(current_phi, current_loss_true)

            proposed_loss_true = self.true_model(denormalize_vector(proposed_phi, self.bounds))
            merit_proposed = self._compute_merit_function(proposed_phi, proposed_loss_true)
            actual_reduction = merit_current - merit_proposed

            predicted_objective_reduction = -(torch.dot(grad_norm, p) + 0.5 * torch.dot(p, hvp_func(p)))
            
            predicted_penalty_value = self.lambda_constraints * torch.sum(torch.relu(-(constraints_current + constraint_jac @ p)))
            current_penalty_value = self.lambda_constraints * torch.sum(torch.relu(-constraints_current))
            predicted_penalty_reduction = current_penalty_value - predicted_penalty_value
            
            pred_merit_reduction = predicted_objective_reduction + predicted_penalty_reduction

            if pred_merit_reduction.item() <= 1e-10: 
                return -1.0 if actual_reduction < 0 else 1.0, proposed_loss_true

            rho = actual_reduction / pred_merit_reduction
            return rho.item(), proposed_loss_true
        
        rho, proposed_loss = get_rho_analytical()
        self.rhos.append(rho)

        # 4. Update trust radius and accept/reject step (same as original)
        if rho < 0.25:
            self.trust_radius = max(self.trust_radius * 0.25, 0.001)
            print(f"Step has poor agreement (rho={rho}). Shrinking trust radius to {self.trust_radius:.3f}")
        elif rho > 0.75 and torch.linalg.norm(p).item() >= 0.8*self.trust_radius:
            self.trust_radius = min(1.25 * self.trust_radius, np.sqrt(self.true_model.dim)) # Cap max radius
            print(f"Excellent step (rho={rho}). Expanding trust radius to {self.trust_radius:.3f}")

        if rho > 0.1:
            with torch.no_grad():
                self._current_phi.data  = proposed_phi
                self.update_history(self.current_phi.detach(), proposed_loss.detach())
            print("Step accepted.")
        else:
            print("Step rejected. Proposed loss:", proposed_loss.item(), "Current loss:", self.history[1][-1].item())
            
        if debug:
            return self.history[0][-1].detach().clone(), self.history[1][-1].detach().clone(), cos_step
        return self.history[0][-1].detach().clone(), self.history[1][-1].detach().clone()
    def _solve_trust_region_subproblem(self, g, hvp_func, trust_radius):
        """
        Solves the trust-region subproblem using the Steihaug CG method.
        """
        p = torch.zeros_like(g)
        r = g.clone()
        d = -r.clone()
        current_phi = self._current_phi.detach().clone()    

        if torch.linalg.norm(r) < 1e-10:
            return p

        for _ in range(len(g)): # Max CG iterations
            Bd = hvp_func(d)
            d_dot_Bd = torch.dot(d, Bd)

            # Safety Check 1: Negative Curvature Detected
            if d_dot_Bd <= 0:
                # Find intersection with trust-region boundary and exit
                p_dot_d = torch.dot(p, d)
                d_dot_d = torch.dot(d, d)
                p_dot_p = torch.dot(p, p)
                rad = torch.sqrt(p_dot_d**2 + d_dot_d * (trust_radius**2 - p_dot_p))
                tau = (rad - p_dot_d) / d_dot_d
                return p + tau * d

            alpha = torch.dot(r, r) / d_dot_Bd

            d_masked = d.clone()
            d_masked[d.abs() < 1e-8] = 1e-8 
            t_lower = (0.0 - (current_phi + p)) / d_masked
            t_upper = (1.0 - (current_phi + p)) / d_masked
            t = torch.where(d > 0, t_upper, t_lower)
            t[t <= 0] = float('inf') 
            alpha_bound, _ = torch.min(t, dim=0)
            alpha = min(alpha, alpha_bound.item())

            p_new = p + alpha * d

            # Safety Check 2: Step Exceeds Trust Radius
            if torch.linalg.norm(p_new) >= trust_radius:
                # Find intersection with trust-region boundary and exit
                p_dot_d = torch.dot(p, d)
                d_dot_d = torch.dot(d, d)
                p_dot_p = torch.dot(p, p)
                rad = torch.sqrt(p_dot_d**2 + d_dot_d * (trust_radius**2 - p_dot_p))
                tau = (rad - p_dot_d) / d_dot_d
                return p + tau * d

            p = p_new
            r_new = r + alpha * Bd
            
            if torch.linalg.norm(r_new) < 1e-10:
                break

            beta = torch.dot(r_new, r_new) / torch.dot(r, r)
            r = r_new
            d = -r + beta * d
            
        return p

def gpu_worker(device, indexed_genes_batch, problem_fn, results):
    torch.cuda.set_device(device)
    local_results = []
    for idx, genes in indexed_genes_batch:
        phi = torch.tensor(genes, dtype=torch.float32, device=device).unsqueeze(0)
        y = problem_fn(phi)
        local_results.append((idx, y.cpu()))
    results.extend(local_results)

def split_population(indexed_data, num_chunks):
    avg = len(indexed_data) // num_chunks
    return [indexed_data[i*avg : (i+1)*avg if i < num_chunks-1 else len(indexed_data)]
            for i in range(num_chunks)]

def simulate_population_multiprocessing(population, problem_fn, devices):
    num_gpus = len(devices)
    if num_gpus == 0:
        raise RuntimeError("No GPUs available")
    indexed_genes = list(enumerate([ind.genes for ind in population]))
    #Manager list for shared results:
    manager = mp.Manager()
    results = manager.list()
    #Split work into chunks for each GPU:
    chunks = split_population(indexed_genes, num_gpus)
    mp.set_start_method("spawn", force=True)
    #Launch one process per GPU:
    processes = []
    for device, chunk in zip(devices, chunks):
        p = mp.Process(target=gpu_worker, args=(device, chunk, problem_fn, results))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
    #Sort results by original index to preserve population order:
    ordered_results = sorted(list(results), key=lambda x: x[0])
    return [y for _, y in ordered_results]

def compute_simulation(population, problem_fn, devices):
    #Run GPU parallel computation:
    raw_results = simulate_population_multiprocessing(population, problem_fn, devices)
    #Map results back to Individuals:
    for ind, y in zip(population, raw_results):
        loss = y.item() if torch.is_tensor(y) else float(y)
        ind.unique_representation = ind.get_unique_representation_of_genes()
        ind.simulated_fitness = -loss
    #Update computed_fitness_values dictionary:
    for ind in population:
        unique_representation=ind.unique_representation
        if unique_representation in ind.computed_fitness_values:
            iterations,fitness=ind.computed_fitness_values[unique_representation]
            aux=(ind.simulated_fitness+iterations*fitness)/(iterations+1)
            ind.computed_fitness_values[unique_representation]=(iterations+1,aux)
        else:
            ind.computed_fitness_values[unique_representation]=(1,ind.simulated_fitness)

class Individual():
    def __init__(self,problem_fn,genes,computed_fitness_values,device):
        self.problem_fn=problem_fn
        self.genes=genes.copy()
        self.computed_fitness_values=computed_fitness_values
        self.device=device


    def evaluate(self):
        self.fitness=self.computed_fitness_values[self.unique_representation][1]

    def get_unique_representation_of_genes(self):
        return tuple(round(g, 6) for g in self.genes)

    def clone(self):
        cloned_individual=Individual(self.problem_fn,self.genes,self.computed_fitness_values,self.device)
        return cloned_individual

class Population():
    def __init__(self,problem_fn,population_size,generations,phi_bounds,blend_crossover_probability,blend_crossover_alpha,mutation_probability,local_search_period,num_local_searches,local_search_std_deviations_factor,random_immigration_probability,mutation_std_deviations_factor,tournament_size,elite_size,hall_of_fame_size,device,devices,WandB):
        self.num_evaluations=0
        self.device=device
        self.devices=devices
        self.WandB=WandB
        self.problem_fn=problem_fn
        self.population_size=population_size
        self.generations=generations
        self.computed_fitness_values=dict()
        self.phi_bounds=phi_bounds
        self.blend_crossover_probability=blend_crossover_probability#0.7 is a suitable value
        self.blend_crossover_alpha=blend_crossover_alpha#0.3 is a suitable value
        self.mutation_probability=mutation_probability
        self.random_immigration_probability=random_immigration_probability#0.01 would be a suitable value
        self.mutation_std_deviations_factor=mutation_std_deviations_factor#0.05 would be a suitable value
        self.mutation_std_deviations=[self.mutation_std_deviations_factor*(high.item()-low.item()) for low,high in self.phi_bounds.T]
        self.local_search_period=local_search_period
        self.num_local_searches=num_local_searches
        self.local_search_std_deviations_factor=local_search_std_deviations_factor
        self.local_search_std_deviations=[self.local_search_std_deviations_factor*elem for elem in self.mutation_std_deviations]
        self.tournament_size=tournament_size
        self.population=[]
        for _ in range(self.population_size):
            genes=self.get_initial_genes()
            individual=Individual(self.problem_fn,genes,self.computed_fitness_values,self.device)
            self.population.append(individual)
        compute_simulation(self.population,self.problem_fn,self.devices)
        self.num_evaluations+=len(self.population)
        #torch.cuda.empty_cache()
        for individual in self.population:
            individual.evaluate()
        self.elite=[]
        self.elite_size=elite_size
        self.hall_of_fame=[]
        self.hall_of_fame_size=hall_of_fame_size

    def get_initial_genes(self):
        #initial_genes=[random.uniform(low.item(), high.item()) for low,high in self.phi_bounds.T]
        #print(self.problem_fn.initial_phi)
        initial_genes=[self.problem_fn.initial_phi.tolist()[gene_index]+0.0001*np.random.normal(0, self.mutation_std_deviations[gene_index]) for gene_index in range(len(self.phi_bounds.T))]#TO_DO: Check if a more suitable initialization exists (a more spread initialization over the search space maybe)
        for gene_index in range(len(initial_genes)):#Make sure genes don't escape the bounds
            if initial_genes[gene_index]<self.phi_bounds.T[gene_index][0].item():
                initial_genes[gene_index]=self.phi_bounds.T[gene_index][0].item()
            elif initial_genes[gene_index]>self.phi_bounds.T[gene_index][1].item():
                initial_genes[gene_index]=self.phi_bounds.T[gene_index][1].item()
        return initial_genes

    def update_elite(self):#TO_DO: Check if I would need a more suitable elite, like saving all individuals with a fitness value above a threshold
        unique_signatures = set()
        new_elite = []
        for ind in sorted(self.population, key=lambda ind: ind.fitness, reverse=True):
            sig = ind.unique_representation
            if sig not in unique_signatures:
                new_elite.append(ind)
                unique_signatures.add(sig)
            if len(new_elite) == self.elite_size:
                break
        self.elite = new_elite

    def update_hall_of_fame(self):#TO_DO: Check if I would need a more suitable hall of fame, like saving all individuals with a fitness value above a threshold
        combined = self.hall_of_fame + self.elite
        combined_sorted = sorted(combined, key=lambda ind: ind.fitness, reverse=True)
        unique_signatures = set()
        new_hof = []
        for ind in combined_sorted:
            sig = ind.unique_representation
            if sig not in unique_signatures:
                new_hof.append(ind)
                unique_signatures.add(sig)
            if len(new_hof) == self.hall_of_fame_size:
                break
        self.hall_of_fame = new_hof

    def uniform_swap(self,individual1,individual2):
        for gene_index in range(len(individual1.genes)):
            if random.random()<0.5:
                aux=individual1.genes[gene_index]
                individual1.genes[gene_index]=individual2.genes[gene_index]
                individual2.genes[gene_index]=aux
        return individual1,individual2

    def blend_crossover(self, ind1, ind2):
        child1_genes, child2_genes = [], []
        for i, (g1, g2) in enumerate(zip(ind1.genes, ind2.genes)):
            d = abs(g1 - g2)
            low = min(g1, g2) - self.blend_crossover_alpha * d
            high = max(g1, g2) + self.blend_crossover_alpha * d
            #Sample new genes:
            new_g1 = random.uniform(low, high)
            new_g2 = random.uniform(low, high)
            #Clamp to phi_bounds:
            new_g1 = max(min(new_g1, self.phi_bounds.T[i][1].item()), self.phi_bounds.T[i][0].item())
            new_g2 = max(min(new_g2, self.phi_bounds.T[i][1].item()), self.phi_bounds.T[i][0].item())
            child1_genes.append(new_g1)
            child2_genes.append(new_g2)
        ind1.genes = child1_genes
        ind2.genes = child2_genes
        return ind1, ind2

    def crossover(self, ind1, ind2):
        if random.random() < self.blend_crossover_probability:#Do blend crossover
            return self.blend_crossover(ind1, ind2)
        else:# Do uniform swap
            return self.uniform_swap(ind1, ind2)

    def compute_diversity(self):
        #Create a matrix of genes:
        gene_matrix = np.array([ind.genes for ind in self.population])
        #Normalize each gene to [0,1] based on phi_bounds:
        gene_min = self.phi_bounds.T[:, 0].cpu().numpy()
        gene_max = self.phi_bounds.T[:, 1].cpu().numpy()
        gene_range = gene_max - gene_min + 1e-12#avoid division by zero
        normalized_matrix = (gene_matrix - gene_min) / gene_range
        #Compute pairwise Euclidean distances on normalized genes to obtain gene diversity:
        pairwise_distances = pdist(normalized_matrix, metric='euclidean')
        self.gene_diversity = np.mean(pairwise_distances)
        
        #Compute normalized fitness diversity:
        fitness_values = np.array([ind.fitness for ind in self.population])
        std = np.std(fitness_values)
        mean = np.mean(fitness_values)
        fitness_diversity = std / (abs(mean) + 1e-12)
        #Normalize diversity into [0,1]:
        normalized_div = np.clip(fitness_diversity / (fitness_diversity + 1), 0, 1)
        self.fitness_diverstity=normalized_div

    def mutation(self,individual):
        for gene_index in range(len(individual.genes)):
            if random.random() < self.mutation_probability:
                individual.genes[gene_index]+=np.random.normal(0, self.mutation_std_deviations[gene_index])
                if individual.genes[gene_index]<self.phi_bounds.T[gene_index][0].item():#Make sure genes don't escape the bounds
                    individual.genes[gene_index]=self.phi_bounds.T[gene_index][0].item()
                elif individual.genes[gene_index]>self.phi_bounds.T[gene_index][1].item():
                    individual.genes[gene_index]=self.phi_bounds.T[gene_index][1].item()
        return individual

    def local_search(self,individual):
        cloned_individuals = [individual.clone() for _ in range(self.num_local_searches)]
        for ind in cloned_individuals:
            for gene_index in range(len(ind.genes)):
                if random.random() < self.mutation_probability:
                    ind.genes[gene_index]+=np.random.normal(0, self.local_search_std_deviations[gene_index])
                    if ind.genes[gene_index]<self.phi_bounds.T[gene_index][0].item():#Make sure genes don't escape the bounds
                        ind.genes[gene_index]=self.phi_bounds.T[gene_index][0].item()
                    elif ind.genes[gene_index]>self.phi_bounds.T[gene_index][1].item():
                        ind.genes[gene_index]=self.phi_bounds.T[gene_index][1].item()
        return cloned_individuals

    def random_immigration(self,individual):
        initial_genes=self.get_initial_genes()
        individual.genes=initial_genes.copy()
        return individual

    def update_generation(self):
        #Clone the population:
        cloned_population = [ind.clone() for ind in self.population]
        random.shuffle(cloned_population)
        #Compute diversity:
        self.compute_diversity()
        # Apply crossover and mutation to the cloned population:
        for i in range(0,self.population_size,2):
            if random.random()<self.random_immigration_probability:
                cloned_population[i]=self.random_immigration(cloned_population[i])
                cloned_population[i+1]=self.random_immigration(cloned_population[i+1])
            else:
                cloned_population[i], cloned_population[i+1]=self.crossover(cloned_population[i], cloned_population[i+1])
                cloned_population[i] = self.mutation(cloned_population[i])
                cloned_population[i+1] = self.mutation(cloned_population[i+1])
        #Update the fitness values:
        compute_simulation(cloned_population,self.problem_fn,self.devices)#TO_DO: Check that individuals' fitness is being correctly updated
        self.num_evaluations+=len(cloned_population)
        #torch.cuda.empty_cache()#TO_DO: Check if needed
        #Merge original population and offspring
        self.population+=cloned_population
        for individual in self.population:
            individual.evaluate()
        #Update elite and hall of fame:
        self.update_elite()
        self.update_hall_of_fame()
        #Elite individuals automatically survive to the next generation:
        new_generation=[]
        unique_signatures = set()
        for ind in self.elite:
            new_generation.append(ind)
            unique_signatures.add(ind.unique_representation)
        #Selection process based on tournaments:
        attempts = 0
        max_attempts = self.population_size*10#Maximum number of allowed tournaments
        while len(new_generation) < self.population_size and attempts < max_attempts:
            participants = random.sample(self.population, self.tournament_size)
            winner = max(participants, key=lambda ind: ind.fitness)
            signature = winner.unique_representation
            if signature not in unique_signatures:
                new_generation.append(winner)
                unique_signatures.add(signature)
            else:
                attempts += 1
        #If diversity is too low and the maximum number of allowed tournaments is reached, fill population with random immigrants
        if len(new_generation) < self.population_size:
            missing = self.population_size - len(new_generation)
            print(f"Warning: only {len(unique_signatures)} unique individuals found, creating {missing} random immigrants.")
            immigrants_list=[]
            for _ in range(missing):
                immigrant = Individual(self.problem_fn,self.get_initial_genes(),self.computed_fitness_values,self.device)
                immigrants_list.append(immigrant)
            compute_simulation(immigrants_list,self.problem_fn,self.devices)
            self.num_evaluations+=len(immigrants_list)
            #torch.cuda.empty_cache()
            new_generation+=immigrants_list
            for individual in new_generation:
                individual.evaluate()
        self.population=new_generation.copy()#TO_DO: Check if the .copy() can be removed

    def play_evolution(self):
        hist_best_loss=1e16#Huge number
        hist_best_simulated_loss=1e16
        self.no_progress_counter=0
        with wandb.init(reinit = True,**self.WandB) as wb, tqdm(total=self.generations) as pbar:
            for generation in range(self.generations):
                self.update_generation()
                if generation>0:
                    if -self.hall_of_fame[0].fitness<current_best_loss:
                        self.no_progress_counter=0
                    else:
                        self.no_progress_counter+=1
                if self.no_progress_counter>0 and self.no_progress_counter%self.local_search_period==0:
                    locally_searched_individuals=self.local_search(self.elite[0])
                    compute_simulation(locally_searched_individuals,self.problem_fn,self.devices)
                    self.num_evaluations+=len(locally_searched_individuals)
                    for individual in self.population+locally_searched_individuals:
                        individual.evaluate()
                    locally_searched_individuals_sorted = sorted(locally_searched_individuals, key=lambda ind: ind.fitness, reverse=True)
                    print(f"Fitness of best individual: {self.elite[0].fitness}")
                    print("Fitness of locally searched individuals:")
                    print([ind.fitness for ind in locally_searched_individuals_sorted])
                    if locally_searched_individuals_sorted[0].fitness>self.elite[0].fitness:
                        print(f"Local search found a better individual with fitness: {locally_searched_individuals_sorted[0].fitness}")
                        self.no_progress_counter=0
                        self.population[0]=locally_searched_individuals_sorted[0]
                        self.update_elite()
                        self.update_hall_of_fame()
                current_best_loss=-self.hall_of_fame[0].fitness
                if current_best_loss<hist_best_loss:
                    hist_best_loss=current_best_loss
                current_best_simulated_loss=min([-ind.simulated_fitness for ind in self.population])
                if current_best_simulated_loss<hist_best_simulated_loss:
                    hist_best_simulated_loss=current_best_simulated_loss
                print(f"Generation {generation+1} computed!")
                print(f"Current best loss: {current_best_loss}")
                print(f"Historic best loss: {hist_best_loss}")
                print(f"Historic best simulated loss: {hist_best_simulated_loss}")
                log_dict = {
                    'generation': generation + 1,
                    'current_best_loss': current_best_loss,
                    'hist_best_loss': hist_best_loss,
                    'hist_best_simulated_loss': hist_best_simulated_loss,
                    'gene_diversity': self.gene_diversity,
                    'fitness_diverstity': self.fitness_diverstity,
                    'no_progress_counter': self.no_progress_counter,
                    'num_evaluations':self.num_evaluations
                }
                wb.log(log_dict)
                pbar.set_description(f"hist_best_loss: {log_dict['hist_best_loss']} (gen. {log_dict['generation']})")
                pbar.update()
                self.save_population_history(generation)
        wb.finish()
        return self.hall_of_fame

    def save_population_history(self, generation):
        filename=f"outputs/{self.WandB['name']}/population_history.csv"
        os.makedirs(os.path.dirname(filename), exist_ok=True) if os.path.dirname(filename) else None
        write_header = False
        if not os.path.exists(filename):
            write_header = True
        with open(filename, "a", newline="") as f:
            writer = csv.writer(f)
            if write_header:
                header = ["generation"] + [f"gene_{i}" for i in range(len(self.population[0].genes))] + ["fitness"]
                writer.writerow(header)
            for ind in self.population:
                writer.writerow([generation] + ind.genes + [ind.fitness])

class GA():   
    def __init__(self,problem_fn,population_size,generations,phi_bounds,blend_crossover_probability,blend_crossover_alpha,mutation_probability,local_search_period,num_local_searches,local_search_std_deviations_factor,random_immigration_probability,mutation_std_deviations_factor,tournament_size,elite_size,hall_of_fame_size,device,devices,WandB):
        self.problem_fn=problem_fn
        self.population_size=population_size
        self.generations=generations
        self.phi_bounds=phi_bounds
        self.blend_crossover_probability=blend_crossover_probability
        self.blend_crossover_alpha=blend_crossover_alpha
        self.mutation_probability=mutation_probability
        self.local_search_period=local_search_period
        self.num_local_searches=num_local_searches
        self.local_search_std_deviations_factor=local_search_std_deviations_factor
        self.random_immigration_probability=random_immigration_probability
        self.mutation_std_deviations_factor=mutation_std_deviations_factor
        self.tournament_size=tournament_size
        self.elite_size=elite_size
        self.hall_of_fame_size=hall_of_fame_size
        self.device=device
        self.devices=devices
        self.WandB=WandB
        self.the_population=Population(self.problem_fn,self.population_size,self.generations,self.phi_bounds,self.blend_crossover_probability,self.blend_crossover_alpha,self.mutation_probability,self.local_search_period,self.num_local_searches,self.local_search_std_deviations_factor,self.random_immigration_probability,self.mutation_std_deviations_factor,self.tournament_size,self.elite_size,self.hall_of_fame_size,self.device,self.devices,self.WandB)

    def run_optimization(self):        
        hall_of_fame=self.the_population.play_evolution()
        print("Obtained hall of fame:")
        for individual_index in range(len(hall_of_fame)):
            individual=hall_of_fame[individual_index]
            print(f"Individual: {individual}, fitness value: {individual.fitness}, genes: {individual.genes}")
        self.save_last_population()

    def save_last_population(self):
        #Save genes of best individual of hall of fame
        with open(f"outputs/{self.WandB['name']}/phi_optm_GA.txt", "w") as f:
            for gene in self.the_population.hall_of_fame[0].genes:
                f.write(f"{gene}\n")
        #Save genes of best individual of hall of fame with fixed parameters
        with open(f"outputs/{self.WandB['name']}/phi_optm_GA_with_fixed_params.txt", "w") as f:
            for gene in self.problem_fn.add_fixed_params(torch.tensor(self.the_population.hall_of_fame[0].genes, dtype=torch.float32, device=self.device)):
                f.write(f"{gene}\n")
        #Save computed_fitness_values dictionary:
        with open(f"outputs/{self.WandB['name']}/computed_fitness_values.pkl", "wb") as f:
            dump(self.the_population.population[0].computed_fitness_values, f)
        #Save all individuals in the last generation without the dictionary, the device and the problem_fn:
        for ind in self.the_population.population:
            ind.problem_fn = None
            ind.device = None
            ind.computed_fitness_values = None
        with open(f"outputs/{self.WandB['name']}/last_generation.pkl", "wb") as f:
            dump(self.the_population.population, f)
#"""
class RL_muons_env_new(gym.Env):
    def __init__(self, problem_fn):
        super().__init__()
        self.problem_fn=problem_fn
        self.n_magnets=problem_fn.n_magnets
        self.n_params=problem_fn.n_params

        self.low_bounds,self.high_bounds=self.GetBounds()#TO_DO: Check if I need to set the device here
        self.observation_space = gym.spaces.Box(
            low=self.low_bounds, 
            high=self.high_bounds, 
            dtype=np.float32
        )
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(self.n_params,), dtype=np.float32
        )

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
        self.obs = np.zeros(self.observation_space.shape, dtype=np.float32)
        self.current_magnet=0
        return obs, {}

    def step(self, action):#TO_DO: Consider the restrictions of the variables based on parametrization["warm_idx"] and add_fixed_params()
        start = self.current_magnet * self.n_params
        end   = (self.current_magnet + 1) * self.n_params

        low_slice = np.asarray(self.low_bounds[start:end], dtype=np.float32)
        high_slice = np.asarray(self.high_bounds[start:end], dtype=np.float32)
        self.obs[start:end] = self.map_action_to_bounds(action,low_slice,high_slice)
        self.current_magnet+=1

        terminated = (self.current_magnet >= self.n_magnets)
        reward = 0.0
        if terminated:
            phi=torch.tensor(self.obs, dtype=torch.float32).unsqueeze(0)
            reward = self.problem_fn(phi)
        truncated = False
        info = {}
        return self.obs, reward, terminated, truncated, info

    def render(self, mode="human"):
        print(self.phi)

    def seed(self, seed=None):
        np.random.seed(seed)

    def map_action_to_bounds(self, action, low_slice, high_slice):
        return (action + 1.0) * 0.5 * (high_slice - low_slice) + low_slice

    def GetBounds(self,device = torch.device('cpu')):#TO_DO: I copied this function from problems.py, find a better way to extract the bounds as this would cause problems if the original function is modified
        z_gap = (10,50)
        magnet_lengths = (100, 350)
        dY_bounds = (5, 250)
        dY_yoke_bounds = (5, 450)
        if self.use_B_goal: NI_bounds = (0.1,1.9)
        else: NI_bounds = (0.0, 70e3)
        if self.use_diluted:
            dX_bounds = (1, 85)
            gap_bounds = (5, 80)
            inner_gap_bounds = (0., 30.)
            yoke_bounds = (1,141)
        else:
            dX_bounds = (5, 250)
            gap_bounds = (2, 150)
            yoke_bounds = (0.99,3)
            inner_gap_bounds = (0., 150.)

        bounds_low = torch.tensor([[z_gap[0],magnet_lengths[0], 
                       dX_bounds[0], dX_bounds[0], 
                       dY_bounds[0], dY_bounds[0],
                       gap_bounds[0], gap_bounds[0],
                       yoke_bounds[0], yoke_bounds[0],
                       dY_yoke_bounds[0], dY_yoke_bounds[0],
                       inner_gap_bounds[0], inner_gap_bounds[0],
                       NI_bounds[0]] for _ in range(self.n_magnets)],device=device,dtype=torch.get_default_dtype())
        bounds_high = torch.tensor([[z_gap[1],magnet_lengths[1], 
                        dX_bounds[1], dX_bounds[1], 
                        dY_bounds[1], dY_bounds[1],
                        gap_bounds[1], gap_bounds[1],
                        yoke_bounds[1], yoke_bounds[1],
                        dY_yoke_bounds[1], dY_yoke_bounds[1],
                        inner_gap_bounds[1], inner_gap_bounds[1],
                        NI_bounds[1]] for _ in range(self.n_magnets)],device=device,dtype=torch.get_default_dtype())
        bounds_low[0,0] = 0

        inverted_polarity = self.DEFAULT_PHI[:, 14] < 0
        if inverted_polarity.any():
            bounds_low[inverted_polarity, 14] = -NI_bounds[1]
            bounds_high[inverted_polarity, 14] = -NI_bounds[0]
            if not self.use_diluted:
                bounds_low[inverted_polarity, 8] = 1.0 / yoke_bounds[1]
                bounds_high[inverted_polarity, 8] = 1.0 / yoke_bounds[0]
                bounds_low[inverted_polarity, 9] = 1.0 / yoke_bounds[1]
                bounds_high[inverted_polarity, 9] = 1.0 / yoke_bounds[0]

        if self.use_diluted:
            bounds_low[0,6] = 2.0
            bounds_low[0,7] = 2.0
            bounds_low[0,1] = 120.5
            bounds_low[1,1] = 485.5
            bounds_low[2,1] = 285
            bounds_low[3:,1] = 30
            bounds_high[:,1] = 500
        
        if self.SND:
            bounds_low[-2,1] = 90
            bounds_high[-2,1] = 350
            bounds_low[-2,[12,13]] = 30
            bounds_high[-2,[12,13]] = 150
            bounds_low[-1,1] = 170
            bounds_high[-1,1] = 350
            bounds_low[-1,2] = 30
            bounds_high[-1,2] = 250
            bounds_low[-1,3] = 40
            bounds_high[-1,3] = 250

        if self.fSC_mag: 
            bounds_low[1,0] = 30
            bounds_high[1,0] = 300
            bounds_low[2,0] = 30
            bounds_high[2,0] = 300
            bounds_low[1,1] = 50
            bounds_high[1,1] = 400
            bounds_low[1,2] = 30
            bounds_high[1,2] = 50
            bounds_low[1,3] = 30
            bounds_high[1,3] = 50
            bounds_low[1,4] = 15
            bounds_high[1,4] = 30
            bounds_low[1,5] = 15
            bounds_high[1,5] = 30
            bounds_low[1,6] = 15
            bounds_high[1,6] = 150
            bounds_low[1,7] = 1.0
            bounds_high[1,7] = 4
            bounds_low[1,8] = 1.0
            bounds_high[1,8] = 4
            bounds_low[1,9] = 1.0
            bounds_high[1,9] = 4
        return bounds_low,bounds_high
        bounds_low = apply_index(bounds_low, self.params_idx).flatten()
        bounds_high = apply_index(bounds_high, self.params_idx).flatten()
        bounds = torch.stack([bounds_low, bounds_high])
        return bounds


class RL_muons_env(gym.Env):
    def __init__(self, problem_fn, phi_bounds, max_steps, tolerance, step_scale):
        super().__init__()
        self.problem_fn=problem_fn
        self.phi_bounds=phi_bounds
        self.max_steps = max_steps
        self.tolerance = tolerance
        self.step_scale=step_scale#0.05 would be a suitable value

        self.steps = 0
        self.x = None
        self.prev_f = None
        self.best_x = None
        self.best_f = None

        self.dim=len(self.phi_bounds.T)
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(self.dim,), dtype=np.float32
        )
        low_bounds, high_bounds = self.phi_bounds.numpy()
        self.low_bounds = low_bounds.astype(np.float32)
        self.high_bounds = high_bounds.astype(np.float32)
        obs_low = np.concatenate([self.low_bounds, np.array([-np.inf], dtype=np.float32)])
        obs_high = np.concatenate([self.high_bounds, np.array([np.inf], dtype=np.float32)])
        self.observation_space = gym.spaces.Box(
            low=obs_low, high=obs_high, dtype=np.float32
        )
        self.step_scale=[self.step_scale*(high.item()-low.item()) for low,high in self.phi_bounds.T]

        self.reset()#TO_DO: Check if I need to reset here

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
        #TO_DO: Think about proper initialization
        self.x=[self.problem_fn.initial_phi.tolist()[param_index]+0.0001*np.random.normal(0, self.step_scale[param_index]) for param_index in range(len(self.phi_bounds.T))]
        print(type(self.x))
        print(hola)

        #self.x = np.random.uniform(
        #    -self.init_scale, self.init_scale, size=(self.dim,)
        #).astype(np.float32)

        phi=torch.tensor(self.x, dtype=torch.float32).unsqueeze(0)
        y = self.problem_fn(phi)

        self.prev_f = y
        self.best_x = self.x.copy()
        self.best_f = self.prev_f
        self.steps = 0

        obs = np.concatenate([self.x, np.array([self.prev_f], dtype=np.float32)])
        return obs, {}

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        action = np.clip(action, -1.0, 1.0)
        delta = action * self.step_scale
        self.x = np.clip(self.x + delta, self.low_bounds, self.high_bounds)

        phi=torch.tensor(self.x, dtype=torch.float32).unsqueeze(0)
        y = self.problem_fn(phi)
        f_val=y
        reward = self.prev_f - f_val  #Improvement reward: Note that since we are minimizing the loss, the reward is the previous loss minus the current loss
        self.prev_f = f_val
        self.steps += 1

        # track best
        if f_val < self.best_f:
            self.best_f = f_val
            self.best_x = self.x.copy()

        done = bool(f_val <= self.tolerance)
        truncated = bool(self.steps >= self.max_steps)

        obs = np.concatenate([self.x, np.array([f_val], dtype=np.float32)])
        info = {"best_f": self.best_f, "best_x": self.best_x.copy()}

        return obs, float(reward), done, truncated, info

    def render(self, mode="human"):
        print(f"step={self.steps}, x={self.x}, f(x)={self.prev_f:.4f}, best_f={self.best_f:.4f}")

    def seed(self, seed=None):
        np.random.seed(seed)

class RL():
    def __init__(self,problem_fn,phi_bounds,max_steps,tolerance,step_scale,device,devices,WandB):
        self.problem_fn=problem_fn
        self.phi_bounds=phi_bounds
        self.max_steps=max_steps
        self.tolerance=tolerance
        self.step_scale=step_scale
        self.device=device
        self.devices=devices
        self.WandB=WandB

    def run_optimization(self):        
        env = RL_muons_env(self.problem_fn, self.phi_bounds, self.max_steps, self.tolerance, self.step_scale)
        eval_env = RL_muons_env(self.problem_fn, self.phi_bounds, self.max_steps, self.tolerance, self.step_scale)

        # 2) Build IQN Q-function factory
        iqn_q_function = IQNQFunctionFactory(
            n_quantiles=32,         # number of quantile samples for learning
            n_greedy_quantiles=32,  # quantiles used for greedy action evaluation
            embed_size=64           # size of quantile embedding
        )

        # 3) Create SAC with IQN critic
        sac = d3rlpy.algos.SACConfig(
            q_func_factory=iqn_q_function,   # <-- plug IQN in here
            batch_size=256,
            n_critics=2,
            gamma=0.99,
            tau=5e-3,
            #learning_rate=3e-4,
            initial_temperature=0.1
        ).create(device="cuda:0")            # or device="cpu"
        sac.build_with_env(env)

        # 4) Train online (interacts with env)
        sac.fit_online(
            env,
            eval_env=eval_env,
            n_steps=2000,#20000,          # your total interaction steps
            #logdir="logs/iqn_sac",
            #eval_interval=10_000      # evaluate every N steps
        )

        env_evaluator = EnvironmentEvaluator(eval_env)
        rewards = env_evaluator(sac, dataset=None)  # runs a few episodes and returns average reward
        print("Evaluation rewards:", rewards)

        # 5) Save / load
        sac.save_model(f"outputs/{self.WandB['name']}/iqn_sac_model.d3")
#"""

class CMAES():
    def __init__(self,problem_fn,phi_bounds,initial_step_size,population_size,generations,device,devices,WandB):
        self.problem_fn=problem_fn
        self.phi_bounds=phi_bounds
        self.initial_step_size=initial_step_size
        self.population_size=population_size
        self.generations=generations
        self.device=device
        self.devices=devices
        self.WandB=WandB

    def run_optimization(self,previous_optimization_folder=None):
        low_bounds, high_bounds = self.phi_bounds.cpu().numpy()
        if previous_optimization_folder is not None:
            with open(f"outputs/{previous_optimization_folder}/phi_optm_GA.txt", "r") as f:
                initial_phi = [float(line.strip()) for line in f]
                for i in range(len(initial_phi)):
                    if initial_phi[i]<low_bounds[i]:
                        initial_phi[i]=low_bounds[i]
                    elif initial_phi[i]>high_bounds[i]:
                        initial_phi[i]=high_bounds[i]
        else:
            initial_phi=self.problem_fn.initial_phi.tolist()
        sigma_vector = high_bounds - low_bounds
        es = cma.CMAEvolutionStrategy(initial_phi, self.initial_step_size, {'bounds': [low_bounds, high_bounds], 'popsize': self.population_size, 'CMA_stds': sigma_vector})
        with wandb.init(reinit = True,**self.WandB) as wb, tqdm(total=self.generations) as pbar:
            for generation in range(self.generations):
                solutions = es.ask()
                indexed_solutions=list(enumerate([s for s in solutions]))
                #Manager list for shared results:
                manager = mp.Manager()
                results = manager.list()
                #Split work into chunks for each GPU:
                chunks = split_population(indexed_solutions, len(self.devices))
                mp.set_start_method("spawn", force=True)
                #Launch one process per GPU:
                processes = []
                for device, chunk in zip(self.devices, chunks):
                    p = mp.Process(target=gpu_worker, args=(device, chunk, self.problem_fn, results))
                    p.start()
                    processes.append(p)
                for p in processes:
                    p.join()
                #Sort results by original index to preserve population order:
                ordered_results = sorted(list(results), key=lambda x: x[0])
                losses=[y.item() for _, y in ordered_results]
                es.tell(solutions, losses)
                es.logger.add()
                es.disp()
                current_best_loss=min(losses)
                log_dict = {
                    'generation': generation + 1,
                    'current_best_loss': current_best_loss,
                    'hist_best_loss': es.result.fbest,
                    'num_evaluations':es.result.evaluations
                }
                wb.log(log_dict)
                pbar.set_description(f"hist_best_loss: {log_dict['hist_best_loss']} (gen. {log_dict['generation']})")
                pbar.update()
                self.save_population_history(generation,solutions,losses)
        wb.finish()
        #Save genes of best individual:
        with open(f"outputs/{self.WandB['name']}/phi_optm_CMAES.txt", "w") as f:
            f.write("Solution with best simulated loss:\n")
            for variable in es.result.xbest:
                f.write(f"{variable}\n")
            f.write("Solution with favorite mean loss:\n")
            for variable in es.result.xfavorite:
                f.write(f"{variable}\n")
        #Save genes of best individual of hall of fame with fixed parameters:
        with open(f"outputs/{self.WandB['name']}/phi_optm_CMAES_with_fixed_params.txt", "w") as f:
            f.write("Solution with best simulated loss:\n")
            for variable in self.problem_fn.add_fixed_params(torch.tensor(es.result.xbest, dtype=torch.float32, device=self.device)):
                f.write(f"{variable}\n")
            f.write("Solution with favorite mean loss:\n")
            for variable in self.problem_fn.add_fixed_params(torch.tensor(es.result.xfavorite, dtype=torch.float32, device=self.device)):
                f.write(f"{variable}\n")
        return es

    def save_population_history(self, generation, solutions, losses):
        filename=f"outputs/{self.WandB['name']}/population_history.csv"
        os.makedirs(os.path.dirname(filename), exist_ok=True) if os.path.dirname(filename) else None
        write_header = False
        if not os.path.exists(filename):
            write_header = True
        with open(filename, "a", newline="") as f:
            writer = csv.writer(f)
            if write_header:
                header = ["generation"] + [f"gene_{i}" for i in range(len(solutions[0]))] + ["fitness"]
                writer.writerow(header)
            for i in range(len(solutions)):
                s=solutions[i]
                loss=losses[i]
                writer.writerow([generation] + s.tolist() + [loss])

class BayesianOptimizer(OptimizerClass):
    
    def __init__(self,true_model,
                 surrogate_model,
                 bounds:tuple,
                 initial_phi:torch.tensor = None,
                 device = torch.device('cuda'),
                 acquisition_fn = Custom_LogEI,
                 acquisition_params = {'q':1,'num_restarts': 30, 'raw_samples':4096},
                 history:tuple = (),
                 model_scheduler:dict = {},
                 WandB:dict = {'name': 'BayesianOptimization'},
                 reduce_bounds:int = 4000,
                 outputs_dir = 'outputs',
                 multi_fidelity:bool = False,
                 resume:bool = False):
        
        super().__init__(true_model,
                 surrogate_model,
                 bounds,
                 device = device,
                 history = history,
                 WandB = WandB,
                 outputs_dir = outputs_dir,
                 resume = resume)
        if len(history)==0:
            if resume: 
                with open(join(outputs_dir,'history.pkl'), "rb") as f:
                    self.history = load(f)
                    self.history = tuple(tensor.to(torch.get_default_dtype()) for tensor in self.history)
                self._i = len(self.history[0])
            else: 
                self.history = (initial_phi.cpu().view(-1,initial_phi.size(0)),
                true_model(initial_phi).cpu().view(-1,1))
                self._i = 0
        else: self.history = history
        self.acquisition_fn = acquisition_fn
        self.acquisition_params = acquisition_params
        self.model_scheduler = model_scheduler
        self._iter_reduce_bounds = reduce_bounds
        self.multi_fidelity = multi_fidelity
        if resume: 
            for i in model_scheduler:
                if self._i > i and i>0:
                    self.model = model_scheduler[i]
            if self._i > reduce_bounds and reduce_bounds>0:
                self.reduce_bounds() 
        #self.true_model.apply_det_loss = False
    def get_new_phi(self):
        '''Minimize acquisition function, returning the next phi to evaluate'''
        loss_best = self.get_optimal()[1].flatten()*(-1)
        acquisition = self.acquisition_fn(self.model, 
                                        loss_best.to(self.device), 
                                        deterministic_fn=None,#self.true_model.deterministic_loss if hasattr(self.true_model,'deterministic_loss') else None,
                                        constraint_fn=None)#self.true_model.get_constraints if hasattr(self.true_model,'get_constraints') else None)
        return botorch.optim.optimize.optimize_acqf(acquisition, self.bounds.to(self.device),**self.acquisition_params)[0]
    
    def optimization_iteration(self):
        if self._i in self.model_scheduler:
            self.model = self.model_scheduler[self._i](self.bounds,self.device)
        if self._i % 20 == 0 and self._i >= self._iter_reduce_bounds:
            self.reduce_bounds()
        
        t1 = time()
        self.fit_surrogate_model()
        print('model fit time: ', time()-t1)
        t1 = time()
        phi = self.get_new_phi().cpu()
        print('acquisition function optimization time: ', time()-t1)
        start_of_simulation=time()
        y = self.true_model(phi)
        print(time()-start_of_simulation)
        if self.multi_fidelity and y < self.history[1][0] * 10:
            n_samples = self.true_model.n_samples
            self.true_model.n_samples = 0
            y = self.true_model(phi)
            self.true_model.n_samples = n_samples
        self.update_history(phi,y)
        y,idx = self.loss(phi,y).flatten().min(0)
        return phi[idx],y
    
    
            
    def clean_training_data(self):
        '''Remove samples in D that are not contained in the bounds.'''
        idx = self.bounds[0].le(self.history[0]).logical_and(self.bounds[1].ge(self.history[0])).all(-1)
        assert idx.sum()>0, 'No samples in bounds!'
        return (self.history[0][idx],(-1)*self.history[1][idx])
    
    def reduce_bounds(self,local_bounds:float = 0.1):
        '''Reduce the bounds to the region (+-local_bounds) of the current optimal, respecting also the previous bounds.'''
        phi = self.get_optimal()[0]
        original_bounds = self.true_model.GetBounds()
        
        new_bounds = torch.stack([phi*(1-local_bounds),phi*(1+local_bounds)]).sort(dim=0).values #sort due to negative values
        new_bounds[0] = torch.maximum(original_bounds[0],new_bounds[0])
        new_bounds[1] = torch.minimum(original_bounds[1],new_bounds[1])
        new_bounds[1] = torch.maximum(new_bounds[1],0.1*torch.ones_like(new_bounds[1]))
        self.bounds = new_bounds
        self.model.bounds = new_bounds.to(self.device)
    def loss(self,x,y):
        return y
        return self.true_model.deterministic_loss(x,y)
    

class TurboOptimizer(BayesianOptimizer):
    """
    Implements the Trust Region Bayesian Optimization (TuRBO) strategy by extending 
    the BayesianOptimizer class.

    TuRBO maintains a trust region (a hyper-rectangle) around the best observed solution, 
    adjusting its size based on the success of the optimization. This approach helps to 
    balance exploration and exploitation effectively.
    """
    def __init__(self, 
                 true_model, 
                 surrogate_model, 
                 bounds: tuple,
                 initial_phi: torch.tensor = None,
                 device=torch.device('cuda'), 
                 acquisition_fn=botorch.acquisition.UpperConfidenceBound,
                 acquisition_params={'q':1,'num_restarts': 30, 'raw_samples':4096},
                 history: tuple = (), 
                 WandB: dict = {'name': 'TurboOptimization'},
                 outputs_dir='outputs', 
                 resume: bool = False,
                 # TuRBO specific parameters
                 failure_tolerance: int = 20,
                 success_tolerance: int = 3,
                 length_init: float = 0.2,
                 length_min: float = 0.01,
                 length_max: float = 1.):
        """
        Initializes the TuRBO optimizer.

        Args:
            true_model: The true model to be optimized.
            surrogate_model: The surrogate model used for optimization.
            bounds (tuple): The global bounds of the optimization problem.
            initial_phi (torch.tensor, optional): The starting point for the optimization.
            device (torch.device, optional): The device to run the optimization on.
            acquisition_fn (optional): The acquisition function to use.
            acquisition_params (dict, optional): Parameters for the acquisition function.
            history (tuple, optional): The history of previous evaluations.
            WandB (dict, optional): Configuration for Weights & Biases logging.
            outputs_dir (str, optional): Directory to save outputs.
            resume (bool, optional): Whether to resume from a previous state.
            failure_tolerance (int): Number of consecutive failures to shrink the trust region.
            success_tolerance (int): Number of consecutive successes to expand the trust region.
            length_init (float): Initial side length of the trust region hypercube.
            length_min (float): Minimum side length of the trust region.
            length_max (float): Maximum side length of the trust region.
        """
        super().__init__(true_model=true_model,
                         surrogate_model=surrogate_model,
                         bounds=bounds,
                         initial_phi=initial_phi,
                         device=device,
                         acquisition_fn=acquisition_fn,
                         acquisition_params=acquisition_params,
                         history=history,
                         WandB=WandB,
                         outputs_dir=outputs_dir,
                         resume=resume)

        # Store the original global bounds
        self.global_bounds = bounds.clone().cpu()

        # Initialize TuRBO state
        self.length = length_init
        self.length_min = length_min
        self.length_max = length_max
        self.failure_tolerance = failure_tolerance
        self.success_tolerance = success_tolerance
        self.success_counter = 0
        self.failure_counter = 0

        # Set the initial trust region around the best starting point
        self._update_trust_region()

    def _update_trust_region(self):
        """
        Updates the trust region bounds based on the current best solution and length.
        The new bounds are centered around the best point and clipped by the global bounds.
        """
        phi_best, _ = self.get_optimal()
        
        # Center the trust region around the best point
        half_length = self.length / 2.0
        new_bounds_lower = phi_best - half_length
        new_bounds_upper = phi_best + half_length

        # Clip the trust region to be within the global problem bounds
        self.bounds[0] = torch.max(new_bounds_lower, self.global_bounds[0])
        self.bounds[1] = torch.min(new_bounds_upper, self.global_bounds[1])

    def optimization_iteration(self):
        """
        Performs a single iteration of TuRBO.

        This involves:
        1. Fitting the surrogate model.
        2. Optimizing the acquisition function within the current trust region to get a new candidate.
        3. Evaluating the true model at the new candidate point.
        4. Updating the TuRBO state (success/failure counters and trust region size).
        """
        # Get the best loss before this iteration
        _, loss_best_before = self.get_optimal()

        # Perform a standard Bayesian optimization step within the current trust region
        # The parent method uses self.bounds, which now represent the trust region
        phi_new, loss_new = super().optimization_iteration()

        # Check if the new point is better than the best before this iteration
        if loss_new < loss_best_before:
            self.success_counter += 1
            self.failure_counter = 0
        else:
            self.success_counter = 0
            self.failure_counter += 1

        # Update trust region length based on success/failure
        if self.success_counter >= self.success_tolerance:
            self.length = min(self.length * 2.0, self.length_max)
            self.success_counter = 0
        elif self.failure_counter >= self.failure_tolerance:
            self.length = max(self.length / 2.0, self.length_min)
            self.failure_counter = 0

        self.reduce_bounds(self.length)
        
        if self.length < self.length_min:
            print("Trust region length has reached its minimum. Consider restarting.")

        return phi_new, loss_new

class PowellOptimizer(OptimizerClass):
    """
    Optimizer using Powell's conjugate direction method via scipy.optimize.minimize.
    Logs individual function evaluations to WandB in real-time.
    Preserves direction state between iterations to mimic a single longer run.
    """
    def __init__(self, true_model,
                 bounds: torch.Tensor,
                 initial_phi: torch.Tensor,
                 device=torch.device('cuda'),
                 history: tuple = (),
                 WandB: dict = {'name': 'PowellOptimization'},
                 outputs_dir='outputs',
                 resume: bool = False,
                 scipy_options: dict = None):

        # --- History Initialization (same as before) ---
        loaded_history = ()
        if resume:
            history_path = join(outputs_dir,'history.pkl')
            try:
                with open(history_path, "rb") as f:
                    # Load history AND direction state if saved
                    saved_state = load(f)
                    if isinstance(saved_state, dict) and 'history' in saved_state and 'directions' in saved_state:
                         loaded_history = saved_state['history']
                         self._current_directions = saved_state['directions'] # Restore directions
                         print(f"Resumed history ({len(loaded_history[0])} points) and Powell directions from {history_path}")
                    else: # Handle old format history files
                         loaded_history = saved_state
                         self._current_directions = None # Cannot resume directions
                         print(f"Resumed history ({len(loaded_history[0])} points) from {history_path}. Powell directions reset.")

            except FileNotFoundError:
                print(f"Resume requested, but state file not found at {history_path}. Starting fresh.")
                resume = False
                self._current_directions = None
            except Exception as e:
                print(f"Error loading state file: {e}. Starting fresh.")
                resume = False
                self._current_directions = None

        else: # Not resuming
             self._current_directions = None # Initialize directions state

        if len(history) > 0:
             effective_history = history
             print("Using history provided via argument.")
             # If history provided, directions cannot be known unless also provided (complex)
             if self._current_directions is not None:
                  print("Warning: Using provided history, ignoring any potentially resumed Powell directions.")
             self._current_directions = None
        elif resume and len(loaded_history) > 0:
             effective_history = loaded_history
             print("Using resumed history from file.")
             # Directions were already handled during loading
        else:
             if initial_phi is None:
                  raise ValueError("PowellOptimizer requires initial_phi if not resuming or providing history.")
             print("Using initial_phi to start history.")
             initial_phi_clipped = torch.max(torch.min(initial_phi.cpu(), bounds[1]), bounds[0])
             if not torch.equal(initial_phi_clipped.cpu(), initial_phi.cpu()):
                  print("Warning: Initial phi was outside bounds, clipped to:", initial_phi_clipped)
             initial_phi_dev = initial_phi_clipped.to(device)
             if initial_phi_dev.dim() == 1:
                  initial_phi_dev = initial_phi_dev.unsqueeze(0)
             y_init = true_model(initial_phi_dev).cpu()
             effective_history = (initial_phi_clipped.cpu().view(-1, bounds.shape[1]),
                                  y_init.view(-1,1))
             self._current_directions = None # Start fresh

        # --- Base Class Initialization (same as before) ---
        super().__init__(true_model=true_model,
                         surrogate_model=None,
                         bounds=bounds,
                         device=device,
                         history=effective_history,
                         WandB=WandB,
                         outputs_dir=outputs_dir,
                         resume=resume) # Resume status passed to base, though we handled state here

        # --- Powell Specific Attributes (same as before, directions handled above) ---
        self.scipy_options = scipy_options if scipy_options is not None else {}
        if 'maxiter' in self.scipy_options:
            # print("Warning: 'maxiter' in scipy_options ignored; controlled internally by PowellOptimizer.")
            del self.scipy_options['maxiter']
        if 'direc' in self.scipy_options:
             print("Warning: 'direc' in scipy_options ignored; managed internally by PowellOptimizer state.")
             del self.scipy_options['direc']
        self.scipy_options.setdefault('xtol', 1e-4)
        self.scipy_options.setdefault('ftol', 1e-4)
        self.scipy_options.setdefault('disp', False)

        bounds_np = self.bounds.numpy()
        self.scipy_bounds = ScipyBounds(bounds_np[0], bounds_np[1])
        self._last_scipy_result = None
        self._wandb_run_active = wandb.run is not None

    # --- _objective_wrapper remains the same ---
    def _objective_wrapper(self, x_np: np.ndarray) -> float:
        """
        Wrapper for Scipy: Evaluates true_model, updates history, and logs to WandB.
        (Code is identical to previous version)
        """
        phi = torch.from_numpy(x_np).to(dtype=self.history[0].dtype if self.n_calls() > 0 else torch.float32, device=self.device)
        if phi.dim() == 1:
             phi = phi.unsqueeze(0)

        try:
            y = self.true_model(phi)
        except Exception as e:
            print(f"\nError evaluating true_model at {phi.tolist()}: {e}")
            return float('inf')

        loss_val = self.loss(phi.cpu(), y.cpu())
        self.update_history(phi.cpu(), y.cpu())

        if self._wandb_run_active and wandb.run is not None:
            try:
                log_data = {'eval_loss': loss_val.item()}
                wandb.log(log_data, step=self.n_calls())
            except Exception as e:
                 print(f"\nWarning: WandB logging failed in _objective_wrapper: {e}")

        return loss_val.item()

    def optimization_iteration(self):
        """
        Performs one major Powell iteration, preserving direction state.
        """
        print(f"\n--- Powell Iteration {self._i} (Calls: {self.n_calls()}) ---")
        if self.n_calls() == 0:
             raise RuntimeError("PowellOptimizer history is empty, cannot start iteration.")

        phi_current, loss_current = self.get_optimal()
        x0 = phi_current.cpu().numpy()

        print(f"Starting Powell step from: {x0.tolist()}, Current Best Loss: {loss_current.item():.4f}")

        # *** Prepare options, including current directions if available ***
        current_options = {**self.scipy_options, 'maxiter': 1} # Base options + force maxiter=1
        if self._current_directions is not None:
            print("Using stored directions from previous iteration.")
            current_options['direc'] = self._current_directions
        else:
            print("Initializing Powell directions (first iteration or after reset).")
            # SciPy will initialize directions automatically if 'direc' is not provided

        t1 = time()
        try:
            result = minimize(
                fun=self._objective_wrapper,
                x0=x0,
                method='Powell',
                bounds=self.scipy_bounds,
                options=current_options, # Pass options with potential 'direc'
            )
            self._last_scipy_result = result
            scipy_time = time() - t1
            nfev_this_call = result.nfev
            print(f"Scipy minimize (maxiter=1) completed in {scipy_time:.2f}s. Func Evals This Call: {nfev_this_call}. Success: {result.success}. Message: {result.message}")

            # *** Store the final directions for the next iteration ***
            if hasattr(result, 'direc'):
                 self._current_directions = result.direc
            else:
                 # Should not happen with Powell, but handle defensively
                 print("Warning: Result object did not contain 'direc'. Directions may reset.")
                 self._current_directions = None

        except Exception as e:
            print(f"\nError during scipy.optimize.minimize: {e}")
            # Optionally reset directions on error? Or keep the old ones?
            # self._current_directions = None # Safer to reset if minimize failed badly
            phi_best, loss_best = self.get_optimal()
            return phi_best, loss_best # Return current best if minimize fails

        # Get the best point found *overall* after this step
        phi_best_after_step, loss_best_after_step = self.get_optimal()
        return phi_best_after_step, loss_best_after_step

    # --- Update saving/loading in OptimizerClass ---
    # We need the main run_optimization loop to save/load the direction state too.
    # This requires modifying OptimizerClass or making save/load specific here.
    # Let's override save/load within PowellOptimizer for simplicity.

    def save_state(self):
        """Saves history and Powell directions."""
        state_path = join(self.outputs_dir, 'optimizer_state.pkl') # Use a more descriptive name
        state = {
            'history': self.history,
            'directions': self._current_directions
        }
        try:
            with open(state_path, "wb") as f:
                dump(state, f)
            print(f"Saved optimizer state (history & directions) to {state_path}")
        except Exception as e:
            print(f"\nFailed to save optimizer state: {e}")

    def stopping_criterion(self, **convergence_params):
        """Checks stopping criteria: max iterations or Powell convergence."""
        # 1. Check max iterations (from base class)
        max_iter = convergence_params.get('max_iter', 100)
        if self._i >= max_iter:
            print(f"Stopping: Reached max iterations ({self._i} >= {max_iter}).")
            return True

        # 2. Check if the *last* scipy call reported convergence (optional)
        # This might stop early if Powell converges quickly within one call.
        check_scipy_convergence = convergence_params.get('check_scipy_convergence', False)
        if check_scipy_convergence and self._last_scipy_result is not None:
            # Check common scipy success flags or messages indicative of convergence
            # Note: Powell's success/message might be less informative than gradient-based methods
            # Check if function value change or parameter change is below tolerance
            # This requires comparing previous results, which adds complexity.
            # A simpler check: Did the last call succeed and report minimal change?
            # result.success is often True even if maxiter is reached. A better check might be needed.
            # For now, rely primarily on max_iter. Add tolerance checks if needed.
            pass

        return False # Continue optimization
    
    
if __name__ == '__main__':
    mp.set_start_method("spawn", force=True)
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
    
