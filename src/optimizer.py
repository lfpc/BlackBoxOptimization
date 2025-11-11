import botorch
import torch
import torch.multiprocessing as mp
from tqdm import tqdm
from scipy.stats.qmc import LatinHypercube
from matplotlib import pyplot as plt
from pickle import dump,  load
import wandb
from os.path import join
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
import queue
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import os

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
        #mask = y.eq(0).all(dim=1)
        #y = y[:, 0:2]
        #y[mask] = 20*torch.ones((y.size(-1)), device=y.device, dtype=y.dtype)
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
            #assert False, y_pred
            
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
                 initial_lambda_constraints:float = 1e2,
                 initial_lr:float = 1e-3
                 ):
        super().__init__(true_model,
                 surrogate_model,
                 bounds = bounds,
                 device = device,
                 history = history,
                 WandB = WandB,
                 outputs_dir = outputs_dir,
                 resume = resume)
        
        self.local_results = [[],[]]
        assert torch.all(initial_phi >= self.bounds[0].to(self.device)) and torch.all(initial_phi <= self.bounds[1].to(self.device)), "current_phi is out of bounds"
        initial_phi = normalize_vector(initial_phi, bounds)
        if resume: 
            with open(join(outputs_dir,'history.pkl'), "rb") as f:
                self.history = load(f)
        else: self.simulate_and_update(initial_phi)
            
            
        initial_phi = initial_phi if not resume else history[0][-1]
        self._current_phi = torch.nn.Parameter(initial_phi.detach().clone().to(device))
        
        
        self.epsilon = epsilon
        self.lhs_sampler = LatinHypercube(d=initial_phi.size(-1))
        self.samples_phi = samples_phi
        self._samples_phi = samples_phi # in case we need to change it
        self.phi_optimizer = torch.optim.SGD([self._current_phi],lr=initial_lr)
        self.second_order = second_order
        self.lambda_constraints = initial_lambda_constraints
        self.trust_radius = initial_lr
        self.rhos = []
    @property
    def current_phi(self):
        """Returns the current phi in the original bounds."""
        return denormalize_vector(self._current_phi, self.bounds)
    
    def sample_phi(self):
        """Draw samples in a hypercube of side 2*epsilon around current_phi."""
        perturb = self.lhs_sampler.random(self.samples_phi)
        with torch.no_grad():
            perturb = (2*torch.from_numpy(perturb).to(dtype=torch.get_default_dtype()) - 1.0) * self.epsilon
            phis = (self._current_phi.unsqueeze(0).cpu() + perturb).clamp(0.0,1.0)#(self.bounds[0], self.bounds[1])
        return phis
    def n_hits(self,phi,y,x = None):
        w = x[:,7].to(y.device).flatten()
        y = y.flatten()
        return  (w*y).sum()/w.sum()
    def get_model_pred(self, phi):
        phi = normalize_vector(phi, self.bounds)
        x_samp = torch.as_tensor(self.true_model.sample_x(),device=self.device, dtype=torch.get_default_dtype())
        condition = torch.cat([phi.repeat(x_samp.size(0), 1), x_samp[:,:7]], dim=-1)
        y_pred = self.model.predict_proba(condition)
        return self.n_hits(phi, y_pred, x_samp)

    def clean_training_data(self):
        return torch.cat(self.local_results[0], dim=0), torch.cat(self.local_results[1], dim=0)

    def simulate_and_update(self, phi, update_history:bool = True):
        with torch.no_grad():
            x = torch.as_tensor(self.true_model.sample_x(phi),device=self.device, dtype=torch.get_default_dtype())
            condition = torch.cat([phi.repeat(x.size(0), 1), x[:,:7]], dim=-1).to(self.device)
            phi = denormalize_vector(phi, self.bounds).clone().detach()
            y = self.true_model(phi.cpu(),x).view(-1,1)
            print("Simulation performed. Output shape:", y.shape)
            self.local_results[0].append(condition)
            self.local_results[1].append(y)
            loss = self.n_hits(phi, y, x).reshape(1)
            if update_history:
                self.update_history(phi,loss.detach())
        return loss


    def optimization_iteration(self):
        """
        1) Sample locally around current_phi
        2) Evaluate true_model
        3) Fit local surrogate
        4) Update current_phi by minimizing surrogate
        5) Return (phi, loss)
        """
        
        if self._i ==0:
            with open('local_results_test.pkl', 'rb') as f:
                self.local_results = load(f)
            normalized_tensors = []
            for tensor in self.local_results[0]:
                tensor = torch.as_tensor(tensor)
                # Split tensor into first 63 and last 7 elements
                first_63 = tensor[:63] if tensor.dim() == 1 else tensor[:, :63]
                last_7 = tensor[63:] if tensor.dim() == 1 else tensor[:, 63:]
                
                # Normalize only the first 63 elements
                normalized_first_63 = normalize_vector(first_63, self.bounds)
                
                # Concatenate normalized first part with unchanged last part
                if tensor.dim() == 1:
                    normalized_tensor = torch.cat([normalized_first_63, last_7])
                else:
                    normalized_tensor = torch.cat([normalized_first_63, last_7], dim=-1)
                
                normalized_tensors.append(normalized_tensor)
            
            self.local_results[0] = normalized_tensors
        else:
            # 1) local sampling
            phis = self.sample_phi()
            # 2) true evaluations
            for phi in phis:
                self.simulate_and_update(phi, update_history=False)
        
        print('Iteration {} : Finished simulations for local samples.'.format(self._i))
        self.fit_surrogate_model()
        # Save local_results to a file for inspection
        
        x_samp = torch.as_tensor(self.true_model.sample_x(),device=self.device, dtype=torch.get_default_dtype())

        condition = torch.cat([self._current_phi.repeat(x_samp.size(0), 1), x_samp[:,:7]], dim=-1)
        y_pred = self.model.predict_proba(condition)
        
        self.phi_optimizer.zero_grad()
        loss_sur = self.n_hits(self._current_phi, y_pred, x_samp)
        constraints = self.true_model.get_constraints_func(self.current_phi).to(loss_sur.device)
        constraints = -self.lambda_constraints * torch.log(constraints + 1e-4).sum()
        loss_sur = loss_sur + constraints
        loss_sur.backward()
        self.phi_optimizer.step()
        with torch.no_grad():
            self._current_phi.data = self._current_phi.data.clamp(0.0,1.0)#(self.bounds[0], self.bounds[1])
        with torch.no_grad():
            self.local_results = [[],[]]
            self.simulate_and_update(self._current_phi)

        return self.history[0][-1], self.history[1][-1]

    def optimization_iteration_second_order(self):
        """
        Performs ONE optimization step using a trust-region method,
        implemented entirely in PyTorch.
        """
        # 1) & 2) & 3) Same setup

        if self._i ==0:
            with open('local_results_test.pkl', 'rb') as f:
                self.local_results = load(f)
            normalized_tensors = []
            for tensor in self.local_results[0]:
                tensor = torch.as_tensor(tensor)
                # Split tensor into first 63 and last 7 elements
                first_63 = tensor[:63] if tensor.dim() == 1 else tensor[:, :63]
                last_7 = tensor[63:] if tensor.dim() == 1 else tensor[:, 63:]
                
                # Normalize only the first 63 elements
                normalized_first_63 = normalize_vector(first_63, self.bounds)
                
                # Concatenate normalized first part with unchanged last part
                if tensor.dim() == 1:
                    normalized_tensor = torch.cat([normalized_first_63, last_7])
                else:
                    normalized_tensor = torch.cat([normalized_first_63, last_7], dim=-1)
                
                normalized_tensors.append(normalized_tensor)
            
            self.local_results[0] = normalized_tensors
        else:
            phis = self.sample_phi()
            for phi in phis:
                self.simulate_and_update(phi, update_history=False)
        print('Iteration {} : Finished simulations for local samples.'.format(self._i))
        self.fit_surrogate_model()

        plt.plot(self.model.last_train_loss)
        plt.savefig(f'loss_plot_{self._i}.png')
        plt.grid()
        plt.close()

        x_samp = torch.as_tensor(self.true_model.sample_x(), device=self.device, dtype=torch.get_default_dtype())
        def compute_surrogate_loss(phi):
            condition = torch.cat([phi.repeat(x_samp.size(0), 1), x_samp[:,:7]], dim=-1)
            y_pred = self.model.predict_proba(condition)
            loss = self.n_hits(phi, y_pred, x_samp)
            constraints = self.true_model.get_constraints_func(denormalize_vector(phi, self.bounds)).to(loss.device)
            #assert constraints.ge(0).all(), "Constraints must be positive"
            constraints = -self.lambda_constraints * torch.log(constraints + 1e-7).sum()
            return loss + constraints

        # --- Trust-Region Logic ---
        current_phi_detached = self._current_phi.detach().clone()
        initial_loss = compute_surrogate_loss(current_phi_detached)
        g = torch.func.grad(compute_surrogate_loss)(current_phi_detached)

        def hvp_func(v):
            # Hessian-vector product function needed by the subproblem solver
            _, hvp_val = torch.func.jvp(torch.func.grad(compute_surrogate_loss), (current_phi_detached,), (v,))
            return hvp_val

        p = self._solve_trust_region_subproblem(g, hvp_func, self.trust_radius)
        assert p.ne(0).any(), "Trust-region step is zero, no optimization performed"
        proposed_phi = current_phi_detached + p
        proposed_phi = proposed_phi.clamp(0.0,1.0)#(self.bounds[0], self.bounds[1])
        @torch.no_grad()
        def get_rho(p):
            """
            Computes the rho ratio for the trust-region step.
            """
            with torch.no_grad(): constraints = self.true_model.get_constraints_func(denormalize_vector(proposed_phi, self.bounds))
            if constraints.lt(0).any():
                print(f"Constraints violated for phi={proposed_phi}, constraints={constraints}")
                return 0.0, self.history[1][-1].detach().clone()
            proposed_loss = self.simulate_and_update(proposed_phi, update_history=False)
            actual_reduction = self.history[1][-1].item() - proposed_loss.item()
            #actual_reduction *= 1e6  # Scale to match surrogate loss
            #actual_reduction = initial_loss - compute_surrogate_loss(proposed_phi).item()
            predicted_reduction = - (torch.dot(g, p) + 0.5 * torch.dot(p, hvp_func(p)))
            rho = actual_reduction / (predicted_reduction + 1e-9)
            return rho, proposed_loss
        
        # 1. Solve the subproblem to find the proposed step p
        rho, proposed_loss = get_rho(p)
        self.rhos.append(rho.item())

        # 3. Update trust radius and accept/reject step based on rho
        if rho < 0.25:
            # Poor model fit. Shrink radius. Step is not taken unless rho is positive.
            self.trust_radius *= 0.25
            print(f"Step has poor agreement (rho={rho}). Shrinking trust radius to {self.trust_radius:.3f}")
        elif rho > 0.75 and torch.linalg.norm(p).item() >= 0.8*self.trust_radius:
            self.trust_radius = min(1.25 * self.trust_radius, 0.1) # Cap max radius
            print(f"Excellent step (rho={rho}). Expanding trust radius to {self.trust_radius:.3f}")

        if rho > 0.1:
            with torch.no_grad():
                self._current_phi.data  = proposed_phi
                self.local_results = [[self.local_results[0][-1]], [self.local_results[1][-1]]]
                self.update_history(self.current_phi.detach(), proposed_loss.detach())
            print("Step accepted.")
        else:
            [[self.local_results[0][0]], [self.local_results[1][0]]]
            print("Step rejected. Proposed loss:", proposed_loss.item(), "Current loss:", self.history[1][-1].item())
        return self.history[0][-1].detach().clone(), self.history[1][-1].detach().clone()


    def _solve_trust_region_subproblem(self, g, hvp_func, trust_radius):
        """
        Solves the trust-region subproblem using the Steihaug CG method.
        """
        p = torch.zeros_like(g)
        r = g.clone()
        d = -r.clone()

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
    def optimization_iteration_scipy(self):
        """
        A new optimization method that takes a single, second-order step using SciPy.
        
        1) Same setup as your original function (sampling, fitting surrogate).
        2) Defines the surrogate loss as a pure function.
        3) Uses `torch.func` to get gradient and Hessian functions.
        4) Calls SciPy's `trust-constr` optimizer for ONE iteration.
        5) Updates `current_phi` and evaluates against the true model.
        """
        # --- 1. Same initial setup ---
        with torch.no_grad():
            self._current_phi.data = self._current_phi.data.clamp(self.bounds[0], self.bounds[1])

        constraints = self.true_model.get_constraints_func(self._current_phi.detach().cpu().numpy())
        assert (constraints>=0).all(), "Initial phi violates constraints: {}".format(constraints)
        num_constraints = len(constraints)

        
        phis = self.sample_phi()
        for phi in phis:
            self.simulate_and_update(phi, update_history=False)
        print('Iteration {} : Finished simulations for local samples.'.format(self._i))
        self.fit_surrogate_model()

        # Sample a fixed x_samp for this optimization step
        x_samp = torch.as_tensor(self.true_model.sample_x(), device=self.device, dtype=torch.get_default_dtype())

        def compute_surrogate_loss(phi):
            """
            This function contains every step that connects 'phi' to the final loss.
            It is a "pure" function for compatibility with torch.func.
            """
            condition = torch.cat([phi.repeat(x_samp.size(0), 1), x_samp[:,:7]], dim=-1)
            y_pred = self.model.predict_proba(condition)
            loss = self.n_hits(phi, y_pred, x_samp)
            return loss*1e6
        grad_func = torch.func.grad(compute_surrogate_loss)
        hess_func = torch.func.hessian(compute_surrogate_loss)
        def loss_wrapper(p_np):
            p_torch = torch.from_numpy(p_np).to(self.device).float()
            return compute_surrogate_loss(p_torch).item()
        def jacobian_wrapper(p_np):
            p_torch = torch.from_numpy(p_np).to(self.device).float()
            return grad_func(p_torch).detach().cpu().numpy().astype(np.float64)
        def hessian_wrapper(p_np):
            p_torch = torch.from_numpy(p_np).to(self.device).float()
            return hess_func(p_torch).detach().cpu().numpy().astype(np.float64)
            
        # Convert your bounds to the format SciPy expects
        bounds_obj = ScipyBounds(self.bounds[0].cpu().numpy(), self.bounds[1].cpu().numpy())
        constraints= [NonlinearConstraint(
            fun= self.true_model.get_constraints_func, 
            lb=np.zeros(num_constraints),  # Lower bound for each constraint is 0
            ub=np.inf * np.ones(num_constraints), # Upper bound is infinity
            keep_feasible=True)]
        
        # --- 5. Call the optimizer for a single step ---   
        initial_guess = self._current_phi.detach().cpu().numpy()
        result = minimize(
            fun=loss_wrapper,
            x0=initial_guess,
            method='trust-constr',
            jac=jacobian_wrapper,
            hess=hessian_wrapper,
            bounds=bounds_obj,
            constraints=constraints,
            options={'maxiter': 2, 'disp': False} # Key: Perform only ONE iteration
        )
        
        # --- 6. Update state and evaluate against the true model ---
        with torch.no_grad():
            self._current_phi = torch.from_numpy(result.x).to(self.device).float()
            
            self.local_results = [[],[]]
            self.simulate_and_update(self._current_phi)

        self._i += 1
        return self.history[0][-1], self.history[1][-1]
    
    def get_new_phi(self):
        self.phi_optimizer.zero_grad()
        x = torch.as_tensor(self.true_model.sample_x(self._current_phi), device=self.device, dtype=torch.get_default_dtype())
        phi = self._current_phi.repeat(x.size(0), 1)
        l = self.loss(phi,self.model(phi,x))
        l.backward()
        self.phi_optimizer.step()
        return self._current_phi

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
    def __init__(self,problem_fn,population_size,generations,phi_bounds,blend_crossover_probability,blend_crossover_alpha,mutation_probability,random_immigration_probability,mutation_std_deviations_factor,tournament_size,elite_size,hall_of_fame_size,device,devices,WandB):
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
        self.tournament_size=tournament_size
        self.population=[]
        for _ in range(self.population_size):
            genes=self.get_initial_genes()
            individual=Individual(self.problem_fn,genes,self.computed_fitness_values,self.device)
            self.population.append(individual)
        compute_simulation(self.population,self.problem_fn,self.devices)
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
                    'mutation_probability': self.mutation_probability
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
    def __init__(self,problem_fn,population_size,generations,phi_bounds,blend_crossover_probability,blend_crossover_alpha,mutation_probability,random_immigration_probability,mutation_std_deviations_factor,tournament_size,elite_size,hall_of_fame_size,device,devices,WandB):
        self.problem_fn=problem_fn
        self.population_size=population_size
        self.generations=generations
        self.phi_bounds=phi_bounds
        self.blend_crossover_probability=blend_crossover_probability
        self.blend_crossover_alpha=blend_crossover_alpha
        self.mutation_probability=mutation_probability
        self.random_immigration_probability=random_immigration_probability
        self.mutation_std_deviations_factor=mutation_std_deviations_factor
        self.tournament_size=tournament_size
        self.elite_size=elite_size
        self.hall_of_fame_size=hall_of_fame_size
        self.device=device
        self.devices=devices
        self.WandB=WandB
        self.the_population=Population(self.problem_fn,self.population_size,self.generations,self.phi_bounds,self.blend_crossover_probability,self.blend_crossover_alpha,self.mutation_probability,self.random_immigration_probability,self.mutation_std_deviations_factor,self.tournament_size,self.elite_size,self.hall_of_fame_size,self.device,self.devices,self.WandB)

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
"""
class RL_muons_env(gym.Env):
    def __init__(self, dim, problem_fn, phi_bounds, max_steps, tolerance):
        super().__init__()
        self.dim=dim
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

        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(self.dim,), dtype=np.float32
        )
        low_bounds, high_bounds = self.phi_bounds.numpy()
        self.low_bounds = low.astype(np.float32)
        self.high_bounds = high.astype(np.float32)
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
        initial_point=[self.problem_fn.initial_phi.tolist()[param_index]+0.0001*np.random.normal(0, self.step_scale[param_index]) for param_index in range(len(self.phi_bounds.T))]
        print(type(initial_point))
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
"""
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
        if self._i % 100 == 0 and self._i >= self._iter_reduce_bounds:
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
    




[249.6988,  71.5368,  49.0927,  28.9864,  45.4430,  11.1451,   6.3051,1.0029,   0.9900,   0.0000, 249.1017,  50.6768,  30.0974,  45.8390,124.7403,  13.5402,  10.0050,   0.9922,   0.9900,   0.0000, 248.3697,7.9946,  31.5727,  36.3253,  28.4407,  51.6227,  12.1411,   0.9911,0.9900,   7.3188, 147.2961,   5.0000,  29.2230,  55.9471,  25.3608,7.4055,   7.0033,   1.0029,   1.0091,   0.0000,  -1.9084, 148.1647,18.5110,  31.2957, 131.5283,  35.1510,   8.7111,  14.8781,   0.9824,0.9994,  29.7593,  -1.8206, 247.2571,  30.5880,  73.6448,  85.5782,91.4696,   8.6192,  26.0465,   0.9899,   1.0009,   2.2481,  -1.8436]