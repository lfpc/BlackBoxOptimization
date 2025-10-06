import botorch
import torch
from tqdm import tqdm
from scipy.stats.qmc import LatinHypercube
from matplotlib import pyplot as plt
from pickle import dump,  load
import wandb
from os.path import join
from scipy.optimize import minimize, Bounds as ScipyBounds, NonlinearConstraint
import numpy as np
import sys
sys.path.append('..')
from utils.acquisition_functions import Custom_LogEI
from utils import normalize_vector, denormalize_vector
#torch.set_default_dtype(torch.float64)
from time import time


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
        y = self.true_model(phi)
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