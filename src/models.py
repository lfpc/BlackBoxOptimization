import torch
import botorch
import gpytorch
from math import sqrt
from utils.models_priors import KumaAlphaPrior, KumaBetaPrior, AngularWeightsPrior
from utils.nets import Generator, Discriminator,GANLosses, Encoder, Decoder, IBNN_ReLU
from tqdm import trange
from matplotlib import pyplot as plt
from utils import standardize
from gpytorch.kernels import ScaleKernel

class GP_RBF(botorch.models.SingleTaskGP):
    def __init__(self,bounds,device = 'cpu'):
        self.device = device
        self.bounds = bounds.to(device)
    def fit(self,X:torch.tensor,Y:torch.tensor,use_scipy = True,options:dict = None,**kwargs):
        X = X.to(self.device)
        Y = Y.to(self.device)
        #Y = standardize(Y)
        X = self.normalization(X,self.bounds)
        super().__init__(X,Y,outcome_transform = botorch.models.transforms.Standardize(m=1),**kwargs)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self)
        if use_scipy:
            botorch.fit.fit_gpytorch_mll(mll)
        else:
            botorch.fit.fit_gpytorch_mll(mll,optimizer=botorch.optim.fit.fit_gpytorch_mll_torch, options=options)
        return self
    @staticmethod
    def normalization(X:torch.tensor, bounds):
        if X.le(1).all(): print('Warning: X is already normalized')
        return (X - bounds[0,:]) / (bounds[1,:] - bounds[0,:])
    def _predict(self,x,return_std = False,**kwargs):
        #self.eval()
        observed_pred = self.posterior(x,**kwargs)
        y_pred = observed_pred.mean.cpu()
        std_pred = observed_pred.variance.diag().sqrt().cpu()
        if return_std: return y_pred,std_pred
        else: return 
    def posterior(self, X, **kwargs):
        return super().posterior(self.normalization(X, self.bounds), **kwargs)
    def __repr__(self):
        return (f"{self.__class__.__name__}("
                f"bounds={self.bounds}, "
                f"device={self.device})")


class GP_Cylindrical_Custom(GP_RBF):
    def __init__(self, bounds:torch.tensor,device = 'cpu', deterministic_loss = None):
        super().__init__(bounds,device)
        self.bounds = self.bounds.t().to(device)
        if self.bounds.dim() == 1: self.bounds = self.bounds.unsqueeze(-1)
    def fit(self,x,y,**kwargs):
        #self.train()
        super().fit(x,y,**kwargs)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.CylindricalKernel(
            num_angular_weights=4,
            alpha_prior=KumaAlphaPrior(),
            alpha_constraint=gpytorch.constraints.constraints.Interval(lower_bound=0.5, upper_bound=1.),
            beta_prior=KumaBetaPrior(),
            beta_constraint=gpytorch.constraints.constraints.Interval(lower_bound=1., upper_bound=2.),
            radial_base_kernel=gpytorch.kernels.MaternKernel(),
            # angular_weights_constraint=gpytorch.constraints.constraints.Interval(lower_bound=exp(-12.),
            #                                                                      upper_bound=exp(20.)),
            angular_weights_prior=AngularWeightsPrior()
        ))
        return self.to(self.device)
    @staticmethod
    def normalization(x, bounds):
        dim = len(bounds)
        # from bounds to [-1, 1]^d
        x = (x - bounds.mean(dim=-1)) / ((bounds[:,1] - bounds[:,0]) / 2)
        # from [-1, 1]^d to Ball(0, 1)
        x = x / sqrt(dim)
        return x
    

class SingleTaskIBNN(GP_RBF):
    def __init__(self, bounds, device = torch.device('cpu'),
                 var_w = 10.,var_b = 5.,depth = 3, deterministic_loss = None
                ):
        super().__init__(bounds,device)
        self.model_args = {'var_w':var_w,'var_b':var_b, 'depth':depth}
        self._kernel = None
    def fit(self,X,Y,use_scipy = True,options:dict = None,**kwargs):
        if self._kernel is None: ibnn_kernel = botorch.models.kernels.InfiniteWidthBNNKernel(depth=self.model_args['depth'])
        else: ibnn_kernel = self._kernel
        ibnn_kernel.weight_var = self.model_args['var_w']
        ibnn_kernel.bias_var = self.model_args['var_b']
        kernel = ScaleKernel(ibnn_kernel, device=self.device)
        super().fit(X,Y,use_scipy = use_scipy,options= options,covar_module=kernel,**kwargs)
        self._kernel = kernel
        return self
    
    
class VAEModel(torch.nn.Module):
    """
    A Conditional VAE to model p(y | phi, X) as a surrogate for Bayesian Optimization.

    - The condition `c` is formed by concatenating the optimization parameter `phi` and
      the fixed parameters `X`.
    - The Encoder learns a latent representation from the simulation output `y` given `c`.
    - The Decoder learns to reconstruct `y` from a latent vector `z` and the condition `c`.
    """
    def __init__(self,
                 phi_dim: int,
                 x_dim: int,
                 y_dim: int,
                 latent_dim: int,
                 encoder_cls=Encoder,
                 decoder_cls=Decoder,
                 epochs: int = 50,
                 batch_size: int = 32,
                 lr: float = 1e-3,
                 propagate_last:int = 0,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        super().__init__()
        self.device = device

        y_dim -= propagate_last
        condition_dim = phi_dim + x_dim
        encoder_input_dim = y_dim + condition_dim
        decoder_input_dim = latent_dim + condition_dim
        self.propagate_last = propagate_last

        self.Encoder = encoder_cls(input_dim=encoder_input_dim, latent_dim=latent_dim).to(self.device)
        self.Decoder = decoder_cls(input_dim=decoder_input_dim, output_dim=y_dim).to(self.device)

        self._epochs = epochs
        self.batch_size = batch_size
        self._lr = lr
        self.latent_dim = latent_dim
        self.last_train_loss = []

    def reparameterization(self, mean, var):
        epsilon = torch.randn_like(var).to(self.device) # sampling epsilon
        z = mean + var * epsilon                       # reparameterization trick
        return z

    def forward(self, y, condition):
        # 1. Encode y given the condition
        encoder_input = torch.cat([y, condition], dim=-1)
        mean, log_var = self.Encoder(encoder_input)

        # 2. Reparameterization trick
        z = self.reparameterization(mean, torch.exp(0.5 * log_var))

        # 3. Decode z given the condition to reconstruct y
        decoder_input = torch.cat([z, condition], dim=-1)
        y_hat = self.Decoder(decoder_input)

        return y_hat, mean, log_var

    def loss_fn(self, y, y_hat, mean, log_var):
        # Use MSE for reconstruction loss of continuous data, as in a simulation output
        reproduction_loss = torch.nn.functional.mse_loss(y_hat, y, reduction='sum')
        # KL Divergence
        kld = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
        return reproduction_loss + kld

    def fit(self, phi, y, x):
        self.train() # Set model to training mode
        y = y[:, :-self.propagate_last] if self.propagate_last > 0 else y
        self.last_train_loss = []
        condition = torch.cat([phi, x], dim=-1)
        num_samples = y.size(0)
        indices = torch.arange(num_samples)
        LR = self._lr
        optimizer = torch.optim.Adam(self.parameters(), lr=LR)
        # Add a learning rate scheduler (ReduceLROnPlateau as an example)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=2)

        print(f"--- Starting CVAE Training on {self.device} ---")
        pbar = trange(self._epochs, position=0, leave=True, desc='CVAE Training Progress')
        for epoch in pbar:
            average_loss = 0.0
            # Shuffle indices for each epoch
            shuffled_indices = indices[torch.randperm(num_samples)]
            for start in range(0, num_samples, self.batch_size):
                end = min(start + self.batch_size, num_samples)
                batch_idx = shuffled_indices[start:end]
                y_batch = y[batch_idx].to(self.device)
                condition_batch = condition[batch_idx].to(self.device)
                optimizer.zero_grad()
                y_hat, mean, log_var = self.forward(y_batch, condition_batch)
                loss = self.loss_fn(y_batch, y_hat, mean, log_var)
                assert not torch.isnan(loss), "Loss is NaN, check your model and data."
                loss.backward()
                optimizer.step()
                average_loss += loss.item()/ (end - start)
            self.last_train_loss.append(average_loss)
            # Step the scheduler with the average loss for this epoch
            scheduler.step(average_loss)
            pbar.set_description(f"Epoch {epoch+1}, Loss: {average_loss:.4f}")
            pbar.set_postfix(loss=f"{average_loss:.4f}", lr=f"{LR:.6f}")
        print(f"Final training loss: {self.last_train_loss}")
        print("--- CVAE Training Finished ---")
        return self

    def generate(self, phi, x, n_samples=1):
        """Generates `n_samples` of `y` for each given `phi` and `x`."""
        self.eval() # Set model to evaluation mode
        phi = phi.unsqueeze(0).repeat(x.size(0), 1)
        
        # Determine batch size from input
        batch_size = phi.shape[0]
        
        # Prepare condition and move to device
        condition = torch.cat([phi, x], dim=-1).to(self.device)
        
        # Repeat condition for each sample we want to generate
        condition = condition.repeat_interleave(n_samples, dim=0)
        
        # Sample z from the prior N(0, I)
        z = torch.randn(batch_size * n_samples, self.latent_dim).to(self.device)
        
        # Create the decoder input
        decoder_input = torch.cat([z, condition], dim=-1)
        
        # Generate y_hat from the decoder
        y_hat = self.Decoder(decoder_input)
        
        # Reshape to (batch_size, n_samples, y_dim)
        y_hat = y_hat.view(batch_size, n_samples, -1)
        if self.propagate_last > 0:
            # If we have a last dimension to propagate, we need to handle it
            # Expand x to match y_hat's dimensions for concatenation along dim=2
            x_last = x[:, -self.propagate_last:].to(self.device).unsqueeze(1).expand(-1, n_samples, -1)
            y_hat = torch.cat([y_hat, x_last], dim=2)
        return y_hat.squeeze(1) # Remove sample dim if n_samples=1
    

from pyro.distributions import ConditionalTransformedDistribution
from pyro.distributions.transforms import conditional_spline_autoregressive
from pyro.distributions.conditional import ConditionalTransformedDistribution
from pyro.distributions import Normal
class NormalizingFlowModel(torch.nn.Module):
    """
    A Conditional Normalizing Flow to model p(y | phi, X) as a surrogate for Bayesian Optimization.
    - The condition `c` is formed by concatenating the optimization parameter `phi` and
      the fixed parameters `X`.
    - The model learns a transformation from a simple base distribution (e.g., Gaussian)
      to the complex conditional distribution of the simulation output `y`.
    - **This version includes data standardization for numerical stability.**
    """
    def __init__(self,
                 phi_dim: int,
                 x_dim: int,
                 y_dim: int,
                 flow_transforms: int = 5,
                 hidden_dim: int = 128,
                 epochs: int = 50,
                 batch_size: int = 32,
                 lr: float = 1e-3,
                 propagate_last: int = 0,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        super().__init__()
        self.device = device
        self.propagate_last = propagate_last
        self.y_dim_original = y_dim
        y_dim_flow = y_dim - propagate_last
        condition_dim = phi_dim + x_dim

        # 1. Define the base distribution (a standard normal distribution)
        base_dist = Normal(torch.zeros(y_dim_flow).to(device), torch.ones(y_dim_flow).to(device))

        # 2. Define the series of transformations.
        # We must wrap the list of transforms in an nn.ModuleList to register them as parameters.
        self.transforms = torch.nn.ModuleList([conditional_spline_autoregressive(y_dim_flow, context_dim=condition_dim, hidden_dims=[hidden_dim, hidden_dim]) for _ in range(flow_transforms)])
        
        # 3. Create the ConditionalTransformedDistribution
        self.flow = ConditionalTransformedDistribution(base_dist, self.transforms)

        self._epochs = epochs
        self.batch_size = batch_size
        self._lr = lr
        self.last_train_loss = []
        
        # Buffers for data standardization
        self.register_buffer('y_mean', torch.zeros(y_dim_flow))
        self.register_buffer('y_std', torch.ones(y_dim_flow))
        self.is_fitted = False

        self.to(device)

    def loss_fn(self, y, condition):
        """
        The loss for a normalizing flow is the negative log-likelihood of the data.
        We add a correction term for the standardization.
        """
        # The log_prob method gives the log-likelihood of y under the distribution
        # conditioned on `condition`. We want to maximize this, which is equivalent
        # to minimizing the negative log-likelihood.
        nll = -self.flow.condition(condition).log_prob(y)
        
        # The change of variables for standardization y_scaled = (y - mu)/std adds a
        # log-determinant term to the NLL, which is sum(log(std)).
        log_det_jacobian = torch.sum(torch.log(self.y_std))
        
        return (nll + log_det_jacobian).mean()

    def fit(self, phi, y, x):
        """
        Trains the normalizing flow model.
        """
        self.train() # Set model to training mode
        
        # Handle propagated dimensions
        y_flow = y[:, :-self.propagate_last] if self.propagate_last > 0 else y
        
        # --- Data Standardization ---
        # Calculate and store mean and std for y
        self.y_mean = y_flow.mean(dim=0)
        self.y_std = y_flow.std(dim=0)
        # Add a small epsilon to std to prevent division by zero
        self.y_std[self.y_std == 0] = 1e-6
        self.is_fitted = True

        y_scaled = (y_flow - self.y_mean) / self.y_std
        # --- End Standardization ---

        self.last_train_loss = []
        condition = torch.cat([phi, x], dim=-1)
        
        num_samples = y_scaled.size(0)
        indices = torch.arange(num_samples)
        
        optimizer = torch.optim.Adam(self.parameters(), lr=self._lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)

        print(f"--- Starting Conditional Normalizing Flow Training on {self.device} ---")
        pbar = trange(self._epochs, position=0, leave=True, desc='CNF Training Progress')

        for epoch in pbar:
            average_loss = 0.0
            # Shuffle indices for each epoch
            shuffled_indices = indices[torch.randperm(num_samples)]
            
            for start in range(0, num_samples, self.batch_size):
                end = min(start + self.batch_size, num_samples)
                batch_idx = shuffled_indices[start:end]
                
                y_batch = y_scaled[batch_idx].to(self.device)
                condition_batch = condition[batch_idx].to(self.device)
                
                optimizer.zero_grad()
                
                loss = self.loss_fn(y_batch, condition_batch)
                
                assert not torch.isnan(loss), "Loss is NaN, check your model and data."
                
                loss.backward()
                # Optional: Gradient clipping for more stability
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=5.0)
                optimizer.step()
                
                average_loss += loss.item() * (end - start)
            
            average_loss /= num_samples
            self.last_train_loss.append(average_loss)
            scheduler.step()
            
            pbar.set_description(f"Epoch {epoch+1}, Loss: {average_loss:.4f}")
            pbar.set_postfix(loss=f"{average_loss:.4f}", lr=f"{optimizer.param_groups[0]['lr']:.6f}")
            
        print(f"Final training loss: {self.last_train_loss[-1]:.4f}")
        print("--- Conditional Normalizing Flow Training Finished ---")
        return self

    def generate(self, phi, x, n_samples=1):
        """
        Generates `n_samples` of `y` for each given `phi` and `x`.
        """
        if not self.is_fitted:
            raise RuntimeError("The model has not been fitted yet. Call fit() before generating samples.")

        self.eval() # Set model to evaluation mode
        with torch.no_grad():
            # If phi is a single vector, unsqueeze it to make it a batch of 1
            if phi.dim() == 1:
                phi = phi.unsqueeze(0)

            # If x has fewer samples than phi, repeat x to match phi's batch size
            if x.size(0) < phi.size(0):
                 x = x.repeat(phi.size(0), 1)

            # Prepare condition and move to device
            condition = torch.cat([phi, x], dim=-1).to(self.device)
            
            # Generate samples from the conditional flow
            y_hat_scaled = self.flow.condition(condition).sample(torch.Size([n_samples]))

            # --- Un-standardize the generated samples ---
            y_hat = y_hat_scaled * self.y_std + self.y_mean
            # --- End Un-standardization ---

            # The shape will be (n_samples, batch_size, y_dim), so we permute it
            # to get (batch_size, n_samples, y_dim)
            y_hat = y_hat.permute(1, 0, 2)

            if self.propagate_last > 0:
                # If we have a last dimension to propagate, handle it
                x_last = x[:, -self.propagate_last:].to(self.device).unsqueeze(1).expand(-1, n_samples, -1)
                y_hat = torch.cat([y_hat, x_last], dim=2)
            
            return y_hat.squeeze(1) # Remove sample dim if n_samples=1

class BinaryClassifierModel:
    """
    A wrapper class to handle training and prediction for a given binary classification model.
    It is not an nn.Module itself, but it manages one.
    """
    def __init__(self,
                phi_dim: int,
                 x_dim: int,
                 n_epochs: int = 50,
                 batch_size: int = 32,
                 lr: float = 1e-3,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        
        self.device = device

        self.model = torch.nn.Sequential(
            torch.nn.Linear(phi_dim + x_dim, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 1)  # Output layer for binary classification
        ).to(self.device)

        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self._lr = lr
        # Use BCEWithLogitsLoss for better numerical stability. It combines Sigmoid and BCELoss.
        self.loss_fn = torch.nn.BCEWithLogitsLoss() 
        self.last_train_loss = []

    def fit(self, condition, y):
        """
        Trains the classifier model.
        """
        self.model.train() # Set the wrapped model to training mode
        self.last_train_loss = []
        
        # Ensure y is the correct shape [batch_size, 1] and type for the loss function
        y = y.view(-1, 1).float() 

        num_samples = y.size(0)
        indices = torch.arange(num_samples)
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self._lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

        print(f"--- Starting Binary Classifier Training on {self.device} ---")
        pbar = trange(self.n_epochs, position=0, leave=True, desc='Classifier Training')

        for epoch in pbar:
            average_loss = 0.0
            # Shuffle indices for each epoch
            shuffled_indices = indices[torch.randperm(num_samples)]
            
            for start in range(0, num_samples, self.batch_size):
                end = min(start + self.batch_size, num_samples)
                batch_idx = shuffled_indices[start:end]
                
                y_batch = y[batch_idx].to(self.device)
                condition_batch = condition[batch_idx].to(self.device)
                optimizer.zero_grad()
                
                # Get model prediction (logits)
                y_pred_logits = self.model(condition_batch)
                
                # Calculate loss
                loss = self.loss_fn(y_pred_logits, y_batch)
                
                assert not torch.isnan(loss), "Loss is NaN, check your model and data."
                
                loss.backward()
                optimizer.step()
                
                average_loss += loss.item() * (end - start)
            
            average_loss /= num_samples
            self.last_train_loss.append(average_loss)
            scheduler.step()
            
            pbar.set_description(f"Epoch {epoch+1}, Loss: {average_loss:.4f}")
            pbar.set_postfix(loss=f"{average_loss:.4f}", lr=f"{optimizer.param_groups[0]['lr']:.6f}")
            
        print(f"Final training loss: {self.last_train_loss[-1]:.4f}")
        print("--- Binary Classifier Training Finished ---")
        return self

    def predict_proba(self, condition):
        """
        Predicts the probability of Y=1 for each given `phi` and `x`.
        """
        self.model.eval()
        condition = condition.to(self.device)
        y_pred_logits = self.model(condition)
        y_pred_prob = torch.sigmoid(y_pred_logits)
        
        return y_pred_prob


if __name__ == '__main__':
    from problems import stochastic_RosenbrockProblem,RosenbrockProblem,stochastic_ThreeHump
    problem = stochastic_ThreeHump()#RosenbrockProblem(noise=0.1)#stochastic_RosenbrockProblem()
    dev = torch.device('cuda')
    phi_range = torch.as_tensor([-10.,10.],device=dev).view(2,-1)  
    dimensions_phi = 2
    if phi_range.size(0) != dimensions_phi: phi_range = phi_range.repeat(1,dimensions_phi)
    model = GP_RBF(phi_range,device=dev)#VAEModel(11,epochs = 50)
    #GANModel(problem,10,1,10,epochs = 50,iters_discriminator=25,iters_generator=5)
    phi = torch.rand(100,dimensions_phi,device=dev)
    n_samples_x = 11
    x = problem.sample_x(phi,n_samples_x).view(-1,1)
    phi = phi.repeat(n_samples_x,1)
    y = problem(phi,x)
    model.fit(phi,y)#(phi,y,x)
    #y_gen = model.generate(torch.cat((phi,x),dim=-1))
    with torch.no_grad():
        y_gen = model.posterior(phi)
        y_gen = y_gen.mean.flatten().cpu()
        y = y.cpu()
    plt.scatter(y_gen.numpy(),y.numpy())
    plt.grid()
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.plot([y_gen.min(),y_gen.max()],[y_gen.min(),y_gen.max()],'k--')
    plt.savefig('testGP.png')
    plt.show()

