import torch
import botorch
import gpytorch
from utils.nets import Generator, Discriminator,GANLosses, Encoder, Decoder, IBNN_ReLU
from tqdm import trange
from matplotlib import pyplot as plt
from utils import HDF5Dataset
import h5py
import numpy as np
import os

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

class GP_BOCK(botorch.models.SingleTaskGP):
    """
    Gaussian Process model using a Cylindrical (BOCK-style) kernel
    for Bayesian Optimization in bounded, moderately/high-dimensional spaces.
    """

    def __init__(self, bounds, device='cpu', num_angular_weights=10):
        self.device = device
        self.bounds = bounds.to(device)
        self.num_angular_weights = num_angular_weights
        self.is_fitted = False

    def fit(self, X: torch.Tensor, Y: torch.Tensor, use_scipy=True, options: dict = None, **kwargs):
        X = X.to(self.device)
        Y = Y.to(self.device)
        X_norm = self.normalization(X, self.bounds)

        # Build cylindrical kernel
        radial_kernel = gpytorch.kernels.RBFKernel(ard_num_dims=None)
        cyl_kernel = gpytorch.kernels.CylindricalKernel(
            num_angular_weights=self.num_angular_weights,
            radial_base_kernel=radial_kernel,
            ard_num_dims=X_norm.size(-1)
        )
        covar_module = gpytorch.kernels.ScaleKernel(cyl_kernel)

        # Initialize the GP using SingleTaskGP directly
        super().__init__(
            X_norm,
            Y,
            covar_module=covar_module,
            outcome_transform=botorch.models.transforms.Standardize(m=1),
            **kwargs
        )

        # Optimize marginal log-likelihood
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self)
        if use_scipy:
            botorch.fit.fit_gpytorch_mll(mll)
        else:
            default_options = {'maxiter': 100}
            if options is not None:
                default_options.update(options)
            botorch.fit.fit_gpytorch_mll(
                mll,
                optimizer=botorch.optim.fit.fit_gpytorch_mll_torch,
                options=default_options
            )
        self.is_fitted = True
        return self

    def normalization(self, X: torch.Tensor, bounds):
        if X.le(1).all():
            print('Warning: X is already normalized')
        X_norm = (X - bounds[0, :]) / (bounds[1, :] - bounds[0, :])
        if not self.is_fitted:
            # Set scale to sqrt(d) to ensure max norm <= 1 for any data in [0,1]^d
            d = X_norm.size(-1)
            self.scale = torch.sqrt(torch.tensor(d, dtype=X_norm.dtype, device=X_norm.device))
        X_norm = X_norm / self.scale
        return X_norm

    def _predict(self, x, return_std=False, **kwargs):
        observed_pred = self.posterior(x, **kwargs)
        y_pred = observed_pred.mean.cpu()
        std_pred = observed_pred.variance.diag().sqrt().cpu()
        return (y_pred, std_pred) if return_std else y_pred

    def posterior(self, X, **kwargs):
        return super().posterior(self.normalization(X, self.bounds), **kwargs)

    def __repr__(self):
        return (f"{self.__class__.__name__}("
                f"bounds={self.bounds}, "
                f"num_angular_weights={self.num_angular_weights}, "
                f"device={self.device})")

class GP_IBNN(GP_RBF):
    """
    A SingleTaskGP model that uses the InfiniteWidthBNNKernel.
    
    Inherits from GP_RBF to reuse normalization, posterior, and other
    helper methods.
    """
    
    def __init__(self, bounds, depth=3, device='cpu'):
        """
        Args:
            bounds (torch.tensor): A 2 x d tensor of lower and upper bounds.
            depth (int): The depth of the virtual BNN.
            device (str): The torch device to use ('cpu' or 'cuda').
        """
        # Call the parent __init__ to set bounds and device
        super().__init__(bounds, device)
        self.depth = depth
        self.var_w = 5.0  # Weight variance default 10
        self.var_b = 1.6   # Bias variance

    def fit(self, X: torch.tensor, Y: torch.tensor, use_scipy=True, options: dict = None, **kwargs):
        """
        Fits the GP model with the InfiniteWidthBNNKernel.
        
        This method overrides the GP_RBF.fit method to inject the
        IBNN kernel during the SingleTaskGP initialization.
        """
        X = X.to(self.device)
        Y = Y.to(self.device)
        
        X_norm = self.normalization(X, self.bounds)
        num_dims = X_norm.shape[-1]
        
        #base_kernel = botorch.models.kernels.InfiniteWidthBNNKernel(depth=self.depth)
        #base_kernel.weight_var = 10.0
        #base_kernel.bias_var = 1.6
        base_kernel = IBNN_ReLU(num_dims, self.var_w, self.var_b, self.depth)
        covar_module = base_kernel#gpytorch.kernels.ScaleKernel(base_kernel)
        super(GP_RBF, self).__init__(
            X_norm, 
            Y, 
            outcome_transform=botorch.models.transforms.Standardize(m=1), 
            covar_module=covar_module, 
            **kwargs
        )
        
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self)
        if use_scipy:
            botorch.fit.fit_gpytorch_mll(mll)
        else:
            default_options = {'maxiter': 100}
            if options is not None:
                default_options.update(options)
            botorch.fit.fit_gpytorch_mll(
                mll, 
                optimizer=botorch.optim.fit.fit_gpytorch_mll_torch, 
                options=default_options
            )
            
        return self

    def __repr__(self):
        # Override repr to include depth
        return (f"{self.__class__.__name__}("
                f"bounds={self.bounds}, "
                f"depth={self.depth}, "
                f"device={self.device})")
    


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
                step_lr: int = 20,
                 lr: float = 1e-3,
                 data_from_file: bool = True,
                 activation: str = 'relu',
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
                 wandb = None):
        
        self.device = device
        self.wandb = wandb

        if activation == 'relu':
            act_layer = torch.nn.ReLU()
        elif activation in ('silu', 'swish'):
            act_layer = torch.nn.SiLU()
        elif activation == 'gelu':
            act_layer = torch.nn.GELU()

        self.model = torch.nn.Sequential(
            torch.nn.Linear(phi_dim + x_dim, 128),
            act_layer,
            torch.nn.Linear(128, 64),
            act_layer,
            torch.nn.Linear(64, 1)  # Output layer for binary classification
        ).to(self.device)

        self.n_epochs = n_epochs
        self.gen_epoch = 0
        self.batch_size = batch_size
        self._lr = lr
        self.step_lr = step_lr
        # Use BCEWithLogitsLoss for better numerical stability. It combines Sigmoid and BCELoss.
        self.loss_fn = torch.nn.BCEWithLogitsLoss() 
        self.last_train_loss = []
        self.data_from_file = data_from_file
    def load_weights(self, path: str, strict: bool = True):
        """
        Load model weights from a .pt/.pth file into the classifier.
        Accepts either a raw state_dict or a checkpoint dict containing
        keys like 'model_state_dict' or 'state_dict'.
        The tensors are mapped to the current device automatically.
        Returns self for chaining.
        """
        map_location = torch.device(self.device)
        checkpoint = torch.load(path, map_location=map_location)

        # Support plain state_dict or a checkpoint with common keys
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                # assume the dict is itself a state_dict
                state_dict = checkpoint
        else:
            # If torch.load returned something unexpected, try to use it directly
            state_dict = checkpoint

        # Load into the internal model
        self.model.load_state_dict(state_dict, strict=strict)
        return self.model
    def save_weights(self, path: str, as_checkpoint: bool = False, extra: dict = None):
        """
        Save model weights to `path`. By default saves the raw state_dict.
        If as_checkpoint=True, saves a checkpoint dict with key 'model_state_dict'
        and any additional keys provided via `extra`.
        Returns self for chaining.
        """
        dirpath = os.path.dirname(path)
        if dirpath:
            os.makedirs(dirpath, exist_ok=True)

        if as_checkpoint:
            chk = {'model_state_dict': self.model.state_dict()}
            if extra:
                chk.update(extra)
            torch.save(chk, path)
        else:
            torch.save(self.model.state_dict(), path)

        return self
    def fit(self,*args, **kwargs):
        """
        Trains the classifier model.
        """
        if self.data_from_file:
            return self._fit_from_file(*args, **kwargs)
        else:
            return self._fit_from_data(*args, **kwargs)
    def _fit_from_file(self, h5_path, **kwargs):
        """
        Trains the classifier model by reading data from an HDF5 file using PyTorch DataLoader.
        """
        
        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self._lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.step_lr, gamma=0.1)
        print(f"--- Starting Binary Classifier Training from HDF5 on {self.device} ---")

        with h5py.File(h5_path, "r") as dataset:
            condition_dset = dataset["condition"]
            y_dset = dataset["y"]
            n_samples = condition_dset.shape[0]
            pbar = trange(self.n_epochs, position=0, leave=True, desc='Classifier Training')
            for epoch in pbar:
                average_loss = 0.0
                n_batches = 0
                order = np.arange(0, n_samples, self.batch_size)
                np.random.shuffle(order)
                
                for start in order:
                    end = min(start + self.batch_size, n_samples)
                    condition_batch = torch.from_numpy(condition_dset[start:end]).to(self.device, non_blocking=True)
                    y_batch = torch.from_numpy(y_dset[start:end]).to(self.device, dtype=torch.float32, non_blocking=True).view(-1, 1)
                    
                    optimizer.zero_grad()
                    y_pred_logits = self.model(condition_batch)
                    loss = self.loss_fn(y_pred_logits, y_batch)
                    loss.backward()
                    optimizer.step()
                    
                    average_loss += loss.item()
                    n_batches += 1
                
                average_loss /= n_batches
                self.gen_epoch += 1
                self.last_train_loss.append(average_loss)
                scheduler.step()
                
                pbar.set_description(f"Epoch {epoch+1}, Loss: {average_loss:.4f}")
                pbar.set_postfix(loss=f"{average_loss:.4f}", lr=f"{optimizer.param_groups[0]['lr']:.6f}")
        print(f"Final training loss: {self.last_train_loss[-1]:.4f}")
        print("--- Binary Classifier Training Finished ---")
        return self

    def _fit_from_data(self, condition, y, **kwargs):
        """
        Trains the classifier model.
        """
        self.model.train() # Set the wrapped model to training mode
        
        # Ensure y is the correct shape [batch_size, 1] and type for the loss function
        y = y.view(-1, 1).float() 

        num_samples = y.size(0)
        indices = torch.arange(num_samples)
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self._lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.step_lr, gamma=0.1)
        pbar = trange(self.n_epochs, position=0, leave=True, desc='Classifier Training on ' + str(self.device))

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
            
            self.gen_epoch += 1
            average_loss /= num_samples
            if self.wandb is not None:
                self.wandb.log({'classifier_train_loss': average_loss}, step=self.gen_epoch, commit=False)
            self.last_train_loss.append(average_loss)
            scheduler.step()
            
            pbar.set_description(f"Epoch {epoch+1}, Loss: {average_loss:.4f}")
            pbar.set_postfix(loss=f"{average_loss:.4f}", lr=f"{optimizer.param_groups[0]['lr']:.6f}")
        return self

    def _predict_proba_cond(self, condition):
        self.model.eval()
        condition = condition.to(self.device)
        y_pred_logits = self.model(condition)
        y_pred_prob = torch.sigmoid(y_pred_logits)
        return y_pred_prob
    def predict_proba(self, phi,x):
        """
        Predicts the probability of Y=1 for each given `phi` and `x`.
        """
        self.model.eval()
        condition = torch.cat([phi.repeat(x.size(0), 1), x], dim=-1)
        return self._predict_proba_cond(condition)


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

