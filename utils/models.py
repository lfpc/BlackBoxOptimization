import torch
import botorch
import gpytorch
from math import sqrt
from .models_priors import KumaAlphaPrior, KumaBetaPrior, AngularWeightsPrior
from .nets import Generator, Discriminator,GANLosses, Encoder, Decoder, IBNN_ReLU
from tqdm import trange
from matplotlib import pyplot as plt
from utils import standardize


class GP_RBF(botorch.models.SingleTaskGP):
    def __init__(self,bounds,device = 'cpu'):
        self.device = device
        self.bounds = bounds
    def fit(self,X,Y,use_scipy = True,options:dict = None,**kwargs):
        self.train()
        Y = standardize(Y)
        X = self.normalization(X,self.bounds)
        super().__init__(X,Y,**kwargs)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self)
        if use_scipy:
            botorch.fit.fit_gpytorch_mll(mll)
        else:
            botorch.fit.fit_gpytorch_mll(mll,optimizer=botorch.optim.fit.fit_gpytorch_mll_torch, options=options)
        return self
    def forward(self,x):
        x = self.normalization(x, bounds=self.bounds)
        return super().forward(x)
    @staticmethod
    def normalization(X, bounds):
        return (X - bounds[0,:]) / (bounds[1,:] - bounds[0,:])
    def predict(self,x,return_std = False,**kwargs):
        self.eval()
        x = self.normalization(x,self.bounds)
        observed_pred = self.posterior(x,**kwargs)
        y_pred = observed_pred.mean.cpu()
        std_pred = observed_pred.mvn.covariance_matrix.diag().sqrt().cpu()
        if return_std: return y_pred,std_pred
        else: return y_pred
    
class GP_Cylindrical_Custom(GP_RBF):#, botorch.models.gpytorch.GPyTorchModel):  # FixedNoiseGP SingleTaskGP
    def __init__(self, bounds:torch.tensor,device = 'cpu'):
        super().__init__(bounds,device)
        self.bounds = self.bounds.t()
        if self.bounds.dim() == 1: self.bounds = self.bounds.unsqueeze(-1)
    def fit(self,x,y,**kwargs):
        self.train()
        y = standardize(y)
        x = self.normalization(x,self.bounds)
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
    
    def forward(self, x):
        x = self.normalization(x, bounds=self.bounds)
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
    @staticmethod
    def normalization(x, bounds):
        dim = len(bounds)
        # from bounds to [-1, 1]^d
        x = (x - bounds.mean(dim=-1)) / ((bounds[:,1] - bounds[:,0]) / 2)
        # from [-1, 1]^d to Ball(0, 1)
        x = x / sqrt(dim)
        return x
    

class SingleTaskIBNN(GP_RBF):
    def __init__(self, bounds,
                 var_w = 10.,var_b = 1.3,depth = 3,
                device = torch.device('cpu')):
        super().__init__(bounds,device)
        self.model_args = {'var_w':var_w,'var_b':var_b, 'depth':depth}
        self._kernel = None #IBNN_ReLU(bounds.size(-1), var_w, var_b, depth)
    def fit(self,X,Y):
        if self._kernel is None: kernel = IBNN_ReLU(self.bounds.size(-1), **self.model_args)
        else: kernel = self._kernel
        super().fit(X,Y,covar_module=kernel, outcome_transform=botorch.models.transforms.outcome.Standardize(m=1))
        self._kernel = kernel
        return self
    def eval(self,*args):
        super().eval(*args)
        self._kernel.eval()
    def train(self,*args):
        super().train(*args)
        self._kernel.train()
    



class GANModel():
    def __init__(self,
                 y_model,
                 psi_dim: int,
                 x_dim: int,
                 batch_size: int,
                 noise_dim: int = 150,
                 y_dim: int = 1,
                 task: str = 'CRAMER',
                 epochs: int = 5,
                 lr: float = 1e-4 * 8,
                 iters_discriminator: int = 1,
                 iters_generator: int = 1,
                 grad_penalty: bool = True,
                 zero_centered_grad_penalty: bool = False,
                 instance_noise_std: float = 0.01,
                 logger=None,
                 burn_in_period=None,
                 averaging_coeff=0.,
                 dis_output_dim=256,
                 gp_reg_coeff=10,
                 device = 'cpu'):
        #super(GANModel, self).__init__(y_model=y_model, x_dim=x_dim, psi_dim=psi_dim, y_dim=y_dim)
        if task in ['WASSERSTEIN', "CRAMER"]:
            output_logits = True
        else:
            output_logits = False
        self._task = task
        self._noise_dim = noise_dim
        self._psi_dim = psi_dim
        self._y_dim = y_dim
        self._x_dim = x_dim
        self._epochs = epochs
        self._lr = lr
        self._batch_size = batch_size
        self._grad_penalty = grad_penalty
        self._zero_centered_grad_penalty = zero_centered_grad_penalty
        self._instance_noise_std = instance_noise_std
        self._iters_discriminator = iters_discriminator
        self._iters_generator = iters_generator
        self._ganloss = GANLosses(task=task)
        self.lambda_reg = gp_reg_coeff
        self._generator = Generator(noise_dim=self._noise_dim,
                                    out_dim=self._y_dim,
                                    psi_dim=self._psi_dim,
                                    x_dim=self._x_dim).to(device)
        self._discriminator = Discriminator(in_dim=self._y_dim,
                                            output_logits=output_logits,
                                            psi_dim=self._psi_dim,
                                            x_dim=self._x_dim,
                                            output_dim=dis_output_dim).to(device)

        self.logger = logger
        self.burn_in_period = burn_in_period
        self.averaging_coeff = averaging_coeff
        self.gen_average_weights = []
        self._cond_mean = torch.zeros(self._x_dim + self._psi_dim,device=device).float()
        self._cond_std = torch.ones(self._x_dim + self._psi_dim,device=device).float()
        self._y_mean = torch.zeros(self._y_dim,device=device).float()
        self._y_std = torch.ones(self._y_dim,device=device).float()
        self.device = device

    @staticmethod
    def instance_noise(data, std):
        return data + torch.randn_like(data) * data.std(dim=0) * std
    def eval(self):
        self._generator.eval()
        self._discriminator.eval()
        return self
    def train(self):
        self._generator.train()
        self._discriminator.train()
        return self
    def loss(self, y, condition):
        # y = (y - self._y_mean) / self._y_std
        # condition = (condition - self._cond_mean) / self._cond_std
        return self._discriminator(y, condition)

    def fit(self, phi,y,x):
        self.train()
        condition = torch.cat([phi,x],dim=-1)
        self._cond_mean = condition.mean(0).detach().clone()
        self._cond_std = condition.std(0).detach().clone()
        self._y_mean = y.mean(0).detach().clone()
        self._y_std = y.std(0).detach().clone()


        g_optimizer = torch.optim.Adam(self._generator.parameters(), lr=self._lr, betas=(0.5, 0.999))
        d_optimizer = torch.optim.Adam(self._discriminator.parameters(), lr=self._lr, betas=(0.5, 0.999))

        y = (y - self._y_mean) / self._y_std
        condition = (condition - self._cond_mean) / self._cond_std
        condition[torch.isnan(condition)] = 0
        #dataset = torch.utils.data.TensorDataset(y_normed, cond_normed)
        dataset = torch.utils.data.TensorDataset(y, condition)
        dataloader = torch.utils.data.DataLoader(dataset,
                                                 batch_size=self._batch_size,
                                                 shuffle=True)

        for epoch in trange(self._epochs, position=1, leave=False, desc = 'GAN training'):
            dis_epoch_loss = []
            gen_epoch_loss = []
            for y_batch, cond_batch in dataloader:
                # print(y_batch.shape, cond_batch.shape)
                for _ in range(self._iters_discriminator):
                    y_gen = self.generate(condition=cond_batch, normalise=False)
                    if self._instance_noise_std:
                        y_batch = self.instance_noise(y_batch, self._instance_noise_std)
                        y_gen = self.instance_noise(y_gen, self._instance_noise_std)

                    if self._task == "CRAMER":
                        y_gen_prime = self.generate(condition=cond_batch, normalise=False)
                        if self._instance_noise_std:
                            y_gen_prime = self.instance_noise(y_gen_prime, self._instance_noise_std)
                        loss = self._ganloss.d_loss(self.loss(y_gen, cond_batch),
                                                    self.loss(y_batch, cond_batch),
                                                    self.loss(y_gen_prime, cond_batch))
                        if self._grad_penalty:
                            loss += self._ganloss.calc_gradient_penalty(self._discriminator,
                                                                        y_gen.data,
                                                                        y_batch.data,
                                                                        cond_batch.data,
                                                                        data_gen_prime=y_gen_prime.data,
                                                                        lambda_reg=self.lambda_reg)
                    else:
                        loss = self._ganloss.d_loss(self.loss(y_gen, cond_batch),
                                                    self.loss(y_batch, cond_batch))
                        if self._grad_penalty:
                            loss += self._ganloss.calc_gradient_penalty(self._discriminator,
                                                                        y_gen.data,
                                                                        y_batch.data,
                                                                        cond_batch.data,
                                                                        lambda_reg=self.lambda_reg)

                    d_optimizer.zero_grad()
                    loss.backward()
                    d_optimizer.step()
                dis_epoch_loss.append(loss.item())

                for _ in range(self._iters_generator):
                    y_gen = self.generate(cond_batch, normalise=False)
                    if self._instance_noise_std:
                        y_batch = self.instance_noise(y_batch, self._instance_noise_std)
                    if self._task == "CRAMER":
                        y_gen_prime = self.generate(condition=cond_batch, normalise=False)
                        loss = self._ganloss.g_loss(self.loss(y_gen, cond_batch),
                                                    self.loss(y_gen_prime, cond_batch),
                                                    self.loss(y_batch, cond_batch))
                    else:
                        loss = self._ganloss.g_loss(self.loss(y_gen, cond_batch))
                    g_optimizer.zero_grad()
                    loss.backward()
                    g_optimizer.step()
                gen_epoch_loss.append(loss.item())

            if self.burn_in_period is not None:
                if epoch >= self.burn_in_period:
                    if not self.gen_average_weights:
                        for param in self._generator.parameters():
                            self.gen_average_weights.append(param.detach().clone())
                    else:
                        for av_weight, weight in zip(self.gen_average_weights, self._generator.parameters()):
                            av_weight.data = self.averaging_coeff * av_weight.data + \
                                             (1 - self.averaging_coeff) * weight.data


        if self.burn_in_period is not None:
            for av_weight, weight in zip(self.gen_average_weights, self._generator.parameters()):
                weight.data = av_weight.data
        return self

    def generate(self, condition, normalise=True):
        if normalise:
            condition = (condition - self._cond_mean) / self._cond_std
            condition[torch.isnan(condition)] = 0
        n = len(condition)
        z = torch.randn(n, self._noise_dim,device=self.device)
        y = self._generator(z, condition)
        if normalise:
            y = y * self._y_std + self._y_mean
        return y
    
    
class VAEModel(torch.nn.Module):
    '''essa VAE prediz o input. Tem que adaptar'''
    def __init__(self, 
                 input_dim: int,
                  encoder = Encoder,
                  decoder = Decoder, 
                  epochs:int = 10, 
                  batch_size:int = 16,
                  lr:float = 1e-3):
        super().__init__()
        self.Encoder = encoder(input_dim)
        self.Decoder = decoder()
        self._epochs = epochs
        self.batch_size = batch_size
        self._lr = lr
    def reparameterization(self, mean, var):
        epsilon = torch.randn_like(var)        # sampling epsilon        
        z = mean + var*epsilon                          # reparameterization trick
        return z     
    def forward(self, x):
        mean, log_var = self.Encoder(x)
        z = self.reparameterization(mean, torch.exp(0.5 * log_var)) # takes exponential function (log var -> var)
        x_hat            = self.Decoder(z)        
        return x_hat, mean, log_var
    def loss_fn(self,x, x_hat, mean, log_var):
        reproduction_loss = torch.nn.functional.binary_cross_entropy(x_hat, x, reduction='sum')
        KLD      = - 0.5 * torch.sum(1+ log_var - mean.pow(2) - log_var.exp())

        return reproduction_loss + KLD
    def fit(self, phi,y,x):
        condition = torch.cat([phi,x],dim=-1)
        dataset = torch.utils.data.TensorDataset(condition,y)
        dataloader = torch.utils.data.DataLoader(dataset,
                                                 batch_size=self.batch_size,
                                                 shuffle=True)
        optimizer = torch.optim.Adam(self.parameters(), lr=self._lr)
        for epoch in trange(self._epochs, position=1, leave=False, desc = 'VAE training'):
            for X,Y in dataloader:
                optimizer.zero_grad()
                x_hat, mean, log_var = self.forward(X)
                loss = self.loss_fn(X, x_hat, mean, log_var)
                loss.backward()
                optimizer.step()
        return self
    def generate(self,phi,x):
        return self.Decoder(phi,x)



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
    print(phi)
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

