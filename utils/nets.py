import torch.nn as nn
import torch.nn.functional as F
import torch
from gpytorch.kernels import Kernel
from . import normalize_vector


class PsiCompressor(nn.Module):
    def __init__(self, input_param, out_dim):
        super().__init__()
        self.fc = nn.Linear(input_param, out_dim)
        self.out_dim = out_dim
        
    def forward(self, psi):
        return torch.tanh(self.fc(psi))


class Net(nn.Module):
    def __init__(self, out_dim, psi_dim, hidden_dim=100, X_dim=1):
        super().__init__()
        self.fc1 = nn.Linear(X_dim + psi_dim, hidden_dim)
        nn.init.xavier_normal_(self.fc1.weight)
        nn.init.constant_(self.fc1.bias, 0.0)
        
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        nn.init.xavier_normal_(self.fc2.weight)
        nn.init.constant_(self.fc2.bias, 0.0)
        
        self.fc3 = nn.Linear(hidden_dim, out_dim)
        nn.init.xavier_normal_(self.fc3.weight)
        nn.init.constant_(self.fc3.bias, 0.0)    
        
        self.fc4 = nn.Linear(hidden_dim, hidden_dim)
        nn.init.xavier_normal_(self.fc4.weight)
        nn.init.constant_(self.fc4.bias, 0.0)
        
    def forward(self, params):
        """
            Generator takes a vector of noise and produces sample
        """
        # psi_embedding = self.pc(params[:, :self.psi_dim])
        # z = torch.cat([z, psi_embedding, params[:, self.psi_dim:]], dim=1)
        h1 = torch.tanh(self.fc1(params))
        h2 = torch.tanh(self.fc2(h1))
        h3 = F.leaky_relu(self.fc4(h2))
        y_gen = self.fc3(h3)
        return y_gen        


# Idea is kind of taken from https://arxiv.org/pdf/1805.08318.pdf and applied to conditions
class Attention(nn.Module):
    def __init__(self, psi_dim, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self._psi_dim = psi_dim

        self.f_net = nn.Conv1d(1, hidden_dim, 1, stride=1, padding=0, bias=False)
        self.g_net = nn.Conv1d(1, hidden_dim, 1, stride=1, padding=0, bias=False)
        self.h_net = nn.Conv1d(1, hidden_dim, 1, stride=1, padding=0, bias=False)
        self.v_net = nn.Conv1d(hidden_dim, 1, 1, stride=1, padding=0, bias=False)
        self.softmax = nn.Softmax(dim=2)
        self.gamma = nn.Parameter(torch.Tensor([0.]), requires_grad=True)

    def forward(self, params):
        params_psi = params[:, :self._psi_dim].unsqueeze(1)
        attention_map = self.softmax(torch.bmm(self.f_net(params_psi).transpose(1, 2), self.g_net(params_psi)))
        self_attention = self.v_net(torch.bmm(attention_map, self.h_net(params_psi).transpose(1, 2)).transpose(1, 2))
        psi_with_attention = (self.gamma * self_attention + params_psi)[:, 0, :]
        return torch.cat([psi_with_attention, params[:, self._psi_dim:]], dim=1)


class SimpleAttention(nn.Module):
    def __init__(self, psi_dim, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self._psi_dim = psi_dim

        self._attention_net = nn.Sequential(
            nn.Linear(psi_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, psi_dim),
            nn.Softmax(dim=1)
        )
        self.gamma = nn.Parameter(torch.Tensor([0.]), requires_grad=True)

    def forward(self, params):
        psi_with_attention = params[:, :self._psi_dim] * self._attention_net(params[:, :self._psi_dim])
        return torch.cat([psi_with_attention, params[:, self._psi_dim:]], dim=1)


class Generator(nn.Module):
    def __init__(self, noise_dim, out_dim, psi_dim, hidden_dim=100, x_dim=1, attention_net=None):
        super(Generator, self).__init__()
        
        self.fc1 = nn.Linear(noise_dim + x_dim + psi_dim, hidden_dim)
        # nn.init.xavier_normal_(self.fc1.weight)
        # nn.init.constant_(self.fc1.bias, 0.0)
        
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        # nn.init.xavier_normal_(self.fc2.weight)
        # nn.init.constant_(self.fc2.bias, 0.0)
        
        self.fc3 = nn.Linear(hidden_dim, out_dim)
        # nn.init.xavier_normal_(self.fc3.weight)
        # nn.init.constant_(self.fc3.bias, 0.0)
        
        self.fc4 = nn.Linear(hidden_dim, hidden_dim)
        # nn.init.xavier_normal_(self.fc4.weight)
        # nn.init.constant_(self.fc4.bias, 0.0)
        
        # self.pc = psi_compressor
        self.psi_dim = psi_dim
        self.attention_net = attention_net

    def forward(self, z, params):
        """
            Generator takes a vector of noise and produces sample
        """
        # psi_embedding = self.pc(params[:, :self.psi_dim])
        # z = torch.cat([z, psi_embedding, params[:, self.psi_dim:]], dim=1)
        if self.attention_net:
            params = self.attention_net(params)
        z = torch.cat([z, params], dim=1)
        h1 = torch.tanh(self.fc1(z))
        h4 = torch.tanh(self.fc4(h1))
        h2 = F.leaky_relu(self.fc2(h4))
        y_gen = self.fc3(h2)
        return y_gen


class Discriminator(nn.Module):
    def __init__(self,
                 in_dim,
                 psi_dim,
                 hidden_dim=100,
                 output_logits=False,
                 x_dim=1,
                 output_dim=1,
                 attention_net=None):
        super(Discriminator, self).__init__()
        self.output_logits = output_logits

        self.fc1 = nn.Linear(in_dim + x_dim + psi_dim, hidden_dim)
        nn.init.xavier_normal_(self.fc1.weight)
        nn.init.constant_(self.fc1.bias, 0.0)
        
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        nn.init.xavier_normal_(self.fc2.weight)
        nn.init.constant_(self.fc2.bias, 0.0)
        
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        nn.init.xavier_normal_(self.fc3.weight)
        nn.init.constant_(self.fc3.bias, 0.0)
        
        self.fc4 = nn.Linear(hidden_dim, output_dim)
        nn.init.xavier_normal_(self.fc4.weight)
        nn.init.constant_(self.fc4.bias, 0.0)
        
        # self.pc = psi_compressor
        self.psi_dim = psi_dim
        self.attention_net = attention_net

    def forward(self, x, params):
        if self.attention_net:
            params = self.attention_net(params)
        print('KSDJKALSDKADHASD')
        print(x)
        assert not torch.isnan(x).any(), "x have NaN values"
        print(params)
        assert not torch.isnan(params).any(), "params have NaN values"
        x = torch.cat([x, params], dim=1)
        # psi_embedding = self.pc(params[:, :self.psi_dim])
        # x = torch.cat([x, psi_embedding, params[:, self.psi_dim:]], dim=1)
        print(self.fc1(x))
        assert not torch.isnan(self.fc1.weight).any(), "fc1 weights have NaN values"
        h1 = torch.tanh(self.fc1(x))
        h2 = F.leaky_relu(self.fc2(h1))
        # h3 = F.leaky_relu(self.fc3(h2))
        if self.output_logits:
            return self.fc4(h2)
        else:
            return torch.sigmoid(self.fc4(h2))


class GANLosses(object):
    def __init__(self, task):
        self.TASK = task

    def g_loss(self, discrim_output, discrim_output_gen_prime=None, discrim_output_real=None):
        eps = 1e-10
        if self.TASK == 'KL':
            loss = torch.log(1 - discrim_output + eps).mean()
        elif self.TASK == 'REVERSED_KL':
            loss = - torch.log(discrim_output + eps).mean()
        elif self.TASK == 'WASSERSTEIN':
            loss = - discrim_output.mean()
        elif self.TASK == "CRAMER":
            # loss = (torch.norm(discrim_output_real - discrim_output, dim=1) + \
            #        torch.norm(discrim_output_real - discrim_output_gen_prime, dim=1) - \
            #        torch.norm(discrim_output - discrim_output_gen_prime, dim=1)).mean()
            loss =   (torch.norm(discrim_output_real - discrim_output_gen_prime, dim=1) -
                      torch.norm(discrim_output_real, dim=1) -
                      torch.norm(discrim_output - discrim_output_gen_prime, dim=1) +
                      torch.norm(discrim_output, dim=1)).mean()
        return loss

    def d_loss(self, discrim_output_gen, discrim_output_real, discrim_output_gen_prime=None):
        eps = 1e-10
        if self.TASK in ['KL', 'REVERSED_KL']:
            loss = - torch.log(discrim_output_real + eps).mean() - torch.log(1 - discrim_output_gen + eps).mean()
        elif self.TASK == 'WASSERSTEIN':
            loss = - (discrim_output_real.mean() - discrim_output_gen.mean())
        elif self.TASK == "CRAMER":
            loss = - (torch.norm(discrim_output_real - discrim_output_gen_prime, dim=1) -
                      torch.norm(discrim_output_real, dim=1) -
                      torch.norm(discrim_output_gen - discrim_output_gen_prime, dim=1) +
                      torch.norm(discrim_output_gen, dim=1)).mean()
        return loss

    def calc_gradient_penalty(self, discriminator, data_gen, inputs_batch, inp_data, lambda_reg=.1, data_gen_prime=None):
        device = data_gen.device
        alpha = torch.rand(inp_data.shape[0], 1).to(device)
        dims_to_add = len(inp_data.size()) - 2
        for i in range(dims_to_add):
            alpha = alpha.unsqueeze(-1)

        interpolates = (alpha * inputs_batch + ((1 - alpha) * data_gen)).to(device)
        interpolates.requires_grad_(True)
        disc_interpolates = discriminator(interpolates, inp_data)
        if self.TASK == "WASSERSTEIN":
            gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                                            grad_outputs=torch.ones(disc_interpolates.size()).to(device),
                                            create_graph=True, retain_graph=True, only_inputs=True)[0]
        elif self.TASK == "CRAMER":
            f_val = torch.norm(discriminator(data_gen_prime, inp_data) - disc_interpolates, dim=1) - \
                    torch.norm(disc_interpolates, dim=1)
            gradients = torch.autograd.grad(outputs=f_val, inputs=interpolates,
                                            grad_outputs=torch.ones(f_val.size()).to(device),
                                            create_graph=True, retain_graph=True, only_inputs=True)[0]
        grad_norm_diff = gradients.norm(2, dim=1) - 1
        grad_norm_diff[grad_norm_diff < 0] = 0
        gradient_penalty = (grad_norm_diff ** 2).mean() * lambda_reg
        return gradient_penalty

    def calc_zero_centered_GP(self, discriminator, data_gen, inputs_batch, inp_data, gamma_reg=.1):
        # TODO: data_gen is not used!
        device = inputs_batch.device
        local_input = inp_data.clone().detach().requires_grad_(True)
        disc_interpolates = discriminator(local_input, inputs_batch)
        gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=local_input,
                                        grad_outputs=torch.ones(disc_interpolates.size()).to(device),
                                        create_graph=True, retain_graph=True, only_inputs=True)[0]
        return gamma_reg / 2 * (gradients.norm(2, dim=1) ** 2).mean() 


class Encoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc_mean = nn.Linear(64, latent_dim)
        self.fc_log_var = nn.Linear(64, latent_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        h = self.relu(self.fc1(x))
        h = self.relu(self.fc2(h))
        return self.fc_mean(h), self.fc_log_var(h)

class Decoder(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        h = self.relu(self.fc1(x))
        h = self.relu(self.fc2(h))
        return self.fc3(h)
    
class IBNN_ReLU(Kernel):
    is_stationary = False

    def __init__(self, d, var_w:float = 10., var_b:float = 1.6, depth:int = 3, **kwargs):
        super().__init__(**kwargs)
        self.d = d
        self.var_w = var_w
        self.var_b = var_b
        self.depth = depth

    def k(self, l, x1, x2):
        # base case
        if l == 0:
            return self.var_b + self.var_w * (x1 * x2).sum(-1) / self.d
        else:
            K_12 = self.k(l - 1, x1, x2)
            K_11 = self.k(l - 1, x1, x1)
            K_22 = self.k(l - 1, x2, x2)
            sqrt_term = torch.sqrt(K_11 * K_22)
            fraction = K_12 / sqrt_term
            epsilon = 1e-7
            theta = torch.acos(torch.clamp(fraction, min=-1 + epsilon, max=1 - epsilon))
            theta_term = torch.sin(theta) + (torch.pi - theta) * fraction
            result = self.var_b + self.var_w / (2 * torch.pi) * sqrt_term * theta_term
            return result
        
    def forward(self, x1, x2, **params):
        d2 = x2.shape[-2]
        x1_shape = tuple(x1.shape)
        d1, dim = x1_shape[-2:]
        new_shape = x1_shape[:-2] + (d1, d2, dim)
        new_x1 = x1.unsqueeze(-2).expand(new_shape)
        new_x2 = x2.unsqueeze(-3).expand(new_shape)
        result = self.k(self.depth, new_x1, new_x2)
        return result

class Classifier(torch.nn.Module):
    def __init__(self, phi_dim,x_dim = 3, hidden_dim=128):
        super().__init__()
        self.fc1 = torch.nn.Linear(x_dim + phi_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        #self.fc3 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = torch.nn.Linear(hidden_dim, 1)
        self.activation = torch.nn.SiLU()

    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        #x = self.activation(self.fc3(x))
        x = self.fc3(x)
        return x
    def predict(self, phi, position):
        #position = self.normalize_position(position)
        #phi = self.normalize_params(phi)
        input_tensor = torch.cat([phi, position], dim=1)
        return torch.sigmoid(self.forward(input_tensor))
    def normalize_params(self, params):
        return normalize_vector(params, self.bounds_phi)

class DeepONetClassifier(torch.nn.Module):
    def __init__(self, phi_dim, x_dim = 3,
                 layers:list = [[128,128],[128,128]], p = 64, layer_norm:bool=False):
        super().__init__()
        self.output_dim = 1
        self.p = p
        branch_input_dim = phi_dim
        trunk_input_dim = x_dim
        branch_layers, trunk_layers = layers
        self.layer_dims = [(phi_dim, x_dim), branch_layers, trunk_layers]
        self.branch_net = self._build_network(branch_input_dim, branch_layers, self.output_dim*self.p, activation='silu')
        self.trunk_net = self._build_network(trunk_input_dim, trunk_layers, self.output_dim*self.p, activation='relu')
        self.bias = torch.nn.Parameter(torch.zeros(self.output_dim))
        self._init_weights()
        self.branch_ln = torch.nn.LayerNorm(self.output_dim * self.p) if layer_norm else None
        self.trunk_ln = torch.nn.LayerNorm(self.output_dim * self.p) if layer_norm else None
        self.last_layer = torch.nn.Linear(self.output_dim * self.p, self.output_dim)
        self.head = torch.nn.Linear(2*self.p, 1)#torch.nn.Sequential(torch.nn.Linear(2*self.p, 2*self.p), torch.nn.SiLU(),torch.nn.Linear(2*self.p, 1))
        
    def _build_network(self, input_dim, layers, final_dim, activation = 'silu'):
        modules = []
        in_size = input_dim
        for h_dim in layers:
            modules.append(torch.nn.Linear(in_size, h_dim))
            modules.append(torch.nn.SiLU() if activation == 'silu' else torch.nn.ReLU())
            in_size = h_dim
        modules.append(torch.nn.Linear(in_size, final_dim))
        return torch.nn.Sequential(*modules)
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_normal_(m.weight)
                torch.nn.init.zeros_(m.bias)
        with torch.no_grad():
            self.branch_net[-1].weight.div_(self.p**0.5)
            self.trunk_net[-1].weight.div_(self.p**0.5)
    
    def forward(self, branch_input, trunk_input):
        assert branch_input.le(1.0).all() and branch_input.ge(0.0).all(), "Branch input should be in [0, 1]"
        b = self.branch_net(branch_input)
        t = self.trunk_net(trunk_input)
        if self.branch_ln is not None:
            b = self.branch_ln(b)
        if self.trunk_ln is not None:
            t = self.trunk_ln(t)
        #b = b.unsqueeze(1).expand_as(t)                 # [B, N, H]
        #feats = torch.cat([t, b], dim=-1)    # [B, N, 3H]
        #logits = self.head(feats).squeeze(-1) + self.bias   # [B, N]
        #return logits
        prediction = torch.einsum('bp,bnp->bnp', b, t)  # [B, N, p]
        prediction = self.last_layer(prediction).squeeze(-1)  # [B, N]
        prediction = prediction + self.bias
        return prediction
    
class QuadraticModel(nn.Module):
    def __init__(self, phi_dim, x_dim=3,
                 trunk_layers: list = [128, 128], p=64, layer_norm: bool = False):
        super().__init__()
        self.output_dim = 1
        self.p = p
        
        branch_input_dim = phi_dim
        trunk_input_dim = x_dim
        
        self.branch_net = self._build_poly(branch_input_dim, self.output_dim * self.p)
        
        self.trunk_net = self._build_network(trunk_input_dim, trunk_layers, self.output_dim * self.p, activation='relu')
        
        self.bias = nn.Parameter(torch.zeros(self.output_dim))
        
        self._init_weights()
        
        self.trunk_ln = nn.LayerNorm(self.output_dim * self.p) if layer_norm else None
        self.head = torch.nn.Linear(2*self.p, 1)

    def _build_network(self, input_dim, layers, final_dim, activation='silu'):
        """Standard MLP construction for the Trunk."""
        modules = []
        in_size = input_dim
        for h_dim in layers:
            modules.append(nn.Linear(in_size, h_dim))
            modules.append(nn.SiLU() if activation == 'silu' else nn.ReLU())
            in_size = h_dim
        modules.append(nn.Linear(in_size, final_dim))
        return nn.Sequential(*modules)

    def _build_poly(self, input_dim, final_dim):
        """
        Builds a Polynomial Expansion layer followed by a Linear projection.
        This guarantees that the mapping from phi -> embedding is quadratic.
        """
        class QuadraticExpansion(nn.Module):
            def __init__(self, dim):
                super().__init__()
                self.dim = dim
            
            def forward(self, x):
                # 1. Linear terms
                linear = x
                # 2. Quadratic terms (Full Outer Product: x_i * x_j)
                # We use triu_indices to take only unique combinations (upper triangle)
                B, D = x.shape
                # Compute outer product [B, D, D]
                outer = torch.einsum('bi,bj->bij', x, x)
                # Extract upper triangle indices
                idx = torch.triu_indices(D, D, device=x.device)
                quad = outer[:, idx[0], idx[1]]
                
                # Concatenate [x, x^2 + cross_terms]
                return torch.cat([linear, quad], dim=1)

        # Calculate expansion dimension: D + D*(D+1)/2
        poly_dim = input_dim + (input_dim * (input_dim + 1)) // 2
        
        return nn.Sequential(
            QuadraticExpansion(input_dim),
            nn.Linear(poly_dim, final_dim)
        )
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        # Scale final layers for stability
        with torch.no_grad():
            self.branch_net[-1].weight.div_(self.p**0.5)
            self.trunk_net[-1].weight.div_(self.p**0.5)
    
    def forward(self, branch_input, trunk_input):
        assert branch_input.le(1.0).all() and branch_input.ge(0.0).all(), "Branch input should be in [0, 1]"
        
        b = self.branch_net(branch_input)

        t = self.trunk_net(trunk_input) 
        
        if self.trunk_ln is not None:
            t = self.trunk_ln(t)
        if t.dim() == 3:
            b = b.unsqueeze(1) # [B, 1, p]
        logits = (b * t).sum(dim=-1, keepdim=False) + self.bias
        return logits
        b = b.unsqueeze(1).expand_as(t)                 # [B, N, H]
        feats = torch.cat([t, b], dim=-1)    # [B, N, 3H]
        logits = self.head(feats).squeeze(-1) + self.bias   # [B, N]
        return logits