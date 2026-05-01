import torch.nn as nn
import torch.nn.functional as F
import torch
from gpytorch.kernels import Kernel
from . import normalize_vector
from math import sqrt


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
    def __init__(self, phi_dim,x_dim = 3, hidden_dim=128, activation='silu'):
        super().__init__()
        self.fc1 = torch.nn.Linear(x_dim + phi_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        #self.fc3 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = torch.nn.Linear(hidden_dim, 1)
        self.activation = {'silu': torch.nn.SiLU(), 'relu': torch.nn.ReLU()}.get(activation)

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
                 layers:list = [[128,128],[128,128]], p = 64, 
                 activation='silu', layer_norm:bool=False):
        super().__init__()
        self.output_dim = 1
        self.p = p
        branch_input_dim = phi_dim
        trunk_input_dim = x_dim
        branch_layers, trunk_layers = layers
        self.layer_dims = [(phi_dim, x_dim), branch_layers, trunk_layers]
        self.branch_net = self._build_network(branch_input_dim, branch_layers, self.output_dim*self.p, activation=activation)
        self.trunk_net = self._build_network(trunk_input_dim, trunk_layers, self.output_dim*self.p, activation=activation)
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
    




class StochasticTaylor(torch.nn.Module):
    def __init__(self, phi_dim, x_dim, phi0, hidden_dim=128, p=64, use_residual=True):
        super().__init__()
        self.D = phi_dim
        self.K = x_dim
        
        # Register phi0 as a buffer so it automatically moves to the correct device
        self.register_buffer('phi0', phi0.view(-1))
        
        # 1. The Shared Trunk (Particle Latent Representation)
        self.trunk = torch.nn.Sequential(
            torch.nn.Linear(self.K, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.GELU()
        )
        
        # 2. The Taylor Heads
        self.head_a = torch.nn.Linear(hidden_dim, 1)
        self.head_b = torch.nn.Linear(hidden_dim, self.D)
        
        # W_C shape: (hidden_dim, D, D)
        self.W_C = torch.nn.Parameter(torch.empty(hidden_dim, self.D, self.D))
        self.bias_C = torch.nn.Parameter(torch.zeros(self.D, self.D))
        torch.nn.init.kaiming_uniform_(self.W_C, a=sqrt(5))
        
        # 3. DeepONet Residual Architecture (Memory Saver)
        self.res_branch = torch.nn.Sequential(
            torch.nn.Linear(self.K, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_dim, p)
        )
        self.res_trunk = torch.nn.Sequential(
            torch.nn.Linear(self.D, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_dim, p)
        )
        self.use_residual = use_residual

    def forward(self, phi, x):
        """
        phi: shape (n_phi, D)
        x: shape (n_phi, S, K) or (S, K)
        """
        # Ensure phi is 2D and compute dphi internally
        if phi.dim() == 1:
            phi = phi.unsqueeze(0)
            
        dphi = phi - self.phi0
        
        # --- Taylor Expansion Part ---
        h = self.trunk(x)
        a = self.head_a(h) 
        b = self.head_b(h) 
        
        # Expand dphi for the sample dimension if x is 3D
        lin_term = torch.sum(b * dphi.unsqueeze(1), dim=2, keepdim=True) 
        
        W_dphi = torch.einsum('lij, nj -> nli', self.W_C, dphi)
        v = torch.einsum('ni, nli -> nl', dphi, W_dphi)
        quad_main = torch.einsum('nsl, nl -> ns', h, v).unsqueeze(-1)
        
        bias_dphi = torch.einsum('ij, nj -> ni', self.bias_C, dphi)
        b_term = torch.einsum('ni, ni -> n', dphi, bias_dphi).view(-1, 1, 1)

        quad_term = 0.5 * (quad_main + b_term)
        logit = a + lin_term + quad_term 
        # --- DeepONet Residual Part ---
        if self.use_residual:
            norm_dphi_cubed = (torch.norm(dphi, p=2, dim=1) ** 3).view(-1, 1, 1)
        
            y_branch = self.res_branch(x)    
            y_trunk = self.res_trunk(dphi)   
        
            residual = torch.einsum('nsp, np -> ns', y_branch, y_trunk).unsqueeze(-1) * norm_dphi_cubed 
        
            logit = logit + residual
        return logit.squeeze(-1) 

    @torch.no_grad()
    def get_taylor_coefficients(self, x):
        h = self.trunk(x)
        a = self.head_a(h)
        b = self.head_b(h)
        
        # Dynamically handle both 2D (S, K) and 3D (n_phi, S, K) batches
        if h.dim() == 3:
            C_unsym = torch.einsum('nsl, lij -> nsij', h, self.W_C) + self.bias_C
            C = 0.5 * (C_unsym + C_unsym.transpose(2, 3))
        else:
            C_unsym = torch.einsum('sl, lij -> sij', h, self.W_C) + self.bias_C
            C = 0.5 * (C_unsym + C_unsym.transpose(1, 2))
        return a, b, C

    @torch.no_grad()
    def grad_phi(self, phi, x):
        """
        Analytical Gradient of the expected hits evaluated strictly at phi0.
        Requires zero autograd overhead.
        """
        a, b, _ = self.get_taylor_coefficients(x)
        
        # Sigmoid derivatives
        p = torch.sigmoid(a)
        dp = p * (1.0 - p)
        
        # Sum over the sample dimension
        sample_dim = 1 if x.dim() == 3 else 0
        grad = torch.sum(dp * b, dim=sample_dim) 
        return grad

    @torch.no_grad()
    def hess_phi(self, phi, x):
        """
        Analytical Hessian of the expected hits evaluated strictly at phi0.
        Outputs the exact DxD matrix to run your eigenvalue compensation analysis.
        """
        a, b, C = self.get_taylor_coefficients(x)
        
        p = torch.sigmoid(a)
        dp = p * (1.0 - p)
        ddp = p * (1.0 - p) * (1.0 - 2.0 * p)
        
        if x.dim() == 3:
            b_outer = torch.einsum('nsi, nsj -> nsij', b, b)
            hess = ddp.unsqueeze(-1) * b_outer + dp.unsqueeze(-1) * C
            return torch.sum(hess, dim=1) # Shape: (n_phi, D, D)
        else:
            b_outer = torch.einsum('si, sj -> sij', b, b)
            hess = ddp.unsqueeze(-1) * b_outer + dp.unsqueeze(-1) * C
            return torch.sum(hess, dim=0) # Shape: (D, D)

    def predict_proba(self, phi, x):
        logits = self.forward(phi, x)
        return torch.sigmoid(logits)

    def predict_hits(self, phi, x):
        return self.predict_proba(phi, x).sum(dim=-1)

class SharedBasisStochasticTaylor(torch.nn.Module):
    """
    Local stochastic Taylor model around phi0.

    For a muon x and parameter vector phi, define dphi = phi - phi0.
    The model predicts the local logit

        eta(x, phi) =
            a(x)
            + u(x)^T dphi
            + 0.5 * sum_k lambda_k(x) * (v_k^T dphi)^2
            + 0.5 * sum_j diag_j(x) * dphi_j^2
            + residual_3(x, dphi)

    where:
      - a(x) is a scalar offset,
      - u(x) is the linear coefficient,
      - {v_k} are GLOBAL shared quadratic directions,
      - lambda_k(x) are muon-dependent amplitudes for those directions,
      - diag_j(x) is an optional muon-dependent diagonal correction,
      - residual_3 is constrained to scale like ||dphi||^3.

    The probability is sigmoid(eta).
    """

    def __init__(
        self,
        phi_dim,
        x_dim,
        phi0,
        hidden_dim=128,
        rank=8,
        residual_width=64,
        use_diagonal=True,
        use_residual=True,
        eps_norm=1e-12,
    ):
        super().__init__()

        self.D = phi_dim
        self.K = x_dim
        self.rank = rank
        self.use_diagonal = use_diagonal
        self.use_residual = use_residual
        self.eps_norm = eps_norm

        # Center of the local Taylor model
        self.register_buffer("phi0", phi0.view(-1))

        # ---------------------------
        # 1) Shared trunk over muon features
        # ---------------------------
        self.trunk = torch.nn.Sequential(
            torch.nn.Linear(self.K, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.GELU(),
        )

        # ---------------------------
        # 2) Heads for local Taylor coefficients
        # ---------------------------
        self.head_a = torch.nn.Linear(hidden_dim, 1)          # a(x)
        self.head_u = torch.nn.Linear(hidden_dim, self.D)     # u(x)
        self.head_lambda = torch.nn.Linear(hidden_dim, rank)  # lambda(x)

        if self.use_diagonal:
            self.head_diag = torch.nn.Linear(hidden_dim, self.D)  # diagonal correction d(x)

        # ---------------------------
        # 3) Global shared quadratic basis V in R^{D x rank}
        #    These are the common curvature directions.
        # ---------------------------
        self.V_raw = torch.nn.Parameter(torch.empty(self.D, rank))
        torch.nn.init.orthogonal_(self.V_raw)

        # ---------------------------
        # 4) Third-order residual branch
        #    This branch is multiplied by ||dphi||^3 so it cannot dominate
        #    first- and second-order behavior near phi0.
        # ---------------------------
        if self.use_residual:
            self.res_branch = torch.nn.Sequential(
                torch.nn.Linear(self.K, hidden_dim),
                torch.nn.GELU(),
                torch.nn.Linear(hidden_dim, residual_width),
            )
            self.res_trunk = torch.nn.Sequential(
                torch.nn.Linear(self.D, hidden_dim),
                torch.nn.GELU(),
                torch.nn.Linear(hidden_dim, residual_width),
            )

    # ------------------------------------------------------------------
    # Utility: orthonormalized shared basis
    # ------------------------------------------------------------------
    def basis(self):
        """
        Returns an orthonormal basis V derived from V_raw using QR.
        Shape: (D, rank)
        """
        # Reduced QR gives D x rank with orthonormal columns if D >= rank
        V, _ = torch.linalg.qr(self.V_raw, mode="reduced")
        return V

    # ------------------------------------------------------------------
    # Utility: local coefficients
    # ------------------------------------------------------------------
    def local_coefficients(self, x):
        """
        Compute the muon-dependent coefficients a(x), u(x), lambda(x), diag(x).

        x:
          - shape (S, K), or
          - shape (n_phi, S, K)

        returns:
          a       : (..., 1)
          u       : (..., D)
          lam     : (..., rank)
          diag    : (..., D) or None
        """
        h = self.trunk(x)
        a = self.head_a(h)
        u = self.head_u(h)
        lam = self.head_lambda(h)

        diag = self.head_diag(h) if self.use_diagonal else None
        return a, u, lam, diag

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(self, phi, x):
        """
        phi: shape (n_phi, D) or (D,)
        x  : shape (S, K) or (n_phi, S, K)

        returns:
          logits of shape
            - (n_phi, S) if x is (n_phi, S, K)
            - (n_phi, S) if x is (S, K)
        """
        if phi.dim() == 1:
            phi = phi.unsqueeze(0)

        dphi = phi - self.phi0                     # (n_phi, D)
        n_phi = dphi.shape[0]

        V = self.basis()                           # (D, rank)

        # Local coefficients from muon features
        a, u, lam, diag = self.local_coefficients(x)

        # --------------------------------------------------------------
        # Broadcast x if needed:
        # - if x is (S, K), coefficients are (S, *)
        # - if x is (n_phi, S, K), coefficients are (n_phi, S, *)
        # --------------------------------------------------------------
        if x.dim() == 2:
            # a:    (S, 1)
            # u:    (S, D)
            # lam:  (S, rank)
            # diag: (S, D) or None
            S = x.shape[0]

            # Linear term: u(x)^T dphi
            # -> (n_phi, S)
            lin_term = torch.einsum("sd,nd->ns", u, dphi)

            # Shared low-rank quadratic term
            # z = V^T dphi, shape (n_phi, rank)
            z = dphi @ V
            z2 = z ** 2
            quad_lr = 0.5 * torch.einsum("sr,nr->ns", lam, z2)

            # Optional diagonal quadratic correction
            if self.use_diagonal:
                dphi2 = dphi ** 2
                quad_diag = 0.5 * torch.einsum("sd,nd->ns", diag, dphi2)
            else:
                quad_diag = 0.0

            logit = a.transpose(0, 1) + lin_term + quad_lr + quad_diag

            # Third-order residual
            if self.use_residual:
                norm_dphi = torch.norm(dphi, p=2, dim=1, keepdim=True)   # (n_phi, 1)
                scale3 = norm_dphi ** 3                                  # (n_phi, 1)

                # Direction only, to avoid the residual learning lower-order scale effects
                dphi_unit = dphi / (norm_dphi + self.eps_norm)           # (n_phi, D)

                y_branch = self.res_branch(x)                            # (S, residual_width)
                y_trunk = self.res_trunk(dphi_unit)                      # (n_phi, residual_width)

                residual = torch.einsum("sp,np->ns", y_branch, y_trunk)  # (n_phi, S)
                logit = logit + scale3 * residual

            return logit

        elif x.dim() == 3:
            # x is paired with phi
            # a:    (n_phi, S, 1)
            # u:    (n_phi, S, D)
            # lam:  (n_phi, S, rank)
            # diag: (n_phi, S, D) or None
            S = x.shape[1]

            # Linear term
            lin_term = torch.sum(u * dphi.unsqueeze(1), dim=-1)          # (n_phi, S)

            # Shared low-rank quadratic term
            z = dphi @ V                                                 # (n_phi, rank)
            z2 = z ** 2
            quad_lr = 0.5 * torch.sum(lam * z2.unsqueeze(1), dim=-1)     # (n_phi, S)

            # Optional diagonal correction
            if self.use_diagonal:
                dphi2 = dphi ** 2
                quad_diag = 0.5 * torch.sum(diag * dphi2.unsqueeze(1), dim=-1)
            else:
                quad_diag = 0.0

            logit = a.squeeze(-1) + lin_term + quad_lr + quad_diag

            # Third-order residual
            if self.use_residual:
                norm_dphi = torch.norm(dphi, p=2, dim=1, keepdim=True)   # (n_phi, 1)
                scale3 = norm_dphi ** 3                                  # (n_phi, 1)
                dphi_unit = dphi / (norm_dphi + self.eps_norm)           # (n_phi, D)

                y_branch = self.res_branch(x)                            # (n_phi, S, residual_width)
                y_trunk = self.res_trunk(dphi_unit)                      # (n_phi, residual_width)

                residual = torch.einsum("nsp,np->ns", y_branch, y_trunk)
                logit = logit + scale3 * residual

            return logit

        else:
            raise ValueError("x must have shape (S, K) or (n_phi, S, K)")

    # ------------------------------------------------------------------
    # Probability / expected hits
    # ------------------------------------------------------------------
    def predict_proba(self, phi, x):
        return torch.sigmoid(self.forward(phi, x))

    def predict_hits(self, phi, x):
        return self.predict_proba(phi, x).sum(dim=-1)

    # ------------------------------------------------------------------
    # Explicit quadratic logit Hessian C(x) at phi0
    # ------------------------------------------------------------------
    @torch.no_grad()
    def get_logit_hessian_coefficients(self, x):
        """
        Returns the muon-dependent Hessian of the LOGIT at phi0:
            C(x) = V diag(lambda(x)) V^T + diag(diag(x))

        Shapes:
          if x is (S, K):         returns (S, D, D)
          if x is (n_phi, S, K):  returns (n_phi, S, D, D)
        """
        V = self.basis()
        _, _, lam, diag = self.local_coefficients(x)

        if x.dim() == 2:
            # low-rank part: sum_r lam[s,r] * v_r v_r^T
            C_lr = torch.einsum("sr,dr,er->sde", lam, V, V)

            if self.use_diagonal:
                C_diag = torch.diag_embed(diag)
                C = C_lr + C_diag
            else:
                C = C_lr
            return C

        elif x.dim() == 3:
            C_lr = torch.einsum("nsr,dr,er->nsde", lam, V, V)

            if self.use_diagonal:
                C_diag = torch.diag_embed(diag)
                C = C_lr + C_diag
            else:
                C = C_lr
            return C

        else:
            raise ValueError("x must have shape (S, K) or (n_phi, S, K)")

    # ------------------------------------------------------------------
    # Analytical gradient of expected hits at phi0
    # ------------------------------------------------------------------
    @torch.no_grad()
    def grad_phi(self, phi, x):
        """
        Analytical gradient of expected hits evaluated at phi0.

        Uses:
            grad p = sigma'(a) * u
        because dphi = 0 kills quadratic and residual contributions in first derivative.

        Returns:
          if x is (S, K):         (D,)
          if x is (n_phi, S, K):  (n_phi, D)
        """
        a, u, _, _ = self.local_coefficients(x)
        p = torch.sigmoid(a)
        dp = p * (1.0 - p)

        if x.dim() == 2:
            grad = torch.sum(dp * u, dim=0)          # (D,)
            return grad

        elif x.dim() == 3:
            grad = torch.sum(dp * u, dim=1)          # (n_phi, D)
            return grad

        else:
            raise ValueError("x must have shape (S, K) or (n_phi, S, K)")

    # ------------------------------------------------------------------
    # Analytical Hessian of expected hits at phi0
    # ------------------------------------------------------------------
    @torch.no_grad()
    def hess_phi(self, phi, x):
        """
        Analytical Hessian of expected hits at phi0.

        For one muon:
            Hess p = sigma''(a) * u u^T + sigma'(a) * C

        where
            C = V diag(lambda) V^T + diag(diag)

        Returns:
          if x is (S, K):         (D, D)
          if x is (n_phi, S, K):  (n_phi, D, D)
        """
        a, u, _, _ = self.local_coefficients(x)
        C = self.get_logit_hessian_coefficients(x)

        p = torch.sigmoid(a)
        dp = p * (1.0 - p)
        ddp = dp * (1.0 - 2.0 * p)

        if x.dim() == 2:
            u_outer = torch.einsum("sd,se->sde", u, u)              # (S, D, D)
            hess_per_muon = ddp.unsqueeze(-1) * u_outer + dp.unsqueeze(-1) * C
            return torch.sum(hess_per_muon, dim=0)                  # (D, D)

        elif x.dim() == 3:
            u_outer = torch.einsum("nsd,nse->nsde", u, u)           # (n_phi, S, D, D)
            hess_per_muon = ddp.unsqueeze(-1) * u_outer + dp.unsqueeze(-1) * C
            return torch.sum(hess_per_muon, dim=1)                  # (n_phi, D, D)

        else:
            raise ValueError("x must have shape (S, K) or (n_phi, S, K)")

    # ------------------------------------------------------------------
    # Optional regularizer to call from training loop
    # ------------------------------------------------------------------
    def orthogonality_penalty(self):
        """
        Penalize non-orthogonality of the shared basis.
        Useful if you do not rely only on QR in basis().
        """
        G = self.V_raw.transpose(0, 1) @ self.V_raw
        I = torch.eye(self.rank, device=G.device, dtype=G.dtype)
        return torch.sum((G - I) ** 2)