import torch
import torch.nn as nn
import time

# Define the MLP
class MLP(nn.Module):
    def __init__(self, input_size=106, hidden_size1=128, hidden_size2=128, output_size=1):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size2, output_size)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu1(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.fc3(out)
        return out

model = MLP()

def total_G(phi, z_batch):
    """
    This function computes sum_i G(phi, z_i) where G is an MLP that takes (phi, z_i) as input.
    """
    N = z_batch.shape[0]
    phi_expanded = phi.unsqueeze(0).expand(N, -1) # Shape (N, 100)
    mlp_input = torch.cat((phi_expanded, z_batch), dim=1) # Shape (N, 106)
    mlp_output = model(mlp_input) # Shape (N, 1)
    return torch.sum(mlp_output)

N = 1_000_000
phi = torch.randn(100, requires_grad=True)
z = torch.randn(N, 6) # z is the batch of vectors
v = torch.randn(100) # The vector 'v' for the H*v product

# 1. Compute the gradient w.r.t. phi
start_time = time.time()
grad_phi, = torch.autograd.grad(total_G(phi, z), phi, create_graph=True)
end_time = time.time()
print(f"Time for grad: {end_time - start_time:.6f} seconds")

# 2. Compute the HVP
# This is the gradient of (grad_phi • v) w.r.t. phi
start_time = time.time()
hvp, = torch.autograd.grad(torch.dot(grad_phi, v), phi, retain_graph=True)
end_time = time.time()
print(f"Time for HVP: {end_time - start_time:.6f} seconds")

print("Hessian-vector product shape:", hvp.shape)

# 3. Compute the full Hessian w.r.t. phi
def total_G_phi_only(p):
    return total_G(p, z)

start_time = time.time()
H = torch.autograd.functional.hessian(total_G_phi_only, phi)
end_time = time.time()
print(f"Time for Hessian: {end_time - start_time:.6f} seconds")
print("Hessian shape:", H.shape)