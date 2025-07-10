import torch
import matplotlib.pyplot as plt
import sys
import os
import pickle
sys.path.append(os.path.abspath(os.path.join('..', 'src')))
from optimizer import LGSO, LCSO
from problems import ShipMuonShield
from models import VAEModel, NormalizingFlowModel, BinaryClassifierModel
import torch
import subprocess
import re

torch.autograd.set_detect_anomaly(True)

def get_freest_gpu():
    if not torch.cuda.is_available():
        return torch.device('cpu')
    result = subprocess.run(
        ['nvidia-smi', '--query-gpu=memory.free', '--format=csv,nounits,noheader'],
        stdout=subprocess.PIPE, encoding='utf-8'
    )
    # Parse free memory for each GPU
    mem_free = [int(x) for x in result.stdout.strip().split('\n')]
    max_idx = mem_free.index(max(mem_free))
    return torch.device(f'cuda:{max_idx}')

dev = get_freest_gpu()
dim = 60

# Minimal ShipMuonShield setup for fast LGSO test
problem = ShipMuonShield(
    simulate_fields=False,  # No real field simulation
    n_samples=500_000,           # Very low number of samples for speed
    apply_det_loss=False,  # Skip detector loss for speed
    cost_loss_fn=None,     # No cost penalty
    dimensions_phi=dim,
    default_phi=torch.tensor(ShipMuonShield.warm_baseline), 
    cores = 100,
    parallel = False,
    fSC_mag = False,
    sensitive_plane = {'dz': 0.01, 'dx': 4.3, 'dy': 6.1,'position': 82}
)

# Get bounds for the reduced problem
bounds = problem.GetBounds(device=torch.device('cpu'))

# Initial phi
initial_phi = problem.initial_phi

# Set up a simple generative model (GANModel)
generative_model = BinaryClassifierModel(phi_dim=dim,
                            x_dim = 7,
                            n_epochs = 15,
                            batch_size = 2048,
                            lr = 1e-3,
                            device = dev)

# Set up LGSO optimizer
optimizer = LCSO(
    true_model=problem,
    surrogate_model=generative_model,
    bounds=bounds,
    samples_phi=50,        
    epsilon=0.2,       # Local search radius
    initial_phi=initial_phi,
    device='cpu',
    outputs_dir='/home/hep/lprate/projects/BlackBoxOptimization/outputs/test_robustness',
    resume=False)

# Run a short LGSO optimization and record losses
n_iters = 20
train_losses = []

objective_losses = [optimizer.loss(*optimizer.history).item()]
print(f"Initial Objective loss = {objective_losses[0]}")

for i in range(n_iters):
    if i > 0: generative_model.n_epochs = 3
    phi, obj_loss = optimizer.optimization_iteration()
    optimizer._i += 1
    print("Difference between current_phi and initial_phi:", (optimizer.current_phi - initial_phi).detach().cpu().numpy())
    # For generative model training loss, use the surrogate's fit loss if available, else dummy
    train_losses.extend(optimizer.model.last_train_loss)

    objective_losses.append(obj_loss.item())
    print(f"Iter {i+1}: Objective loss = {obj_loss}")
    with open(os.path.join(optimizer.outputs_dir,f'history.pkl'), "wb") as f:
        pickle.dump(optimizer.history, f)

print(f"Final phi: {optimizer.current_phi}")
# Plot the losses



plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.plot(train_losses, marker='o')
plt.title('Generative Model Training Loss')
plt.xlabel('Iteration')
plt.ylabel('Train Loss')
plt.grid(True)
plt.axvline(x=15, color='red', linestyle='--')
for idx in range(15, len(train_losses), 3):
    plt.axvline(x=idx, color='red', linestyle='--')

plt.subplot(1,2,2)
plt.plot(objective_losses, marker='o', color='orange')
plt.title('Objective Loss (to Minimize)')
plt.xlabel('Iteration')
plt.ylabel('Objective Loss')
plt.grid(True)

plt.tight_layout()
plt.savefig('robustness_test_losses.png')

print('Figure saved as robustness_test_losses.png')

print(f"Final Objective loss = {objective_losses[-1]}") 


