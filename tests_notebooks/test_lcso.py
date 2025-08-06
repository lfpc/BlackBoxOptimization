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
import wandb
import argparse

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
dim = 63
epsilon = 0.1  # Local search radius
samples_phi = 50  # Number of samples for phi
n_muons = 100_000
initial_lambda_constraints = 1e-3
initial_lr = 0.1


parser = argparse.ArgumentParser()
parser.add_argument('--name', type=str, default='TestSecondOrder', help='WandB run name')
args = parser.parse_args()

WANDB = {'project': 'LCSOTests_warm_baseline_normalized', 'name': args.name, 'config': {
    'dim': dim,
    'epsilon': epsilon,
    'samples_phi': samples_phi}}


# Minimal ShipMuonShield setup for fast LGSO test
problem = ShipMuonShield(
    simulate_fields=False,  # No real field simulation
    n_samples=n_muons,           # Very low number of samples for speed
    apply_det_loss=False,  # Skip detector loss for speed
    cost_loss_fn=None,     # No cost penalty
    dimensions_phi=dim,
    default_phi=torch.tensor(ShipMuonShield.warm_baseline), 
    cores = 80,
    SND = False,
    parallel = False,
    fSC_mag = False,
    cavern = True,
    sensitive_plane = [{'dz': 0.01, 'dx': 4, 'dy': 6,'position': 82}]
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
    samples_phi=samples_phi,
    epsilon=epsilon,
    initial_phi=initial_phi,
    initial_lambda_constraints=initial_lambda_constraints,  # Initial lambda for constraints
    initial_lr=initial_lr,  # Initial learning rate or trust_radius for phi optimization
    device='cpu',
    outputs_dir='/home/hep/lprate/projects/BlackBoxOptimization/outputs/test_robustness',
    resume=False)

# Run a short LGSO optimization and record losses
n_iters = 10
train_losses = []

objective_losses = [optimizer.loss(*optimizer.history).item()*n_muons]
print(f"Initial Objective loss = {objective_losses[0]}")

wandb.login()
with wandb.init(reinit = True,**WANDB) as wb:
    wb.log({'objective_loss': objective_losses[0], 
            'constraints': problem.get_constraints(initial_phi).item(), 
            'violated_bounds': initial_phi.lt(bounds[0]).logical_or(initial_phi.gt(bounds[1])).sum().float().item(),
            'distance_from_origin': 0.0,
            'trust_radius': optimizer.trust_radius})
    old_phi = initial_phi.clone().detach()
    for i in range(n_iters):
        if i > 0: generative_model.n_epochs = 6
        #if i in [15, 20, 25]:  # Reduce learning rate at specific iterations
        #    optimizer.phi_optimizer.param_groups[0]['lr'] *= 0.2
        if i in [3,6, 9]:
            optimizer.lambda_constraints *= 0.1
        phi, obj_loss = optimizer.optimization_iteration_second_order()
        obj_loss *= n_muons
        print("Optimized phi:", phi)
        predicted_loss = optimizer.get_model_pred(phi).item()*n_muons

        with torch.no_grad(): contraints = problem.get_constraints(phi)
        assert torch.all(phi >= bounds[0].to(phi.device)) and torch.all(phi <= bounds[1].to(phi.device)), "current_phi is out of bounds after optimization step"
        #assert contraints.gt(0).all(), "Constraints must be positive after optimization step"
        violated_bounds = phi.lt(bounds[0]).logical_or(phi.gt(bounds[1])).sum().float().item()
        step_norm = torch.norm(phi - old_phi, p=2)
        distance_from_origin = torch.norm(phi - initial_phi, p=2)

        lr = optimizer.trust_radius
        wb.log({'objective_loss': obj_loss.item(), 
                'constraints': contraints, 
                'violated_bounds': violated_bounds, 
                'step_norm' : step_norm.item(), 
                'distance_from_origin' : distance_from_origin.item(), 
                'rho': optimizer.rhos[-1], 'trust_radius': lr,
                'predicted_loss': predicted_loss,})

        optimizer._i += 1
        print("Difference between current_phi and initial_phi:", (optimizer.current_phi - initial_phi).detach().cpu().numpy())
        # For generative model training loss, use the surrogate's fit loss if available, else dummy
        train_losses.extend(optimizer.model.last_train_loss)

        objective_losses.append(obj_loss.item())
        print(f"Iter {i+1}: Objective loss = {obj_loss}")
        with open(os.path.join(optimizer.outputs_dir,f'history.pkl'), "wb") as f:
            pickle.dump(optimizer.history, f)
        old_phi = phi.clone().detach()

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


