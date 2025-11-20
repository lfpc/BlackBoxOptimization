import json
import torch
import matplotlib.pyplot as plt
import sys
import os
import pickle
sys.path.append(os.path.abspath(os.path.join('..', 'src')))
from optimizer import AnalyticalOptimizer
from problems import ThreeHump, RosenbrockProblem, HelicalValleyProblem
import torch
import subprocess
import wandb
import argparse
import numpy as np

torch.autograd.set_detect_anomaly(True)
torch.manual_seed(42)
np.random.seed(42)

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




parser = argparse.ArgumentParser()
parser.add_argument('--name', type=str, default=None, help='WandB run name')
parser.add_argument('--second_order', action='store_true', help='Use second order optimization')
parser.add_argument('--n_iters', type=int, default=30, help='Number of optimization iterations')
parser.add_argument('--lr', type=float, default=0.1, help='Initial learning rate for phi optimization')
parser.add_argument('--problem', type=str, choices=['ThreeHump', 'Rosenbrock', 'HelicalValley'], default='ThreeHump', help='Optimization problem to use')
args = parser.parse_args()

dev = get_freest_gpu()
outputs_dir = '/home/hep/lprate/projects/BlackBoxOptimization/outputs/test_lcso'
os.makedirs(outputs_dir, exist_ok=True)
initial_lambda_constraints = 1e-3
initial_lr = args.lr

if args.name is None:
    name = 'SecondOrder_' if args.second_order else 'FirstOrder_'
    name += f'lr{str(initial_lr).replace(".", "_")}'
else:
    name = args.name

if args.problem == 'ThreeHump':
    dim = 2
    problem = ThreeHump(phi_bounds = ((-2.3,-2),(2.3,2)))
    initial_phi = torch.tensor([-1.2, 1.0])
elif args.problem == 'Rosenbrock':
    dim = 20
    problem = RosenbrockProblem(dim = dim, 
                         phi_bounds = ((-2.0, 2.0)))
    initial_phi = torch.tensor([[-1.2, 1.8]*(dim//2)]).flatten()
elif args.problem == 'HelicalValley':
    dim = 3
    problem = HelicalValleyProblem(phi_bounds = ((-10.0, 10.0)))
    initial_phi = torch.tensor([-6, -4.6, 2.1]) #torch.tensor([-9, -1.0, -9.0])




WANDB = {'project': 'AnalyticalGradientsTests_' + args.problem, 'name': name, 'config': {
    'dim': dim,
    'initial_lambda_constraints': initial_lambda_constraints,
    'initial_lr': initial_lr}}

bounds = problem.GetBounds(device=torch.device('cpu'))


optimizer = AnalyticalOptimizer(
    true_model=problem,
    bounds=bounds,
    initial_phi=initial_phi,
    initial_lambda_constraints=initial_lambda_constraints,
    initial_lr=initial_lr,
    device='cpu',
    outputs_dir=outputs_dir,
    resume=False)

train_losses = []

objective_losses = [optimizer.loss(*optimizer.history).item()]
print(f"Initial Objective loss = {objective_losses[0]}")

wandb.login()
with wandb.init(reinit = True,**WANDB) as wb:
    wb.log({'objective_loss': objective_losses[0], 
            'constraints': problem.get_constraints(initial_phi).item(), 
            'violated_bounds': initial_phi.lt(bounds[0]).logical_or(initial_phi.gt(bounds[1])).sum().float().item(),
            'distance_from_origin': 0.0,
            'trust_radius': optimizer.trust_radius,
            'phi_1': initial_phi[0].item(),
            'phi_2': initial_phi[1].item()})
    old_phi = initial_phi.clone().detach()
    for i in range(args.n_iters):
        if i in [20]:  # Reduce learning rate at specific iterations
            optimizer.phi_optimizer.param_groups[0]['lr'] *= 0.2
        if i in [3,6, 9]:
            optimizer.lambda_constraints *= 0.1
        if args.second_order:
            phi, obj_loss, cos_step = optimizer.optimization_iteration_second_order(True)
        else:
            phi, obj_loss = optimizer.optimization_iteration()
            cos_step = -1.0
        print("Optimized phi:", phi)

        with torch.no_grad(): contraints = problem.get_constraints(phi)
        assert torch.all(phi >= bounds[0].to(phi.device)) and torch.all(phi <= bounds[1].to(phi.device)), f"current_phi is out of bounds after optimization step, {phi} not in {bounds}"
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
                #'rho': optimizer.rhos[-1], 
                'trust_radius': lr,
                'cosine_step': cos_step,
                'phi_1': phi[0].item(),
                'phi_2': phi[1].item()})

        optimizer._i += 1

        objective_losses.append(obj_loss.item())
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
plt.savefig('figs/test_lcso.png')

print('Figure saved as figs/test_lcso.png')

print(f"Final Objective loss = {objective_losses[-1]}") 


