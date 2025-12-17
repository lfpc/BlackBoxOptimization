import json
import torch
import matplotlib.pyplot as plt
import sys
import os
import pickle
sys.path.append(os.path.abspath(os.path.join('..', 'src')))
from optimizer import LCSO_sampled
from problems import ThreeHump_stochastic_hits, Rosenbrock_stochastic_hits, HelicalValley_stochastic_hits, ShipMuonShieldCuda
from models import BinaryClassifierModel
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
    mem_free = [int(x) for x in result.stdout.strip().split('\n')]
    max_idx = mem_free.index(max(mem_free))
    return torch.device(f'cuda:{max_idx}')




parser = argparse.ArgumentParser()
parser.add_argument('--name', type=str, default=None, help='WandB run name')
parser.add_argument('--second_order', action='store_true', help='Use second order optimization')
parser.add_argument('--n_epochs', type=int, default=10, help='Number of epochs for classifier training per iteration')
parser.add_argument('--analytical', action='store_true', help='Use analytical gradients')
parser.add_argument('--n_iters', type=int, default=30, help='Number of optimization iterations')
parser.add_argument('--subsamples', type=int, default=100_000, help='Number of subsamples per iteration')
parser.add_argument('--n_samples', type=int, default=20_000_000, help='Number of samples per phi')
parser.add_argument('--lr', type=float, default=1.0, help='Initial learning rate for phi optimization')
parser.add_argument('--problem', type=str, choices=['ThreeHump', 'Rosenbrock', 'HelicalValley', 'MuonShield'], default='Rosenbrock', help='Optimization problem to use')
parser.add_argument('--samples_phi', type=int, default=5, help='Number of samples for phi in each iteration')
parser.add_argument('--epsilon', type=float, default=0.05, help='Trust region radius for phi optimization')
parser.add_argument('--batch_size', type=int, default=4096, help='Batch size for training the surrogate model')
parser.add_argument('--local_file_storage', action='store_true', help='Use local file storage for outputs')
args = parser.parse_args()

dev = get_freest_gpu()
outputs_dir = '/home/hep/lprate/projects/BlackBoxOptimization/outputs/test_lcso'
os.makedirs(outputs_dir, exist_ok=True)
epsilon = args.epsilon
n_samples = args.n_samples
initial_lambda_constraints = 1e-3
initial_lr = args.lr
weight_hits = True

if args.name is None:
    name = 'SecondOrder_' if args.second_order else 'FirstOrder_'
    name += f'lr{str(initial_lr).replace(".", "_")}'
else:
    name = args.name

if args.problem == 'ThreeHump':
    dim = 2
    samples_phi = args.samples_phi  
    x_dim = 2
    problem = ThreeHump_stochastic_hits(n_samples = n_samples, 
                                    phi_bounds = ((-2.3,-2),(2.3,2)), 
                                    x_bounds = (-3,3),
                                    reduction = 'none')
    initial_phi = torch.tensor([-2.1,2.0])#torch.tensor([-1.2, 1.0])
elif args.problem == 'Rosenbrock':
    dim = 20
    samples_phi = args.samples_phi
    x_dim = 2
    problem = Rosenbrock_stochastic_hits(dim = dim, n_samples = n_samples, 
                                    phi_bounds = ((-2.0, 2.0)), 
                                    x_bounds = (-3,3),
                                    reduction = 'none')
    initial_phi = torch.tensor([[-1.2, 1.8]*(dim//2)]).flatten()
elif args.problem == 'HelicalValley':
    dim = 3
    samples_phi = args.samples_phi
    x_dim = 2
    problem = HelicalValley_stochastic_hits(n_samples = n_samples, 
                                    phi_bounds = ((-10.0, 10.0)), 
                                    x_bounds = (-5,5),
                                    reduction = 'none')
    initial_phi = torch.tensor([-9, -1.0, -9.0])

elif args.problem == 'MuonShield':
    dim = 63
    samples_phi = args.samples_phi 
    x_dim = 8
    config_file = "/home/hep/lprate/projects/BlackBoxOptimization/outputs/config.json"
    with open(config_file, 'r') as f:
        CONFIG = json.load(f)
    CONFIG.pop("data_treatment", None)
    CONFIG.pop('results_dir', None)
    CONFIG['dimensions_phi'] = 63
    CONFIG['initial_phi'] = ShipMuonShieldCuda.params['warm_baseline']
    CONFIG['n_samples'] = n_samples
    CONFIG['reduction'] = 'none'
    CONFIG['cost_as_constraint'] = False
    problem = ShipMuonShieldCuda(**CONFIG)
    initial_phi = problem.initial_phi



WANDB = {'project': 'LCSOTests_' + args.problem, 'name': name, 'config': {
    'dim': dim,
    'epsilon': epsilon,
    'samples_phi': samples_phi,
    'n_samples': n_samples,
    'initial_lambda_constraints': initial_lambda_constraints,
    'initial_lr': initial_lr,
    'weight_hits': weight_hits}}

bounds = problem.GetBounds(device=torch.device('cpu'))

classifier = BinaryClassifierModel(phi_dim=dim,
                            x_dim = x_dim,
                            batch_size = args.batch_size,
                            lr = 1e-2,
                            device = dev,
                            activation = 'silu' if args.second_order else 'relu',
                            data_from_file = args.local_file_storage
                            )

optimizer = LCSO_sampled(
    true_model=problem,
    surrogate_model=classifier,
    bounds=bounds,
    samples_phi=samples_phi,
    n_subsamples = args.subsamples,
    n_epochs = args.n_epochs,
    epsilon=epsilon,
    initial_phi=initial_phi,
    initial_lambda_constraints=initial_lambda_constraints,  # Initial lambda for constraints
    initial_lr=initial_lr,  # Initial learning rate or trust_radius for phi optimization
    weight_hits=weight_hits,
    device='cpu',
    outputs_dir=outputs_dir,
    second_order=args.second_order,
    local_file_storage = '/scratch/lprate/local_rosenbrock.h5' if args.local_file_storage else None,
    resume=False)

# Run a short LGSO optimization and record losses
n_iters = args.n_iters
train_losses = []

objective_losses = [optimizer.loss(*optimizer.history).item()*1e5]
print(f"Initial Objective loss = {objective_losses[0]}")

wandb.login()
with wandb.init(reinit = True,**WANDB) as wb:
    #classifier.wandb = wb
    log = {'objective_loss': objective_losses[0], 
            'constraints': problem.get_constraints(initial_phi).item(), 
            'violated_bounds': initial_phi.lt(bounds[0]).logical_or(initial_phi.gt(bounds[1])).sum().float().item(),
            'distance_from_origin': 0.0,
            'trust_radius': optimizer.trust_radius,
            'phi_1': initial_phi[0].item(),
            'phi_2': initial_phi[1].item()}
    if args.problem != 'MuonShield':
        log['real_loss'] = super(type(problem), problem).loss(initial_phi).item()
    wb.log(log)
    old_phi = initial_phi.clone().detach()
    for i in range(n_iters):
        if i > 0: classifier.n_epochs = 6
        #if i in [15, 20, 25]:  # Reduce learning rate at specific iterations
        #    optimizer.phi_optimizer.param_groups[0]['lr'] *= 0.2
        if i in [3,6, 9]:
            optimizer.lambda_constraints *= 0.1
        if args.second_order:
            phi, obj_loss, cos_step = optimizer.optimization_iteration_second_order(True)
        else:
            phi, obj_loss = optimizer.optimization_iteration()
            cos_step = -1.0
        obj_loss *= 1e5
        print("Optimized phi:", phi)
        predicted_loss = optimizer.get_model_pred(phi,normalize = True).item()*1e5

        with torch.no_grad(): contraints = problem.get_constraints(phi)
        assert torch.all(phi >= bounds[0].to(phi.device)) and torch.all(phi <= bounds[1].to(phi.device)), f"current_phi is out of bounds after optimization step, {phi} not in {bounds}"
        #assert contraints.gt(0).all(), "Constraints must be positive after optimization step"
        violated_bounds = phi.lt(bounds[0]).logical_or(phi.gt(bounds[1])).sum().float().item()
        step_norm = torch.norm(phi - old_phi, p=2)
        distance_from_origin = torch.norm(phi - initial_phi, p=2)

        lr = optimizer.trust_radius
        log = {'objective_loss': obj_loss.item(), 
            'constraints': contraints, 
            'violated_bounds': violated_bounds, 
            'step_norm' : step_norm.item(), 
            'distance_from_origin' : distance_from_origin.item(), 
            'trust_radius': lr,
            'phi_1': phi[0].item(),
            'phi_2': phi[1].item(),
            'cosine_step': cos_step,
            'predicted_loss': predicted_loss,}
        if args.problem != 'MuonShield':
            log['real_loss'] = super(type(problem), problem).loss(phi).item()

        wb.log(log)

        optimizer._i += 1
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
plt.savefig('figs/test_lcso.png')

print('Figure saved as figs/test_lcso.png')

print(f"Final Objective loss = {objective_losses[-1]}") 


