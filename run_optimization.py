import torch
from botorch import acquisition, settings
from src.optimizer import BayesianOptimizer,LGSO
from src import problems
from src.models import GP_RBF, GP_Cylindrical_Custom, SingleTaskIBNN, GANModel
from utils.acquisition_functions import Custom_LogEI
from matplotlib import pyplot as plt
import argparse
import wandb
import os

PROJECTS_DIR = os.getenv('PROJECTS_DIR', default = '~')

from warnings import filterwarnings
filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument("--cpu",dest = 'cuda', action = 'store_false')
parser.add_argument("--seed", type=int, default=13)
parser.add_argument("--maxiter", type=int, default=1000)
parser.add_argument("--dimensions", type=int, default=34)
parser.add_argument("--n_initial", type=int, default=-1)
parser.add_argument('--phi_bounds', nargs='+', type=float, default=None)
parser.add_argument('--problem', type=str, default='ship')
parser.add_argument('--optimization', type=str, default='bayesian')
parser.add_argument('--model', type=str, default='gp_rbf')
parser.add_argument('--noise', type=float, default=1.0)
parser.add_argument('--n_samples', type=int, default=0)
parser.add_argument('--torch_optimizer',dest='scipy', action = 'store_false')
parser.add_argument('--float64', action = 'store_true')
parser.add_argument('--name', type=str, default='optimizationtest')
parser.add_argument('--group', type=str, default='BayesianOptimization')
parser.add_argument("--nodes",type=int,default = 16)
parser.add_argument("--n_tasks_per_node", type=int, default=32)
parser.add_argument("--n_tasks", type=int, default=None)
parser.add_argument("--save_history", action='store_true')
parser.add_argument("--model_switch", type=int,default = -1)
parser.add_argument("--resume", action='store_true')
parser.add_argument("--reduce_bounds", type=int, default=-1)
parser.add_argument("--parallel", type=int, default = 1)
parser.add_argument("--simulated_fields", action='store_true')
args = parser.parse_args()

wandb.login()
WANDB = {'project': 'MuonShieldOptimization', 'group': args.group, 'config': vars(args), 'name': args.name}

if args.cuda: assert torch.cuda.is_available()
if torch.cuda.is_available() and args.cuda: dev = torch.device('cuda')
else: dev = torch.device('cpu')
print('Device:', dev)
torch.manual_seed(args.seed);
if args.float64: torch.set_default_dtype(torch.float64)

OUTPUTS_DIR = os.path.join(PROJECTS_DIR,'BlackBoxOptimization/outputs',args.name)
if not os.path.exists(OUTPUTS_DIR):
    os.makedirs(OUTPUTS_DIR)


def main(model,problem_fn,dimensions_phi,max_iter,N_initial_points,phi_range, model_scheduler):
    if N_initial_points == -1: initial_phi = problem_fn.DEFAULT_PHI.to(dev)
    else: initial_phi = (phi_range[1]-phi_range[0])*torch.rand(N_initial_points,dimensions_phi,device=dev)+phi_range[0]
    #assert initial_phi.ge(phi_range[0]).logical_and(initial_phi.le(phi_range[1])).all()

    if args.optimization == 'bayesian':
        acquisition_fn = Custom_LogEI#acquisition.qLogExpectedImprovement if args.parallel>1 else acquisition.LogExpectedImprovement
        q = min(args.parallel,problem_fn.cores)
        optimizer = BayesianOptimizer(problem_fn,model,
                                      phi_range,acquisition_fn=acquisition_fn,
                                      initial_phi = initial_phi,device = dev, 
                                      model_scheduler=model_scheduler,
                                      outputs_dir=OUTPUTS_DIR,
                                      reduce_bounds=args.reduce_bounds,
                                      WandB = WANDB,
                                      acquisition_params = {'q':q,'num_restarts': 30, 'raw_samples':5000},
                                      resume = args.resume)

    elif args.optimization == 'lgso':
        optimizer = LGSO(problem_fn,
                         model,
                         phi_range,
                         epsilon= 0.2,
                         samples_phi = 42,
                         initial_phi = initial_phi,
                         device = dev, 
                         WandB = WANDB,
                         resume=args.resume)

    optimizer.run_optimization(save_optimal_phi=True,save_history=args.save_history,
                               max_iter = max_iter,use_scipy=args.scipy,)

    return optimizer


if __name__ == "__main__":

    dimensions_phi = args.dimensions
    if args.n_tasks is None: n_tasks = args.nodes*args.n_tasks_per_node
    else: n_tasks = args.n_tasks

    if args.problem == 'stochastic_rosenbrock': problem_fn = problems.stochastic_RosenbrockProblem(n_samples=args.n_samples,std = args.noise)
    elif args.problem == 'rosenbrock': problem_fn = problems.RosenbrockProblem(args.noise)
    elif args.problem == 'stochastic_threehump': problem_fn = problems.stochastic_ThreeHump(n_samples=args.n_samples,std = args.noise)
    elif args.problem == 'ship': problem_fn = problems.ShipMuonShieldCluster(cores = n_tasks,seed=args.seed, parallel=args.parallel, dimensions_phi=dimensions_phi,simulate_fields=args.simulated_fields, fSC_mag=True)
    elif args.problem == 'ship_warm': problem_fn = problems.ShipMuonShieldCluster(cores = n_tasks,seed=args.seed, parallel=args.parallel, dimensions_phi=dimensions_phi,simulate_fields=args.simulated_fields, fSC_mag=False)

    if args.phi_bounds is None: phi_range = problem_fn.GetBounds(device=dev); WANDB['config']['phi_bounds'] = phi_range
    #add phi initial here?
    else:
        phi_range = torch.as_tensor(args.phi_bounds,device=dev,dtype=torch.get_default_dtype()).view(2,-1)  
        if phi_range.size(1) != args.dimensions: phi_range = phi_range.repeat(1,args.dimensions)
    if args.model == 'gp_rbf': model = GP_RBF(phi_range,device = dev)
    elif args.model == 'gp_bock': model = GP_Cylindrical_Custom(phi_range,device = dev)
    elif args.model == 'ibnn': model = SingleTaskIBNN(phi_range,device = dev)
    elif args.model == 'gan': model = GANModel(42,484449,64,device = dev)
    model_scheduler = {args.model_switch:SingleTaskIBNN,
                       #args.reduce_bounds:GP_RBF,
                       }

    optimizer = main(model,problem_fn,args.dimensions,args.maxiter,args.n_initial,phi_range, model_scheduler)

    phi,y,idx = optimizer.get_optimal(return_idx=True)
    with open(os.path.join(OUTPUTS_DIR,"phi_optm.txt"), "w") as txt_file:
        for p in phi.flatten():
            txt_file.write(str(p.item()) + "\n")
    print('Optimal phi', phi)
    print('Optimal y', y.item(),f' at iteration {idx}')
    print(f'Calls to the function: {optimizer.n_calls()}')
    min_loss = torch.cummin(optimizer.history[1],dim=0).values
    
    if True:
        plt.plot(min_loss.cpu().numpy())
        plt.ylabel('Optimal Loss')
        plt.xlabel('Iteration')
        plt.savefig(os.path.join(OUTPUTS_DIR,'min_loss.png'))
        plt.close()







