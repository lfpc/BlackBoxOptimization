import torch
from botorch import acquisition, settings
from src.optimizer import BayesianOptimizer,LGSO,GA
from src import problems
from src.models import GP_RBF, GP_Cylindrical_Custom, SingleTaskIBNN
from utils.acquisition_functions import Custom_LogEI
from matplotlib import pyplot as plt
import argparse
import wandb
import os
import json
import sys

PROJECTS_DIR = os.getenv('PROJECTS_DIR', default = '~')

from warnings import filterwarnings
filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument("--cpu",dest = 'cuda', action = 'store_false')
parser.add_argument("--seed", type=int, default=13)
parser.add_argument("--maxiter", type=int, default=2000)#1000
parser.add_argument('--problem', type=str, default='ship_cuda')
parser.add_argument('--optimization', type=str, default='bayesian')
parser.add_argument('--model', type=str, default='gp_rbf')
parser.add_argument('--name', type=str, default='optimizationtest')
parser.add_argument('--group', type=str, default='BayesianOptimization')
parser.add_argument("--dont_save_history", action='store_false', dest='save_history')
parser.add_argument("--resume", action='store_true')
parser.add_argument("--reduce_bounds", type=int, default=1000)
parser.add_argument("--multi_fidelity", type=int, nargs='?', const=-1, default=None)
parser.add_argument("--parallel", type=int, default = 1)
parser.add_argument("--model_switch", type=int,default = -1)
parser.add_argument('--n_samples', type=int, default=0)
parser.add_argument("--n_initial", type=int, default=-1)
parser.add_argument('--float64', action = 'store_true')
parser.add_argument('--config_file', type=str, default='outputs/config.json')

args = parser.parse_args()

OUTPUTS_DIR = os.path.join(PROJECTS_DIR,'BlackBoxOptimization/outputs',args.name)
config_file = os.path.join(OUTPUTS_DIR,'config.json')

run_in_background=True#False

if args.resume:
    with open(config_file, 'r') as f:
        CONFIG = json.load(f)
else: 
    if not os.path.exists(OUTPUTS_DIR):
        os.makedirs(OUTPUTS_DIR)
    with open(args.config_file, 'r') as src, open(config_file, 'w') as dst:
        CONFIG = json.load(src)
        if run_in_background:
            CONFIG['W0'] = float(12E6)
            CONFIG['L0'] = float(29.7)
            CONFIG['dimensions_phi'] = int(63)
            default_phi_name = str('warm_baseline')
        else:
            CONFIG['W0'] = float(input("Enter Reference Cost (W0) [default: 12E6]: ") or 12E6)
            CONFIG['L0'] = float(input("Enter Maximum Length (L0) [default: 29.7]: ") or 29.7)
            CONFIG['dimensions_phi'] = int(input("Enter number of dimensions [default: 63]: ") or 63)
            default_phi_name = str(input("Enter name of initial phi [default: see DEFAULT_PHI of Ship class]: ") or 'warm_baseline')
        print('default_phi_name', default_phi_name)
        if default_phi_name == '': CONFIG['initial_phi'] = problems.ShipMuonShield.DEFAULT_PHI.tolist()
        else: CONFIG['initial_phi'] = problems.ShipMuonShield.params[default_phi_name]
        if args.multi_fidelity is not None:
            if args.multi_fidelity > 0:
                CONFIG['n_samples'] = args.multi_fidelity
            elif args.multi_fidelity == -1 and CONFIG.get('n_samples', 0) == 0:
                CONFIG['n_samples'] = int(float(input("Enter number of samples for low_fidelity [default: 5E5]: ") or 5E5) )
        json.dump(CONFIG, dst, indent=4)
    


wandb.login()
if args.optimization == 'GA':
    GA_dict={}
    GA_dict["population_size"]=20
    GA_dict["generations"]=100#50
    GA_dict["blend_crossover_probability"]=0#0.7
    GA_dict["blend_crossover_alpha"]=0.3
    GA_dict["mutation_probability"]=0.1
    GA_dict["local_search_period"]=8
    GA_dict["num_local_searches"]=20
    GA_dict["local_search_std_deviations_factor"]=0.05
    GA_dict["random_immigration_probability"]=0.01
    GA_dict["mutation_std_deviations_factor"]=0.05
    GA_dict["tournament_size"]=3
    GA_dict["elite_size"]=3
    GA_dict["hall_of_fame_size"]=3
    WANDB = {'project': 'MuonShieldOptimization', 'group': args.optimization, 'config': {**vars(args), **CONFIG, **GA_dict}, 'name': args.name}
else:
    WANDB = {'project': 'MuonShieldOptimization', 'group': args.group, 'config': {**vars(args), **CONFIG}, 'name': args.name}

if args.cuda: assert torch.cuda.is_available()
if torch.cuda.is_available() and args.cuda: dev = torch.device('cuda')
else: dev = torch.device('cpu')
print('Device:', dev)
torch.manual_seed(args.seed)
if args.float64: torch.set_default_dtype(torch.float64)



def main(model,problem_fn,dimensions_phi,max_iter,N_initial_points,phi_bounds, model_scheduler):
    if N_initial_points == -1: 
        initial_phi = problem_fn.initial_phi.to(dev)
    else: 
        initial_phi = (phi_bounds[1]-phi_bounds[0])*torch.rand(N_initial_points,dimensions_phi,device=dev)+phi_bounds[0]

    # Check if initial_phi is within phi_bounds
    within_lower = initial_phi.ge(phi_bounds[0])
    within_upper = initial_phi.le(phi_bounds[1])
    within_bounds = within_lower.logical_and(within_upper)
    if not within_bounds.all():
        violating_indices = (~within_bounds).nonzero(as_tuple=True)
        raise ValueError(f"Some initial_phi values are out of bounds at indices: {violating_indices}\nViolating values: {initial_phi[violating_indices]}")
    
    
    if args.optimization == 'bayesian':
        acquisition_fn = Custom_LogEI#acquisition.qLogExpectedImprovement if args.parallel>1 else acquisition.LogExpectedImprovement
        q = min(args.parallel,problem_fn.cores)
        optimizer = BayesianOptimizer(problem_fn,model,
                                      phi_bounds,
                                      acquisition_fn=acquisition_fn,
                                      initial_phi = initial_phi,
                                      device = dev, 
                                      model_scheduler=model_scheduler,
                                      outputs_dir=OUTPUTS_DIR,
                                      reduce_bounds=args.reduce_bounds,
                                      WandB = WANDB,
                                      acquisition_params = {'q':q,'num_restarts': 30, 'raw_samples':5000},
                                      multi_fidelity= args.multi_fidelity,
                                      resume = args.resume)

    elif args.optimization == 'lgso':
        optimizer = LGSO(problem_fn,
                         model,
                         phi_bounds,
                         epsilon= 0.2,
                         samples_phi = 42,
                         initial_phi = initial_phi,
                         device = dev, 
                         WandB = WANDB,
                         resume=args.resume)

    optimizer.run_optimization(save_optimal_phi=True,save_history=args.save_history,
                               max_iter = max_iter)

    return optimizer


if __name__ == "__main__":
    
    
    config_file = os.path.join(OUTPUTS_DIR,'config.json')
    with open(config_file, 'r') as f:
        CONFIG = json.load(f)
    CONFIG.pop("data_treatment", None)

    #if args.n_tasks is None: n_tasks = args.nodes*args.n_tasks_per_node
    #else: n_tasks = args.n_tasks

    if args.problem == 'stochastic_rosenbrock': problem_fn = problems.stochastic_RosenbrockProblem(n_samples=args.n_samples,std = args.noise)
    elif args.problem == 'rosenbrock': problem_fn = problems.RosenbrockProblem(args.noise)
    elif args.problem == 'stochastic_threehump': problem_fn = problems.stochastic_ThreeHump(n_samples=args.n_samples,std = args.noise)
    elif args.problem == 'ship': 
        problem_fn = problems.ShipMuonShieldCluster(parallel=args.parallel,**CONFIG)
    elif args.problem == 'ship_cuda':
        CONFIG.pop('results_dir', None)
        problem_fn = problems.ShipMuonShieldCuda(parallel=args.parallel, **CONFIG)
    phi_bounds = CONFIG.get('phi_bounds',None) 
    dimensions = CONFIG.get('dimensions_phi')
    if phi_bounds is None: phi_bounds = problem_fn.GetBounds(device=dev); WANDB['config']['phi_bounds'] = phi_bounds
    else:
        phi_bounds = torch.as_tensor(args.phi_bounds,device=dev,dtype=torch.get_default_dtype()).view(2,-1)  
        if phi_bounds.size(1) != dimensions: phi_bounds = phi_bounds.repeat(1,dimensions)

    if args.model == 'gp_rbf': model = GP_RBF(phi_bounds,device = dev)
    elif args.model == 'gp_bock': model = GP_Cylindrical_Custom(phi_bounds,device = dev)
    elif args.model == 'ibnn': model = SingleTaskIBNN(phi_bounds,device = dev)
    model_scheduler = {args.model_switch:SingleTaskIBNN
                       }

    if args.optimization == 'GA':#Genetic Algorithms
        num_gpus = torch.cuda.device_count()
        devices = [torch.device(f'cuda:{i}') for i in range(num_gpus)]
        GA(problem_fn=problem_fn,
            population_size=GA_dict["population_size"],
            generations=GA_dict["generations"],
            phi_bounds=phi_bounds,
            blend_crossover_probability=GA_dict["blend_crossover_probability"],
            blend_crossover_alpha=GA_dict["blend_crossover_alpha"],
            mutation_probability=GA_dict["mutation_probability"],
            local_search_period=GA_dict["local_search_period"],
            num_local_searches=GA_dict["num_local_searches"],
            local_search_std_deviations_factor=GA_dict["local_search_std_deviations_factor"],
            random_immigration_probability=GA_dict["random_immigration_probability"],
            mutation_std_deviations_factor=GA_dict["mutation_std_deviations_factor"],
            tournament_size=GA_dict["tournament_size"],
            elite_size=GA_dict["elite_size"],
            hall_of_fame_size=GA_dict["hall_of_fame_size"],
            device=dev,
            devices=devices,
            WandB=WANDB).run_optimization()
        sys.exit()

    optimizer = main(model,problem_fn,dimensions,args.maxiter,args.n_initial,phi_bounds, model_scheduler)

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







