import torch
from botorch import acquisition, settings
from utils.optimizer import BayesianOptimizer,LGSO
from utils import problems
from matplotlib import pyplot as plt
import argparse
from utils.models import GP_RBF, GP_Cylindrical_Custom
import wandb

#from warnings import filterwarnings
#filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument("--cpu",dest = 'cuda', action = 'store_false')
parser.add_argument("--seed", type=int, default=13)
parser.add_argument("--maxiter", type=int, default=100)
parser.add_argument("--dimensions", type=int, default=10)
parser.add_argument("--n_initial", type=int, default=-1)
parser.add_argument('--phi_bounds', nargs='+', type=float, default=None)
parser.add_argument('--problem', type=str, default='stochastic_rosenbrock')
parser.add_argument('--optimization', type=str, default='bayesian')
parser.add_argument('--model', type=str, default='gp_rbf')
parser.add_argument('--noise', type=float, default=1.0)
parser.add_argument('--n_samples', type=int, default=1)
parser.add_argument('--torch_optimizer',dest='scipy', action = 'store_false')
parser.add_argument('--float64', action = 'store_true')

args = parser.parse_args()

wandb.login()
WANDB = {'project': 'MuonShieldOptimization', 'group': 'BayesianOptimization', 'config': vars(args), 'name': 'test1'}
OUTPUTS_FILE = 'outputs'

if torch.cuda.is_available() and args.cuda: dev = torch.device('cuda')
else: dev = torch.device('cpu')
print('Device:', dev)
torch.manual_seed(args.seed);
if args.float64: torch.set_default_dtype(torch.float64)




def main(model,problem_fn,dimensions_phi,max_iter,N_initial_points,phi_range):
    if N_initial_points == -1: initial_phi = problem_fn.DEFAULT_PHI
    else: initial_phi = (phi_range[1]-phi_range[0])*torch.rand(N_initial_points,dimensions_phi,device=dev)+phi_range[0]
    #assert initial_phi.ge(phi_range[0]).logical_and(initial_phi.le(phi_range[1])).all()

    if args.optimization == 'bayesian':
        acquisition_fn = acquisition.ExpectedImprovement
        optimizer = BayesianOptimizer(problem_fn,model,phi_range,acquisition_fn=acquisition_fn,initial_phi = initial_phi,device = dev, WandB = WANDB)

    elif args.optimization == 'lgso':
        optimizer = LGSO(problem_fn,model,phi_range,acquisition_fn=acquisition_fn,initial_phi = initial_phi,device = dev, WandB = WANDB)

    optimizer.run_optimization(max_iter = max_iter,use_scipy=args.scipy)

    return optimizer


if __name__ == "__main__":

    dimensions_phi = args.dimensions

    if args.problem == 'stochastic_rosenbrock': problem_fn = problems.stochastic_RosenbrockProblem(n_samples=args.n_samples,std = args.noise)
    elif args.problem == 'rosenbrock': problem_fn = problems.RosenbrockProblem(args.noise)
    elif args.problem == 'stochastic_threehump': problem_fn = problems.stochastic_ThreeHump(n_samples=args.n_samples,std = args.noise)
    elif args.problem == 'ship': problem_fn = problems.ShipMuonShield()

    if args.phi_bounds is None: phi_range = problem_fn.GetBounds(device=dev); WANDB['config']['phi_bounds'] = phi_range
    else:
        phi_range = torch.as_tensor(args.phi_bounds,device=dev,dtype=torch.get_default_dtype()).view(2,-1)  
        if phi_range.size(1) != args.dimensions: phi_range = phi_range.repeat(1,args.dimensions)

    if args.model == 'gp_rbf': model = GP_RBF(phi_range,dev)
    elif args.model == 'gp_cylindrical': model = GP_Cylindrical_Custom(phi_range,dev)
    optimizer = main(model,problem_fn,args.dimensions,args.maxiter,args.n_initial,phi_range)
    idx = optimizer.D[1].argmin()
    phi,y = optimizer.D[0][idx],optimizer.D[1][idx]
    with open("phi_optm.txt", "w") as txt_file:
        for p in phi.view(-1):
            txt_file.write(" ".join(p.item()) + "\n")
    print('Optimal phi', phi)
    print('Optimal y', y.item(),'|')
    print(f'Calls to the function: {max(args.n_initial,1)}(initial set) + {optimizer.n_iterations()}')
    min_loss = torch.cummin(optimizer.D[1],dim=0).values
    
    plt.plot(optimizer.D[1].cpu().numpy())
    plt.ylabel('Loss')
    plt.xlabel('Iteration')
    plt.savefig('loss.png')
    plt.close()
    plt.plot(min_loss.cpu().numpy())
    plt.ylabel('Loss')
    plt.xlabel('Iteration')
    plt.savefig('min_loss.png')
    plt.close()
    


    








