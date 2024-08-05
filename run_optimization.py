import torch
from botorch import acquisition, settings
from utils.optimizer import BayesianOptimizer,LGSO
from utils import problems
from matplotlib import pyplot as plt
import argparse
from utils.models import GP_RBF, GP_Cylindrical_Custom

from warnings import filterwarnings
filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument("--cpu",dest = 'cuda', action = 'store_false')
parser.add_argument("--seed", type=int, default=13)
parser.add_argument("--maxiter", type=int, default=100)
parser.add_argument("--dimensions", type=int, default=10)
parser.add_argument("--n_initial", type=int, default=-1)
parser.add_argument('--phi_bounds', nargs='+', type=float, default=[-10.,10.])
parser.add_argument('--problem', type=str, default='stochastic_rosenbrock')
parser.add_argument('--optimization', type=str, default='bayesian')
parser.add_argument('--model', type=str, default='gp_rbf')
parser.add_argument('--noise', type=float, default=1.0)
parser.add_argument('--n_samples', type=int, default=1)
parser.add_argument('--torch_optimizer',dest='scipy', action = 'store_false')
parser.add_argument('--float64', action = 'store_true')


args = parser.parse_args()

if torch.cuda.is_available() and args.cuda: dev = torch.device('cuda')
else: dev = torch.device('cpu')
print('Device:', dev)
torch.manual_seed(args.seed);
if args.float64: torch.set_default_dtype(torch.float64)





def main(model,problem_fn,dimensions_phi,max_iter,N_initial_points,phi_range):
    if N_initial_points == -1: initial_phi = 2*torch.ones(1,dimensions_phi,device=dev)
    else: initial_phi = (phi_range[1]-phi_range[0])*torch.rand(N_initial_points,dimensions_phi,device=dev)+phi_range[0]
    if args.optimization == 'bayesian':
        acquisition_fn = acquisition.ExpectedImprovement
        optimizer = BayesianOptimizer(problem_fn,model,phi_range,acquisition_fn=acquisition_fn,initial_phi = initial_phi,device = dev)
    elif args.optimization == 'lgso':
        optimizer = LGSO(problem_fn,model,phi_range,acquisition_fn=acquisition_fn,initial_phi = initial_phi,device = dev)
    return *optimizer.run_optimization(max_iter = max_iter,use_scipy=args.scipy),optimizer

if __name__ == "__main__":

    N_initial_points = args.n_initial
    phi_range = torch.as_tensor(args.phi_bounds,device=dev,dtype=torch.get_default_dtype()).view(2,-1)  
    max_iter = args.maxiter
    dimensions_phi = args.dimensions
    if phi_range.size(1) != dimensions_phi: phi_range = phi_range.repeat(1,dimensions_phi)

    if args.problem == 'stochastic_rosenbrock': problem_fn = problems.stochastic_RosenbrockProblem(n_samples=args.n_samples,std = args.noise)
    if args.problem == 'rosenbrock': problem_fn = problems.RosenbrockProblem(args.noise)
    elif args.problem == 'stochastic_threehump': problem_fn = problems.stochastic_ThreeHump(n_samples=args.n_samples,std = args.noise)
    if args.model == 'gp_rbf': model = GP_RBF(phi_range,dev)
    elif args.model == 'gp_cylindrical': model = GP_Cylindrical_Custom(phi_range,dev)
    phi,y,BayesOpt = main(model,problem_fn,dimensions_phi,max_iter,N_initial_points,phi_range)
    
    print('Optimal phi', phi)
    print('Optimal y', y.item(),'|' ,problem_fn(phi.view(1,-1)).item())
    print(f'Calls to the function: {N_initial_points}(initial set) + {BayesOpt.n_iterations()}')
    print('[1]^D : ', problem_fn(torch.ones(1,dimensions_phi,device=dev)))
    min_loss = torch.cummin(BayesOpt.D[1],dim=0).values
    
    plt.plot(BayesOpt.D[1].cpu().numpy())
    plt.ylabel('Loss')
    plt.xlabel('Iteration')
    plt.savefig('loss.png')
    plt.close()
    plt.plot(min_loss.cpu().numpy())
    plt.ylabel('Loss')
    plt.xlabel('Iteration')
    plt.savefig('min_loss.png')
    plt.close()

    if dimensions_phi == 1:
        with torch.no_grad():
            phi = torch.linspace(*phi_range.flatten(),10000,device=dev).view(-1,1)
            observed_pred = model.posterior(phi)
            phi_sample = BayesOpt.D[0].cpu()
            y_sample = BayesOpt.D[1].cpu()
            y_true = problem_fn(phi).cpu()
            y_pred = observed_pred.mean.cpu()
            std = observed_pred.mvn.covariance_matrix.diag().sqrt().cpu()
            phi = phi.cpu()
            plt.plot(phi,y_true, color = 'green', label = 'True function',linewidth = 3)
            plt.plot(phi,y_pred, color = 'blue', label = 'Predicted function')
            plt.fill_between(phi.flatten(),y_pred.flatten()-std,y_pred.flatten()+std, alpha = 0.25, color = 'blue')
            plt.scatter(phi_sample,y_sample, color = 'black', label = 'Sampled points', marker = 'x')
            true_minimal = (phi[torch.argmin(y_true)].item(),y_true.min())
            plt.axvline(true_minimal[0],color = 'red', linestyle = '--')#,label = f'Minimum: x = {true_minimal[0]:.2f}, f(x) = {true_minimal[1]:.2f}')
            plt.title(f'Minimum: x = {true_minimal[0]:.2f}, f(x) = {true_minimal[1]:.2f}', color = 'red')
            plt.grid()
            plt.legend()
            plt.xlabel('x')
            plt.ylabel('f(x)')
            plt.savefig('GP.png')
            plt.close()
    


    








