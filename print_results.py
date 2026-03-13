#Script must be executed inside the container


"""
import pickle
with open("hall_of_fame_3.pkl", "rb") as f:
    hall_of_fame = pickle.load(f)
for individual_index in range(len(hall_of_fame)):
    individual=hall_of_fame[individual_index]
    print(f"Individual: {individual}, fitness value: {individual.fitness}, genes: {individual.genes}")
"""
import os
import argparse
import json
import torch
from src import problems

parser = argparse.ArgumentParser()
parser.add_argument("--cpu",dest = 'cuda', action = 'store_false')
parser.add_argument("--seed", type=int, default=13)
parser.add_argument("--maxiter", type=int, default=1000)
parser.add_argument('--problem', type=str, default='ship_cuda')
parser.add_argument('--optimizer', type=str, default='bayesian')
parser.add_argument('--model', type=str, default='gp_rbf')
parser.add_argument('--name', type=str, default='optimizationtest')
parser.add_argument('--group', type=str, default='BayesianOptimization')
parser.add_argument("--dont_save_history", action='store_false', dest='save_history')
parser.add_argument("--resume", action='store_true')
parser.add_argument("--reduce_bounds", type=int, default=-1)
parser.add_argument("--multi_fidelity", type=int, nargs='?', const=-1, default=None)
parser.add_argument("--parallel", type=int, default = 1)
parser.add_argument("--model_switch", type=int,default = -1)
parser.add_argument('--n_samples', type=int, default=0)
parser.add_argument("--n_initial", type=int, default=-1)
parser.add_argument('--float64', action = 'store_true')
parser.add_argument('--config_file', type=str, default='outputs/config.json')
args = parser.parse_args()

PROJECTS_DIR = os.getenv('PROJECTS_DIR', default = '~')
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
            CONFIG['W0'] = float(20E6)
            CONFIG['L0'] = float(29.8)
            CONFIG['dimensions_phi'] = int(31)#int(28)
            default_phi_name = str('stellatryon_try2')
        else:
            CONFIG['W0'] = float(input("Enter Reference Cost (W0) [default: 11E6]: ") or 11E6)
            CONFIG['L0'] = float(input("Enter Maximum Length (L0) [default: 29.7]: ") or 29.7)
            CONFIG['dimensions_phi'] = int(input("Enter number of dimensions [default: 63]: ") or 63)
            default_phi_name = str(input("Enter name of initial phi [default: see DEFAULT_PHI of Ship class]: ") or '')
        print('default_phi_name', default_phi_name)
        if default_phi_name == '': CONFIG['initial_phi'] = problems.ShipMuonShield.DEFAULT_PHI.tolist()
        else: CONFIG['initial_phi'] = problems.ShipMuonShield.params[default_phi_name]
        if args.multi_fidelity is not None:
            if args.multi_fidelity > 0:
                CONFIG['n_samples'] = args.multi_fidelity
            elif args.multi_fidelity == -1 and CONFIG.get('n_samples', 0) == 0:
                CONFIG['n_samples'] = int(float(input("Enter number of samples for low_fidelity [default: 5E5]: ") or 5E5) )
        json.dump(CONFIG, dst, indent=4)
CONFIG.pop("data_treatment", None)
CONFIG.pop('results_dir', None)
problem_fn = problems.ShipMuonShieldCuda(parallel=args.parallel, **CONFIG)


#genes=[275.2312226113086,69.91090367087457,68.7399805062417,19.999945419746112,1.6699954721239034,280.09548651370534,71.1379857389719,72.89994838259452,19.999803905628237,1.5200166380374687,287.70989335305256,60.629828716171204,59.58986351216028,30.00040917263062,2.2200151374579673,113.8505034182601,31.431268364964456,7.903081833469876,28.607117444203386,11.750051558644099,4.340139428157812,-0.8800001168390332,199.98787123735244,40.0,50.0,50.00050235951396,2.9695089725794013,-1.8699930795220259]
#genes=[264.5375333104282,70.66981291296914,68.7403818256359,20.00026116433447,1.6700415063480543,295.84519885489885,73.53185563662855,77.13484260452282,18.41628697924543,1.648384433809095,287.7103138313564,57.16676929582479,59.636929488122256,29.988251281186038,2.2200803777599702,122.41241566868703,30.00022240617547,7.999945395747138,28.790275867540036,11.749942157738143,4.021386920075127,-0.8800019198328507,195.13063197654273,39.99988900575596,49.999884713478544,49.9999128337096,3.9399315811608457,-1.8699743587480147]
genes=[251.97321193442926,22.977274385888443,10.0,8.0,8.0,1.9744703905496466,282.48085949006844,85.0,16.342180834554675,9.346828381497136,13.053077718538038,1.9082176652454346,227.8023119979162,33.17585834782546,18.83310777265273,12.089575590709108,8.0,7.23603517138795,100.0,6.088016150388447,18.0,22.51443805975549,6.029837706884618,300.0,40.0,43.178955700028055,48.98035646871749,16.07584468419444,31.529229738130155,4.805809934762963,-1.899999976158142]
with open(f"provisional_phi_optm_GA_with_fixed_params.txt", "w") as f:
    for gene in problem_fn.add_fixed_params(torch.tensor(genes, dtype=torch.float32)):
        f.write(f"{gene}\n")
print(hola)

for individual_index in range(len(hall_of_fame)):
    individual=hall_of_fame[individual_index]
    phi=torch.tensor(individual.genes, dtype=torch.float32).unsqueeze(0)
    constraints=problem_fn.get_constraints(phi)
    loss = problem_fn(phi)
    print("hola")
    print(f"Individual: {individual}, fitness value: {individual.fitness}, genes: {individual.genes}")
    print(constraints)
    print(loss)
    print(hola)