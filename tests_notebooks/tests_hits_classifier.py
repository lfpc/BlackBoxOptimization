import sys
import os
PROJECTS_DIR = '/home/hep/lprate/projects'   
os.environ['PROJECTS_DIR'] = PROJECTS_DIR
sys.path.append('/home/hep/lprate/projects/BlackBoxOptimization/src')
from problems import ShipMuonShield, ShipMuonShieldCluster
import torch
from sklearn.calibration import calibration_curve
from tqdm import tqdm
import argparse
from torch.utils.data import DataLoader, TensorDataset
import time
import numpy as np
import matplotlib.pyplot as plt
import pickle
import gzip
dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_float32_matmul_precision('high')
def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
MUONS_FILE = '/home/hep/lprate/projects/MuonsAndMatter/data/full_sample/full_sample_0.pkl'
from scipy.stats.qmc import LatinHypercube


def plot_hist_errors(hits, predictions, n_bins=15):
    hits_sum = hits.sum(axis = 1)
    expected_hits = predictions.sum(axis = 1)
    #print(f"Mean of Hits: {hits_sum.mean().item()}, Std of Hits: {hits_sum.std().item()}")
    #print(f"Mean of Expected Hits: {expected_hits.mean().item()}, Std of Expected Hits: {expected_hits.std().item()}")
    plt.bar(range(len(hits_sum)), hits_sum.detach().cpu().flatten().numpy(), alpha=0.5, label='Hits')
    plt.bar(range(len(expected_hits)), expected_hits.detach().cpu().flatten().numpy(), alpha=0.5, label='Expected Hits')
    plt.legend()
    error = 100*((expected_hits - hits_sum)/hits_sum).detach().cpu().flatten().numpy()
    #plt.bar(range(len(error)),error)
    plt.xlabel('Sample Index')
    plt.ylabel('Percentage Error')
    plt.title(f'Error List Bar Plot {np.sum(error).item()}')
    plt.grid()
    plt.savefig('errors_per_phi.png')
    plt.close()
    

def plot_calibration_curve(hits_train, predictions_train, hits_test, predictions_test, n_bins=15, name='calibration_plot.png'):
    hits_train = hits_train.detach().cpu().numpy().flatten()
    predictions_train = predictions_train.detach().cpu().numpy().flatten()
    hits_test = hits_test.detach().cpu().numpy().flatten()
    predictions_test = predictions_test.detach().cpu().numpy().flatten()
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    prob_true_test, prob_pred_test = calibration_curve(hits_test, predictions_test, n_bins=n_bins)
    axes[0, 0].plot(prob_pred_test, prob_true_test, marker='o', label='Classifier')
    axes[0, 0].plot([0, 1], [0, 1], linestyle='--', label='Perfectly calibrated')
    axes[0, 0].set_xlabel('Mean predicted probability')
    axes[0, 0].set_ylabel('Fraction of positives')
    axes[0, 0].set_title('Test Set Calibration curve')
    axes[0, 0].legend()
    axes[0, 0].grid()

    # Test set histograms
    axes[0, 1].hist(hits_test, n_bins, alpha=0.5, label='Hits', histtype='step', linewidth=2, density=False)
    axes[0, 1].hist(predictions_test, n_bins, alpha=0.5, label='Predictions', histtype='step', linewidth=2, density=False)
    axes[0, 1].set_xlabel('Value')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Test Set Histograms of Hits and Predictions')
    axes[0, 1].legend()
    axes[0, 1].grid()

    # Train set calibration curve
    prob_true_train, prob_pred_train = calibration_curve(hits_train, predictions_train, n_bins=n_bins)
    axes[1, 0].plot(prob_pred_train, prob_true_train, marker='o', label='Classifier')
    axes[1, 0].plot([0, 1], [0, 1], linestyle='--', label='Perfectly calibrated')
    axes[1, 0].set_xlabel('Mean predicted probability')
    axes[1, 0].set_ylabel('Fraction of positives')
    axes[1, 0].set_title('Train Set Calibration curve')
    axes[1, 0].legend()
    axes[1, 0].grid()

    # Train set histograms
    axes[1, 1].hist(hits_train, bins=n_bins, alpha=0.5, label='Hits', histtype='step', linewidth=2, density=False)
    axes[1, 1].hist(predictions_train, bins=n_bins, alpha=0.5, label='Predictions', histtype='step', linewidth=2, density=False)
    axes[1, 1].set_xlabel('Value')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('Train Set Histograms of Hits and Predictions')
    axes[1, 1].legend()
    axes[1, 1].grid()

    fig.tight_layout()
    plt.savefig(name)
    plt.close()

class Classifier(torch.nn.Module):
    def __init__(self, phi_dim,x_dim = 8, hidden_dim=256):
        super().__init__()
        self.fc1 = torch.nn.Linear(x_dim + phi_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        #self.fc3 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = torch.nn.Linear(hidden_dim, 1)
        self.activation = torch.nn.ReLU()

    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        #x = self.activation(self.fc3(x))
        x = self.fc4(x)
        return x
    
class HitsClassifier():
    def __init__(self,
                 device = 'cpu',
                 **classifier_kargs) -> None:
        self.model = torch.compile(Classifier(**classifier_kargs).to(device))
        self.loss_fn = torch.nn.BCEWithLogitsLoss()
    def fit(self, inputs,y, n_epochs: int = 10, batch_size: int = 2048):
        self.model.train()
        losses = []
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)#torch.optim.SGD(self.model.parameters(), lr=0.01)#
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
        for e in tqdm(range(n_epochs), desc="Training Progress"):
            epoch_losses = []
            permutation = torch.randperm(inputs.size(0))
            for i in range(0, inputs.size(0), batch_size):
                indices = permutation[i:i + batch_size]
                batch_inputs, batch_y = inputs[indices].to(dev), y[indices].to(dev)
                optimizer.zero_grad()
                p_hits = self.model(batch_inputs)
                loss = self.loss_fn(p_hits, batch_y)
                loss.backward()
                optimizer.step()
                epoch_losses.append(loss.item())
            scheduler.step()
            losses.append(sum(epoch_losses) / len(epoch_losses))
        return losses
    def get_predictions(self, inputs, batch_size=200000):
        self.model.eval()
        predictions = []
        for i in range(0, inputs.size(0), batch_size):
            batch_inputs = inputs[i:i + batch_size].to(dev)
            batch_predictions = self.model(batch_inputs).sigmoid().cpu()
            predictions.append(batch_predictions)
        return torch.cat(predictions, dim=0)
    def __call__(self,inputs, return_unc = False):
        predictions = self.get_predictions(inputs)
        return predictions  

def sample_phi_uniform(bounds, num_samples=1):
    if num_samples<=0: return torch.tensor([])
    lower = bounds[0]
    upper = bounds[1]
    return lower + (upper - lower) * torch.rand((num_samples, 54), device=bounds.device)
def sample_phi_lhs(phi,epsilon:float=0.2, num_samples:int=84):
    lhs_sampler = LatinHypercube(d=phi.size(-1))
    sample = (2*lhs_sampler.random(n=num_samples)-1)*epsilon
    sample = torch.as_tensor(sample,device=phi.device,dtype=phi.dtype)
    return sample+phi

def get_data(size,n_samples, cluster:bool = False, local_sample:bool = False):
    if cluster: problem = ShipMuonShieldCluster(cores=512, fSC_mag=False, dimensions_phi=54,simulate_fields=False, 
                                                apply_det_loss=False,n_samples = n_samples,muons_file = MUONS_FILE, 
                                                return_files = "/home/hep/lprate/projects/MuonsAndMatter/data/outputs/results")
    else:
        problem = ShipMuonShield(cores=45, fSC_mag=False, dimensions_phi=54,
                             sensitive_plane=82, simulate_fields=False, apply_det_loss=False,
                             muons_file = MUONS_FILE,n_samples = n_samples)
    hits = []
    muons = []
    load_data = os.path.exists('data.pt')
    size_load = 0
    if load_data:
        data = torch.load('data.pt')
        size_load = data['phi'].shape[0]
        for h,m in zip(data['hits'],data['muons']):
            hits.append(h.view(-1, 1))
            muons.append(m)
    if local_sample: 
        initial_phi = problem.DEFAULT_PHI
        phi = sample_phi_lhs(initial_phi)
        
    else: phi = sample_phi_uniform(problem.GetBounds(), size - size_load) 
    for p in phi:
        if cluster: 
            m = problem.sample_x(p, slice = False)
            muons_file = os.path.join(PROJECTS_DIR,'cluster/muons.pkl')
            with gzip.open(muons_file,'wb') as f: pickle.dump(m,f)
            muons.append(torch.as_tensor(m, dtype=p.dtype).reshape(n_samples,-1))
        else:
            m = problem.sample_x(p)
            muons.append(torch.as_tensor(m, dtype=p.dtype).reshape(n_samples,-1))
        if cluster: hits.append(problem.simulate(p, None, reduction = None).float().view(-1, 1))
        else: hits.append(problem.simulate(p, m, return_nan=True).T.eq(0).all(-1).logical_not().float().view(-1, 1))
        assert hits[-1].shape[0] == n_samples, hits[-1].shape
        
    hits = torch.stack(hits).float()
    muons = torch.stack(muons)
    if size_load>0: phi = torch.cat([data['phi'].to(torch.get_default_dtype()), phi])
    if size-size_load>0: torch.save({'phi': phi, 'muons': muons, 'hits': hits}, 'data.pt')
    return phi[:size], hits[:size], muons[:size,:,:3]

def get_subsample(muons,hits = None, size:int = 100000):
    idx = torch.randperm(muons.shape[0])[:size]
    if hits is None: return muons[idx]
    else: return muons[idx], hits[idx]
def print_metrics(predictions, hits):
    hits_sum = hits.sum().item()
    expected_hits = predictions.sum().item()
    error = expected_hits - hits_sum

    accuracy = ((predictions > 0.5) == hits).float().mean().item() * 100
    perc_error = (error / hits_sum) * 100
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"Hits: {hits_sum}")
    print(f"Expected number of hits: {expected_hits}")
    print(f"Error: {error}, Percentage Error: {perc_error:.2f}%")

if __name__ == '__main__':
    hits = []
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--test_size', type=int, default=1, help='Number of samples to test')
    parser.add_argument('--train_size', type=int, default=199, help='Number of samples to train on')
    parser.add_argument('--n_epochs_train', type=int, default=5, help='Number of epochs to train')
    parser.add_argument('--n_epochs_fn', type=int, default=5, help='Number of epochs to finetune')
    parser.add_argument('--batch_size', type=int, default=1000000, help='Batch size')
    parser.add_argument('--initial_sample_size', type=int, default=1000000, help='Initial muons sample size')
    parser.add_argument('--subsample_size', type=int, default=100000, help='FineTuning subsample size')
    parser.add_argument('--cluster', action='store_true', help='Run simulations with cluster')
    parser.add_argument('--local_sample', action='store_true', help='Local sampling (LGSO)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()
    set_seed(args.seed)
    size = args.train_size + args.test_size
    phi, hits, muons = get_data(size, args.initial_sample_size, cluster = args.cluster, local_sample = args.local_sample)
    muons_train, muons_test = muons[:args.train_size].view(-1,muons.shape[-1]), muons[args.train_size:size].view(-1,muons.shape[-1])
    inputs_train = torch.cat((phi[:args.train_size].repeat_interleave(muons.shape[1],dim=0), muons_train), dim=-1)
    inputs_test = torch.cat((phi[args.train_size:size].repeat_interleave(muons.shape[1],dim=0), muons_test), dim=-1)
    hits_train, hits_test = hits[:args.train_size].view(-1,1), hits[args.train_size:size].view(-1,1)
    classifier = HitsClassifier(phi_dim=phi.shape[-1], x_dim=muons.shape[-1], device=dev)
    start_time = time.time()
    losses = classifier.fit(inputs_train, hits_train, n_epochs=args.n_epochs_train, batch_size=args.batch_size)
    end_time = time.time()
    print(f"Total training time: {end_time - start_time} seconds")

    with torch.no_grad():
        predictions_test = classifier.get_predictions(inputs_test)
        predictions_train = classifier.get_predictions(inputs_train)

        test_hits_sum = hits_test.sum().item()
        test_expected_hits = predictions_test.sum().item()
        test_error = test_expected_hits - test_hits_sum

        train_hits_sum = hits_train.sum().item()
        train_expected_hits = predictions_train.sum().item()
        train_error = train_expected_hits - train_hits_sum

        train_accuracy = ((predictions_train > 0.5) == hits_train).float().mean().item() * 100
        test_accuracy = ((predictions_test > 0.5) == hits_test).float().mean().item() * 100
        train_perc_error = (train_error / train_hits_sum) * 100
        test_perc_error = (test_error / test_hits_sum) * 100
        print(f"Train Accuracy: {train_accuracy:.2f}%")
        print(f"Train Hits: {train_hits_sum}")
        print(f"Expected number of hits (Train): {train_expected_hits}")
        print(f"Error: {train_error}, Percentage Error: {train_perc_error:.2f}%")
        print(f"Test Accuracy: {test_accuracy:.2f}%")
        print(f"Test Hits: {test_hits_sum}")
        print(f"Expected number of hits (Test): {test_expected_hits}")
        print(f"Error: {test_error}, Percentage Error: {test_perc_error:.2f}%")
    plot_hist_errors(hits_train.view(args.train_size,-1), predictions_train.view(args.train_size,-1))
    plot_calibration_curve(hits_train, predictions_train, hits_test, predictions_test, name='calibration_plot_raw.png')
    muons_sample, hits_sample = get_subsample(muons_test,hits_test, args.subsample_size)

    inputs_train = torch.cat([inputs_train, torch.cat([phi[args.train_size:size].repeat_interleave(muons_sample.shape[0],dim=0), muons_sample], dim=-1)])
    hits_train = torch.cat([hits_train,hits_sample])
    start_time = time.time()
    losses_fn = classifier.fit(inputs_train, hits_train, n_epochs=args.n_epochs_fn, batch_size=args.batch_size)

    plt.plot(losses, color = 'b', label = 'Train')
    plt.plot(range(args.n_epochs_train-1,args.n_epochs_train+args.n_epochs_fn-1),losses_fn, color = 'r', label = 'FineTune')
    plt.axvline(args.n_epochs_train-1, color='k', linestyle='--', label='Finetune start')
    plt.grid()
    plt.legend()
    plt.savefig('losses.png')  
    plt.close()

    end_time = time.time()
    print(f"Total finetunning time: {end_time - start_time} seconds")
    with torch.no_grad():
        predictions_test = classifier.get_predictions(inputs_test)
        predictions_train = classifier.get_predictions(inputs_train)

        test_hits_sum = hits_test.sum().item()
        test_expected_hits = predictions_test.sum().item()
        test_error = test_expected_hits - test_hits_sum

        train_hits_sum = hits_train.sum().item()
        train_expected_hits = predictions_train.sum().item()
        train_error = train_expected_hits - train_hits_sum

        train_accuracy = ((predictions_train > 0.5) == hits_train).float().mean().item() * 100
        test_accuracy = ((predictions_test > 0.5) == hits_test).float().mean().item() * 100
        train_perc_error = (train_error / train_hits_sum) * 100
        test_perc_error = (test_error / test_hits_sum) * 100
        print(f"Train Accuracy: {train_accuracy:.2f}%")
        print(f"Train Hits: {train_hits_sum}")
        print(f"Expected number of hits (Train): {train_expected_hits}")
        print(f"Error: {train_error}, Percentage Error: {train_perc_error:.2f}%")
        print(f"Test Accuracy: {test_accuracy:.2f}%")
        print(f"Test Hits: {test_hits_sum}")
        print(f"Expected number of hits (Test): {test_expected_hits}")
        print(f"Error: {test_error}, Percentage Error: {test_perc_error:.2f}%")
        
    plot_calibration_curve(hits_train, predictions_train, hits_test, predictions_test, name='calibration_plot_finetunned.png')
    #plot_hist_errors(hits_train.view(size,-1), predictions_train.view(size,-1))





   
 