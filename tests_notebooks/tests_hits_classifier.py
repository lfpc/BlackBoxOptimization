import torch
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.abspath(os.path.join('..', 'src')))
sys.path.append(os.path.abspath(os.path.join('..')))
#from optimizer import LGSO, LCSO, normalize_vector
#from problems import ShipMuonShield, make_index, ShipMuonShieldCuda
import torch
import pickle
import numpy as np
import time
from tqdm import trange

class BinaryClassifierModel:
    """
    A wrapper class to handle training and prediction for a given binary classification model.
    It is not an nn.Module itself, but it manages one.
    """
    def __init__(self,
                phi_dim: int,
                 x_dim: int,
                 n_epochs: int = 50,
                 batch_size: int = 32,
                 lr: float = 1e-3,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        
        self.device = device

        self.model = torch.nn.Sequential(
            torch.nn.Linear(phi_dim + x_dim, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 1)  # Output layer for binary classification
        ).to(self.device)

        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self._lr = lr
        # Use BCEWithLogitsLoss for better numerical stability. It combines Sigmoid and BCELoss.
        self.loss_fn = torch.nn.BCEWithLogitsLoss() 
        self.last_train_loss = []

    def fit(self, condition, y):
        """
        Trains the classifier model.
        """
        self.model.train() # Set the wrapped model to training mode
        self.last_train_loss = []
        
        # Ensure y is the correct shape [batch_size, 1] and type for the loss function
        y = y.view(-1, 1).float() 

        num_samples = y.size(0)
        indices = torch.arange(num_samples)
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self._lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
        pos_weight = torch.sum(y == 0) / torch.sum(y == 1)
        self.loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(self.device))

        print(f"--- Starting Binary Classifier Training on {self.device} ---")
        pbar = trange(self.n_epochs, position=0, leave=True, desc='Classifier Training')

        for epoch in pbar:
            average_loss = 0.0
            # Shuffle indices for each epoch
            shuffled_indices = indices[torch.randperm(num_samples)]
            
            for start in range(0, num_samples, self.batch_size):
                end = min(start + self.batch_size, num_samples)
                batch_idx = shuffled_indices[start:end]
                
                y_batch = y[batch_idx].to(self.device)
                condition_batch = condition[batch_idx].to(self.device)
                optimizer.zero_grad()
                
                # Get model prediction (logits)
                y_pred_logits = self.model(condition_batch)
                
                # Calculate loss
                loss = self.loss_fn(y_pred_logits, y_batch)
                
                assert not torch.isnan(loss), "Loss is NaN, check your model and data."
                
                loss.backward()
                optimizer.step()
                
                average_loss += loss.item() * (end - start)
            
            average_loss /= num_samples
            self.last_train_loss.append(average_loss)
            scheduler.step()
            
            pbar.set_description(f"Epoch {epoch+1}, Loss: {average_loss:.4f}")
            pbar.set_postfix(loss=f"{average_loss:.4f}", lr=f"{optimizer.param_groups[0]['lr']:.6f}")
            
        print(f"Final training loss: {self.last_train_loss[-1]:.4f}")
        print("--- Binary Classifier Training Finished ---")
        return self

    def predict_proba(self, condition):
        """
        Predicts the probability of Y=1 for each given `phi` and `x`.
        """
        self.model.eval()
        condition = condition.to(self.device)
        y_pred_logits = self.model(condition)
        y_pred_prob = torch.sigmoid(y_pred_logits)
        
        return y_pred_prob

files_dir = "/home/hep/lprate/projects/BlackBoxOptimization/tests_notebooks/"
print("Loading from ", files_dir)
start_time = time.time()
local_info = (torch.from_numpy(np.load(files_dir + "local_info_0.npy")),
              torch.from_numpy(np.load(files_dir + "local_info_1.npy")))
print(f"File loaded in {time.time() - start_time:.4f} seconds.")
print("Data shape: ", local_info[0].shape, local_info[1].shape)
generative_model = BinaryClassifierModel(phi_dim=80,
                            x_dim = 7,
                            n_epochs = 15,
                            batch_size = 2048,
                            lr = 1e-2,
                            device = 'cuda')
generative_model.fit(local_info[0].reshape(-1, 87), local_info[1].reshape(-1, 1))

plt.figure()
plt.plot(generative_model.last_train_loss)
plt.xlabel('Epoch')
plt.ylabel('Training Loss')
plt.title('Surrogate Model Training Loss')
plt.grid(True)
plt.show() 

sur_loss = []
with torch.no_grad():
    for b in local_info[0]:
        batch_loss = generative_model.predict_proba(b).sum().cpu()
        sur_loss.append(batch_loss)

sur_loss = torch.stack(sur_loss)
true_loss = local_info[1].sum(1).flatten()

plt.figure(figsize=(6, 6))
plt.scatter(true_loss, sur_loss, color='blue', label='Number of hits')
plt.plot([true_loss.min().item(), true_loss.max().item()],
         [true_loss.min().item(), true_loss.max().item()],
         'k--', label='y = x')
plt.xlabel('True Loss')
plt.ylabel('Surrogate Loss')
plt.title('Surrogate vs True Loss')
plt.legend()
plt.grid(True)
plt.show()


