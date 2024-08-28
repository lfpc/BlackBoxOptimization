import torch
import numpy as np

def standardize(y:torch.tensor):
    std = y.std() if y.std()>0 else 1
    return (y-y.mean())/std
def split_array_old(array,n_div:int):
    division = int(len(array) / (n_div)) 
    workloads = []
    for i in range(n_div):
        workloads.append(array[i * division:(i + 1) * division, :])
    for j,w in enumerate(array[(i + 1) * division:, :]):    
        workloads[j] = np.append(workloads[j],w.reshape(1,-1),axis=0)
    return workloads

def get_split_indices(num_splits, N):
    '''Get indices that divide array of size N into num_splits splits.
    If R := N%num_splits > 0, the rest is divided into the R new divisions'''
    base_size = N // num_splits
    remainder = N % num_splits
    sizes = [base_size + 1 if i < remainder else base_size for i in range(num_splits)]
    indices = []
    start_idx = 0
    for size in sizes:
        end_idx = start_idx + size
        indices.append((start_idx, end_idx))
        start_idx = end_idx
    return indices

def split_array_idx(phi, 
                    indices = None, 
                    num_splits = None,
                    N_samples = None):
    if indices is None: indices = get_split_indices(num_splits,N_samples)
    splits = []
    for idx in indices:
        splits.append([phi,idx]) #can we only pass phi once?
    return splits

def split_array(arr, K):
    N = len(arr)
    base_size = N // K
    remainder = N % K
    sizes = [base_size + 1 if i < remainder else base_size for i in range(K)]
    splits = np.split(arr, np.cumsum(sizes)[:-1])
    return splits