import torch
import numpy as np

def standardize(y:torch.tensor):
    std = y.std() if y.std()>0 else 1
    return (y-y.mean())/std

def soft_clamp(x, max_mean = 1.E6):
    return torch.tanh(x/max_mean)*max_mean

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
                    N_samples = None,
                    file = None):
    if indices is None: indices = get_split_indices(num_splits,N_samples)
    phi = phi.view(-1,phi.size(-1))
    splits = []
    for p in phi:
        for idx in indices:
            input = [p,idx]
            if file is not None: input.append(file)
            splits.append(input)
    return splits

def split_array_parallel(phi, 
                    N_samples = -1):
    indices = (0,N_samples)
    splits = []
    for p in phi:
        splits.append([p,indices])
    return splits


def split_array(arr, K):
    N = len(arr)
    base_size = N // K
    remainder = N % K
    sizes = [base_size + 1 if i < remainder else base_size for i in range(K)]
    splits = np.split(arr, np.cumsum(sizes)[:-1])
    return splits