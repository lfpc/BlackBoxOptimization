import torch
import numpy as np
import h5py

def uniform_sample(shape, bounds, device):
    """
    Samples from a uniform distribution within the given bounds.
    """
    min_val, max_val = bounds
    return (max_val - min_val) * torch.rand(shape, device=device) + min_val

def make_index(row: int, cols):
    """Make a list of (row, col) index pairs for given row and columns."""
    return [(row, c) for c in cols]
def apply_index(params, index):
    """Extract values from params using a list of (row, col) indices or slice(None)."""
    if index == slice(None):
        return params
    are_tensors = torch.is_tensor(params) and torch.is_tensor(index)
    return params[index[:,0], index[:,1]] if are_tensors else [params[r][c] for (r, c) in index]

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

def split_idx_parallel(phi, 
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

def split_array_parallel(phi, arr, K_total):
    """
    Distributes a total of K_total workloads among the elements of phi.
    For each phi, it splits the array 'arr' into a calculated number of chunks.

    Args:
        phi (iterable): An iterable of parameters.
        arr (np.ndarray): The NumPy array to be split for each workload.
        K_total (int): The total number of workloads (e.g., cores) to be created.

    Returns:
        list: A list of workloads. The total length of the list will be K_total.
              Each workload is a list [chunk_of_array, parameter_from_phi].
    """
    num_phi = len(phi)
    if num_phi == 0:
        return []
    if K_total < 0:
        raise ValueError("Total number of workloads (K_total) cannot be negative.")

    # Determine how many workloads (splits) each phi element gets.
    # This distributes the remainder, just like in split_array.
    base_splits_per_phi = K_total // num_phi
    remainder_splits = K_total % num_phi
    splits_for_each_phi = [base_splits_per_phi + 1 if i < remainder_splits else base_splits_per_phi for i in range(num_phi)]

    workloads = []
    # Iterate through each parameter in phi and its assigned number of splits.
    for p, num_splits_for_p in zip(phi, splits_for_each_phi):
        # For each phi, split the *entire* data array into its assigned number of chunks.
        array_chunks = split_array(arr, num_splits_for_p)
        for chunk in array_chunks:
            # Inverted the order to [data_chunk, phi_parameter]
            workloads.append([chunk, p])
            
    return workloads

from scipy.spatial import ConvexHull
def compute_solid_volume_numpy(vertices):
    vertices = np.asarray(vertices)
    """Compute the volume of the convex solid formed by two non-aligned rectangles using ConvexHull."""
    hull = ConvexHull(vertices)
    return torch.as_tensor(hull.volume)


def compute_solid_volume(vertices):
    """
    Compute the volume of a convex hull in a differentiable way using PyTorch.
    This implementation uses the fact that for convex polyhedra, we can triangulate
    from a point inside the polyhedron to each face.
    """
    # Convert vertices to tensor if not already
    if not isinstance(vertices, torch.Tensor):
        vertices = torch.tensor(vertices, dtype=torch.float32)
    
    # Compute centroid of the vertices
    centroid = torch.mean(vertices, dim=0)
    
    # For a convex hull with two non-aligned rectangles, we can triangulate the solid
    # We'll use tetrahedra from the centroid to each triangular face
    
    # Define the faces for the two rectangles
    # Bottom rectangle
    bottom_faces = torch.tensor([
        [0, 1, 2], [0, 2, 3],  # Bottom rectangle
    ])
    
    # Top rectangle
    top_faces = torch.tensor([
        [4, 6, 5], [4, 7, 6],  # Top rectangle
    ])
    
    # Side faces connecting the rectangles
    side_faces = torch.tensor([
        [0, 4, 7], [0, 7, 3],  # Side 1
        [3, 7, 6], [3, 6, 2],  # Side 2 
        [2, 6, 5], [2, 5, 1],  # Side 3
        [1, 5, 4], [1, 4, 0],  # Side 4
    ])
    
    # Combine all faces
    all_faces = torch.cat([bottom_faces, top_faces, side_faces], dim=0)
    
    # Calculate volume by summing tetrahedra volumes
    total_volume = torch.tensor(0.0, requires_grad=True)
    
    for face in all_faces:
        # Get the vertices of the face
        v1 = vertices[face[0]]
        v2 = vertices[face[1]]
        v3 = vertices[face[2]]
        
        # Calculate the volume of the tetrahedron formed by the face and the centroid
        # Volume = 1/6 * |((v2-v1) × (v3-v1)) · (centroid-v1)|
        cross_product = torch.linalg.cross(v2 - v1, v3 - v1)
        volume = torch.abs(torch.dot(cross_product, centroid - v1)) / 6.0
        total_volume = total_volume + volume
    
    return total_volume

def normalize_vector(x:torch.tensor, bounds:tuple):
    """
    Normalize a tensor x to the range defined by bounds.
    :param x: Input tensor to normalize.
    :param bounds: A tuple (min_bound, max_bound) defining the normalization range.
    :return: Normalized tensor.
    """
    min_bound, max_bound = bounds
    return (x - min_bound) / (max_bound - min_bound)

def denormalize_vector(x:torch.tensor, bounds:tuple):
    """
    Denormalize a tensor x from the range defined by bounds.
    :param x: Input tensor to denormalize.
    :param bounds: A tuple (min_bound, max_bound) defining the normalization range.
    :return: Denormalized tensor.
    """
    min_bound, max_bound = bounds
    return x * (max_bound - min_bound) + min_bound

def fn_pen(x): return torch.nn.functional.relu(x,inplace=False).pow(2)

class HDF5Dataset(torch.utils.data.IterableDataset):
    """
    PyTorch IterableDataset for parallel HDF5 loading.
    Designed for num_workers > 0.
    """
    def __init__(self, h5_path, batch_size, shuffle=True, buffer_size_factor=100):
        super().__init__()
        self.h5_path = h5_path
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.buffer_size = batch_size * buffer_size_factor
        
        try:
            with h5py.File(self.h5_path, 'r') as f:
                self.n_samples = f['condition'].shape[0]
        except Exception as e:
            print(f"Error opening HDF5 file {self.h5_path} to read metadata: {e}")
            self.n_samples = 0
        
        print(f"HDF5IterableDataset initialized: {self.n_samples} samples.")

    def __iter__(self):
        """
        Called once per worker to yield batches.
        """
        
        # 1. Get worker info and open file handle
        worker_info = torch.utils.data.get_worker_info()
        try:
            h5_file = h5py.File(self.h5_path, 'r')
            condition_dset = h5_file['condition']
            y_dset = h5_file['y']
        except Exception as e:
            print(f"Worker failed to open HDF5 file {self.h5_path}: {e}")
            return
            
        # 2. Create this worker's indices
        indices = np.arange(self.n_samples)
        # NOTE: We do NOT shuffle the main index list if self.shuffle is True.
        # We shuffle inside the buffer instead.

        # 3. Split the work among workers
        if worker_info is None: # num_workers = 0
            iter_start, iter_end = 0, self.n_samples
        else: # num_workers > 0
            per_worker = int(np.ceil(self.n_samples / float(worker_info.num_workers)))
            worker_id = worker_info.id
            iter_start = worker_id * per_worker
            iter_end = min(iter_start + per_worker, self.n_samples)
        
        worker_indices = indices[iter_start:iter_end]
        
        if not self.shuffle:
            # If no shuffling, just read batches sequentially (original fast-path)
            for i in range(0, len(worker_indices), self.batch_size):
                batch_indices = worker_indices[i : i + self.batch_size]
                if len(batch_indices) == 0: continue
                try:
                    # No sort needed, already sequential
                    condition_batch = condition_dset[batch_indices]
                    y_batch = y_dset[batch_indices]
                    yield torch.from_numpy(condition_batch).float(), torch.from_numpy(y_batch).float()
                except Exception as e:
                    print(f"Worker read error (no-shuffle): {e}.")
                    continue
        else:
            # --- NEW BUFFERED SHUFFLING LOGIC ---
            # 4. Iterate over the worker's indices in large buffer-sized steps
            for i in range(0, len(worker_indices), self.buffer_size):
                
                # 4a. Get indices for the sequential buffer read
                buffer_indices = worker_indices[i : i + self.buffer_size]
                if len(buffer_indices) == 0:
                    continue
                
                try:
                    # 4b. Read the large sequential buffer (FAST)
                    # No np.sort(), these are sequential
                    condition_buffer = condition_dset[buffer_indices]
                    y_buffer = y_dset[buffer_indices]

                    # 4c. Shuffle the buffer in-memory (FAST)
                    rand_indices = np.arange(len(condition_buffer))
                    np.random.shuffle(rand_indices)
                    condition_buffer = condition_buffer[rand_indices]
                    y_buffer = y_buffer[rand_indices]

                    # 4d. Yield batches from the shuffled buffer
                    for j in range(0, len(condition_buffer), self.batch_size):
                        condition_batch = condition_buffer[j : j + self.batch_size]
                        y_batch = y_buffer[j : j + self.batch_size]
                        
                        if len(condition_batch) == 0:
                            continue
                            
                        yield torch.from_numpy(condition_batch).float(), torch.from_numpy(y_batch).float()
                        
                except Exception as e:
                    print(f"Worker read error (shuffle): {e}.")
                    continue
        
        # 5. Clean up
        h5_file.close()