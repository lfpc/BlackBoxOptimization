import torch
import numpy as np

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
