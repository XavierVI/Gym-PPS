import numpy as np
from typing import Tuple


def make_periodic(x: np.array, L: float) -> np.array:
    x[x > L] -= 2 * L
    x[x < -L] += 2 * L
    return x


def normalize_angle(x: np.array) -> np.array:
    return ((x + np.pi) % (2 * np.pi)) - np.pi


def get_sizes(size_p, size_e, size_o, n_p, n_e, n_o):
    n_peo = n_p + n_e + n_o
    size = np.concatenate(( # concatenate sizes for predators, prey, and obstacles
        np.full(n_p, size_p), # creates an array of size n_p filled with size_p
        np.full(n_e, size_e),
        np.full(n_o, size_o)
    ))
    # stacks duplicates of 1D arrays on top of each other to make a 2D array
    sizes = np.tile(size.reshape(n_peo, 1), (1, n_peo))
    sizes = sizes + sizes.T
    np.fill_diagonal(sizes, 0)
    return size, sizes


def get_mass(m_p, m_e, m_o, n_p, n_e, n_o):
    masses = np.concatenate((
        np.full(n_p, m_p),
        np.full(n_e, m_e),
        np.full(n_o, m_o)
    ))
    return masses


def get_focused(Pos, Vel, norm_threshold, width, remove_self):
    norms = np.sqrt(Pos[0, :]**2 + Pos[1, :]**2)
    sorted_seq = np.argsort(norms)
    Pos = Pos[:, sorted_seq]
    norms = norms[sorted_seq]
    Pos = Pos[:, norms < norm_threshold]
    sorted_seq = sorted_seq[norms < norm_threshold]
    if remove_self == True:
        Pos = Pos[:, 1:]
        sorted_seq = sorted_seq[1:]
    Vel = Vel[:, sorted_seq]
    target_Pos = np.zeros((2, width))
    target_Vel = np.zeros((2, width))
    until_idx = np.min([Pos.shape[1], width])
    target_Pos[:, :until_idx] = Pos[:, :until_idx]
    target_Vel[:, :until_idx] = Vel[:, :until_idx]
    return target_Pos, target_Vel

def get_dist_b2b(p: np.ndarray, L: float, is_periodic: bool, sizes: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate distances between bodies (b2b).
    
    Args:
        p: Position array of shape (2, n_peo)
        L: Domain size
        is_periodic: Whether periodic boundary conditions are used
        sizes: Size matrix of shape (n_peo, n_peo)
        
    Returns:
        Tuple of (d_b2b_center, d_b2b_edge, is_collide_b2b)
    """
    # Use broadcasting to compute pairwise differences
    # Shape: (2, N, 1) - (2, 1, N) -> (2, N, N)
    diff = p[:, :, np.newaxis] - p[:, np.newaxis, :]
    
    if is_periodic:
        # Enforce periodicity
        # In-place modification is safe here as diff is a new array
        diff[diff > L] -= 2 * L
        diff[diff < -L] += 2 * L
        
    # Calculate Euclidean distances between centers
    # Shape: (N, N)
    d_center = np.sqrt(np.sum(diff**2, axis=0))
    
    # Calculate edge distances (penetration depth)
    # Positive values indicate collision (overlap)
    # Negative values indicate separation
    # sizes is matrix of (r_i + r_j)
    d_edge = sizes - d_center
    
    # Check for collisions (penetration > 0)
    is_collide = d_edge > 0
    
    # Ignore self-collisions
    np.fill_diagonal(is_collide, False)
    
    return d_center, d_edge, is_collide


def get_dist_b2w(p: np.ndarray, size: np.ndarray, L: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate distances between bodies and walls (b2w).
    
    Args:
        p: Position array of shape (2, n_peo)
        size: Size array of shape (n_peo,)
        L: Domain size
        
    Returns:
        Tuple of (d_b2w, is_collide_b2w)
    """
    # Calculate penetration depth for each wall
    # Walls at -L (Left/Bottom) and +L (Right/Top)
    
    # Left wall (x = -L): Penetration if x - r < -L => (-L + r) - x > 0
    d_left = (-L + size) - p[0]
    
    # Top wall (y = L): Penetration if y + r > L => y - (L - r) > 0
    d_top = p[1] - (L - size)
    
    # Right wall (x = L): Penetration if x + r > L => x - (L - r) > 0
    d_right = p[0] - (L - size)
    
    # Bottom wall (y = -L): Penetration if y - r < -L => (-L + r) - y > 0
    d_bottom = (-L + size) - p[1]
    
    # Stack results: (4, N)
    d_b2w = np.array([d_left, d_top, d_right, d_bottom])
    
    # Check for collisions (penetration > 0)
    is_collide = d_b2w > 0
    
    return d_b2w, is_collide
