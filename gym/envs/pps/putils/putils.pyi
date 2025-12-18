import numpy as np
from typing import Tuple

def make_periodic(x: np.ndarray, L: float) -> np.ndarray:
    """Apply periodic boundary conditions to positions."""
    ...

def normalize_angle(x: np.ndarray) -> np.ndarray:
    """Normalize angles to [-π, π] range."""
    ...

def get_sizes(size_p: float, size_e: float, size_o: float, n_p: int, n_e: int, n_o: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate size arrays for all agents.
    
    Returns:
        Tuple of (size, sizes) where:
        - size: 1D array of sizes for each agent
        - sizes: 2D matrix of combined sizes (size[i] + size[j])
    """
    ...

def get_mass(m_p: float, m_e: float, m_o: float, n_p: int, n_e: int, n_o: int) -> np.ndarray:
    """
    Create mass array for all agents.
    
    Returns:
        1D array of masses for each agent
    """
    ...

def get_focused(Pos: np.ndarray, Vel: np.ndarray, norm_threshold: float, width: int, remove_self: bool) -> Tuple[np.ndarray, np.ndarray]:
    """
    Filter and focus on nearest neighbors within threshold.
    
    Args:
        Pos: Position array of shape (2, n)
        Vel: Velocity array of shape (2, n)
        norm_threshold: Maximum distance threshold
        width: Maximum number of neighbors to return
        remove_self: Whether to remove self from results
        
    Returns:
        Tuple of (target_Pos, target_Vel) both of shape (2, width)
    """
    ...

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
    ...

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
    ...

def seeding(seed: int) -> Tuple[np.random.RandomState, int]:
    """
    Seed the random number generator.
    
    Args:
        seed: Seed value
        
    Returns:
        Tuple of (random state, seed)
    """
    ...