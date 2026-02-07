import numpy as np


def make_periodic(x: np.array, L: float) -> np.array:
    x[x > L] -= 2 * L
    x[x < -L] += 2 * L
    return x


def normalize_angle(x: np.array) -> np.array:
    return ((x + np.pi) % (2 * np.pi)) - np.pi


def get_sizes(size_p, size_e, size_o, n_p, n_e, n_o):
    n_peo = n_p + n_e + n_o
    size = np.concatenate((
        np.full(n_p, size_p),
        np.full(n_e, size_e),
        np.full(n_o, size_o)
    ))
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

# TODO: optimize / fix these functions because it's AI generated
def get_dist_b2b(p, L, is_periodic, sizes):
    n = p.shape[1]

    # Expand dimensions for broadcasting
    # p shape: (2, N) -> (2, N, 1) and (2, 1, N)
    p_col = p[:, :, np.newaxis]
    p_row = p[:, np.newaxis, :]

    # Calculate differences
    diff = p_row - p_col  # (2, N, N)

    if is_periodic:
        # diff = make_periodic(diff, L) # This modifies in place, but diff is new array so ok
        # Actually make_periodic expects array, logic:
        diff[diff > L] -= 2 * L
        diff[diff < -L] += 2 * L

    # Calculate distances
    d_center = np.sqrt(np.sum(diff**2, axis=0))  # (N, N)

    # Calculate edge distances
    d_edge = d_center - sizes

    # Check collisions
    # Collision if distance < sum of radii (d_edge < 0)
    # We ignore self-collisions (diagonal)
    is_collide = d_edge < 0
    np.fill_diagonal(is_collide, False)

    return d_center, d_edge, is_collide


def get_dist_b2w(p, size, L):
    """
    Calculate distance to walls.
    Order: 0: Left (-L), 1: Top (+L), 2: Right (+L), 3: Bottom (-L).
    force matrix indices:
    row 0 (x): 1*Left, -1*Right. 
    row 1 (y): -1*Top, 1*Bottom.
    """
    n = p.shape[1]
    # size is (N,) or (1, N) ? pps.py passes self._size
    # In get_sizes: size is passed as flat array?
    # self._size from get_sizes is (N,).

    # 0 Left: x < -L + r  => pen = (-L + r) - x
    d_left = (-L + size) - p[0, :]

    # 1 Top: y > L - r => pen = y - (L - r)
    d_top = p[1, :] - (L - size)

    # 2 Right: x > L - r => pen = x - (L - r)
    d_right = p[0, :] - (L - size)

    # 3 Bottom: y < -L + r => pen = (-L + r) - p[1]
    d_bottom = (-L + size) - p[1, :]

    d_b2w = np.array([d_left, d_top, d_right, d_bottom])  # (4, N)

    is_collide = d_b2w > 0

    # Ensure d_b2w is only positive where colliding (or maybe force calc handles 0?)
    # pps.py: sf_b2w = ... (self.is_collide_b2w * self.d_b2w)
    # So we just return d_b2w as is (penetration depth), masking handles the rest.

    return d_b2w, is_collide
