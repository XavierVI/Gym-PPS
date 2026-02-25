import gymnasium as gym
from gym_pps.wrappers import PredatorPreySwarmCustomizer
import numpy as np

class Agent:
    def __init__(self, adversary=False):
        self.adversary = adversary


class NJPEnvironment(PredatorPreySwarmCustomizer):
    """Custom environment extending PredatorPreySwarmCustomizer with NJP metrics for multi-agent analysis."""

    def __init__(self, env, args):
        """
        Initialize the NJPEnvironment with agents and environment parameters.
        
        Args:
            env: The base environment to wrap
            args: Configuration arguments for the environment
        """
        super(NJPEnvironment, self).__init__(env, args)
        env = env.unwrapped
        self.num_prey = env.n_e
        self.num_predator = env.n_p
        # Create agent objects: prey agents followed by predator (adversary) agents
        self.agents = [Agent() for _ in range(self.num_prey)] + [Agent(adversary=True) for _ in range(self.num_predator)]
        self.agent_types = ['adversary', 'agent']
        self.action_space=env.action_space
        self.observation_space=env.observation_space
        print('NJP functions added successfully.')

    def dos_and_doa(self, x, h, T, N, D):
        """
        Calculate Degree of Sparsity (DOS) and Degree of Alignment (DOA) over multiple episodes.
        
        This function computes Sparsity and alignment metrics across all agents over T time steps.
        It iterates through each timestep and computes nearest neighbor distances for each agent.
        
        Args:
            x: Position array of shape [dimensions, agents, timesteps] - agent positions
            h: Heading array of shape [dimensions, agents, timesteps] - agent headings/velocities
            T: episode length
            N: is the total number which is equal to n1 for prey
            D: is the environment size defined as the maximum possible distance for two agents.
            
        Returns:
            tuple: (DOS, DOA) where
                - DOS: Degree of Sparsity (normalized sum of nearest neighbor distances in positions)
                - DOA: Degree of Alignment (normalized sum of nearest neighbor distances in headings)
        """
        # Get actual number of agents from data shape
        num_agents = np.shape(x)[1]
        k = [0] * num_agents
        k_h = [0] * num_agents
        distances = []
        distances_h = []
        assert np.shape(x)[1] == np.shape(h)[1]
        # Iterate through all timesteps and agents to compute nearest neighbor distances
        for t in range(np.shape(x)[2]):
            for j in range(np.shape(x)[1]):
                k[j] = self._find_nearest_neighbors_DOS(x[:, :, t], j)
                k_h[j] = self._find_nearest_neighbors_DOA(h[:, :, t], j)
                distances.append(k[j])
                distances_h.append(k_h[j])

        # Normalize by total number of samples (T*N*D for DOS, 2*T*N for DOA)
        DOS = np.sum(distances) / (T * N * D)
        DOA = np.sum(distances_h) / (2 * T * N)
        return DOS, DOA
        

    def periodic_dos_and_doa(self, x, h, T, N, D):
        """
        x: [dims, agents, timesteps]
        h: [dims, agents, timesteps]
        L: box size (assumed 2.0 based on paper)
        """
        L = 2.0
        total_dist = 0
        total_align = 0

        for t in range(T):
            pos = x[:, :, t]  # [2, N]
            head = h[:, :, t]  # [2, N]

            for j in range(N):
                # 1. Compute distances to all other agents using Minimum Image Convention
                # Broadcasting: [2, 1] - [2, N] -> [2, N]
                diff = pos[:, j:j+1] - pos

                # Minimum Image Convention for Periodic Boundaries
                diff = diff - L * np.round(diff / L)

                dist_sq = np.sum(diff**2, axis=0)
                dist_sq[j] = np.inf  # Ignore self

                # 2. Find index k of the spatially nearest neighbor
                k = np.argmin(dist_sq)
                nearest_dist = np.sqrt(dist_sq[k])

                # 3. Calculate metrics using the SAME neighbor k
                total_dist += nearest_dist
                total_align += np.linalg.norm(head[:, j] + head[:, k])

        # Normalization according to Eq (2) and (3)
        DOS = total_dist / (T * N * D)   # Eq (2)
        DOA = total_align / (2 * T * N)  # Eq (3)

        return DOS, DOA

    def dos_and_doa_one_episode(self, x, h, N, D):
        """
        Calculate Degree of Sparsity (DOS) and Degree of Alignment (DOA) for a single episode.
        
        Similar to dos_and_doa() but operates on a single episode's data without timestep iteration.
        
        Args:
            x: Position array of shape [dimensions, agents] - agent positions in this episode
            h: Heading array of shape [dimensions, agents] - agent headings/velocities in this episode
            N: is the total number which is equal to n1 for prey
            D: is the environment size defined as the maximum possible distance for two agents.
            
            
        Returns:
            tuple: (DOS, DOA) where
                - DOS: Degree of Sparsity for this episode
                - DOA: Degree of Alignment for this episode
        """
        # Get actual number of agents from data shape
        num_agents = np.shape(x)[1]
        k = [0] * num_agents
        k_h = [0] * num_agents
        distances = []
        distances_h = []
        assert np.shape(x)[1] == np.shape(h)[1]
        # Compute nearest neighbor distances for each agent in the episode
        for i in range(np.shape(x)[1]):
            k[i] = self._find_nearest_neighbors_DOS(x, i)
            k_h[i] = self._find_nearest_neighbors_DOA(h, i)
            distances.append(k[i])
            distances_h.append(k_h[i])

        # Normalize by number of agents and dimensions
        DOS = np.sum(distances) / (N * D)
        DOA = np.sum(distances_h) / (2 * N)
        return DOS, DOA
    
    def _find_nearest_neighbors_DOS(self, x, i):
        """
        Find the nearest neighbor distance for Degree of Sparsity (position-based metric).
        
        Computes the Euclidean distance from agent i to all other agents and returns the minimum.
        This metric measures how close agents are to their nearest neighbor in position space.
        
        Args:
            x: Position array of shape [dimensions, agents]
            i: Index of the agent for which to find nearest neighbor
            
        Returns:
            float: Minimum distance from agent i to any other agent
        """
        distances = []
        # Calculate distances from agent i to all other agents
        for j in range(np.shape(x)[1]):
            if j != i:
                distances.append(np.linalg.norm(x[:, i] - x[:, j]))

        return np.min(distances)
    
    def _find_nearest_neighbors_DOA(self, x, i):
        """
        Find the nearest neighbor distance for Degree of Alignment (heading-based metric).
        
        Computes a metric based on the sum of agent i's heading/velocity and other agents' headings,
        and returns the minimum across all other agents. This measures heading alignment between agents.
        
        Args:
            x: Heading/velocity array of shape [dimensions, agents]
            i: Index of the agent for which to find alignment metric
            
        Returns:
            float: Minimum alignment-based distance from agent i to any other agent
        """
        distances = []
        # Calculate alignment distances (based on heading/velocity sums) from agent i to all other agents
        for j in range(np.shape(x)[1]):
            if j != i:
                distances.append(np.linalg.norm(x[:, i] + x[:, j]))

        return np.min(distances)
