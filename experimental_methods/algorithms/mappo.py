"""
MAPPO (Multi-Agent Proximal Policy Optimization) implementation for multi-agent environments.

MAPPO is a policy gradient algorithm that extends PPO to multi-agent settings with:
- Centralized training, decentralized execution
- Shared value function for training stability
- Local observations (agents only see others within a radius)
- Actor-critic architecture with GAE for advantage estimation
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from utils.networks import MLPNetwork


class PPOAgent:
    """
    Single agent for MAPPO algorithm.
    
    Each agent maintains its own policy but shares a value function during training.
    """

    def __init__(self, num_in_pol, num_out_pol, num_in_critic, hidden_dim=128, 
                 lr_actor=1e-4, lr_critic=1e-3, device='cpu'):
        """
        Initialize a PPO agent.
        
        Args:
            num_in_pol (int): Input dimension for policy network
            num_out_pol (int): Output dimension for policy network (action space)
            num_in_critic (int): Input dimension for value network (typically obs + other agents' obs)
            hidden_dim (int): Hidden dimension for networks
            lr_actor (float): Learning rate for actor (policy) network
            lr_critic (float): Learning rate for critic (value) network
            device (str): 'cpu' or 'cuda'
        """
        self.device = device
        
        # Policy network: outputs mean and log_std for continuous actions
        self.policy = MLPNetwork(
            input_size=num_in_pol,
            output_size=num_out_pol * 2,  # mean and log_std for each action
            hidden_sizes=[hidden_dim, hidden_dim],
            output_activation=None
        ).to(device)
        
        # Value network: estimates state value for advantage computation
        self.value_net = MLPNetwork(
            input_size=num_in_critic,
            output_size=1,
            hidden_sizes=[hidden_dim, hidden_dim],
            output_activation=None
        ).to(device)
        
        # Optimizers
        self.actor_optimizer = optim.Adam(self.policy.parameters(), lr=lr_actor)
        self.critic_optimizer = optim.Adam(self.value_net.parameters(), lr=lr_critic)
        
        # Store dimensions for reference
        self.num_out_pol = num_out_pol
        self.log_std = nn.Parameter(torch.zeros(num_out_pol)).to(device)
    
    def get_action(self, observation, exploration=True):
        """
        Sample an action from the policy.
        
        Args:
            observation: Current observation (tensor)
            exploration (bool): Whether to add exploration noise
            
        Returns:
            action (tensor), log_prob (tensor), value (tensor)
        """
        with torch.no_grad():
            policy_output = self.policy(observation)
            mean = policy_output[:, :self.num_out_pol]
            
            if exploration:
                # Sample from Gaussian policy
                std = torch.exp(self.log_std.expand_as(mean))
                dist = torch.distributions.Normal(mean, std)
                action = dist.sample()
                log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)
            else:
                # Deterministic policy for evaluation
                action = mean
                log_prob = torch.zeros(action.shape[0], 1).to(self.device)
            
            # Get value estimate
            value = self.value_net(observation)
        
        return action, log_prob, value
    
    def evaluate_action(self, observation, action, critic_input):
        """
        Evaluate the log probability and value of an action.
        
        Args:
            observation: Agent's observation (for policy)
            action: Action taken (for log probability)
            critic_input: Input for critic (may include other agents' info)
            
        Returns:
            log_prob (tensor), value (tensor), entropy (tensor)
        """
        policy_output = self.policy(observation)
        mean = policy_output[:, :self.num_out_pol]
        std = torch.exp(self.log_std.expand_as(mean))
        
        dist = torch.distributions.Normal(mean, std)
        log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)
        entropy = dist.entropy().sum(dim=-1, keepdim=True)
        
        value = self.value_net(critic_input)
        
        return log_prob, value, entropy
    
    def get_params(self):
        """Return network parameters for checkpointing."""
        return {
            'policy': self.policy.state_dict(),
            'value_net': self.value_net.state_dict(),
            'log_std': self.log_std.data.cpu()
        }
    
    def load_params(self, params):
        """Load network parameters from checkpoint."""
        self.policy.load_state_dict(params['policy'])
        self.value_net.load_state_dict(params['value_net'])
        self.log_std.data = params['log_std'].to(self.device)


class MAPPO:
    """
    Multi-Agent Proximal Policy Optimization (MAPPO) coordinator.
    
    Centralized training with decentralized execution:
    - Each agent has its own policy that observes other agents only within a radius
    - During training, a centralized value function can see all agent information
    - Actions are shared only within radius during execution
    """

    def __init__(self, agent_init_params, num_agents, observation_radius=None,
                 gamma=0.995, gae_lambda=0.97, clip_ratio=0.2, 
                 lr_actor=1e-4, lr_critic=1e-3, hidden_dim=128, 
                 entropy_coeff=0.01, device='cpu'):
        """
        Initialize MAPPO algorithm.
        
        Args:
            agent_init_params (list of dict): Initialization parameters for each agent
            num_agents (int): Total number of agents in the environment
            observation_radius (float): Radius within which agents observe each other (None = all visible)
            gamma (float): Discount factor
            gae_lambda (float): Lambda for Generalized Advantage Estimation
            clip_ratio (float): PPO clipping ratio
            lr_actor (float): Actor learning rate
            lr_critic (float): Critic learning rate
            hidden_dim (int): Hidden dimension for networks
            entropy_coeff (float): Coefficient for entropy regularization
            device (str): 'cpu' or 'cuda'
        """
        self.nagents = len(agent_init_params)
        self.num_agents = num_agents
        self.observation_radius = observation_radius
        
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_ratio = clip_ratio
        self.entropy_coeff = entropy_coeff
        self.device = device
        
        # Initialize PPO agents
        self.agents = [
            PPOAgent(
                num_in_pol=params['num_in_pol'],
                num_out_pol=params['num_out_pol'],
                num_in_critic=params['num_in_critic'],
                hidden_dim=hidden_dim,
                lr_actor=lr_actor,
                lr_critic=lr_critic,
                device=device
            )
            for params in agent_init_params
        ]
        
        self.agent_init_params = agent_init_params
        self.niter = 0
    
    def filter_observations_by_radius(self, observations, positions, agent_idx, is_adversary):
        """
        Filter observations to only include agents within observation radius.
        
        Args:
            observations: Full observations array [batch, num_agents, obs_dim]
            positions: Agent positions [batch, num_agents, 2] (x, y coordinates)
            agent_idx (int): Index of the agent whose observation to filter
            is_adversary (bool): Whether this is an adversary (predator) agent
            
        Returns:
            filtered_observation: Observation including only nearby agents
        """
        if self.observation_radius is None:
            # No radius restriction - return full observation
            return observations
        
        batch_size = observations.shape[0]
        agent_pos = positions[:, agent_idx:agent_idx+1, :]  # [batch, 1, 2]
        
        # Compute distances to all other agents
        all_positions = positions  # [batch, num_agents, 2]
        distances = torch.norm(agent_pos - all_positions, dim=2)  # [batch, num_agents]
        
        # Create mask for agents within radius (always include self)
        mask = (distances <= self.observation_radius) | (torch.arange(self.num_agents).unsqueeze(0) == agent_idx)
        
        # Filter observations based on mask
        # For agents outside radius, zero out their observation contribution
        filtered_obs = observations.clone()
        for b in range(batch_size):
            filtered_obs[b, ~mask[b]] = 0.0
        
        return filtered_obs
    
    def step(self, observations, positions, agent_slices, explore=True):
        """
        Get actions for all agents given observations.
        
        Args:
            observations: [num_agents, obs_dim] or [batch, num_agents, obs_dim]
            positions: Agent positions for radius filtering
            agent_slices (list): List of slices separating predators and prey
            explore (bool): Whether to explore (sample) or exploit (deterministic)
            
        Returns:
            List of actions for each agent type
        """
        actions = []
        
        for agent_idx, agent in enumerate(self.agents):
            is_adversary = agent_idx < agent_slices[0].stop
            
            # Filter observations by radius
            filtered_obs = self.filter_observations_by_radius(
                observations, positions, agent_idx, is_adversary
            )
            
            # Handle batching if needed
            if filtered_obs.dim() == 1:
                filtered_obs = filtered_obs.unsqueeze(0)
            
            # Get action from policy
            action, _, _ = agent.get_action(filtered_obs, exploration=explore)
            actions.append(action)
        
        return actions
    
    def compute_gae(self, rewards, values, dones, next_values, gamma, gae_lambda):
        """
        Compute Generalized Advantage Estimation (GAE).
        
        Args:
            rewards: Trajectory rewards [trajectory_length]
            values: Value estimates [trajectory_length]
            dones: Episode termination flags [trajectory_length]
            next_values: Value estimate at trajectory end
            gamma: Discount factor
            gae_lambda: GAE lambda parameter
            
        Returns:
            advantages: Computed advantages [trajectory_length]
            returns: Computed returns [trajectory_length]
        """
        advantages = []
        advantage = 0.0
        
        # Compute TD residuals and advantages backward through trajectory
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = next_values
            else:
                next_value = values[t + 1]
            
            # TD residual
            delta = rewards[t] + gamma * next_value * (1 - dones[t]) - values[t]
            
            # Accumulate advantage with decay
            advantage = delta + gamma * gae_lambda * (1 - dones[t]) * advantage
            advantages.insert(0, advantage)
        
        advantages = torch.tensor(advantages, dtype=torch.float32, device=self.device)
        returns = advantages + torch.tensor(values, dtype=torch.float32, device=self.device)
        
        # Normalize advantages for stability
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        return advantages, returns
    
    def update(self, trajectories, num_epochs=5, batch_size=64):
        """
        Update agent policies using PPO.
        
        Args:
            trajectories: Dict mapping agent_idx to lists of (obs, action, reward, next_obs, done, value)
            num_epochs (int): Number of optimization epochs
            batch_size (int): Batch size for updates
        """
        for agent_idx, trajectory in trajectories.items():
            if not trajectory:
                continue
            
            agent = self.agents[agent_idx]
            
            observations = torch.tensor(
                np.array([t[0] for t in trajectory]), 
                dtype=torch.float32, 
                device=self.device
            )
            actions = torch.tensor(
                np.array([t[1] for t in trajectory]), 
                dtype=torch.float32, 
                device=self.device
            )
            rewards = np.array([t[2] for t in trajectory])
            values = np.array([t[5] for t in trajectory])
            dones = np.array([t[4] for t in trajectory])
            
            # Get value estimate at trajectory end
            with torch.no_grad():
                next_obs = torch.tensor(trajectory[-1][3], dtype=torch.float32, device=self.device).unsqueeze(0)
                next_value = agent.value_net(next_obs).item()
            
            # Compute advantages and returns
            advantages, returns = self.compute_gae(
                rewards, values, dones, next_value, self.gamma, self.gae_lambda
            )
            
            # Store old policy log probs
            with torch.no_grad():
                old_log_probs, _, _ = agent.evaluate_action(observations, actions, observations)
            
            # PPO optimization over multiple epochs
            num_batches = max(1, len(trajectory) // batch_size)
            indices = np.arange(len(trajectory))
            
            for epoch in range(num_epochs):
                np.random.shuffle(indices)
                
                for batch_idx in range(num_batches):
                    start_idx = batch_idx * batch_size
                    end_idx = min((batch_idx + 1) * batch_size, len(trajectory))
                    batch_indices = indices[start_idx:end_idx]
                    
                    # Get batch data
                    batch_obs = observations[batch_indices]
                    batch_actions = actions[batch_indices]
                    batch_advantages = advantages[batch_indices]
                    batch_returns = returns[batch_indices]
                    batch_old_log_probs = old_log_probs[batch_indices]
                    
                    # Compute new log probs and values
                    new_log_probs, batch_values, entropy = agent.evaluate_action(
                        batch_obs, batch_actions, batch_obs
                    )
                    
                    # PPO clipping
                    ratio = torch.exp(new_log_probs - batch_old_log_probs)
                    clipped_ratio = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio)
                    actor_loss = -torch.min(
                        ratio * batch_advantages,
                        clipped_ratio * batch_advantages
                    ).mean()
                    
                    # Value function loss
                    critic_loss = 0.5 * ((batch_values.squeeze() - batch_returns) ** 2).mean()
                    
                    # Entropy bonus
                    entropy_loss = -self.entropy_coeff * entropy.mean()
                    
                    # Total loss
                    total_loss = actor_loss + critic_loss + entropy_loss
                    
                    # Optimize
                    agent.actor_optimizer.zero_grad()
                    agent.critic_optimizer.zero_grad()
                    total_loss.backward()
                    agent.actor_optimizer.step()
                    agent.critic_optimizer.step()
        
        self.niter += 1
    
    def prep_rollouts(self):
        """Prepare agents for rollout (set to eval mode)."""
        for agent in self.agents:
            agent.policy.eval()
            agent.value_net.eval()
    
    def prep_training(self):
        """Prepare agents for training (set to train mode)."""
        for agent in self.agents:
            agent.policy.train()
            agent.value_net.train()
    
    def save(self, filename):
        """Save all agent parameters."""
        save_dict = {
            'agent_params': [agent.get_params() for agent in self.agents],
            'init_dict': {
                'agent_init_params': self.agent_init_params,
                'num_agents': self.num_agents,
                'observation_radius': self.observation_radius,
                'gamma': self.gamma,
                'gae_lambda': self.gae_lambda,
                'clip_ratio': self.clip_ratio,
                'entropy_coeff': self.entropy_coeff
            }
        }
        torch.save(save_dict, filename)
    
    @classmethod
    def init_from_env(cls, env, observation_radius=None, gamma=0.995, gae_lambda=0.97, 
                      clip_ratio=0.2, lr_actor=1e-4, lr_critic=1e-3, hidden_dim=128, 
                      entropy_coeff=0.01, device='cpu'):
        """
        Factory method to create MAPPO from environment.
        
        Args:
            env: The multi-agent environment
            observation_radius (float): Radius for local observations
            gamma (float): Discount factor
            gae_lambda (float): GAE lambda
            clip_ratio (float): PPO clip ratio
            lr_actor (float): Actor learning rate
            lr_critic (float): Critic learning rate
            hidden_dim (int): Hidden dimension
            entropy_coeff (float): Entropy coefficient
            device (str): Device to use
            
        Returns:
            MAPPO: Initialized algorithm instance
        """
        # Extract environment specs
        num_in_pol = env.observation_space.shape[0]
        num_out_pol = env.action_space.shape[0]
        # Critic sees all agents' observations
        num_in_critic = num_in_pol
        
        # Get number of agents
        num_agents = env.num_predator + env.num_prey
        
        # Create agent parameters
        agent_init_params = [
            {
                'num_in_pol': num_in_pol,
                'num_out_pol': num_out_pol,
                'num_in_critic': num_in_critic
            }
            for _ in range(len(env.agent_types))
        ]
        
        return cls(
            agent_init_params=agent_init_params,
            num_agents=num_agents,
            observation_radius=observation_radius,
            gamma=gamma,
            gae_lambda=gae_lambda,
            clip_ratio=clip_ratio,
            lr_actor=lr_actor,
            lr_critic=lr_critic,
            hidden_dim=hidden_dim,
            entropy_coeff=entropy_coeff,
            device=device
        )
    
    @classmethod
    def init_from_save(cls, filename, device='cpu'):
        """Load MAPPO from checkpoint."""
        save_dict = torch.load(filename)
        init_dict = save_dict['init_dict']
        
        instance = cls(**init_dict, device=device)
        
        for agent, params in zip(instance.agents, save_dict['agent_params']):
            agent.load_params(params)
        
        return instance
