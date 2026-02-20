import torch
import torch.nn.functional as F
from gymnasium.spaces import Box, Discrete
from utils.networks import MLPNetwork
from utils.misc import soft_update, average_gradients, onehot_from_logits, gumbel_softmax
from utils.agents import DDPGAgent

MSELoss = torch.nn.MSELoss()

class MADDPG(object):
    """
    Wrapper class for DDPG-esque (i.e. also MADDPG) agents in multi-agent task
    """
    def __init__(self, agent_init_params, alg_types, epsilon, noise,
                 gamma=0.95, tau=0.01, lr_actor=1e-4, lr_critic=1e-3,  hidden_dim=64, 
                 discrete_action=False):
        """
        Initialize the MADDPG multi-agent training wrapper.
        
        This constructor sets up all agents with their respective policies and critics,
        initializes hyperparameters for training, and prepares device tracking for GPU/CPU.

        Args:
            agent_init_params (list of dict): List of parameter dictionaries for each agent, containing:
                num_in_pol (int): Input dimensions to policy (observation space size)
                num_out_pol (int): Output dimensions to policy (action space size)
                num_in_critic (int): Input dimensions to critic (observation + action space size)
            alg_types (list of str): Learning algorithm for each agent ('DDPG' or 'MADDPG')
            epsilon (float): Initial exploration epsilon for action noise
            noise (float): Initial action noise scale
            gamma (float): Discount factor for future rewards (default: 0.95)
            tau (float): Soft update rate for target networks (default: 0.01)
            lr_actor (float): Learning rate for actor/policy networks (default: 1e-4)
            lr_critic (float): Learning rate for critic networks (default: 1e-3)
            hidden_dim (int): Number of hidden dimensions for neural networks (default: 64)
            discrete_action (bool): Whether the action space is discrete (default: False)
        """
        # Store number of agents and algorithm types
        self.nagents = len(alg_types)
        self.alg_types = alg_types
        
        # Store exploration and noise parameters
        self.epsilon = epsilon
        self.noise = noise
        
        # Initialize individual DDPG agents for each actor in the multi-agent system
        # Each agent has its own policy and critic networks
        self.agents = [DDPGAgent(lr_actor=lr_actor, lr_critic=lr_critic, discrete_action=discrete_action,
                                 hidden_dim=hidden_dim, epsilon=self.epsilon, noise=self.noise,
                                 **params)
                       for params in agent_init_params]
        
        # Store initialization parameters for later reconstruction (e.g., loading from checkpoint)
        self.agent_init_params = agent_init_params
        
        # Store training hyperparameters
        self.gamma = gamma
        self.tau = tau
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.discrete_action = discrete_action
        
        # Track device placement for different network components to avoid device mismatch errors
        # PyTorch requires all tensor operations to be on the same device
        self.pol_dev = 'cpu'          # device for active policy networks
        self.critic_dev = 'cpu'       # device for active critic networks
        self.trgt_pol_dev = 'cpu'     # device for target policy networks
        self.trgt_critic_dev = 'cpu'  # device for target critic networks
        
        # Counter for total training iterations (used for logging)
        self.niter = 0

    @property
    def policies(self):
        return [a.policy for a in self.agents]

    
    def target_policies(self, agent_i, obs):
        return self.agents[agent_i].target_policy(obs)

    def scale_noise(self, scale, new_epsilon):
        """
        Scale noise for each agent
        Inputs:
            scale (float): scale of noise
        """
        for a in self.agents:
            a.scale_noise(scale)
            a.epsilon = new_epsilon

    def reset_noise(self):
        for a in self.agents:
            a.reset_noise()

    def step(self, observations, start_stop_num, explore=False):
        """
        Take a step forward in environment with all agents
        Inputs:
            observations: List of observations for each agent
            explore (boolean): Whether or not to add exploration noise
        Outputs:
            actions: List of actions for each agent
        """
        return [self.agents[i].step(observations[:, start_stop_num[i]].t(), explore=explore) for i in range(len(start_stop_num))]
    
    def show_hidden_feature(self, observations, start_stop_num):
        return [self.agents[i].truncated_policy(observations[:, start_stop_num[i]].t()) for i in range(len(start_stop_num))]
    
    def show_action_value(self, obs, acs, start_stop_num):
        actual_value= []
        for i in range(len(start_stop_num)):
            curr_agent = self.agents[i]   
            vf_in = torch.cat((obs[:, start_stop_num[i]].t(), acs[i].t()), dim=1)
            actual_value.append(curr_agent.critic(vf_in))
        return actual_value
    
    def model_predict_kernel_explainer(self, data):
        data_tensor = torch.tensor(data, dtype=torch.float32)
        action = self.agents[0].policy
        model =self.agents[0].critic
        model.eval()
        vf_in = torch.cat((data_tensor, action(data_tensor)), dim=1)
        with torch.no_grad():
            predictions = model(vf_in)
        return predictions.numpy()



    def update(self, obs, acs, rews, next_obs, dones, agent_i, parallel=False, logger=None):
        """
        Update parameters of agent model based on sample from replay buffer
        Inputs:
            sample: tuple of (observations, actions, rewards, next
                    observations, and episode end masks) sampled randomly from
                    the replay buffer. Each is a list with entries
                    corresponding to each agent
            agent_i (int): index of agent to update
            parallel (bool): If true, will average gradients across threads
            logger (SummaryWriter from Tensorboard-Pytorch):
                If passed in, important quantities will be logged
        """
        # obs, acs, rews, next_obs, dones = sample            
        curr_agent = self.agents[agent_i]

        curr_agent.critic_optimizer.zero_grad()
        all_trgt_acs = self.target_policies(agent_i, next_obs)
        trgt_vf_in = torch.cat((next_obs, all_trgt_acs), dim=1)
        target_value = (rews + self.gamma *
                        curr_agent.target_critic(trgt_vf_in) *
                        (1 - dones))
        vf_in = torch.cat((obs, acs), dim=1)
        actual_value = curr_agent.critic(vf_in)
        vf_loss = MSELoss(actual_value, target_value.detach())
        # vf_loss = (actual_value-target_value.detach()) ** 2
        vf_loss.backward()
        if parallel:
            average_gradients(curr_agent.critic)
        # torch.nn.utils.clip_grad_norm(curr_agent.critic.parameters(), 0.5)
        curr_agent.critic_optimizer.step()

        curr_agent.policy_optimizer.zero_grad()

        if not self.discrete_action:
            # Forward pass as if onehot (hard=True) but backprop through a differentiable
            # Gumbel-Softmax sample. The MADDPG paper uses the Gumbel-Softmax trick to backprop
            # through discrete categorical samples, but I'm not sure if for i, pi in zip(range(self.nagents), self.policies):that is
            # correct since it removes the assumption of a deterministic policy for
            # DDPG. Regardless, discrete policies don't seem to learn properly without it.

            curr_pol_out = curr_agent.policy(obs)
            curr_pol_vf_in = curr_pol_out

        all_pol_acs = curr_pol_vf_in
        vf_in = torch.cat((obs, all_pol_acs), dim=1)

        pol_loss = -curr_agent.critic(vf_in).mean()
        pol_loss.backward()
        if parallel:
            average_gradients(curr_agent.policy)
        curr_agent.policy_optimizer.step()
        if logger is not None:
            logger.add_scalars('agent%i/losses' % agent_i,
                               {'vf_loss': vf_loss,
                                'pol_loss': pol_loss},
                               self.niter)

    def update_all_targets(self):
        """
        Update all target networks (called after normal updates have been
        performed for each agent)
        """
        for a in self.agents:
            soft_update(a.target_critic, a.critic, self.tau)   
            soft_update(a.target_policy, a.policy, self.tau)
        self.niter += 1

    def prep_training(self, device='gpu'):
        for a in self.agents:
            a.policy.train()
            a.critic.train()
            a.target_policy.train()
            a.target_critic.train()
        if device == 'gpu' or device == 'cuda':
            fn = lambda x: x.cuda()
        else:
            fn = lambda x: x.cpu()
        if not self.pol_dev == device:
            for a in self.agents:
                a.policy = fn(a.policy)
            self.pol_dev = device
        if not self.critic_dev == device:
            for a in self.agents:
                a.critic = fn(a.critic)
            self.critic_dev = device
        if not self.trgt_pol_dev == device:
            for a in self.agents:
                a.target_policy = fn(a.target_policy)
            self.trgt_pol_dev = device
        if not self.trgt_critic_dev == device:
            for a in self.agents:
                a.target_critic = fn(a.target_critic)
            self.trgt_critic_dev = device

    def prep_rollouts(self, device='cpu'):
        for a in self.agents:
            a.policy.eval()
        if device == 'gpu' or device == 'cuda':
            fn = lambda x: x.cuda()
        else:
            fn = lambda x: x.cpu()
        # only need main policy for rollouts
        if not self.pol_dev == device:
            for a in self.agents:
                a.policy = fn(a.policy)
            self.pol_dev = device

    def save(self, filename):
        """
        Save trained parameters of all agents into one file
        """
        self.prep_training(device='cpu')  # move parameters to CPU before saving
        save_dict = {'init_dict': self.init_dict,
                     'agent_params': [a.get_params() for a in self.agents]}
        torch.save(save_dict, filename)

    @classmethod
    def init_from_env(cls, env, agent_alg="MADDPG", adversary_alg="MADDPG",
                      gamma=0.95, tau=0.01, lr_actor=1e-4, lr_critic=1e-3, hidden_dim=64, epsilon=0.1, noise=0.1):
        """
        Factory method to instantiate MADDPG from a multi-agent environment.
        
        This class method simplifies initialization by automatically extracting environment
        specifications (observation and action space dimensions) and creating the necessary
        agent initialization parameters. This is the recommended way to initialize MADDPG
        for a new training run.
        
        The method distinguishes between two types of agents:
        - Adversarial agents (predators): Use the specified adversary_alg
        - Regular agents (prey): Use the specified agent_alg
        
        Args:
            env: The multi-agent gymnasium environment with 'agent_types' attribute
            agent_alg (str): Algorithm for prey/regular agents ('MADDPG' or 'DDPG')
            adversary_alg (str): Algorithm for predator/adversary agents ('MADDPG' or 'DDPG')
            gamma (float): Discount factor for future rewards
            tau (float): Soft update rate for target networks
            lr_actor (float): Actor/policy learning rate
            lr_critic (float): Critic learning rate
            hidden_dim (int): Hidden dimension size for neural networks
            epsilon (float): Initial exploration epsilon
            noise (float): Initial action noise scale
            
        Returns:
            MADDPG: Initialized MADDPG instance ready for training
        """
        # Extract environment specifications (dimensions of observation and action spaces)
        agent_init_params = []
        num_in_pol = env.observation_space.shape[0]          # Policy input = observation space
        num_out_pol = env.action_space.shape[0]              # Policy output = action space
        num_in_critic = env.observation_space.shape[0] + env.action_space.shape[0]  # Critic input = obs + actions

        # Assign algorithm type to each agent based on agent_type
        # Adversaries use adversary_alg, others use agent_alg
        alg_types = [adversary_alg if atype == 'adversary' else agent_alg
                     for atype in env.agent_types]
        print("=" * 50)
        print(alg_types)
        print("=" * 50)
    
        
        # Create initialization parameters for each agent in the environment
        # All agents share the same network dimensions but may use different algorithms
        for algtype in alg_types:
            agent_init_params.append({'num_in_pol': num_in_pol,
                                      'num_out_pol': num_out_pol,
                                      'num_in_critic': num_in_critic})
        
        # Prepare the initialization dictionary with all hyperparameters
        init_dict = {'gamma': gamma,
                     'tau': tau,
                     'lr_actor': lr_actor,
                     'lr_critic': lr_critic,
                     'epsilon': epsilon,
                     'noise': noise,
                     'hidden_dim': hidden_dim,
                     'alg_types': alg_types,
                     'agent_init_params': agent_init_params}
        
        # Create the MADDPG instance with the prepared parameters
        instance = cls(**init_dict)
        # Store init_dict for later saving/loading
        instance.init_dict = init_dict
        return instance

    @classmethod
    def init_from_save(cls, filename):
        """
        Factory method to instantiate MADDPG from a previously saved checkpoint.
        
        This class method loads a model that was previously saved using the save() method.
        It reconstructs the MADDPG instance with the same architecture and loads all
        trained weights into the networks. This is used for resuming training or
        performing inference with trained models.
        
        Args:
            filename (str): Path to the saved checkpoint file created by save()
            
        Returns:
            MADDPG: Initialized and restored MADDPG instance with trained weights
        """
        # Load the saved checkpoint containing initialization and trained parameters
        save_dict = torch.load(filename)
        
        # Reconstruct the MADDPG instance using the stored initialization dictionary
        # This recreates the network architecture identical to the saved model
        instance = cls(**save_dict['init_dict'])
        instance.init_dict = save_dict['init_dict']
        
        # Restore trained weights for each agent
        for agent, params in zip(instance.agents, save_dict['agent_params']):
            agent.load_params(params)
        
        return instance