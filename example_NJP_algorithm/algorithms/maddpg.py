import torch
import torch.nn.functional as F
from gym.spaces import Box, Discrete
from utils.networks import MLPNetwork
from utils.misc import soft_update, average_gradients, onehot_from_logits, gumbel_softmax
from utils.agents import DDPGAgent
import numpy as np

MSELoss = torch.nn.MSELoss()


class MADDPG(object):
    """
    Wrapper class for DDPG-esque (i.e. also MADDPG) agents in multi-agent task
    """

    def __init__(
        self,
        agent_init_params,
        alg_types,
        epsilon,
        noise,
        gamma=0.95,
        tau=0.01,
        lr_actor=1e-4,
        lr_critic=1e-3,
        hidden_dim=64,
        discrete_action=False,
    ):
        self.nagents = len(alg_types)
        self.alg_types = alg_types
        self.epsilon = epsilon
        self.noise = noise

        self.agents = [
            DDPGAgent(
                lr_actor=lr_actor,
                lr_critic=lr_critic,
                discrete_action=discrete_action,
                hidden_dim=hidden_dim,
                epsilon=self.epsilon,
                noise=self.noise,
                **params,
            )
            for params in agent_init_params
        ]

        self.agent_init_params = agent_init_params
        self.gamma = gamma
        self.tau = tau
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.discrete_action = discrete_action

        # Track devices (kept for compatibility)
        self.pol_dev = "cpu"
        self.critic_dev = "cpu"
        self.trgt_pol_dev = "cpu"
        self.trgt_critic_dev = "cpu"

        self.niter = 0
        self.init_dict = None  # will be set in init_from_env

    @property
    def policies(self):
        return [a.policy for a in self.agents]

    def target_policies(self, agent_i, obs):
        """
        Compute target actions from target policy, ensuring obs on same device.
        """
        dev = next(self.agents[agent_i].target_policy.parameters()).device
        if isinstance(obs, np.ndarray):
            obs = torch.from_numpy(obs).float()
        obs = obs.to(dev)
        return self.agents[agent_i].target_policy(obs)

    def scale_noise(self, scale, new_epsilon):
        for a in self.agents:
            a.scale_noise(scale)
            a.epsilon = new_epsilon

    def reset_noise(self):
        for a in self.agents:
            a.reset_noise()

    def step(self, observations, start_stop_num, explore=False):
        """
        Take a step forward in environment with all agents
        observations: Tensor shaped like [?, obs_dim, ...] per your code usage
        """
        return [
            self.agents[i].step(observations[:, start_stop_num[i]].t(), explore=explore)
            for i in range(len(start_stop_num))
        ]

    def show_hidden_feature(self, observations, start_stop_num):
        return [
            self.agents[i].truncated_policy(observations[:, start_stop_num[i]].t())
            for i in range(len(start_stop_num))
        ]

    def show_action_value(self, obs, acs, start_stop_num):
        actual_value = []
        for i in range(len(start_stop_num)):
            curr_agent = self.agents[i]
            vf_in = torch.cat((obs[:, start_stop_num[i]].t(), acs[i].t()), dim=1)
            actual_value.append(curr_agent.critic(vf_in))
        return actual_value

    def model_predict_kernel_explainer(self, data):
        data_tensor = torch.tensor(data, dtype=torch.float32)
        action = self.agents[0].policy
        model = self.agents[0].critic
        model.eval()
        vf_in = torch.cat((data_tensor, action(data_tensor)), dim=1)
        with torch.no_grad():
            predictions = model(vf_in)
        return predictions.numpy()

    def update(self, obs, acs, rews, next_obs, dones, agent_i, parallel=False, logger=None):
        """
        Update parameters of agent model based on sample from replay buffer
        """

        curr_agent = self.agents[agent_i]

        # ------------------ CRITICAL FIX: device alignment ------------------
        # Put ALL batch tensors on the SAME device as the agent networks
        dev = next(curr_agent.critic.parameters()).device

        # Convert numpy->tensor and move to dev; if already tensor, just move/cast
        obs = torch.as_tensor(obs, device=dev, dtype=torch.float32)
        next_obs = torch.as_tensor(next_obs, device=dev, dtype=torch.float32)
        acs = torch.as_tensor(acs, device=dev, dtype=torch.float32)
        rews = torch.as_tensor(rews, device=dev, dtype=torch.float32)
        dones = torch.as_tensor(dones, device=dev, dtype=torch.float32)
        # -------------------------------------------------------------------

        # ---------------------- Critic update ----------------------
        curr_agent.critic_optimizer.zero_grad()

        # target actions (already safe due to next_obs on dev; target_policies also guards)
        all_trgt_acs = self.target_policies(agent_i, next_obs)

        # Ensure cat inputs same dev (just in case)
        all_trgt_acs = all_trgt_acs.to(dev)
        trgt_vf_in = torch.cat((next_obs, all_trgt_acs), dim=1)

        target_value = rews + self.gamma * curr_agent.target_critic(trgt_vf_in) * (1 - dones)

        vf_in = torch.cat((obs, acs), dim=1)
        actual_value = curr_agent.critic(vf_in)

        vf_loss = MSELoss(actual_value, target_value.detach())
        vf_loss.backward()

        if parallel:
            average_gradients(curr_agent.critic)

        curr_agent.critic_optimizer.step()

        # ---------------------- Policy update ----------------------
        curr_agent.policy_optimizer.zero_grad()

        if not self.discrete_action:
            curr_pol_out = curr_agent.policy(obs)
            curr_pol_vf_in = curr_pol_out
        else:
            # If you later enable discrete_action, you'll need the gumbel/onehot logic here.
            # For now keep a reasonable default:
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
            logger.add_scalars(
                f"agent{agent_i}/losses",
                {"vf_loss": vf_loss, "pol_loss": pol_loss},
                self.niter,
            )

    def update_all_targets(self):
        for a in self.agents:
            soft_update(a.target_critic, a.critic, self.tau)
            soft_update(a.target_policy, a.policy, self.tau)
        self.niter += 1

    def prep_training(self, device="gpu"):
        """
        Put all networks in train mode and move them to desired device.
        Uses DDPGAgent.to() to keep policy/critic/targets/truncated_policy consistent.
        """
        for a in self.agents:
            a.policy.train()
            a.critic.train()
            a.target_policy.train()
            a.target_critic.train()

        dev = torch.device("cuda:0") if device == "gpu" else torch.device("cpu")

        for a in self.agents:
            a.to(dev)

        # track for compatibility
        self.pol_dev = device
        self.critic_dev = device
        self.trgt_pol_dev = device
        self.trgt_critic_dev = device

    def prep_rollouts(self, device="cpu"):
        """
        Rollouts only need main policy; but we move the whole agent for safety/consistency.
        """
        dev = torch.device("cuda:0") if device == "gpu" else torch.device("cpu")
        for a in self.agents:
            a.policy.eval()
            a.to(dev)

        self.pol_dev = device

    def save(self, filename):
        """
        Save trained parameters of all agents into one file
        """
        self.prep_training(device="cpu")  # move parameters to CPU before saving
        save_dict = {
            "init_dict": self.init_dict,
            "agent_params": [a.get_params() for a in self.agents],
        }
        torch.save(save_dict, filename)

    @classmethod
    def init_from_env(
        cls,
        env,
        agent_alg="MADDPG",
        adversary_alg="MADDPG",
        gamma=0.95,
        tau=0.01,
        lr_actor=1e-4,
        lr_critic=1e-3,
        hidden_dim=64,
        epsilon=0.1,
        noise=0.1,
    ):
        agent_init_params = []
        num_in_pol = env.observation_space.shape[0]
        num_out_pol = env.action_space.shape[0]
        num_in_critic = env.observation_space.shape[0] + env.action_space.shape[0]

        alg_types = [adversary_alg if atype == "adversary" else agent_alg for atype in env.agent_types]

        for algtype in alg_types:
            agent_init_params.append(
                {"num_in_pol": num_in_pol, "num_out_pol": num_out_pol, "num_in_critic": num_in_critic}
            )

        init_dict = {
            "gamma": gamma,
            "tau": tau,
            "lr_actor": lr_actor,
            "lr_critic": lr_critic,
            "epsilon": epsilon,
            "noise": noise,
            "hidden_dim": hidden_dim,
            "alg_types": alg_types,
            "agent_init_params": agent_init_params,
        }

        instance = cls(**init_dict)
        instance.init_dict = init_dict
        return instance

    @classmethod
    def init_from_save(cls, filename):
        save_dict = torch.load(filename)
        instance = cls(**save_dict["init_dict"])
        instance.init_dict = save_dict["init_dict"]
        for a, params in zip(instance.agents, save_dict["agent_params"]):
            a.load_params(params)
        return instance
