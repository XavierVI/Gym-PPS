from torch import Tensor
from torch.autograd import Variable
from torch.optim import Adam
from .networks import MLPNetwork, TruncatedModel
from .misc import hard_update, gumbel_softmax, onehot_from_logits
from .noise import OUNoise, GaussianNoise
import numpy as np


class DDPGAgent(object):
    """
    General class for DDPG agents (policy, critic, target policy, target
    critic, exploration noise)
    """

    def __init__(
        self,
        num_in_pol,
        num_out_pol,
        num_in_critic,
        lr_actor,
        lr_critic,
        hidden_dim=64,
        discrete_action=False,
        epsilon=0.1,
        noise=0.1,
    ):
        """
        Inputs:
            num_in_pol (int): number of dimensions for policy input
            num_out_pol (int): number of dimensions for policy output
            num_in_critic (int): number of dimensions for critic input
        """
        self.policy = MLPNetwork(
            num_in_pol,
            num_out_pol,
            hidden_dim=hidden_dim,
            constrain_out=True,
            discrete_action=discrete_action,
        )
        self.truncated_policy = TruncatedModel(self.policy)

        self.critic = MLPNetwork(
            num_in_critic, 1, hidden_dim=hidden_dim, constrain_out=False
        )

        self.target_policy = MLPNetwork(
            num_in_pol,
            num_out_pol,
            hidden_dim=hidden_dim,
            constrain_out=True,
            discrete_action=discrete_action,
        )

        self.target_critic = MLPNetwork(
            num_in_critic, 1, hidden_dim=hidden_dim, constrain_out=False
        )

        hard_update(self.target_policy, self.policy)
        hard_update(self.target_critic, self.critic)

        self.policy_optimizer = Adam(self.policy.parameters(), lr_actor)
        self.critic_optimizer = Adam(self.critic.parameters(), lr_critic)

        self.epsilon = epsilon
        self.noise = noise

        if not discrete_action:
            self.exploration = GaussianNoise(num_out_pol, noise)
        else:
            self.exploration = 0.3  # epsilon for eps-greedy

        self.discrete_action = discrete_action

    # ---------------- NEW: move all networks to the same device ----------------
    def to(self, device):
        """
        Move all networks (policy/critic/targets) to device.
        This is CRITICAL to avoid cpu/cuda mismatch in rollout & update.
        """
        self.policy.to(device)
        self.truncated_policy.to(device)
        self.critic.to(device)
        self.target_policy.to(device)
        self.target_critic.to(device)
        return self
    # --------------------------------------------------------------------------

    def reset_noise(self):
        if not self.discrete_action:
            self.exploration.reset()

    def scale_noise(self, scale):
        if self.discrete_action:
            self.exploration = scale
        else:
            self.exploration.scale = scale

    def step(self, obs, explore=False):
        """
        Take a step forward in environment for a minibatch of observations
        Inputs:
            obs (PyTorch Tensor): Observations for this agent
            explore (boolean): Whether or not to add exploration noise
        Outputs:
            action (PyTorch Tensor): Actions for this agent
        """
        # Ensure obs and policy are on the same device
        dev = next(self.policy.parameters()).device
        obs = obs.to(dev)

        action = self.policy(obs)

        if self.discrete_action:
            if explore:
                action = gumbel_softmax(action, hard=True)
            else:
                action = onehot_from_logits(action)

        else:  # continuous action
            if explore:
                # IMPORTANT: create noise/random tensors on SAME device as action
                if np.random.rand() < self.epsilon:
                    action = (
                        Tensor(np.random.uniform(-1, 1, size=action.shape))
                        .to(dev)
                        .requires_grad_(False)
                    )
                else:
                    action = action + (
                        Tensor(self.exploration.noise()).to(dev).requires_grad_(False)
                    )
                    action = action.clamp(-1, 1)

        return action.t()

    def get_params(self):
        return {
            "policy": self.policy.state_dict(),
            "critic": self.critic.state_dict(),
            "target_policy": self.target_policy.state_dict(),
            "target_critic": self.target_critic.state_dict(),
            "policy_optimizer": self.policy_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
        }

    def load_params(self, params):
        self.policy.load_state_dict(params["policy"])
        self.critic.load_state_dict(params["critic"])
        self.target_policy.load_state_dict(params["target_policy"])
        self.target_critic.load_state_dict(params["target_critic"])
        self.policy_optimizer.load_state_dict(params["policy_optimizer"])
        self.critic_optimizer.load_state_dict(params["critic_optimizer"])
