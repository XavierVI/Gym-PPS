# algorithms/ddpg.py
import torch
from utils.misc import soft_update, average_gradients
from utils.agents import DDPGAgent

MSELoss = torch.nn.MSELoss()


class IDDPG(object):
    """
    Independent DDPG for multi-agent tasks (controller-level in your project):
    each agent has its own actor and critic, critic sees only (obs, act).

    IMPORTANT (to be runnable with your current pipeline):
    - ReplayBuffer.sample() returns obs/next_obs already in the network-consumable format
      (same assumption as your working MADDPG.update()).
    - Therefore, update() does NOT slice obs by obs_slices (those are rollout-time entity slices).
    - Actions may be stored as:
        * [B, act_dim]                (most likely, since MADDPG works)
        * [B, act_dim * nagents]      (concat controllers)
      We handle both safely.
    """

    def __init__(
        self,
        agent_init_params,
        epsilon,
        noise,
        gamma=0.95,
        tau=0.01,
        lr_actor=1e-4,
        lr_critic=1e-3,
        hidden_dim=64,
        discrete_action=False,
    ):
        self.nagents = len(agent_init_params)
        self.epsilon = epsilon
        self.noise = noise

        # NOTE: this is creating two agents as well
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
        print(len(self.agents))
        self.agent_init_params = agent_init_params
        self.gamma = gamma
        self.tau = tau
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.discrete_action = discrete_action

        # Track devices (compat)
        self.pol_dev = "cpu"
        self.critic_dev = "cpu"
        self.trgt_pol_dev = "cpu"
        self.trgt_critic_dev = "cpu"

        self.niter = 0
        self.init_dict = None  # set in init_from_env

    @property
    def policies(self):
        return [a.policy for a in self.agents]

    def scale_noise(self, scale, new_epsilon):
        for a in self.agents:
            a.scale_noise(scale)
            a.epsilon = new_epsilon

    def reset_noise(self):
        for a in self.agents:
            a.reset_noise()

    def step(self, observations, obs_slices, explore=False):
        """
        Rollout-time step (keeps your original behavior):
        observations is shaped like [obs_dim, total_entities] in your main.py.
        obs_slices slice ENTITY columns, then transpose to [n_entities, obs_dim].
        """
        return [
            self.agents[i].step(
                observations[:, obs_slices[i]].t(), explore=explore)
            for i in range(len(obs_slices))
        ]

    @staticmethod
    def _to_dev(x, dev):
        # Convert numpy->tensor if needed and move to device
        return torch.as_tensor(x, device=dev, dtype=torch.float32)

    def _select_action_for_agent(self, acs, agent_i, act_slices=None):
        """
        Make sure we get an action tensor of shape [B, act_dim] for this agent.
        Handles:
          - acs: [B, act_dim] -> return acs
          - acs: [B, act_dim * nagents] -> slice by act_slices[agent_i] if valid
          - otherwise -> safe fallback: take first act_dim
        """
        act_dim = self.agent_init_params[agent_i]["num_out_pol"]

        if acs.dim() != 2:
            raise RuntimeError(
                f"Expected acs to be 2D [B, K], got shape {tuple(acs.shape)}")

        B, K = acs.shape
        if K == act_dim:
            return acs

        # If concatenated and slices provided, use them if in range
        if act_slices is not None and len(act_slices) > agent_i:
            sl = act_slices[agent_i]
            if sl is not None and sl.stop is not None and K >= sl.stop:
                out = acs[:, sl]
                if out.size(1) == act_dim:
                    return out

        # Fallback: take first act_dim (prevents empty slice -> 35 vs 37 crash)
        if K >= act_dim:
            return acs[:, :act_dim]

        raise RuntimeError(
            f"acs has too few dims: K={K} < act_dim={act_dim}. "
            f"acs shape={tuple(acs.shape)}"
        )

    def update(
        self,
        obs,
        acs,
        rews,
        next_obs,
        dones,
        agent_i,
        obs_slices=None,   # kept for API compatibility; intentionally unused
        act_slices=None,   # used only if acs is concatenated
        parallel=False,
        logger=None,
    ):
        """
        Update parameters of one agent based on a batch.

        Expected (same as your working MADDPG.update()):
          obs, next_obs: [B, obs_dim]
          acs:           [B, act_dim] OR [B, act_dim*nagents]
          rews, dones:   [B, 1] OR [B] OR [B, nagents]
        """
        curr_agent = self.agents[agent_i]

        # ----- device alignment (same idea as your MADDPG) -----
        dev = next(curr_agent.critic.parameters()).device
        obs = self._to_dev(obs, dev)
        next_obs = self._to_dev(next_obs, dev)
        acs = self._to_dev(acs, dev)
        rews = self._to_dev(rews, dev)
        dones = self._to_dev(dones, dev)

        # obs must be [B, obs_dim]
        if obs.dim() != 2:
            raise RuntimeError(
                f"Expected obs to be [B, obs_dim], got {tuple(obs.shape)}")
        if next_obs.dim() != 2:
            raise RuntimeError(
                f"Expected next_obs to be [B, obs_dim], got {tuple(next_obs.shape)}")

        # choose action for this agent robustly
        act_i = self._select_action_for_agent(
            acs, agent_i, act_slices=act_slices)

        # choose reward/done for this agent if multi-column
        if rews.dim() == 2 and rews.size(1) > 1:
            rew_i = rews[:, agent_i].unsqueeze(1)
        else:
            rew_i = rews.unsqueeze(1) if rews.dim() == 1 else rews

        if dones.dim() == 2 and dones.size(1) > 1:
            done_i = dones[:, agent_i].unsqueeze(1)
        else:
            done_i = dones.unsqueeze(1) if dones.dim() == 1 else dones

        # ---------------------- Critic update ----------------------
        curr_agent.critic_optimizer.zero_grad()

        # target action from target policy (on same device)
        # NOTE: here we only fetch information for agent(i)
        # so it is more decentralized?
        trgt_act_i = curr_agent.target_policy(next_obs)
        trgt_vf_in = torch.cat((next_obs, trgt_act_i), dim=1)

        target_q = rew_i + self.gamma * \
            curr_agent.target_critic(trgt_vf_in) * (1.0 - done_i)

        vf_in = torch.cat((obs, act_i), dim=1)
        current_q = curr_agent.critic(vf_in)

        vf_loss = MSELoss(current_q, target_q.detach())
        vf_loss.backward()

        if parallel:
            average_gradients(curr_agent.critic)

        curr_agent.critic_optimizer.step()

        # ---------------------- Policy update ----------------------
        curr_agent.policy_optimizer.zero_grad()

        curr_act = curr_agent.policy(obs)
        pol_vf_in = torch.cat((obs, curr_act), dim=1)

        pol_loss = -curr_agent.critic(pol_vf_in).mean()
        pol_loss.backward()

        if parallel:
            average_gradients(curr_agent.policy)

        curr_agent.policy_optimizer.step()

        if logger is not None:
            logger.add_scalars(
                f"agent{agent_i}/losses",
                {"vf_loss": float(vf_loss.item()),
                 "pol_loss": float(pol_loss.item())},
                self.niter,
            )

    def update_all_targets(self):
        for a in self.agents:
            soft_update(a.target_critic, a.critic, self.tau)
            soft_update(a.target_policy, a.policy, self.tau)
        self.niter += 1

    def prep_training(self, device="cuda"):
        """
        Compatible with main.py passing device="cuda" or "cpu".
        """
        dev = torch.device(device) if not isinstance(
            device, torch.device) else device

        for a in self.agents:
            a.policy.train()
            a.critic.train()
            a.target_policy.train()
            a.target_critic.train()
            a.to(dev)

        self.pol_dev = str(dev)
        self.critic_dev = str(dev)
        self.trgt_pol_dev = str(dev)
        self.trgt_critic_dev = str(dev)

    def prep_rollouts(self, device="cpu"):
        """
        Rollouts only need policy in eval; move whole agent for consistency.
        """
        dev = torch.device(device) if not isinstance(
            device, torch.device) else device
        for a in self.agents:
            a.policy.eval()
            a.to(dev)
        self.pol_dev = str(dev)

    def save(self, filename):
        self.prep_training(device="cpu")
        save_dict = {
            "init_dict": self.init_dict,
            "agent_params": [a.get_params() for a in self.agents],
        }
        torch.save(save_dict, filename)

    @classmethod
    def init_from_env(
        cls,
        env,
        obs_slices=None,  # kept for signature compatibility with your main.py
        act_slices=None,
        gamma=0.95,
        tau=0.01,
        lr_actor=1e-4,
        lr_critic=1e-3,
        hidden_dim=64,
        epsilon=0.1,
        noise=0.1,
        discrete_action=False,
    ):
        """
        Build two controllers (pred-controller, prey-controller) by default:
          policy input:  obs_dim (env.observation_space.shape[0])
          policy output: act_dim (env.action_space.shape[0])
          critic input:  obs_dim + act_dim
        """
        agent_init_params = []

        obs_dim = env.observation_space.shape[0]
        act_dim = env.action_space.shape[0]
        num_in_critic = obs_dim + act_dim

        # If obs_slices provided, use its length; else default to 2 controllers
        nagents = len(obs_slices) if obs_slices is not None else 2

        for _ in range(nagents):
            agent_init_params.append(
                {"num_in_pol": obs_dim, "num_out_pol": act_dim,
                    "num_in_critic": num_in_critic}
            )

        init_dict = dict(
            gamma=gamma,
            tau=tau,
            lr_actor=lr_actor,
            lr_critic=lr_critic,
            epsilon=epsilon,
            noise=noise,
            hidden_dim=hidden_dim,
            discrete_action=discrete_action,
            agent_init_params=agent_init_params,
        )

        instance = cls(**init_dict)
        instance.init_dict = init_dict
        return instance

    @classmethod
    def init_from_save(cls, filename):
        save_dict = torch.load(filename, map_location="cpu")
        instance = cls(**save_dict["init_dict"])
        instance.init_dict = save_dict["init_dict"]
        for a, params in zip(instance.agents, save_dict["agent_params"]):
            a.load_params(params)
        return instance


# compatibility alias
DDPG = IDDPG
