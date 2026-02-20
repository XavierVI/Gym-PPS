import gymnasium as gym
import gym_pps
from custom_env import NJPEnvironment
import argparse
import torch
import time
import os
import numpy as np
from algorithms.maddpg import MADDPG
from pathlib import Path
from utils.buffer import ReplayBuffer
from tqdm import tqdm
from algorithms import ddpg as ddpg_mod
"""
This repository provides a reference implementation of the MARL algorithm for the PPS environment, 
adapted from “Predator-prey survival pressure is sufficient to evolve swarming behaviors” (New Journal of Physics).
https://iopscience.iop.org/article/10.1088/1367-2630/acf33a
"""

## Define the Predator-Prey Swarm (PPS) environment
scenario_name = 'PredatorPreySwarm-v0'  
custom_param = 'custom_param.json'      
env = gym.make(scenario_name)
custom_param = os.path.dirname(os.path.realpath(__file__)) + '/' + custom_param
env = NJPEnvironment(env, custom_param)

USE_CUDA = torch.cuda.is_available()
device = "cuda" if torch.cuda.is_available() else "cpu"

start_time = time.time()


# --------------------
# Helpers
# --------------------
def _force_to_device(algo, dev):
    """Move all known modules to a device to avoid cpu/cuda mismatch."""
    def _to_device(obj, dev_):
        if obj is None:
            return
        if hasattr(obj, "to"):
            try:
                obj.to(dev_)
            except Exception:
                pass

    for attr in ["policy", "critic", "target_policy", "target_critic",
                 "actor", "actor_target", "critic_target"]:
        _to_device(getattr(algo, attr, None), dev)

    for ag in getattr(algo, "agents", []):
        for attr in ["policy", "critic", "target_policy", "target_critic",
                     "actor", "actor_target", "critic_target"]:
            _to_device(getattr(ag, attr, None), dev)


def _load_algo(config, env, start_stop_num):
    """
    Create algorithm object based on config.alg.
    This function is intentionally defensive to support different ddpg.py signatures.
    """
    if config.alg.upper() == "MADDPG":
        algo = MADDPG.init_from_env(
            env,
            agent_alg="MADDPG",
            adversary_alg="MADDPG",
            tau=config.tau,
            lr_actor=config.lr_actor,
            lr_critic=config.lr_critic,
            epsilon=config.epsilon,
            noise=config.noise,
            hidden_dim=config.hidden_dim,
        )
        return algo

    if config.alg.upper() == "DDPG":
        # Try to import the class you saved.
        from algorithms import ddpg as ddpg_mod
        DDPGCls = getattr(ddpg_mod, "IDDPG", None) or getattr(
            ddpg_mod, "DDPG", None)
        if DDPGCls is None:
            raise ImportError(
                "Cannot find class IDDPG or DDPG in algorithms/ddpg.py")

        # Build slices for 2 controllers: predators-controller and prey-controller
        # obs are organized by agent columns, and you already have start_stop_num for that.
        obs_slices = start_stop_num

        # Action slices:
        # In your codebase, maddpg.step returns 2 action blocks and you column_stack them.
        # The controller action dim is typically env.action_space.shape[0].
        # If your ddpg.py expects flattened [batch, act_dim_total], these slices must match.
        #
        # Here we assume 2 controllers, each outputs act_dim_ctrl = env.action_space.shape[0]
        # (common in the NJP wrapper where each "side" is treated as one agent).
        act_dim_ctrl = env.action_space.shape[0]
        act_slices = [slice(0, act_dim_ctrl), slice(
            act_dim_ctrl, 2 * act_dim_ctrl)]

        # Try init_from_env signatures:
        try:
            # If your ddpg.py is my "稳妥版", it likely wants obs_slices & act_slices.
            algo = DDPGCls.init_from_env(
                env,
                obs_slices=obs_slices,
                act_slices=act_slices,
                tau=config.tau,
                lr_actor=config.lr_actor,
                lr_critic=config.lr_critic,
                epsilon=config.epsilon,
                noise=config.noise,
                hidden_dim=config.hidden_dim,
            )
        except TypeError:
            # Otherwise, fall back to a simpler init
            algo = DDPGCls.init_from_env(
                env,
                tau=config.tau,
                lr_actor=config.lr_actor,
                lr_critic=config.lr_critic,
                epsilon=config.epsilon,
                noise=config.noise,
                hidden_dim=config.hidden_dim,
            )

        # Attach for later (so update can access if needed)
        algo._obs_slices = obs_slices
        algo._act_slices = act_slices
        return algo

    raise ValueError(
        f"Unknown --alg {config.alg}. Choose from MADDPG or DDPG.")


def _algo_update(algo, obs_s, acs_s, rews_s, next_obs_s, dones_s, a_i, start_stop_num):
    """
    Call update() with a best-effort compatibility layer.
    Supports:
      - update(..., start_stop_num=...)
      - update(..., obs_slices=..., act_slices=...)
      - update(...) plain
    """
    # Try (obs_slices/act_slices) style
    if hasattr(algo, "_obs_slices") and hasattr(algo, "_act_slices"):
        try:
            return algo.update(obs_s, acs_s, rews_s, next_obs_s, dones_s,
                               agent_i=a_i,
                               obs_slices=algo._obs_slices,
                               act_slices=algo._act_slices)
        except TypeError:
            pass

    # Try start_stop_num style
    try:
        return algo.update(obs_s, acs_s, rews_s, next_obs_s, dones_s,
                           agent_i=a_i,
                           start_stop_num=start_stop_num)
    except TypeError:
        pass

    # Plain update
    return algo.update(obs_s, acs_s, rews_s, next_obs_s, dones_s, a_i)


def move_atts_to_device(maddpg, device):
    """
    force move ALL model modules to the same device (fix cpu/cuda mismatch)
    """
    def _to_device(obj, dev):
        if obj is None:
            return
        if hasattr(obj, "to"):
            try:
                obj.to(dev)
            except Exception:
                pass

    for attr in ["policy", "critic", "target_policy", "target_critic",
                 "actor", "actor_target", "critic_target"]:
        _to_device(getattr(maddpg, attr, None), device)

    for ag in getattr(maddpg, "agents", []):
        for attr in ["policy", "critic", "target_policy", "target_critic",
                     "actor", "actor_target", "critic_target"]:
            _to_device(getattr(ag, attr, None), device)

    printed = False
    for ag in getattr(maddpg, "agents", []):
        for attr in ["policy", "actor", "critic"]:
            m = getattr(ag, attr, None)
            if m is not None and hasattr(m, "parameters"):
                try:
                    print("First agent", attr, "param device:",
                          next(m.parameters()).device)
                    printed = True
                    break
                except StopIteration:
                    pass
        if printed:
            break


# --------------------
# Train loop
# --------------------
def run(config):
    model_dir = Path('./models') / config.env_id
    if not model_dir.exists():
        curr_run = 'run1'
    else:
        exst_run_nums = [
            int(str(folder.name).split('run')[1])
            for folder in model_dir.iterdir()
            if str(folder.name).startswith('run')
        ]
        curr_run = 'run1' if len(
            exst_run_nums) == 0 else f'run{max(exst_run_nums) + 1}'

    run_dir = model_dir / curr_run
    log_dir = run_dir / 'logs'
    os.makedirs(log_dir, exist_ok=True)

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    if not USE_CUDA:
        torch.set_num_threads(config.n_training_threads)

    # 2 controllers: predators-controller and prey-controller
    start_stop_num = [
        slice(0, env.num_predator),
        slice(env.num_predator, env.num_predator + env.num_prey),
    ]

    algo = _load_algo(config, env, start_stop_num)
    _force_to_device(algo, device)

    # Buffers (pred-side / prey-side), consistent with the original implementation.
    adversary_buffer = ReplayBuffer(
        config.buffer_length,
        env.num_predator,
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.shape[0],
        start_stop_index=start_stop_num[0],
    )
    agent_buffer = ReplayBuffer(
        config.buffer_length,
        env.num_prey,
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.shape[0],
        start_stop_index=start_stop_num[1],
    )
    buffer_total = [adversary_buffer, agent_buffer]

    t = 0

    for ep_i in tqdm(range(0, config.n_episodes, config.n_rollout_threads)):
        print(
            "\rEpisodes %i-%i of %i"
            % (ep_i + 1, ep_i + 1 + config.n_rollout_threads, config.n_episodes),
            end="",
            flush=True,
        )

        obs = env.reset()
        algo.prep_rollouts(device=device)

        algo.scale_noise(algo.noise, algo.epsilon)
        algo.reset_noise()

        for et_i in range(config.episode_length):
            if config.render and (ep_i % config.render_interval == 0):
                env.render()

            torch_obs = torch.as_tensor(
                obs, dtype=torch.float32, device=device)

            # actions for [predators_controller, prey_controller]
            torch_agent_actions = algo.step(
                torch_obs, start_stop_num, explore=True)
            agent_actions = np.column_stack(
                [ac.data.cpu().numpy() for ac in torch_agent_actions])

            next_obs, rewards, dones, infos = env.step(agent_actions)

            agent_buffer.push(obs, agent_actions, rewards, next_obs, dones)
            adversary_buffer.push(obs, agent_actions, rewards, next_obs, dones)

            obs = next_obs
            t += config.n_rollout_threads

        # Training updates
        for _ in range(config.n_updates_per_episode):
            algo.prep_training(device=device)

            # In your original codebase this is effectively 2 "agents" (pred-controller, prey-controller)
            for a_i in range(getattr(algo, "nagents", 2)):
                buf = buffer_total[a_i]
                if len(buf) < config.batch_size:
                    continue

                obs_s, acs_s, rews_s, next_obs_s, dones_s = buf.sample(
                    config.batch_size, to_gpu=USE_CUDA
                )
                _algo_update(algo, obs_s, acs_s, rews_s,
                             next_obs_s, dones_s, a_i, start_stop_num)

            algo.update_all_targets()
            algo.prep_rollouts(device=device)

        # Exploration decay
        algo.noise = max(config.min_noise, algo.noise - config.noise_decay)
        algo.epsilon = max(config.min_epsilon,
                           algo.epsilon - config.epsilon_decay)

        # Save
        if ep_i % config.save_interval < config.n_rollout_threads:
            os.makedirs(run_dir / 'incremental', s,
                        next_obs_s, dones_s, a_i, start_stop_num)

            algo.update_all_targets()
            algo.prep_rollouts(device=device)

        # Exploration decay
        algo.noise = max(config.min_noise, algo.noise - config.noise_decay)
        algo.epsilon = max(config.min_epsilon,
                           algo.epsilon - config.epsilon_decay)

        # Saveexist_ok=True)
        algo.save(run_dir / 'incremental' / ('model_ep%i.pt' % (ep_i + 1)))

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"\nElapsed time: {elapsed_time/60:.2f} min")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_id", default="model_1", type=str)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--n_rollout_threads", default=1, type=int)
    parser.add_argument("--n_training_threads", default=10, type=int)
    parser.add_argument("--buffer_length", default=int(5e5), type=int)
    parser.add_argument("--n_episodes", default=201, type=int)
    parser.add_argument("--episode_length", default=200, type=int)
    parser.add_argument("--batch_size", default=256, type=int)
    parser.add_argument("--n_exploration_eps", default=25000, type=int)
    parser.add_argument("--init_noise_scale", default=0.3, type=float)
    parser.add_argument("--final_noise_scale", default=0.0, type=float)
    parser.add_argument("--save_interval", default=50, type=int)
    parser.add_argument("--hidden_dim", default=128, type=int) 
    parser.add_argument("--lr_actor", default=1e-4, type=float)
    parser.add_argument("--lr_critic", default=1e-3, type=float)
    parser.add_argument("--epsilon", default=0.1, type=float)
    parser.add_argument("--noise", default=0.1, type=float)
    parser.add_argument("--tau", default=0.01, type=float)
    parser.add_argument("--agent_alg", default="MADDPG", type=str,choices=['MADDPG', 'DDPG'])
    parser.add_argument("--adversary_alg", default="MADDPG", type=str,choices=['MADDPG', 'DDPG'])
    parser.add_argument(
        "--render",
        action="store_true",
        help="Enable env.render(). On headless servers leave this OFF, or run with xvfb-run.",
    )
    parser.add_argument(
        "--render_interval",
        default=20,
        type=int,
        help="Render every N episodes (only when --render is set). Default: 20",
    )
    parser.add_argument("--alg", default="MADDPG",
                        type=str, choices=["MADDPG", "DDPG"])

    parser.add_argument("--n_updates_per_episode", default=50, type=int)

    # Exploration decay
    parser.add_argument("--min_noise", default=0.05, type=float)
    parser.add_argument("--min_epsilon", default=0.05, type=float)
    parser.add_argument("--noise_decay", default=5e-5, type=float)
    parser.add_argument("--epsilon_decay", default=5e-5, type=float)

    config = parser.parse_args()

    run(config)