import gymnasium as gym
import gym_pps
from custom_env import NJPEnvironment
import argparse
import torch
import time
import os
import numpy as np
from algorithms.maddpg import MADDPG
from algorithms.mappo import MAPPO
from algorithms import ddpg as ddpg_mod
from pathlib import Path

from tqdm import tqdm
from utils.helpers import *


from utils.device_management import move_model_to_device

"""
This repository provides a reference implementation of the MARL algorithm for the PPS environment, 
adapted from "Predator-prey survival pressure is sufficient to evolve swarming behaviors" (New Journal of Physics).
https://iopscience.iop.org/article/10.1088/1367-2630/acf33a
"""

# ============================================================================
# GLOBAL CONFIGURATION
# ============================================================================

USE_CUDA = torch.cuda.is_available()
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
START_TIME = time.time()


# ============================================================================
# ENVIRONMENT SETUP
# ============================================================================

def setup_environment():
    """Initialize and wrap the Predator-Prey Swarm environment."""
    scenario_name = 'PredatorPreySwarm-v0'
    custom_param_name = 'custom_param.json'

    environment = gym.make(scenario_name)
    custom_param_path = os.path.dirname(
        os.path.realpath(__file__)) + '/' + custom_param_name
    environment = NJPEnvironment(environment, custom_param_path)

    return environment

# ============================================================================
# TRAINING SETUP
# ============================================================================

def initialize_seeds_and_threads(config):
    """Set random seeds and thread count."""
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    if not USE_CUDA:
        torch.set_num_threads(config.n_training_threads)


def initialize_maddpg_model(env, config, agent_slices):
    """Initialize the MADDPG model with environment parameters."""
    maddpg = MADDPG.init_from_env(
        env,
        agent_alg=config.agent_alg,
        adversary_alg=config.adversary_alg,
        tau=config.tau,
        lr_actor=config.lr_actor,
        lr_critic=config.lr_critic,
        epsilon=config.epsilon,
        noise=config.noise,
        hidden_dim=config.hidden_dim
    )
    move_model_to_device(maddpg, DEVICE)
    return maddpg


def initialize_mappo_model(env, config, agent_slices):
    mappo = MAPPO.init_from_env(
        env,
        observation_radius=5.0,  # Agents see others within 5 units
        gamma=0.995,
        lr_actor=1e-4,
        lr_critic=1e-3
    )
    move_model_to_device(mappo, DEVICE)
    return mappo


def initialize_ddpg_model(env, config, agent_slices):
    """Initialize the DDPG model with environment parameters."""
    DDPGCls = getattr(ddpg_mod, "IDDPG", None) or getattr(
        ddpg_mod, "DDPG", None)
    if DDPGCls is None:
        raise ImportError(
            "Cannot find class IDDPG or DDPG in algorithms/ddpg.py")

    # Build slices for 2 controllers: predators-controller and prey-controller
    obs_slices = agent_slices

    # Action slices: each controller outputs env.action_space.shape[0] actions
    act_dim_ctrl = env.action_space.shape[0]
    act_slices = [slice(0, act_dim_ctrl), slice(
        act_dim_ctrl, 2 * act_dim_ctrl)]

    # Try init_from_env with obs_slices and act_slices first
    try:
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
        # Fall back to simpler init signature
        algo = DDPGCls.init_from_env(
            env,
            tau=config.tau,
            lr_actor=config.lr_actor,
            lr_critic=config.lr_critic,
            epsilon=config.epsilon,
            noise=config.noise,
            hidden_dim=config.hidden_dim,
        )

    # Attach slices for later use in update
    algo._obs_slices = obs_slices
    algo._act_slices = act_slices
    move_model_to_device(algo, DEVICE)
    return algo


# ============================================================================
# EPISODE EXECUTION
# ============================================================================

def run_episode_ddpg(env, ddpg, config, agent_slices, buffers):
    """Run a single DDPG training episode and collect trajectory data."""
    observation, _ = env.reset()
    ddpg.prep_rollouts(device=DEVICE)
    ddpg.scale_noise(ddpg.noise, ddpg.epsilon)
    ddpg.reset_noise()
    
    # Initialize storage for trajectory data
    position_dims, num_agents = np.shape(env.unwrapped.p)
    heading_dims, _ = np.shape(env.unwrapped.heading)
    
    positions = np.zeros((position_dims, num_agents, config.episode_length))
    headings = np.zeros((heading_dims, num_agents, config.episode_length))
    
    # Rollout episode
    for step_idx in range(config.episode_length):
        if config.render and (step_idx % config.render_interval == 0):
            env.render()
        
        # Store current state
        positions[:, :, step_idx] = env.unwrapped.p
        headings[:, :, step_idx] = env.unwrapped.heading
        
        # Get actions from policy
        torch_obs = torch.as_tensor(observation, dtype=torch.float32, device=DEVICE)
        torch_actions = ddpg.step(torch_obs, agent_slices, explore=True)
        actions = np.column_stack([action.data.cpu().numpy() for action in torch_actions])
        
        # Step environment
        next_observation, rewards, dones, infos = env.step(actions)
        
        # Store transitions in replay buffers
        buffers[0].push(observation, actions, rewards, next_observation, dones)  # predators
        buffers[1].push(observation, actions, rewards, next_observation, dones)  # prey
        
        observation = next_observation
    
    return positions, headings, num_agents, position_dims


def run_episode_maddpg(env, maddpg, config, agent_slices, buffers):
    """Run a single training episode and collect trajectory data."""
    observation, _ = env.reset()
    maddpg.prep_rollouts(device=DEVICE)
    maddpg.scale_noise(maddpg.noise, maddpg.epsilon)
    maddpg.reset_noise()
    
    # Initialize storage for trajectory data
    position_dims, num_agents = np.shape(env.unwrapped.p)
    heading_dims, _ = np.shape(env.unwrapped.heading)
    
    positions = np.zeros((position_dims, num_agents, config.episode_length))
    headings = np.zeros((heading_dims, num_agents, config.episode_length))
    
    # Rollout episode
    for step_idx in range(config.episode_length):
        if config.render and (step_idx % config.render_interval == 0):
            env.render()
        
        # Store current state
        positions[:, :, step_idx] = env.unwrapped.p
        headings[:, :, step_idx] = env.unwrapped.heading
        
        # Get actions from policy
        torch_obs = torch.as_tensor(observation, dtype=torch.float32, device=DEVICE)
        torch_actions = maddpg.step(torch_obs, agent_slices, explore=True)
        actions = np.column_stack([action.data.cpu().numpy() for action in torch_actions])
        
        # Step environment
        next_observation, rewards, dones, infos = env.step(actions)
        
        # Store transitions in replay buffers
        buffers[0].push(observation, actions, rewards, next_observation, dones)  # predators
        buffers[1].push(observation, actions, rewards, next_observation, dones)  # prey
        
        observation = next_observation
    
    return positions, headings, num_agents, position_dims


def train_maddpg_model(maddpg, config, buffers):
    """Perform training iterations on the MADDPG model."""
    num_training_steps = 50
    
    for _ in range(num_training_steps):
        maddpg.prep_training(device=DEVICE)
        
        for agent_idx in range(maddpg.nagents):
            if len(buffers[agent_idx]) >= config.batch_size:
                obs_batch, actions_batch, rewards_batch, next_obs_batch, dones_batch = buffers[agent_idx].sample(
                    config.batch_size,
                    to_gpu=USE_CUDA
                )
                maddpg.update(obs_batch, actions_batch, rewards_batch, next_obs_batch, dones_batch, agent_idx)
        
        maddpg.update_all_targets()
        maddpg.prep_rollouts(device=DEVICE)


def train_ddpg_model(ddpg, config, buffers, agent_slices):
    """Perform training iterations on the DDPG model."""
    num_training_steps = config.n_updates_per_episode
    
    for _ in range(num_training_steps):
        ddpg.prep_training(device=DEVICE)
        
        # Train both controllers (predators and prey)
        for agent_idx in range(2):
            if len(buffers[agent_idx]) >= config.batch_size:
                obs_batch, actions_batch, rewards_batch, next_obs_batch, dones_batch = buffers[agent_idx].sample(
                    config.batch_size,
                    to_gpu=USE_CUDA
                )
                # Use compatibility layer for update
                _ddpg_update(ddpg, obs_batch, actions_batch, rewards_batch,
                            next_obs_batch, dones_batch, agent_idx, agent_slices)
        
        ddpg.update_all_targets()
        ddpg.prep_rollouts(device=DEVICE)


def _ddpg_update(ddpg, obs_s, acs_s, rews_s, next_obs_s, dones_s, agent_idx, agent_slices):
    """
    Call DDPG update() with compatibility layer.
    Supports different update signatures.
    """
    # Try obs_slices/act_slices style
    if hasattr(ddpg, "_obs_slices") and hasattr(ddpg, "_act_slices"):
        try:
            return ddpg.update(obs_s, acs_s, rews_s, next_obs_s, dones_s,
                             agent_i=agent_idx,
                             obs_slices=ddpg._obs_slices,
                             act_slices=ddpg._act_slices)
        except TypeError:
            pass

    # Try agent_slices style
    try:
        return ddpg.update(obs_s, acs_s, rews_s, next_obs_s, dones_s,
                         agent_i=agent_idx,
                         start_stop_num=agent_slices)
    except TypeError:
        pass

    # Plain update
    return ddpg.update(obs_s, acs_s, rews_s, next_obs_s, dones_s, agent_idx)




def run_episode_mappo(env, mappo, config, agent_slices):
    """Run a single MAPPO episode and collect trajectories for all agents."""
    observation, _ = env.reset()
    mappo.prep_rollouts()

    position_dims, num_agents = np.shape(env.unwrapped.p)
    heading_dims, _ = np.shape(env.unwrapped.heading)

    positions = np.zeros((position_dims, num_agents, config.episode_length))
    headings = np.zeros((heading_dims, num_agents, config.episode_length))

    trajectories = {agent_idx: [] for agent_idx in range(num_agents)}

    for step_idx in range(config.episode_length):
        if config.render and (step_idx % config.render_interval == 0):
            env.render()

        positions[:, :, step_idx] = env.unwrapped.p
        headings[:, :, step_idx] = env.unwrapped.heading

        # Build batch observations and positions
        obs_agents = torch.as_tensor(observation.T, dtype=torch.float32, device=DEVICE)
        obs_batch = obs_agents.unsqueeze(0)
        pos_batch = torch.as_tensor(env.unwrapped.p.T, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        print("OBS batch: ", obs_batch.shape)
        print("POS batch: ", pos_batch.shape)
        actions, values = mappo.step(obs_batch, pos_batch, agent_slices, explore=True)

        # for agent_idx, agent in enumerate(mappo.agents):
        #     is_adversary = agent_idx < agent_slices[0].stop
        #     filtered_obs = mappo.filter_observations_by_radius(
        #         obs_batch, pos_batch, agent_idx, is_adversary
        #     )
        #     agent_obs = filtered_obs[0, agent_idx].unsqueeze(0)
        #     action, _, value = agent.get_action(agent_obs, exploration=True)
        #     actions.append(action)
        #     values.append(value)


        # Actions need to be (action_dim, num_agents) for env.step
        actions_array = torch.cat(actions, dim=0).detach().cpu().numpy().T
        print("Actions: ", actions_array.shape)
        print("OBS batch: ", obs_batch.shape)
        print("POS batch: ", pos_batch.shape)
        next_observation, rewards, dones, infos = env.step(actions_array)

        # Record per-agent transition
        next_obs_agents = torch.as_tensor(next_observation.T, dtype=torch.float32, device=DEVICE)
        next_obs_batch = next_obs_agents.unsqueeze(0)
        next_pos_batch = torch.as_tensor(env.unwrapped.p.T, dtype=torch.float32, device=DEVICE).unsqueeze(0)

        for agent_idx, agent in enumerate(mappo.agents):
            is_adversary = agent_idx < agent_slices[0].stop
            filtered_next_obs = mappo.filter_observations_by_radius(
                next_obs_batch, next_pos_batch, agent_idx, is_adversary
            )
            agent_obs = obs_batch[0, agent_idx].cpu().numpy()
            agent_next_obs = filtered_next_obs[0, agent_idx].cpu().numpy()
            agent_action = actions[agent_idx].squeeze(0).cpu().numpy()
            agent_reward = rewards[agent_idx] if hasattr(rewards, '__len__') else rewards
            agent_done = dones[agent_idx] if hasattr(dones, '__len__') else dones
            agent_value = values[agent_idx].item()

            trajectories[agent_idx].append(
                (agent_obs, agent_action, agent_reward, agent_next_obs, agent_done, agent_value)
            )

        observation = next_observation

    return positions, headings, num_agents, position_dims, trajectories


def train_mappo_model(mappo, config, trajectories):
    """Train MAPPO using collected trajectories."""
    mappo.prep_training()
    mappo.update(trajectories, num_epochs=5, batch_size=config.batch_size)



# ============================================================================
# MAIN TRAINING LOOP
# ============================================================================

def run_maddpg(config):
    """Run the complete training pipeline."""
    # Setup
    run_dir, log_dir = create_run_directory(config)
    initialize_seeds_and_threads(config)
    env = setup_environment()
    agent_slices = create_agent_slices(env)
    
    # Initialize models and buffers
    maddpg = initialize_maddpg_model(env, config, agent_slices)
    buffers = create_replay_buffers(env, config, agent_slices)
    dos_and_doa_vals = [] # values over each episode
    
    # Training loop
    for episode_idx in tqdm(range(0, config.n_episodes, config.n_rollout_threads)):
        print(f"\rEpisodes {episode_idx + 1}-{episode_idx + 1 + config.n_rollout_threads} of {config.n_episodes}",
              end='', flush=True)
        
        # Run episode and collect data
        positions, headings, num_agents, position_dims = run_episode_maddpg(
            env, maddpg, config, agent_slices, buffers
        )
        
        # Train on collected data
        train_maddpg_model(maddpg, config, buffers)
        
        # Decay exploration noise
        decay_exploration_noise(maddpg)
        
        # Save checkpoint
        save_model_checkpoint(maddpg, 'maddpg', run_dir, episode_idx, config)
        
        # Calculate metrics (optional - uncomment to enable)
        dos, doa = env.dos_and_doa_one_episode(positions, headings, num_agents, position_dims)
        dos_and_doa_vals.append([episode_idx, dos, doa])
        # print(f"DOS: {dos:.4f}, DOA: {doa:.4f}")
    
    # Print training summary
    elapsed_time = time.time() - START_TIME
    print(f"\nTraining completed in {elapsed_time / 60:.2f} minutes")

    # save metrics
    save_dos_and_doa(
        dos_and_doa_vals,
        filename=run_dir / 'metrics.csv'
    )


def run_mappo(config):
    """Run the complete MAPPO training pipeline."""
    run_dir, log_dir = create_run_directory(config)
    initialize_seeds_and_threads(config)
    env = setup_environment()
    agent_slices = create_agent_slices(env)

    mappo = initialize_mappo_model(env, config, agent_slices)
    dos_and_doa_vals = []

    for episode_idx in tqdm(range(0, config.n_episodes, config.n_rollout_threads)):
        print(f"\rEpisodes {episode_idx + 1}-{episode_idx + 1 + config.n_rollout_threads} of {config.n_episodes}",
              end='', flush=True)

        positions, headings, num_agents, position_dims, trajectories = run_episode_mappo(
            env, mappo, config, agent_slices
        )

        train_mappo_model(mappo, config, trajectories)
        save_model_checkpoint(mappo, 'mappo', run_dir, episode_idx, config)

        dos, doa = env.dos_and_doa_one_episode(positions, headings, num_agents, position_dims)
        dos_and_doa_vals.append([episode_idx, dos, doa])

    elapsed_time = time.time() - START_TIME
    print(f"\nTraining completed in {elapsed_time / 60:.2f} minutes")

    save_dos_and_doa(
        dos_and_doa_vals,
        filename=run_dir / 'metrics.csv'
    )


def run_ddpg(config):
    """Run the complete DDPG training pipeline."""
    # Setup
    run_dir, log_dir = create_run_directory(config)
    initialize_seeds_and_threads(config)
    env = setup_environment()
    agent_slices = create_agent_slices(env)
    
    # Initialize models and buffers
    ddpg_model = initialize_ddpg_model(env, config, agent_slices)
    buffers = create_replay_buffers(env, config, agent_slices)
    dos_and_doa_vals = []  # values over each episode
    
    # Training loop
    for episode_idx in tqdm(range(0, config.n_episodes, config.n_rollout_threads)):
        print(f"\rEpisodes {episode_idx + 1}-{episode_idx + 1 + config.n_rollout_threads} of {config.n_episodes}",
              end='', flush=True)
        
        # Run episode and collect data
        positions, headings, num_agents, position_dims = run_episode_ddpg(
            env, ddpg_model, config, agent_slices, buffers
        )
        
        # Train on collected data
        train_ddpg_model(ddpg_model, config, buffers, agent_slices)
        
        # Decay exploration noise
        decay_exploration_noise(ddpg_model)
        
        # Save checkpoint
        save_model_checkpoint(ddpg_model, 'ddpg', run_dir, episode_idx, config)
        
        # Calculate metrics
        dos, doa = env.dos_and_doa_one_episode(positions, headings, num_agents, position_dims)
        dos_and_doa_vals.append([episode_idx, dos, doa])
    
    # Print training summary
    elapsed_time = time.time() - START_TIME
    print(f"\nTraining completed in {elapsed_time / 60:.2f} minutes")

    # Save metrics
    save_dos_and_doa(
        dos_and_doa_vals,
        filename=run_dir / 'metrics.csv'
    )


if __name__ == '__main__':
    parser = create_argument_parser()
    config = parser.parse_args()
    if config.algo == 'mappo':
        run_mappo(config)
    elif config.algo == 'ddpg':
        run_ddpg(config)
    else:
        run_maddpg(config)
