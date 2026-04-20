import gymnasium as gym
import imageio
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


def setup_environment(config):
    """Initialize and wrap the Predator-Prey Swarm environment."""
    scenario_name = 'PredatorPreySwarm-v0'

    if config.custom_param_name is None:
        custom_param_name = 'custom_param.json'
    else:
        custom_param_name = config.custom_param_name

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


# ============================================================================
# VIDEO SAVING
# ============================================================================

def save_episode_video(frames, output_path, fps=30):
    """Save collected frames as a gif or video file."""
    if len(frames) == 0:
        print("No frames to save")
        return False

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(
        output_path) else '.', exist_ok=True)

    try:
        imageio.mimsave(output_path, frames, fps=fps)
        print(f"Video saved to {output_path}")
        return True
    except Exception as e:
        print(f"Failed to save video to {output_path}: {e}")
        return False


def save_timestep_dos_doa(rows, filename):
    """Save DOS/DOA rows with header: episode,timestep,dos,doa."""
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    data = np.asarray(rows, dtype=float)
    np.savetxt(
        filename,
        data,
        delimiter=",",
        fmt="%.6f",
        header="episode,timestep,dos,doa",
        comments=""
    )


# ============================================================================
# EPISODE EXECUTION
# ============================================================================
def run_episode_maddpg(env, maddpg, config, agent_slices, buffers):
    """Run a single training episode and collect trajectory data."""
    observation, _ = env.reset()
    maddpg.prep_rollouts(device=DEVICE)
    maddpg.scale_noise(maddpg.noise, maddpg.epsilon)
    maddpg.reset_noise()

    # Initialize storage for trajectory data
    # print(f"{env.unwrapped.p.shape}")
    position_dims, num_agents = np.shape(env.unwrapped.p)
    heading_dims, N_h = np.shape(env.unwrapped.heading)

    positions = np.zeros((position_dims, num_agents, config.episode_length))
    headings = np.zeros((heading_dims, N_h, config.episode_length))
    agent_rewards = []
    frames = []

    # Rollout episode
    for step_idx in range(config.episode_length):
        if config.render:
            frame = env.unwrapped.render(mode='rgb_array')
            frames.append(frame)

        # Store current state
        positions[:, :, step_idx] = env.unwrapped.p
        headings[:, :, step_idx] = env.unwrapped.heading

        # Get actions from policy
        torch_obs = torch.as_tensor(
            observation, dtype=torch.float32, device=DEVICE)
        torch_actions = maddpg.step(torch_obs, agent_slices, explore=True)
        actions = np.column_stack([action.data.cpu().numpy()
                                  for action in torch_actions])

        # Step environment
        next_observation, rewards, dones, infos = env.step(actions)
        agent_rewards.append(rewards)

        # Store transitions in replay buffers
        buffers[0].push(observation, actions, rewards,
                        next_observation, dones)  # predators
        buffers[1].push(observation, actions, rewards,
                        next_observation, dones)  # prey

        observation = next_observation

    return positions, headings, num_agents, position_dims, np.mean(agent_rewards, axis=0).squeeze(), frames


def train_maddpg_model(maddpg, config, buffers):
    """Perform training iterations on the MADDPG model."""
    num_training_steps = 50

    for _ in range(num_training_steps):
        maddpg.prep_training(device=DEVICE)
        # print(maddpg.nagents)
        for agent_idx in range(maddpg.nagents):
            if len(buffers[agent_idx]) >= config.batch_size:
                obs_batch, actions_batch, rewards_batch, next_obs_batch, dones_batch = buffers[agent_idx].sample(
                    config.batch_size,
                    to_gpu=USE_CUDA
                )
                # print(
                #     f"[train] agent {agent_idx} rewards_batch min={rewards_batch.min():.2f} max={rewards_batch.max():.2f} mean={rewards_batch.mean():.2f}")
                # print(rewards_batch.shape)
                maddpg.update(obs_batch, actions_batch, rewards_batch,
                              next_obs_batch, dones_batch, agent_idx)

        maddpg.update_all_targets()
        maddpg.prep_rollouts(device=DEVICE)


# ============================================================================
# MAIN TRAINING LOOP
# ============================================================================

def run(config):
    """Run the complete training pipeline."""
    # Setup
    run_dir, log_dir = create_run_directory(config)
    initialize_seeds_and_threads(config)
    env = setup_environment(config)
    agent_slices = create_agent_slices(env)

    # Initialize models and buffers
    maddpg = initialize_maddpg_model(env, config, agent_slices)
    buffers = create_replay_buffers(env, config, agent_slices)
    # Create output directory for videos
    video_dir = Path('.') / run_dir / 'videos'
    video_dir.mkdir(parents=True, exist_ok=True)
    all_timestep_metrics = []
    metrics = []  # values over each episode

    # Training loop
    for episode_idx in tqdm(range(0, config.n_episodes, config.n_rollout_threads)):
        # print(f"\rEpisodes {episode_idx + 1}-{episode_idx + 1 + config.n_rollout_threads} of {config.n_episodes}",
        #       end='', flush=True)

        # Run episode and collect data
        # NOTE: positions and headings are returned as (2, num_agents, episode_length)
        positions, headings, num_agents, position_dims, rewards, frames = run_episode_maddpg(
            env, maddpg, config, agent_slices, buffers
        )

        # print("Rewards shape: ", rewards.shape)

        # Train on collected data
        train_maddpg_model(maddpg, config, buffers)

        # Decay exploration noise
        decay_exploration_noise(maddpg)

        # Save checkpoint
        save_model_checkpoint(maddpg, 'maddpg', run_dir, episode_idx, config)

        # print(f"Position dims: {positions.shape}, Num agents: {num_agents}")
        # print(f"Agent slices: {agent_slices}")

        # Calculate metrics (optional - uncomment to enable)
        # dos, doa = env.dos_and_doa_one_episode(positions, headings, env.num_prey, np.sqrt(2))

        # print(f"{positions.shape}, {headings.shape}")
        # print(f"Heading: {headings[:, 0, 0]}")
        dos, doa = env.periodic_dos_and_doa(
            # pass it prey positions and headings
            positions[:, env.num_predator:, :],
            # NOTE: agent slices doesn't work
            headings[:, env.num_predator:, :],
            config.episode_length,
            env.num_prey,
            D=np.sqrt(2) * env.unwrapped.L,
            L=env.unwrapped.L
        )

        # print(f"Environment size: {env.unwrapped.L}")
        # print(f"\nEpisode {episode_idx + 1}: DOS={dos:.4f}, DOA={doa:.4f}, Rewards={rewards}")
        metric_row = [episode_idx, dos, doa]
        # Extend the list with reward values
        metric_row.extend(rewards.tolist())
        metrics.append(metric_row)
        # print(f"DOS: {dos:.4f}, DOA: {doa:.4f}")

        # Compute DOS/DOA at each timestep for this episode.
        prey_positions = positions[:, env.num_predator:, :]
        prey_headings = headings[:, env.num_predator:, :]
        for step_idx in range(config.episode_length):
            step_dos, step_doa = env.periodic_dos_and_doa(
                prey_positions[:, :, step_idx:step_idx + 1],
                prey_headings[:, :, step_idx:step_idx + 1],
                1,
                env.num_prey,
                D=np.sqrt(2) * env.unwrapped.L,
                L=env.unwrapped.L
            )
            all_timestep_metrics.append([episode_idx, step_idx, step_dos, step_doa])

        if config.render and len(frames) > 0 and episode_idx % config.video_save_interval == 0:
            video_path = video_dir / f"episode_{episode_idx}.mp4"
            save_episode_video(frames, video_path, fps=config.video_fps)

    # Print training summary
    elapsed_time = time.time() - START_TIME
    print(f"\nTraining completed in {elapsed_time / 60:.2f} minutes")

    # save metrics
    save_dos_and_doa(
        metrics,
        filename=run_dir / 'metrics.csv'
    )

    # Save one CSV containing all episodes and timesteps for DOS/DOA.
    save_timestep_dos_doa(
        all_timestep_metrics,
        filename=str(run_dir / 'timestep_metrics_all.csv')
    )


if __name__ == '__main__':
    parser = create_argument_parser()
    config = parser.parse_args()

    if config.multiple_seeds:
        root_seed = 42
        ss = np.random.SeedSequence(root_seed)

        # Spawn 10 independent child seeds (one for each replicate of your 2^k design)
        child_seeds = ss.spawn(10)

        # To get the actual integer to pass to your MARL env:
        seeds = [s.generate_state(1)[0] for s in child_seeds]
        print(f"Training {len(seeds)} replicates with seeds: {seeds}")

        for seed in seeds:
            print(f"\n=== Starting training with seed {seed} ===")
            config.seed = seed
            run(config)

    else:
        run(config)
