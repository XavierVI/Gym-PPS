import gymnasium as gym
from custom_env import NJPEnvironment
import argparse
import torch
import time
import os
import numpy as np
import imageio
from algorithms.maddpg import MADDPG
from algorithms.mappo import MAPPO
from algorithms import ddpg as ddpg_mod
from pathlib import Path
from tqdm import tqdm
from utils.helpers import (
    create_agent_slices,
    create_replay_buffers,
    create_argument_parser,
    decay_exploration_noise,
    save_dos_and_doa
)
from utils.device_management import move_model_to_device
from utils.helpers import *


"""
python evaluate.py --model_path ./models/eda03/run1/incremental/maddpg_model_ep1951.pt --video_output_dir ./evaluation_videos --video_fps 30 --n_episodes 5 --episode_length 1000 

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
# MODEL LOADING
# ============================================================================

def load_model_checkpoint(model_path, model_type):
    """Load a trained model from checkpoint."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint not found at {model_path}")

    model_type_upper = model_type.upper()

    if model_type_upper == "MADDPG":
        model = MADDPG.init_from_save(model_path)
    elif model_type_upper == "DDPG":
        # Try IDDPG first, then DDPG
        DDPGCls = getattr(ddpg_mod, "IDDPG", None) or getattr(
            ddpg_mod, "DDPG", None)
        if DDPGCls is None:
            raise ImportError(
                "Cannot find IDDPG or DDPG class in algorithms/ddpg.py")
        model = DDPGCls.init_from_save(model_path)
    elif model_type_upper == "MAPPO":
        model = MAPPO.init_from_save(model_path)
    else:
        raise ValueError(f"Unknown model type: {model_type_upper}")

    return model


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
# EVALUATION
# ============================================================================

def run(config):
    """Run evaluation of a trained model."""
    # Setup
    # run_dir, log_dir = create_run_directory(config)
    env = setup_environment(config)
    agent_slices = create_agent_slices(env)
    buffers = create_replay_buffers(env, config, agent_slices)

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    if not USE_CUDA:
        torch.set_num_threads(config.n_training_threads)

    # Load model
    print(f"Loading {config.model_type} model from: {config.model_path}")
    checkpoint_path = os.path.join(config.model_path, 'incremental')
    checkpoint_files = [f for f in os.listdir(
        checkpoint_path) if f.endswith('.pt')]
    if not checkpoint_files:
        raise FileNotFoundError(f"No .pt files found in {checkpoint_path}")
    latest_checkpoint = max(checkpoint_files, key=lambda f: int(
        ''.join(filter(str.isdigit, f))))
    model_checkpoint_path = os.path.join(checkpoint_path, latest_checkpoint)
    print(f"Found checkpoint: {model_checkpoint_path}")
    model = load_model_checkpoint(model_checkpoint_path, config.model_type)
    move_model_to_device(model, DEVICE)

    # Create output directory for videos
    video_dir = Path(
        config.video_output_dir) if config.video_output_dir else Path('.')
    video_dir.mkdir(parents=True, exist_ok=True)
    all_timestep_metrics = []
    metrics = []  # values over each episode

    # Evaluation loop
    for ep_i in tqdm(range(0, config.n_episodes, config.n_rollout_threads)):
        print(f"\rEpisodes {ep_i + 1}-{ep_i + 1 + config.n_rollout_threads} of {config.n_episodes}",
              end='', flush=True)

        obs, _ = env.reset()
        model.prep_rollouts(device=DEVICE)
        # model.scale_noise(model.noise, model.epsilon)
        # model.reset_noise()

        # Initialize storage
        pos_dim, num_agents = np.shape(env.unwrapped.p)
        heading_dim, _ = np.shape(env.unwrapped.heading)

        positions = np.zeros((pos_dim, num_agents, config.episode_length))
        headings = np.zeros((heading_dim, num_agents, config.episode_length))
        frames = []
        agent_rewards = []

        # Run episode
        for step_idx in range(config.episode_length):
            # Capture frame for video

            if config.render:
                frame = env.unwrapped.render(mode='rgb_array')
                frames.append(frame)

            # Store positions and headings
            positions[:, :, step_idx] = env.unwrapped.p
            headings[:, :, step_idx] = env.unwrapped.heading

            # Get action from model
            torch_obs = torch.as_tensor(
                obs, dtype=torch.float32, device=DEVICE)
            torch_actions = model.step(torch_obs, agent_slices, explore=False)
            actions = np.column_stack([ac.data.cpu().numpy()
                                      for ac in torch_actions])

            # Step environment
            next_obs, rewards, dones, infos = env.step(actions)
            agent_rewards.append(rewards)

            # Store in buffers
            buffers[0].push(obs, actions, rewards,
                            next_obs, dones)  # predators
            buffers[1].push(obs, actions, rewards, next_obs, dones)  # prey

            obs = next_obs

        # print(f"{positions.shape}, {headings.shape}")
        # print(f"Heading: {headings[:, 0, 0]}")
        dos, doa = env.periodic_dos_and_doa(
            # pass it prey positions and headings
            positions[:, env.num_predator:, :],
            # NOTE: agent slices doesn't work
            headings[:, env.num_predator:, :],
            config.episode_length,
            env.num_prey,
            # for edge length of 1.0
            1/np.sqrt(2) if env.unwrapped.L == 1.0 else np.sqrt(2),
            # np.sqrt(2),
            L=env.unwrapped.L
        )

        # print(f"Environment size: {env.unwrapped.L}")
        # print(f"\nEpisode {episode_idx + 1}: DOS={dos:.4f}, DOA={doa:.4f}, Rewards={rewards}")
        metric_row = [ep_i, dos, doa]
        # Extend the list with reward values
        # average reward per agent across the episode
        agent_rewards = np.mean(agent_rewards, axis=0).squeeze()
        metric_row.extend(agent_rewards.tolist())
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
                1/np.sqrt(2) if env.unwrapped.L == 1.0 else np.sqrt(2),
                L=env.unwrapped.L
            )
            all_timestep_metrics.append([ep_i, step_idx, step_dos, step_doa])

        # Save video
        if config.render and len(frames) > 0 and ep_i % config.video_save_interval == 0:
            video_path = video_dir / f"episode_{ep_i}.mp4"
            save_episode_video(frames, video_path, fps=config.video_fps)

        # Decay exploration noise
        # decay_exploration_noise(model)

    # Print summary
    elapsed_time = time.time() - START_TIME
    print(f"\nEvaluation completed in {elapsed_time / 60:.2f} minutes")

    # save metrics
    if env.num_predator > 0:
        print("Evaluation with predators completed. Saving metrics...")
        metrics_name = f'eval_metrics_with_predators.csv'
        save_dos_and_doa(
            metrics,
            filename=os.path.join(config.model_path, metrics_name)
        )
        save_timestep_dos_doa(
            all_timestep_metrics,
            filename=os.path.join(
                config.model_path, f'{metrics_name.replace(".csv", "")}_timesteps.csv')
        )
    else:
        print("Zero predators evaluation completed. Saving metrics...")
        metrics_name = f'eval_metrics_zero_predators.csv'
        save_dos_and_doa(
            metrics,
            filename=os.path.join(config.model_path, metrics_name)
        )
        save_timestep_dos_doa(
            all_timestep_metrics,
            filename=os.path.join(
                config.model_path, f'{metrics_name.replace(".csv", "")}_timesteps.csv')
        )

    # Save one CSV containing all episodes and timesteps for DOS/DOA.


if __name__ == '__main__':
    parser = create_argument_parser()

    # Evaluation-specific arguments
    parser.add_argument("--model_type", default="maddpg", type=str,
                        choices=['maddpg', 'mappo', 'ddpg'],
                        help="Type of model to load")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to trained model checkpoint (e.g., ./models/model_1/)")

    config = parser.parse_args()

    if config.multiple_seeds:
        root_seed = 42
        ss = np.random.SeedSequence(root_seed)

        # Spawn 10 independent child seeds (one for each replicate of your 2^k design)
        child_seeds = ss.spawn(10)

        # To get the actual integer to pass to your MARL env:
        seeds = [s.generate_state(1)[0] for s in child_seeds]
        print(f"Training {len(seeds)} replicates with seeds: {seeds}")

        # get run directories for each seed
        run_dirs = os.listdir(config.model_path)
        base_path = config.model_path
        print(run_dirs)

        for seed, run_dir in zip(seeds, run_dirs):
            print(f"\n=== Starting training with seed {seed} ===")
            config.seed = seed
            config.model_path = os.path.join(base_path, run_dir)
            run(config)

    else:
        run(config)
