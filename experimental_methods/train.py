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
    custom_param_path = os.path.dirname(os.path.realpath(__file__)) + '/' + custom_param_name
    environment = NJPEnvironment(environment, custom_param_path)
    
    return environment


# ============================================================================
# DEVICE MANAGEMENT
# ============================================================================

def move_model_to_device(maddpg, device):
    """Move all model modules to the same device (fix cpu/cuda mismatch)."""
    model_attributes = ["policy", "critic", "target_policy", "target_critic",
                        "actor", "actor_target", "critic_target"]
    
    def _move_attribute(obj, dev):
        """Move an object to a device if it has the 'to' method."""
        if obj is None:
            return
        if hasattr(obj, "to"):
            try:
                obj.to(dev)
            except Exception:
                pass

    # Move top-level model attributes
    for attr in model_attributes:
        _move_attribute(getattr(maddpg, attr, None), device)

    # Move agent attributes
    for agent in getattr(maddpg, "agents", []):
        for attr in model_attributes:
            _move_attribute(getattr(agent, attr, None), device)

    # Print device verification
    for agent in getattr(maddpg, "agents", []):
        for attr in ["policy", "actor", "critic"]:
            model = getattr(agent, attr, None)
            if model is not None and hasattr(model, "parameters"):
                try:
                    print(f"First agent {attr} param device: {next(model.parameters()).device}")
                    return
                except StopIteration:
                    pass


# ============================================================================
# TRAINING SETUP
# ============================================================================

def create_run_directory(config):
    """Create the run directory and return the path."""
    model_dir = Path('./models') / config.env_id
    
    if not model_dir.exists():
        current_run = 'run1'
    else:
        existing_run_numbers = [
            int(str(folder.name).split('run')[1])
            for folder in model_dir.iterdir()
            if str(folder.name).startswith('run')
        ]
        current_run = f"run{max(existing_run_numbers) + 1}" if existing_run_numbers else 'run1'
    
    run_dir = model_dir / current_run
    log_dir = run_dir / 'logs'
    os.makedirs(log_dir, exist_ok=True)
    
    return run_dir, log_dir


def initialize_seeds_and_threads(config):
    """Set random seeds and thread count."""
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    if not USE_CUDA:
        torch.set_num_threads(config.n_training_threads)


def create_agent_slices(env):
    """Create slices to separate predators from prey in agent list."""
    predator_slice = slice(0, env.num_predator)
    prey_slice = slice(env.num_predator, env.num_predator + env.num_prey)
    return [predator_slice, prey_slice]


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


def create_replay_buffers(env, config, agent_slices):
    """Create replay buffers for predators and prey."""
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    predator_buffer = ReplayBuffer(
        config.buffer_length,
        env.num_predator,
        state_dim=state_dim,
        action_dim=action_dim,
        start_stop_index=agent_slices[0]
    )
    
    prey_buffer = ReplayBuffer(
        config.buffer_length,
        env.num_prey,
        state_dim=state_dim,
        action_dim=action_dim,
        start_stop_index=agent_slices[1]
    )
    
    return [predator_buffer, prey_buffer]


# ============================================================================
# EPISODE EXECUTION
# ============================================================================

def run_episode(env, maddpg, config, agent_slices, buffers):
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


def decay_exploration_noise(maddpg):
    """Decay exploration noise over time."""
    maddpg.noise = max(0.05, maddpg.noise - 5e-5)
    maddpg.epsilon = max(0.05, maddpg.epsilon - 5e-5)


def save_model_checkpoint(maddpg, run_dir, episode_idx, config):
    """Save model checkpoint if save interval reached."""
    if episode_idx % config.save_interval < config.n_rollout_threads:
        checkpoint_dir = run_dir / 'incremental'
        os.makedirs(checkpoint_dir, exist_ok=True)
        maddpg.save(checkpoint_dir / f'model_ep{episode_idx + 1}.pt')

def save_dos_and_doa(dos_and_doa_vals, filename):
    # convert to a numpy array
    dos_and_doa_vals = np.array(dos_and_doa_vals)
    # save to CSV
    np.savetxt(filename, dos_and_doa_vals, delimiter=",", fmt="%.6f",
               header="episode,dos,doa")


# ============================================================================
# MAIN TRAINING LOOP
# ============================================================================

def run(config):
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
        positions, headings, num_agents, position_dims = run_episode(
            env, maddpg, config, agent_slices, buffers
        )
        
        # Train on collected data
        train_maddpg_model(maddpg, config, buffers)
        
        # Decay exploration noise
        decay_exploration_noise(maddpg)
        
        # Save checkpoint
        save_model_checkpoint(maddpg, run_dir, episode_idx, config)
        
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


# ============================================================================
# ARGUMENT PARSING
# ============================================================================

def create_argument_parser():
    """Create and return the argument parser."""
    parser = argparse.ArgumentParser(
        description="MADDPG training for Predator-Prey Swarm environment"
    )
    
    # Model and environment config
    parser.add_argument("--env_id", default="model_1", type=str,
                        help="Environment ID for model directory")
    parser.add_argument("--seed", default=0, type=int,
                        help="Random seed for reproducibility")
    
    # Training parameters
    parser.add_argument("--n_episodes", default=201, type=int,
                        help="Total number of episodes to train")
    parser.add_argument("--episode_length", default=200, type=int,
                        help="Steps per episode")
    parser.add_argument("--n_rollout_threads", default=1, type=int,
                        help="Number of parallel rollout threads")
    parser.add_argument("--n_training_threads", default=10, type=int,
                        help="Number of training threads (CPU only)")
    
    # Replay buffer and batch config
    parser.add_argument("--buffer_length", default=int(5e5), type=int,
                        help="Maximum replay buffer size")
    parser.add_argument("--batch_size", default=256, type=int,
                        help="Batch size for training")
    
    # Neural network config
    parser.add_argument("--hidden_dim", default=128, type=int,
                        help="Hidden dimension for neural networks")
    
    # Learning rates
    parser.add_argument("--lr_actor", default=1e-4, type=float,
                        help="Actor learning rate")
    parser.add_argument("--lr_critic", default=1e-3, type=float,
                        help="Critic learning rate")
    
    # Exploration and noise config
    parser.add_argument("--epsilon", default=0.1, type=float,
                        help="Initial exploration epsilon")
    parser.add_argument("--noise", default=0.1, type=float,
                        help="Initial action noise scale")
    parser.add_argument("--tau", default=0.01, type=float,
                        help="Target network update rate")
    
    # Algorithm config
    parser.add_argument("--agent_alg", default="MADDPG", type=str,
                        choices=['MADDPG', 'DDPG'],
                        help="Algorithm for prey agents")
    parser.add_argument("--adversary_alg", default="MADDPG", type=str,
                        choices=['MADDPG', 'DDPG'],
                        help="Algorithm for predator agents")
    
    # Checkpoint saving
    parser.add_argument("--save_interval", default=50, type=int,
                        help="Save model every N episodes")
    
    # Visualization
    parser.add_argument("--render", action="store_true",
                        help="Enable env.render(). On headless servers leave this OFF, or run with xvfb-run.")
    parser.add_argument("--render_interval", default=20, type=int,
                        help="Render every N episodes (only when --render is set)")
    
    # Unused parameters (kept for backwards compatibility)
    parser.add_argument("--n_exploration_eps", default=25000, type=int, help="(Unused)")
    parser.add_argument("--init_noise_scale", default=0.3, type=float, help="(Unused)")
    parser.add_argument("--final_noise_scale", default=0.0, type=float, help="(Unused)")
    
    return parser


if __name__ == '__main__':
    parser = create_argument_parser()
    config = parser.parse_args()
    run(config)
