import numpy as np
import os
from utils.buffer import ReplayBuffer
from pathlib import Path

import argparse

# ============================================================================
# SAVING DATA
# ============================================================================
def save_model_checkpoint(model, model_name, run_dir, episode_idx, config):
    """Save model checkpoint if save interval reached."""
    if episode_idx % config.save_interval < config.n_rollout_threads:
        checkpoint_dir = run_dir / 'incremental'
        os.makedirs(checkpoint_dir, exist_ok=True)

        model.save(checkpoint_dir / f'{model_name}_model_ep{episode_idx + 1}.pt')


def save_dos_and_doa(dos_and_doa_vals, filename):
    """
    Saves the DoS and DoA to a simple CSV file.

    NOTE: the file is saved in models/model_i/runi
    """
    # convert to a numpy array
    dos_and_doa_vals = np.array(dos_and_doa_vals)
    header = "episode,dos,doa," + \
        ",".join([f"reward_{i}" for i in range(dos_and_doa_vals.shape[1] - 3)])
    # save to CSV
    np.savetxt(filename, dos_and_doa_vals, delimiter=",", fmt="%.6f",
               header=header)


def create_run_directory(config):
    """Create the run directory and return the path."""
    model_dir = Path('./models') / config.env_id / config.algo

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



def decay_exploration_noise(model):
    """Decay exploration noise over time."""
    model.noise = max(0.05, model.noise - 5e-5)
    model.epsilon = max(0.05, model.epsilon - 5e-5)


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


def create_agent_slices(env):
    """Create slices to separate predators from prey in agent list."""
    predator_slice = slice(0, env.num_predator)
    prey_slice = slice(env.num_predator, env.num_predator + env.num_prey)
    return [predator_slice, prey_slice]


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
    parser.add_argument("--algo", default="maddpg", type=str,
                        choices=['maddpg', 'mappo', 'ddpg'],
                        help="Algorithm to train")
    # NOTE: only have to keep these so MADDPG code doesn't break
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


    parser.add_argument("--n_updates_per_episode", default=50, type=int)

    # Exploration decay
    parser.add_argument("--min_noise", default=0.05, type=float)
    parser.add_argument("--min_epsilon", default=0.05, type=float)
    parser.add_argument("--noise_decay", default=5e-5, type=float)
    parser.add_argument("--epsilon_decay", default=5e-5, type=float)
    
    return parser