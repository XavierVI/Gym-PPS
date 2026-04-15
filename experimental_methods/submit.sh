#!/bin/bash

#SBATCH --job-name=gym-pps
#SBATCH --output=logs/%x_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=48:00:00
#SBATCH --partition=general
#SBATCH --mem=32G
#SBATCH --gpus-per-node=1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=xbarr@unm.edu

source /users/xbarr/python-venvs/pytorch-gpu/bin/activate
echo "Starting job $SLURM_JOB_ID on $SLURM_JOB_NODELIST"

PREY=$1
T=200

if [ $PREY -eq 5 ]; then
  echo "Training with 5 prey"
  python -u train.py --multiple_seeds --env_id=training_5_prey_3_fov --n_episodes=2000 --episode_length=$T --hidden_dim=64 --save_interval=500 --custom_param_name=config/training_5_prey_3_fov.json

  python -u train.py --multiple_seeds --env_id=training_5_prey_8_fov --n_episodes=2000 --episode_length=$T --hidden_dim=64 --save_interval=500 --custom_param_name=config/training_5_prey_8_fov.json

elif [ $PREY -eq 20 ]; then
  echo "Training with 20 prey"
  python -u train.py --multiple_seeds --env_id=training_20_prey_3_fov --n_episodes=2000 --episode_length=$T --hidden_dim=64 --save_interval=500 --custom_param_name=config/training_20_prey_3_fov.json

  python -u train.py --multiple_seeds --env_id=training_20_prey_8_fov --n_episodes=2000 --episode_length=$T --hidden_dim=64 --save_interval=500 --custom_param_name=config/training_20_prey_8_fov.json
else
  python -u train.py --multiple_seeds --env_id=baseline --n_episodes=2000 --episode_length=$T --hidden_dim=64 --save_interval=500 --custom_param_name=config/baseline.json
fi