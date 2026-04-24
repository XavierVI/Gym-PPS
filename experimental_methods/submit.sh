#!/bin/bash

#SBATCH --job-name=gym-pps
#SBATCH --output=logs/%x_%j.out
#SBATCH --array=1-3
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=48:00:00
#SBATCH --partition=l40s
#SBATCH --mem=32G
#SBATCH --gpus-per-node=1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=xbarr@unm.edu

source /users/xbarr/python-venvs/pytorch-gpu/bin/activate
echo "Starting job $SLURM_JOB_ID on $SLURM_JOB_NODELIST"

E=500
T=500

if [ $SLURM_ARRAY_TASK_ID -eq 1 ]; then
  echo "Training with 5 prey"
  xvfb-run python -u train.py --multiple_seeds --env_id=training_5_prey_0-707_fov --n_episodes=$E --episode_length=$T --hidden_dim=64 --save_interval=50 --custom_param_name=config/training_5_prey_0-707_fov.json --render

  xvfb-run python -u train.py --multiple_seeds --env_id=training_5_prey_2-828_fov --n_episodes=$E --episode_length=$T --hidden_dim=64 --save_interval=50 --custom_param_name=config/training_5_prey_2-828_fov.json --render

elif [ $SLURM_ARRAY_TASK_ID -eq 2 ]; then
  echo "Training with 20 prey"
  xvfb-run python -u train.py --multiple_seeds --env_id=training_20_prey_0-707_fov --n_episodes=$E --episode_length=$T --hidden_dim=64 --save_interval=50 --custom_param_name=config/training_20_prey_0-707_fov.json --render

  xvfb-run python -u train.py --multiple_seeds --env_id=training_20_prey_2-828_fov --n_episodes=$E --episode_length=$T --hidden_dim=64 --save_interval=50 --custom_param_name=config/training_20_prey_2-828_fov.json --render
else
  xvfb-run python -u train.py --multiple_seeds --env_id=baseline --n_episodes=$E --episode_length=$T --hidden_dim=64 --save_interval=50 --custom_param_name=config/baseline.json --render
fi