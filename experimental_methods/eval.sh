#!/bin/bash

#SBATCH --job-name=gym-pps-eval
#SBATCH --output=logs/%x_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=48:00:00
#SBATCH --partition=h100
#SBATCH --gpus-per-node=1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=xbarr@unm.edu

source /users/xbarr/python-venvs/pytorch-gpu/bin/activate
echo "Starting job $SLURM_JOB_ID on $SLURM_JOB_NODELIST"

PREY=$1

if [ $PREY -eq 5 ]; then
  echo "Training with 5 prey"
  python evaluate.py --multiple_seeds --model_path=./models/training_5_prey_3_fov/ --n_episodes=100 --episode_length=100 --custom_param_name=config/training_5_prey_3_fov.json

  python evaluate.py --multiple_seeds --model_path=./models/training_5_prey_8_fov/ --n_episodes=100 --episode_length=100 --custom_param_name=config/training_5_prey_8_fov.json

elif [ $PREY -eq 20 ]; then
  echo "Training with 20 prey"
  python evaluate.py --multiple_seeds --model_path=./models/training_20_prey_3_fov/ --n_episodes=100 --episode_length=100 --custom_param_name=config/training_20_prey_3_fov.json

  python evaluate.py --multiple_seeds --model_path=./models/training_20_prey_8_fov/ --n_episodes=100 --episode_length=100 --custom_param_name=config/training_20_prey_8_fov.json
else
  python evaluate.py --multiple_seeds --model_path=./models/baseline/ --n_episodes=100 --episode_length=100 --custom_param_name=config/baseline.json
fi