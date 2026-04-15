#!/bin/bash

#SBATCH --job-name=gym-pps-eval
#SBATCH --output=logs/%x_%j.out
#SBATCH --array=1-3
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

E=100
T=500

if [ $SLURM_ARRAY_TASK_ID -eq 1 ]; then
  echo "Evaluating with 5 prey"
  # zero predators evaluation
  python evaluate.py --multiple_seeds --model_path=./models/training_5_prey_3_fov/ --n_episodes=100 --episode_length=$T --custom_param_name=config/zero_predators_eval/3_fov.json

  python evaluate.py --multiple_seeds --model_path=./models/training_5_prey_8_fov/ --n_episodes=100 --episode_length=$T --custom_param_name=config/zero_predators_eval/8_fov.json
  
  # evaluation with 3 predators
  python evaluate.py --multiple_seeds --model_path=./models/training_5_prey_3_fov/ --n_episodes=100 --episode_length=$T --custom_param_name=config/predators_eval/3_fov.json

  python evaluate.py --multiple_seeds --model_path=./models/training_5_prey_8_fov/ --n_episodes=100 --episode_length=$T --custom_param_name=config/predators_eval/8_fov.json

elif [ $SLURM_ARRAY_TASK_ID -eq 2 ]; then
  echo "Evaluating with 20 prey"
  # evaluation with 0 predators
  python evaluate.py --multiple_seeds --model_path=./models/training_20_prey_3_fov/ --n_episodes=100 --episode_length=$T --custom_param_name=config/zero_predators_eval/3_fov.json

  python evaluate.py --multiple_seeds --model_path=./models/training_20_prey_8_fov/ --n_episodes=100 --episode_length=$T --custom_param_name=config/zero_predators_eval/8_fov.json
  
  # evaluation with 3 predators
  python evaluate.py --multiple_seeds --model_path=./models/training_20_prey_3_fov/ --n_episodes=100 --episode_length=$T --custom_param_name=config/predators_eval/3_fov.json

  python evaluate.py --multiple_seeds --model_path=./models/training_20_prey_8_fov/ --n_episodes=100 --episode_length=$T --custom_param_name=config/predators_eval/8_fov.json
else
  # zero predators evaluation
  python evaluate.py --multiple_seeds --model_path=./models/baseline/ --n_episodes=100 --episode_length=$T --custom_param_name=config/baseline.json --custom_param_name=config/zero_predators_eval/5_fov.json

  # evaluation with 3 predators
  python evaluate.py --multiple_seeds --model_path=./models/baseline/ --n_episodes=100 --episode_length=$T --custom_param_name=config/predators_eval/5_fov.json