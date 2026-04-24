#!/bin/bash
#SBATCH --job-name=gym-pps-eval
#SBATCH --output=logs/%x_%j.out
#SBATCH --array=1-3
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=48:00:00
#SBATCH --partition=l40s
#SBATCH --gpus-per-node=1
#SBATCH --mem=32G
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=xbarr@unm.edu

source /users/xbarr/python-venvs/pytorch-gpu/bin/activate
echo "Starting job $SLURM_JOB_ID on $SLURM_JOB_NODELIST"

E=100
T=500

if [ $SLURM_ARRAY_TASK_ID -eq 1 ]; then
  echo "Evaluating with 5 prey"
  # zero predators evaluation
  xvfb-run python evaluate.py --multiple_seeds --model_path=./models/training_5_prey_0-707_fov/ --n_episodes=$E --episode_length=$T --custom_param_name=config/zero_predators_eval/0-707_fov.json --render --video_output_dir=./models/training_5_prey_0-707_fov/no_predators_videos/

  xvfb-run python evaluate.py --multiple_seeds --model_path=./models/training_5_prey_2-828_fov/ --n_episodes=$E --episode_length=$T --custom_param_name=config/zero_predators_eval/2-828_fov.json --render --video_output_dir=./models/training_5_prey_2-828_fov/no_predators_videos/
  
  # evaluation with 3 predators
  xvfb-run python evaluate.py --multiple_seeds --model_path=./models/training_5_prey_0-707_fov/ --n_episodes=$E --episode_length=$T --custom_param_name=config/predators_eval/0-707_fov.json --render --video_output_dir=./models/training_5_prey_0-707_fov/predators_videos/

  xvfb-run python evaluate.py --multiple_seeds --model_path=./models/training_5_prey_2-828_fov/ --n_episodes=$E --episode_length=$T --custom_param_name=config/predators_eval/2-828_fov.json --render --video_output_dir=./models/training_5_prey_2-828_fov/predators_videos/

elif [ $SLURM_ARRAY_TASK_ID -eq 2 ]; then
  echo "Evaluating with 20 prey"
  # evaluation with 0 predators
  xvfb-run python evaluate.py --multiple_seeds --model_path=./models/training_20_prey_0-707_fov/ --n_episodes=$E --episode_length=$T --custom_param_name=config/zero_predators_eval/0-707_fov.json --render --video_output_dir=./models/training_20_prey_0-707_fov/no_predators_videos/

  xvfb-run python evaluate.py --multiple_seeds --model_path=./models/training_20_prey_2-828_fov/ --n_episodes=$E --episode_length=$T --custom_param_name=config/zero_predators_eval/2-828_fov.json --render --video_output_dir=./models/training_20_prey_2-828_fov/no_predators_videos/
  
  # evaluation with 3 predators
  xvfb-run python evaluate.py --multiple_seeds --model_path=./models/training_20_prey_0-707_fov/ --n_episodes=$E --episode_length=$T --custom_param_name=config/predators_eval/0-707_fov.json --render --video_output_dir=./models/training_20_prey_0-707_fov/predators_videos/

  xvfb-run python evaluate.py --multiple_seeds --model_path=./models/training_20_prey_2-828_fov/ --n_episodes=$E --episode_length=$T --custom_param_name=config/predators_eval/2-828_fov.json --render --video_output_dir=./models/training_20_prey_2-828_fov/predators_videos/
else
  # zero predators evaluation
  xvfb-run python evaluate.py --multiple_seeds --model_path=./models/baseline/ --n_episodes=$E --episode_length=$T --custom_param_name=config/baseline.json --custom_param_name=config/zero_predators_eval/2-828_fov.json --render --video_output_dir=./models/baseline/no_predators_videos/

  # evaluation with 3 predators
  xvfb-run python evaluate.py --multiple_seeds --model_path=./models/baseline/ --n_episodes=$E --episode_length=$T --custom_param_name=config/predators_eval/2-828_fov.json --render --video_output_dir=./models/baseline/predators_videos/

fi