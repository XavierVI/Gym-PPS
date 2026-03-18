

## Training Commands

```bash
python train.py --seed=42 --env_id=training_10_prey --n_episodes=2001 --episode_length=200 --hidden_dim=128 --save_interval=500


python train.py --seed=42 --env_id=training_5_prey --n_episodes=2001 --episode_length=200 --hidden_dim=128 --save_interval=500


python train.py --seed=42 --env_id=training_20_prey --n_episodes=2001 --episode_length=200 --hidden_dim=128 --save_interval=500
```


## Evaluation Commands

```bash
python train.py --seed=42 --env_id=eval_10_prey --model_path=./training_10_prey/run_n --n_episodes=100 --episode_length=200


python train.py --seed=42 --env_id=eval_5_prey --n_episodes=100 --episode_length=200 --model_path=./training_5_prey/run_n


python train.py --seed=42 --env_id=eval_20_prey --n_episodes=100 --episode_length=200 --model_path=./training_20_prey/run_n
```