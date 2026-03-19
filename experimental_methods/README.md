

## Training Commands

```bash
python train.py --seed=42 --env_id=training_10_prey --n_episodes=2001 --episode_length=100 --hidden_dim=64 --save_interval=500


python train.py --seed=42 --env_id=training_5_prey --n_episodes=2001 --episode_length=100 --hidden_dim=64 --save_interval=500


python train.py --seed=42 --env_id=training_20_prey --n_episodes=2001 --episode_length=100 --hidden_dim=64 --save_interval=500
```


## Evaluation Commands

```bash
python evaluate.py --seed=42 --model_path=./models/training_10_prey/run_n --n_episodes=100 --episode_length=100


```