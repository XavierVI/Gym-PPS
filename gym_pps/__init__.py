from gymnasium.envs.registration import register

register(
    id='PredatorPreySwarm-v0',
    entry_point='gym_pps.pps:PredatorPreySwarmEnv',
)
