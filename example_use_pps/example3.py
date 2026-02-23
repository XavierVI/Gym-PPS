import os
import json
import numpy as np
import time
import gymnasium as gym
import gym_pps

from gym_pps.wrappers import PredatorPreySwarmCustomizer
from custom_env import MyObs, MyReward, MyAction, MyEnv

"""
This tutorial shows how to customize the environment class in the PredatorPreySwarm (PPS) environment.
Use the following wrapper to customize the environment class:
- MyEnv
"""

## Define the Predator-Prey Swarm (PPS) environment
scenario_name = 'PredatorPreySwarm-v0'  

## customize PPS environment parameters in the .json file
custom_param = 'custom_param.json'      

## Make the environment 
env = gym.make(scenario_name)
custom_param = os.path.dirname(os.path.realpath(__file__)) + '/' + custom_param
env = PredatorPreySwarmCustomizer(env, custom_param)

## Use the following wrapper to customize the environment class
env = MyEnv(env, custom_param)

if __name__ == '__main__':

    n_p = env.get_param('n_p')
    n_e = env.unwrapped.n_e
    n_pe = env.unwrapped.n_pe
    print("Number of predators: ", n_p)
    print("Number of prey: ", n_e)
    print("Number of total agents: ", n_pe)

    s, _ = env.reset()   # (obs_dim, n_peo)
    print("Observation space shape: ", s.shape)
    print("Action space shape: ", env.action_space.shape)
    
    for _ in range(1):
        for step in range(1000):
            env.render()

            # To separately control 
            a_pred = np.random.uniform(-1,1,(2, n_p)) 
            a_prey = np.random.uniform(-1,1,(2, n_e))
            a = np.concatenate((a_pred, a_prey), axis=-1)

            speed = env.compute_speed()
            print("speed: ", speed)

            s_, r, done, info = env.step(a)
            s = s_