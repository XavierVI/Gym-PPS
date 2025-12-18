import os
import json
import numpy as np
import time
import gym
from gym.wrappers import PredatorPreySwarmCustomizer
from custom_env import MyObs, MyReward, MyAction

"""
This tutorial shows how to use the PredatorPreySwarm (PPS) environment.
"""

## Define the Predator-Prey Swarm (PPS) environment
scenario_name = 'PredatorPreySwarm-v0'  

## customize PPS environment parameters in the .json file
custom_param = 'custom_param.json'      

## Make the environment 
env = gym.make(scenario_name)
custom_param = os.path.dirname(os.path.realpath(__file__)) + '/' + custom_param
env = PredatorPreySwarmCustomizer(env, custom_param)

if __name__ == '__main__':

    n_p = env.get_param('n_p')
    n_e = env.n_e
    n_pe = env.n_pe
    print("Number of predators: ", n_p)
    print("Number of prey: ", n_e)
    print("Number of total agents: ", n_pe)

    s = env.reset()   # (obs_dim, n_peo)
    print("Observation space shape: ", s.shape)
    print("Action space shape: ", env.action_space.shape)
    
    for _ in range(1):
        for step in range(1000):
            env.render( mode='human' )

            # To separately control 
            a_pred = np.random.uniform(-1,1,(2, n_p))       # predator action
            a_prey = np.random.uniform(-1,1,(2, n_e))       # prey action
            a = np.concatenate((a_pred, a_prey), axis=-1)   # total action

            # Sample random actions automatically
            # a = env.action_space.sample()

            s_, r, done, info = env.step(a)                 # state, reward, done, info
            s = s_                                          # update state