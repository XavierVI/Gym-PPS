import os
import json
from typing import Any
import gymnasium as gym
import argparse
from abc import abstractmethod


class PredatorPreySwarmCustomizer(gym.Wrapper):

    def __init__(self, env, args):
        super(PredatorPreySwarmCustomizer, self).__init__(env)
        if isinstance(args, argparse.Namespace):
            args_ = vars(args).items()
        elif isinstance(args, dict):
            args_ = args.items()
        elif isinstance(args, str):
            print(f"Retrieving customized param from '{args}'")
            with open(args, "r") as file:
                args_ = json.load(file).items()
        else:
            raise ValueError(
                "Invalid argument type. Parameters must be a dictionary, or argparse.Namespace, or a json file directory")
        for attr, value in args_:
            self.set_param(attr, value)
        self.__reinit__()
        print('Environment parameter customization finished.')

    def set_param(self, name: str, value: Any) -> None:
        if name not in self.env.param_list:
            raise KeyError(f"Parameter '{name}' does not exist!"
                           )
        setattr(self.env, name, value)
        self.__reinit__()

    def get_param(self, name: str) -> Any:
        if name not in self.env.param_list:
            raise KeyError(f"Parameter '{name}' does not exist!"
                           )
        return getattr(self.env, name)

    def __reinit__(self):
        self.env.__reinit__()
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space


class CustomObservation(PredatorPreySwarmCustomizer):
    def reset(self, **kwargs):
        observation = self.env.reset(**kwargs)
        return self.observation(observation)

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        return self.observation(observation), reward, done, info

    @abstractmethod
    def observation(self, observation):
        raise NotImplementedError


class CustomReward(PredatorPreySwarmCustomizer):
    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        return observation, self.reward(observation, reward, action), done, info

    @abstractmethod
    def reward(self, observation, reward, action):
        raise NotImplementedError


class CustomAction(PredatorPreySwarmCustomizer):
    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def step(self, action):
        return self.env.step(self.action(action))

    @abstractmethod
    def action(self, action):
        raise NotImplementedError

    @abstractmethod
    def reverse_action(self, action):
        raise NotImplementedError
