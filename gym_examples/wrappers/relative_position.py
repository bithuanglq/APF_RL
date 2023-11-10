import gym
from gym import spaces
import numpy as np


class RelativePosition(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = spaces.Box(shape=(2+25*6,), low=-np.inf, high=np.inf)


    def observation(self, obs):
        return np.concatenate((obs["target"] - obs["agent"], obs["loc_obs"]), axis=0)   # (2+25*6,)

