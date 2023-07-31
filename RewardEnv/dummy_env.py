import pickle
import pathlib
import numpy as np

dir_path = pathlib.Path(__file__).parent.resolve()

## DUMMY ENV
class MineEnv:
    def __init__(self, step_penalty, nav_reward_scale, attack_reward, success_reward):
        self.val = 0
        self._elapsed_steps = 0
        self._episode_len = 500
        with open(dir_path.joinpath("observation_dict.pkl"), 'rb') as f:
            self.obs = pickle.load(f)

    def reset(self):
        self._elapsed_steps = 0
        return self.obs

    def step(self, action):
        self._elapsed_steps += 1
        mu, sigma = 0, 0.01

        obs = self.obs
        reward = 1 + np.random.normal(mu, sigma)
        done = False
        info = None

        return obs, reward, done, info

    def close(self):
        return None