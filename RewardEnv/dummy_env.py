import pickle
import pathlib
import numpy as np

dir_path = pathlib.Path(__file__).parent.resolve()

## DUMMY ENV
class MineEnv:
    def __init__(self, step_penalty, nav_reward_scale, attack_reward, success_reward):
        self.val = 0
        self._elapsed_steps = 0
        with open(dir_path.joinpath("episode_dict.pkl"), 'rb') as f:
            self.episode = pickle.load(f)
        self._episode_len = len(self.episode['states'])

    def reset(self):
        self._elapsed_steps = 0
        return self.episode['states'][0][0]

    def step(self, action):
        # mu, sigma = 0, 0.01

        # obs = self.obs
        # reward = 1 + np.random.normal(mu, sigma)
        # if action.item() == 1:
        #     reward = 1
        # else:
        #     reward = 0
        # done = False
        # info = None
        obs = self.episode['states'][self._elapsed_steps+1]
        _, reward, done, info = self.episode['states'][self._elapsed_steps]
        self._elapsed_steps += 1

        return obs, reward, done, info

    def close(self):
        return None