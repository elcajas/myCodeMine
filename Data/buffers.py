import numpy as np
import torch
import matplotlib.pyplot as plt

import pickle
import logging
import pathlib
import imageio
from collections import namedtuple, Counter, deque

from mineclip.mineagent.batch import Batch
from .datasets import PPODataset

def discounted_cumsum(data, discount, tmp=0):
    sum = []
    for d in reversed(data):
        tmp = d + discount * tmp
        sum.append(tmp)
    return sum[::-1]

class PPOBuffer:
    def __init__(self, capacity, gamma, lam, n_actions) -> None:
        self.states = [Batch(tmp=np.array([0]))] * capacity
        self.actions = np.zeros(capacity, dtype=np.float32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.log_probs = np.zeros(capacity, dtype=np.float32)
        self.state_values = np.zeros(capacity, dtype=np.float32)
        
        self.frames = []
        self.last_trajectory = Batch(tmp=np.array([0]))
        self.last_episode = []

        self.advantages = np.zeros(capacity, dtype=np.float32)
        self.returns = np.zeros(capacity, dtype=np.float32)

        self.gamma, self.lam = gamma, lam
        self.number_actions = n_actions
        self.pointer, self.path_start_idx, self.max_capacity = 0, 0, capacity 
    
    def store(self, state, action, reward, log_prob, state_value, frame):
        print(self.pointer, self.max_capacity)
        assert self.pointer < self.max_capacity

        self.states[self.pointer] = state
        self.actions[self.pointer] = action
        self.rewards[self.pointer] = reward
        self.log_probs[self.pointer] = log_prob
        self.state_values[self.pointer] = state_value

        self.frames.append(frame)
        self.pointer += 1

    def store_episode(self, obs, reward, done, info):
        self.last_episode.append((obs, reward, done, info))

    def calc_advantages(self, last_val=0):
        path_slice = slice(self.path_start_idx, self.pointer)
        values = np.append(self.state_values[path_slice], last_val)
        rewards = self.rewards[path_slice]
        rewards[:-2] = rewards[2:]
        rewards[-2:] = [0, 0]

        trajectory = Batch.cat(self.states[path_slice])
        trajectory['actions'] = torch.tensor(self.actions[path_slice])
        trajectory['rewards'] = torch.tensor(rewards)

        traj = {'states': self.last_episode.copy()}
        traj['actions'] = torch.tensor(self.actions[path_slice])
        traj['rewards'] = torch.tensor(rewards)


        if trajectory.rewards.sum() >= 200:
            trajectory['successful'] = torch.tensor([1])
            traj['successful'] = torch.tensor([1])
        else:
            trajectory['successful'] = torch.tensor([0])
            traj['successful'] = torch.tensor([0])

        #GAE
        deltas = rewards + self.gamma * values[1:] - values[:-1]
        self.advantages[path_slice] = discounted_cumsum(deltas, self.gamma*self.lam)

        # Rewards-to-go
        self.returns[path_slice] = discounted_cumsum(rewards, self.gamma)
        trajectory['returns'] = torch.tensor(self.returns[path_slice])
        trajectory['advantages'] = torch.tensor(self.advantages[path_slice])
        trajectory['total_return'] = trajectory.rewards.sum().unsqueeze(dim=0)
        traj['total_return'] = trajectory.rewards.sum().unsqueeze(dim=0)
        trajectory['frames'] = self.frames

        self.path_start_idx = self.pointer
        self.last_trajectory = trajectory
        self.frames = []
        self.last_episode = []

        return traj, trajectory

    def get(self, device):
        
        assert self.pointer == self.max_capacity

        self.pointer, self.path_start_idx = 0, 0

        adv_mean, adv_std = np.mean(self.advantages), np.std(self.advantages)
        self.advantages = (self.advantages - adv_mean) / adv_std
        data = dict(actions = self.actions,
                    rewards = self.rewards,
                    log_probs = self.log_probs,
                    state_values = self.state_values,
                    advantages = self.advantages,
                    returns = self.returns)
        
        data_dict = dict(states = Batch.cat(self.states))
        tmp_dict = {k: torch.as_tensor(v, dtype=torch.float32) for k, v in data.items()}
        data_dict.update(tmp_dict)
        return PPODataset(data_dict, device)
    
    def make_video(self, path: pathlib.Path, reward, episode_number):

        video_dir = path.joinpath("videos")
        if not video_dir.exists():
            video_dir.mkdir()
        
        frames = self.last_trajectory.frames
        video_path = video_dir.joinpath(f"video_{episode_number}_{reward:.2f}.mp4")
        writer = imageio.get_writer(video_path, fps=10)
        
        for frame in frames:
            writer.append_data(frame)
        writer.close()

        stat_dir = path.joinpath("stats")
        if not stat_dir.exists():
            stat_dir.mkdir()

        stat_path = stat_dir.joinpath(f"stats_{episode_number}_{reward:.2f}.png")
        # self.plot_statistics(stat_path)
    
    def get_images(self, path: pathlib.Path, reward, episode_number):
        images_dir = path.joinpath("images")
        if not images_dir.exists():
            images_dir.mkdir()

        frames = self.last_trajectory.frames
        actions = self.last_trajectory.actions
        rewards = self.last_trajectory.rewards

        indices = (rewards > 0).nonzero(as_tuple=True)[0]
        l_ind = len(indices)
        number_images = 5
        r = number_images // 2

        fig, ax = plt.subplots(l_ind, number_images, figsize=(number_images*10, l_ind*7))
        fig.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1, wspace=0.02, hspace=0.07)
        for i, ind in enumerate(indices):
            for c in range(number_images):
                n = ind+c-r
                subplot_index = c if l_ind == 1 else (i, c)
                if n > len(rewards): continue
                if n == len(rewards):
                    ax[subplot_index].imshow(frames[n])
                else:
                    ax[subplot_index].imshow(frames[n])
                    ax[subplot_index].set_title(f'frame: {n}, reward: {rewards[n]}, action: {actions[n]}', fontsize=25)

            # Turn off the axes for the current subplot
            if l_ind == 1:
                for a in ax:
                    a.axis('off')
            else:
                for a in ax[i]:
                    a.axis('off')

        img_path = images_dir.joinpath(f'episode_{episode_number}_{reward:.2f}.png')
        fig.savefig(img_path)
        plt.close(fig)

    def plot_statistics(self, path:pathlib.Path):
        
        c = Counter(self.last_trajectory.actions.numpy())
        d = np.array(list(c.items()))

        fig, ax = plt.subplots(figsize=(10,4))
        ax.stem(d[:,0], d[:,1], markerfmt="", basefmt="-b")  # Plot only positive frequencies
        ax.set_xlabel('action number')
        ax.set_ylabel('frecuency')
        ax.set_xlim(-1, self.number_actions+1)
        ax.set_xticks(np.arange(0, self.number_actions, 5))

        ax.grid(which='major', color='#DDDDDD', linewidth=0.8)
        ax.grid(which='minor', color='#EEEEEE', linestyle=':', linewidth=0.6)
        ax.minorticks_on()
        fig.savefig(path)
        plt.close(fig)
    
    def plot_probs(self, probs, path:pathlib.Path):

        probs = probs.detach().cpu()

        fig, ax = plt.subplots(figsize=(10,4))
        ax.stem(probs*100, markerfmt="", basefmt="-b")  # Plot only positive frequencies
        ax.set_xlabel('action number')
        ax.set_ylabel('probability')
        ax.set_xlim(-1, self.number_actions+1)
        ax.set_xticks(np.arange(0, self.number_actions, 5))

        ax.grid(which='major', color='#DDDDDD', linewidth=0.8)
        ax.grid(which='minor', color='#EEEEEE', linestyle=':', linewidth=0.6)
        ax.minorticks_on()
        fig.savefig(path)
        plt.close(fig)

    def save_data(self, path: pathlib.Path):
        path = path.joinpath('episode_data.pkl')
        with open(path, 'wb') as f:
            pickle.dump(self, f)


class SIBuffer:
    def __init__(self, capacity, mean, std, delt) -> None:
        self.trajectories = [Batch(tmp=np.array([0]))] * capacity
        self.total_rewards = np.zeros(capacity, dtype=np.float32)
        self.probs = np.zeros(capacity, dtype=np.float32)

        self.delt = delt
        self.mean, self.std = mean, std
        self.traj_number, self.step_pos, self.max_capacity = 0, 0, capacity

    def store(self, trajectory):
        total_rew = trajectory.rewards.sum().item()
        threshold = self.mean + self.delt * self.std

        if self.traj_number < self.max_capacity:
            if trajectory.successful == 1 or (total_rew > 0 and total_rew >= threshold):
                self.trajectories[self.traj_number] = trajectory
                self.total_rewards[self.traj_number] = total_rew
                self.traj_number += 1
                self.step_pos += len(trajectory.actions)

        else:
            pos = np.argmin(self.total_rewards)
            if trajectory.successful == 1 or (total_rew > 0 and total_rew >= threshold):
                self.trajectories[pos] = trajectory
                self.total_rewards[pos] = total_rew
    
    def sample(self):

        # Update mean and std
        self.mean = self.total_rewards[:self.traj_number].mean()
        self.std = np.std(self.total_rewards[:self.traj_number])
        thr = self.mean + self.delt * self.std
        logging.info(f" --- mean: {self.mean:.2f}, std: {self.std:.2f}, threshold: {(thr):.2f}, num_traj: {self.traj_number} ---")

        return self.trajectories[:self.traj_number]